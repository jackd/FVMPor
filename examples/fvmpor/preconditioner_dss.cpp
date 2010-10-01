#include "preconditioner_dss.h"
#include <fvm/solver.h>

#include <mkl.h>
#include <set>
#include <map>
#include <utility>
#include <limits>

namespace fvmpor {

template<typename T>
struct block_traits {
    enum {blocksize = T::variables};
};
template<>
struct block_traits<double> {
    enum {blocksize = 1};
};

std::set<int> direct_neighbours(const mesh::Mesh& m, int node_id)
{
    std::set<int> neighbours;
    const mesh::Volume& v = m.volume(node_id);
    for (int i = 0; i < v.scvs(); ++i) {
        const mesh::Element& e = v.scv(i).element();
        for (int j = 0; j < e.nodes(); ++j) {
            //neighbours.insert(e.node(j).id());
            if(e.node(j).id()<m.local_nodes())
                neighbours.insert(e.node(j).id());
        }
    }
    return neighbours;
}

std::vector<int> dependent_columns(const mesh::Mesh& m, int node_id, int blocksize)
{
    std::set<int> columnset;
    std::set<int> neighbours = direct_neighbours(m, node_id);
    std::set<int>::iterator it = neighbours.begin();
    std::set<int>::iterator end = neighbours.end();
    for (; it != end; ++it) {
        if (*it < m.local_nodes()) {
            std::set<int> neighbours2 = direct_neighbours(m, *it);
            columnset.insert(neighbours2.begin(), neighbours2.end());
        }
    }
    // Adjust for blocksize
    std::vector<int> columns;
    it = columnset.begin();
    end = columnset.end();
    for (; it != end; ++it) {
        for (int i = 0; i < blocksize; ++i) {
            columns.push_back(*it * blocksize + i);
        }
    }

    return columns;
}

std::vector<int> sequential_vertex_colouring(const mesh::Mesh& m, int blocksize)
{
    //int N = m.nodes() * blocksize;
    int N = m.local_nodes() * blocksize;

    int max_colour = 0;
    std::vector<int> mark(N, -1);
    std::vector<int> colour(N, N-1);

    for (int i = 0; i < N; ++i) {

        int current = i;    // could use a permutation here: current = perm(i)

        std::vector<int> adjacent = dependent_columns(m, current/blocksize, blocksize);

        int asize = adjacent.size();
        for (int j = 0; j < asize; ++j) {
            mark[colour[adjacent[j]]] = i;
        }
        
        int smallest_colour = 0;
        while (smallest_colour < max_colour && mark[smallest_colour] == i) {
            ++smallest_colour;
        }
        
        if (smallest_colour == max_colour) {
            ++max_colour;
        }
        
        colour[current] = smallest_colour;
    }

    return colour;
}

typedef std::set< std::pair<int,int> > MatrixPattern;

void insert_block(MatrixPattern& matpat, int row, int col, int blocksize)
{
    for (int i = 0; i < blocksize; ++i) {
        for (int j = 0; j < blocksize; ++j) {
            matpat.insert(std::make_pair(blocksize*row + i, blocksize*col + j));
        }
    }
}

ColumnPattern column_pattern(const mesh::Mesh& m, int blocksize)
{
    // Determine the set of (i,j) coordinates
    MatrixPattern matpat;
    for (int i = 0; i < m.elements(); ++i) {
        const mesh::Element& e = m.element(i);
        for (int j = 0; j < e.nodes(); ++j) {
            const mesh::Node& n = e.node(j);
            if (n.id() < m.local_nodes()) {
                for (int k = 0; k < e.nodes(); ++k) {
                    const mesh::Node& nb = e.node(k);
                    if (nb.id() < m.local_nodes()) {
                        insert_block(matpat, n.id(), nb.id(), blocksize);
                    }
                }
            }
        }
    }

    // Create the column patterns
    ColumnPattern pat;

    MatrixPattern::iterator it = matpat.begin();
    MatrixPattern::iterator end = matpat.end();
    for (; it != end; ++it) {
        int j = it->first;
        if (j == pat.size()) {
            pat.push_back(std::vector<int>(1, it->second));
        } else {
            pat.back().push_back(it->second);
        }
    }

    return pat;

}

void Preconditioner::initialise(const mesh::Mesh& m)
{
    blocksize = block_traits<Physics::value_type>::blocksize;
    N = m.local_nodes() * blocksize;
    shift.resize(N);

    // Create DSS data structure
    int opt = MKL_DSS_MSG_LVL_WARNING + MKL_DSS_TERM_LVL_ERROR;
    int flag = dss_create(dss_handle, opt);
    assert(flag == MKL_DSS_SUCCESS);

    // Determine sparsity pattern
    pat = column_pattern(m, blocksize);
    colourvec = sequential_vertex_colouring(m, blocksize);
    assert(colourvec.size() == N);
    num_colours = *std::max_element(colourvec.begin(), colourvec.end()) + 1;

    // Create CSR row and column arrays
    row_index.resize(N+1);
    row_index[0] = 1;   // 1-based indexing
    for (int i = 0; i < N; ++i) {
        row_index[i+1] = row_index[i] + pat[i].size();
        for (int j = 0; j < pat[i].size(); ++j) {
            columns.push_back(pat[i][j] + 1);   // 1-based indexing
        }
    }
    nnz = columns.size();
    values.resize(nnz);

    // Define DSS matrix structure
    opt = MKL_DSS_SYMMETRIC_STRUCTURE;
    flag = dss_define_structure(
        dss_handle, opt, &row_index[0], N, N, &columns[0], nnz);
    assert(flag == MKL_DSS_SUCCESS);

    // Reorder DSS matrix
    opt = MKL_DSS_AUTO_ORDER;
    flag = dss_reorder(dss_handle, opt, 0);
    assert(flag == MKL_DSS_SUCCESS);
}

int Preconditioner::setup(
    const mesh::Mesh& m, double tt, double c, double h,
    const_iterator residual, const_iterator weights,
    iterator sol, iterator derivative,
    iterator temp1, iterator temp2, iterator temp3,
    Callback compute_residual)
{
    ++num_setups;

    if (pat.size() == 0) initialise(m);

    // Save original values
    std::copy(sol, sol + m.local_nodes(), temp1);
    double* sol_vec   = reinterpret_cast<double*>(&temp1[0]);

    // Save original derivatives
    std::copy(derivative, derivative + m.local_nodes(), temp2);
    double* derivative_vec = reinterpret_cast<double*>(&temp2[0]);

    // Original residual
    const double* res = reinterpret_cast<const double*>(&residual[0]);

    // Shifted values, derivatives and residual
    double* sol_shift = reinterpret_cast<double*>(&sol[0]);
    double* derivative_shift = reinterpret_cast<double*>(&derivative[0]);
    const double* shift_res = reinterpret_cast<const double*>(&temp3[0]);

    // Compute shift vector
    const double* weightvec = reinterpret_cast<const double*>(&weights[0]);
    double eps = std::sqrt(std::numeric_limits<double>::epsilon());
    for (int j = 0; j < N; ++j) {
        shift[j] = eps * std::max(
            std::abs(sol_vec[j]), std::max(
            std::abs(h*derivative_vec[j]),
            1.0 / weightvec[j]
        ));
    }

    // A CSR sparse matrix used to assemble the nonzero values
    std::vector< std::map<int, double> > CSR_matrix(N);

    // Process sets of independent columns
    for (int colour = 0; colour < num_colours; ++colour) {

        // Shift each column or not, depending on its colour
        for (int j = 0; j < N; ++j) {
            if (colourvec[j] == colour) {
                sol_shift[j] = sol_vec[j] + shift[j];
                derivative_shift[j] = derivative_vec[j] + c * shift[j];
            } else {
                sol_shift[j] = sol_vec[j];
                derivative_shift[j] = derivative_vec[j];
            }
        }

        // Compute shifted residual
        compute_residual(temp3, false);
        ++num_callbacks;

        // Load the values into the CSR matrix
        for (int j = 0; j < N; ++j) {
            if (colourvec[j] == colour) {
                for (int i = 0; i < pat[j].size(); ++i) {
                    int row = pat[j][i];
                    double value = (shift_res[row] - res[row]) / shift[j];
                    CSR_matrix[row].insert(std::make_pair(j, value));
                }
            }
        }

    }

    // Copy over CSR matrix into DSS array
    int p = 0;
    for (int i = 0; i < N; ++i) {
        std::map<int, double>::iterator it = CSR_matrix[i].begin();
        std::map<int, double>::iterator end = CSR_matrix[i].end();
        for(; it != end; ++it) {
            assert(columns[p] == it->first + 1);    // 1-based indexing
            values[p] = it->second;
            ++p;
        }
    }

    // Factorise
    int opt = MKL_DSS_INDEFINITE;
    int flag = dss_factor_real(dss_handle, opt, &values[0]);
    assert(flag == MKL_DSS_SUCCESS);

    return 0;
}

int Preconditioner::apply(
    const mesh::Mesh& m,
    double t, double c, double h, double delta,
    const_iterator residual, const_iterator weights,
    const_iterator rhs,
    iterator sol, iterator derivative,
    iterator z, iterator temp,
    Callback compute_residual)
{
    ++num_applications;

    const double* r = reinterpret_cast<const double*>(&rhs[0]);
    double* zz = reinterpret_cast<double*>(&z[0]);

    int nrhs = 1;
    int opt = MKL_DSS_REFINEMENT_OFF;
    int flag = dss_solve_real(dss_handle, opt, r, nrhs, zz);
    assert(flag == MKL_DSS_SUCCESS);

    return 0;
}

} // end namespace fvmpor
