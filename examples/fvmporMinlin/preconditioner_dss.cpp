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
    blocksize_ = block_traits<Physics::value_type>::blocksize;
    N_ = m.local_nodes() * blocksize_;
    shift_ = TVecDevice(N_);

    // Create DSS data structure
    int opt = MKL_DSS_MSG_LVL_WARNING + MKL_DSS_TERM_LVL_ERROR;
    int flag = dss_create(dss_handle_, opt);
    assert(flag == MKL_DSS_SUCCESS);

    // Determine sparsity pattern
    pat_ = column_pattern(m, blocksize_);
    colourvec_ = sequential_vertex_colouring(m, blocksize_);
    assert(colourvec_.size() == N_);
    num_colours_ = *std::max_element(colourvec_.begin(), colourvec_.end()) + 1;

    // Create CSR row and column arrays
    row_index_.resize(N_+1);
    row_index_[0] = 1;   // 1-based indexing
    for (int i = 0; i < N_; ++i) {
        row_index_[i+1] = row_index_[i] + pat_[i].size();
        for (int j = 0; j < pat_[i].size(); ++j) {
            columns_.push_back(pat_[i][j] + 1);   // 1-based indexing
        }
    }
    nnz_ = columns_.size();
    values_ = TVecHost(nnz_, lin::row_oriented);
    
    //////////////////////
    //////////////////////
    //////////////////////
    int mi=100;
    int ma=0;
    for(int i=0; i<N_; i++){
        int sz = row_index_[i+1] - row_index_[i];
        if(sz < mi)
            mi = sz;
        if(sz > ma)
            ma = sz;
    }
    std::cerr << "min, max, avg nnz per row of matrix are : " << mi << ", " << ma <<  ", " << (double)nnz_/(double)N_ << std::endl;
    //////////////////////
    //////////////////////
    //////////////////////

    ////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////

    // store lists of the columns associated with each colour
    std::vector<std::vector<int> > colour_p_vec;
    colour_p_vec.resize(num_colours_);
    for(int i=0; i<N_; i++ )
        colour_p_vec[colourvec_[i]].push_back(i);

    std::cerr << "There are " << num_colours_ << " colours" << std::endl;

    // copy the lists into minlin vectors
    colour_p_.resize(num_colours_);
    for(int i=0; i<num_colours_; i++)
        colour_p_[i] = TVecHostIndex( colour_p_vec[i].begin(),
                                      colour_p_vec[i].end() );

    // build an index vector that maps the position of nonzeros in the matrix
    // computed by colour-column to the compressed row storage
    std::vector< std::map<int, int> > CSR_pattern(N_);
    std::vector< int > nnz_per_colour(num_colours_, 0);
    int k=0;
    for(int colour=0; colour<num_colours_; colour++){
        for(int kk=0; kk<colour_p_[colour].size(); kk++){
            int j=colour_p_[colour][kk];
            for (int i = 0; i < pat_[j].size(); ++i) {
                int row = pat_[j][i];
                CSR_pattern[row].insert(std::make_pair(j, k++));
                nnz_per_colour[colour]++;
            }
        }
    }
    k=0;
    TVecHostIndex matrix_p(nnz_); 
    for (int i = 0; i < N_; ++i) {
        std::map<int, int>::iterator it = CSR_pattern[i].begin();
        std::map<int, int>::iterator end = CSR_pattern[i].end();
        for(; it != end; ++it) {
            matrix_p[k] = it->second;
            k++;
        }
    }
    // copy permutation to device
    matrix_p_ = matrix_p;

    // build an index that maps the entries in each residual into the relevant
    // part of the jacobian
    std::vector<TVecHostIndex> res_p;
    res_p.resize(num_colours_);
    res_p_.resize(num_colours_);

    std::vector<TVecHostIndex> shift_p;
    shift_p.resize(num_colours_);
    shift_p_.resize(num_colours_);

    for(int colour=0; colour<num_colours_; colour++){
        int kk=0;
        int n = colour_p_vec[colour].size();
        res_p[colour] = TVecHostIndex(nnz_per_colour[colour]);
        shift_p[colour] = TVecHostIndex(nnz_per_colour[colour]);
        for(int k=0; k<n; k++){
            int j=colour_p_vec[colour][k];
            for(int i=0; i<pat_[j].size(); i++, kk++){
                int row = pat_[j][i];
                res_p[colour][kk] = row;
                shift_p[colour][kk] = j;
            }
        }
        res_p_[colour] = res_p[colour];
        shift_p_[colour] = shift_p[colour];
    }
    
    // make the colour_dist_ vector
    colour_dist_ = TVecHostIndex(num_colours_+1, 0);
    for(int colour=0; colour<num_colours_; colour++)
        colour_dist_[colour+1] = colour_dist_[colour] + nnz_per_colour[colour];
    ////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////

    // Define DSS matrix structure
    opt = MKL_DSS_SYMMETRIC_STRUCTURE;
    flag = dss_define_structure(
        dss_handle_, opt, &row_index_[0], N_, N_, &columns_[0], nnz_);
    assert(flag == MKL_DSS_SUCCESS);

    // Reorder DSS matrix
    opt = MKL_DSS_AUTO_ORDER;
    flag = dss_reorder(dss_handle_, opt, 0);
    assert(flag == MKL_DSS_SUCCESS);
}

int Preconditioner::setup(
    const mesh::Mesh& m, double tt, double c, double h,
    const TVecDevice &residual, const TVecDevice &weights,
    TVecDevice &sol, TVecDevice &derivative,
    TVecDevice &temp1, TVecDevice &temp2, TVecDevice &temp3,
    Callback compute_residual)
{
    ++num_setups_;

    if (pat_.size() == 0) initialise(m);

    util::Timer timer;

    timer.tic();

    // Save original values
    temp1.at(lin::all) = sol;
    TVecDevice sol_vec(sol.size(), temp1.data());

    // Save original derivatives
    temp2.at(lin::all) = derivative;
    TVecDevice derivative_vec(sol.size(), temp2.data());

    // Original residual
    TVecDevice res(sol.size(), const_cast<double*>(residual.data()) );

    // Shifted values, derivatives and residual
    TVecDevice sol_shift(sol.size(), sol.data());
    TVecDevice derivative_shift(sol.size(), derivative.data());
    TVecDevice shift_res(sol.size(), temp3.data());

    // Compute shift vector
    timer.tic();
    double eps = std::sqrt(std::numeric_limits<double>::epsilon());
    if( CoordTraits<CoordDevice>::is_device() ){
        lin::gpu::make_weights_vector(shift_.data(), sol_vec.data(), derivative_vec.data(), weights.data(), eps, h, N_);
    }else{
        for (int j = 0; j < N_; ++j) {
            shift_[j] = eps * std::max(
                std::abs(sol_vec[j]), std::max(
                std::abs(h*derivative_vec[j]),
                1.0 / weights[j]
            ));
        }
    }

    // Process sets of independent columns

    // manually allocate this vector's memory using page locked memory
    // and refer the vector to the memory
    // probably could be a member of the preconditioner class
    TVecHost values_temp(nnz_, lin::row_oriented);
    for (int colour = 0; colour < num_colours_; ++colour) {
        TVecDevice colour_shift(colour_p_[colour].size());
        colour_shift.at(lin::all) = shift_.at(colour_p_[colour]);

        // Shift each column or not, depending on its colour
        sol_shift.at(lin::all) = sol_vec;
        sol_shift.at(colour_p_[colour]) += colour_shift;
        derivative_shift.at(lin::all) = derivative_vec;
        derivative_shift.at(colour_p_[colour]) += c*colour_shift;

        // Compute shifted residual
        compute_residual(temp3, false);
        ++num_callbacks_;

        // find shifted values
        // copy over in the same operation
        TVecDevice r(res_p_[colour].size());
        r.at(lin::all) = res.at(res_p_[colour]) - temp3.at(res_p_[colour]);
        r.at(lin::all) /= shift_.at(shift_p_[colour]);
        // copy to host performed here
        values_temp.at(colour_dist_[colour], colour_dist_[colour+1]-1) = r;
    }

    // copy values to host
    util::Timer timer_copy;
    timer_copy.tic();
    values_.at(lin::all) = values_temp.at(matrix_p_); 
    time_copy_ += timer_copy.toc();
    time_J_ += timer.toc();
   
    timer.tic();
    int opt = MKL_DSS_INDEFINITE;
    int flag = dss_factor_real(dss_handle_, opt, values_.data());
    assert(flag == MKL_DSS_SUCCESS);
    time_M_ += timer.toc();

    return 0;
}

int Preconditioner::apply(
    const mesh::Mesh& m,
    double t, double c, double h, double delta,
    const TVecDevice &residual, const TVecDevice &weights,
    const TVecDevice &rhs,
    TVecDevice &sol, TVecDevice &derivative,
    TVecDevice &z, TVecDevice &temp,
    Callback compute_residual)
{
    ++num_applications_;
    util::Timer timer_apply;
    util::Timer timer_copy;
    timer_apply.tic();

    int nrhs = 1;
    int opt = MKL_DSS_REFINEMENT_OFF;

    timer_copy.tic();
    TVecHost Z(z.size());
    TVecHost RHS(rhs);
    time_copy_ += timer_copy.toc();
    //z.at(lin::all) = rhs;
    int flag = dss_solve_real(dss_handle_, opt, RHS.data(), nrhs, Z.data());
    assert(flag == MKL_DSS_SUCCESS);
    timer_copy.tic();
    z.at(lin::all) = Z;
    time_copy_ += timer_copy.toc();
    time_apply_ += timer_apply.toc();

    return 0;
}

} // end namespace fvmpor
