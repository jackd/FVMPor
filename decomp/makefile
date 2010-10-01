CC= icc
CFLAGS= -g
IS64BIT=1
ifdef IS64BIT
#MKLPATH=/pkg/intel_ia64/mkl/10.0.1.014
#LIBPATH= -L${MKLPATH}/lib/64 -L$(HOME)/lib
#INCLUDEPATH= -I${MKLPATH}/include -I$(HOME)/include -I./include
#LIBPATH= -L/opt/parmetis -L/opt/intel/impi/4.0.0.028/intel64/lib
LIBPATH= -L/opt/parmetis
INCLUDEPATH= -I/opt/parmetis -I./include
LIBS= -lm -lmpi -lparmetis -lmetis -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5
else
MKLPATH=/opt/intel/Compiler/11.1/069/mkl
LIBPATH= -L${MKLPATH}/lib/32 -L$(HOME)/lib -L/opt/intel/impi/3.2/lib
INCLUDEPATH= -I${MKLPATH}/include -I$(HOME)/include -I./include -I/opt/intel/impi/3.2/include
LIBS= -lm -lmpi -lparmetis -lmetis -lpthread -lguide -liomp5 -lmkl_intel_thread -lmkl_intel -lmkl_core
endif
RM= rm -f
AR= ar rc
AR2= ranlib
LINALGOBJ= linalg_sparse.o linalg_dense.o linalg.o fileio.o
FINAL_OBJ=benlib.o indices.o linalg_dense.o linalg_sparse.o linalg.o linalg.a fileio.o  ben_mpi.o linalg_mpi.o linalg_dense_mpi.o MR.o ILU.o precon.o schur.o gmres.o scsl_blas.o
ALLC=gmres.c benlib.c linalg_dense.c linalg_sparse.c linalg.c ben_mpi.c fileio.c indices.c linalg_dense_mpi.c linalg_mpi.c MR.c gmres.c domain_decomp.c schur.c scsl_blas.c


solver : solver.c  gmres.o precon.o fileio.o scsl_blas.o
	$(CC) $(CFLAGS) ${LIBPATH} ${INCLUDEPATH} solver.c fileio.o indices.o linalg.o linalg_dense.o linalg_sparse.o benlib.o gmres.o ben_mpi.o linalg_dense_mpi.o linalg_mpi.o MR.o precon.o  schur.o  ILU.o scsl_blas.o -o  solver -pthread $(LIBS)
	cp ./solver ./bin/solver

split : split.c scsl_blas.o fileio.o $(ALLC)
	$(CC) $(CFLAGS) ${LIBPATH} ${INCLUDEPATH} split.c fileio.o indices.o linalg.o linalg_dense.o linalg_sparse.o benlib.o gmres.o ben_mpi.o linalg_dense_mpi.o linalg_mpi.o MR.o precon.o  ILU.o schur.o  scsl_blas.o -o split -pthread $(LIBS) 
	cp ./split ./bin

decomp : decomp.c linalg_mpi.o schur.o fileio.o benlib.o scsl_blas.o
	$(CC) $(CFLAGS) ${LIBPATH} ${INCLUDEPATH} decomp.c fileio.o indices.o linalg.o linalg_dense.o linalg_sparse.o benlib.o gmres.o ben_mpi.o linalg_dense_mpi.o linalg_mpi.o MR.o precon.o ILU.o schur.o scsl_blas.o -o decomp -pthread $(LIBS)
	cp ./decomp ./bin
	
setenv SGI=1
scsl_blasDEPS=scsl_blas.c
scsl_blas.o : ${sl_blasDEPS}
	$(CC) $(CFLAGS) ${INCLUDEPATH} -c scsl_blas.c -o scsl_blas.o
	
gmresDEPS=gmres.c linalg.a  ben_mpi.o
gmres.o : ${gmresDEPS}
	$(CC) $(CFLAGS) ${INCLUDEPATH} -c gmres.c -o gmres.o

preconDEPS=precon.c MR.o indices.o ILU.o schur.o ben_mpi.o
precon.o : ${preconDEPS}
	$(CC) $(CFLAGS) ${INCLUDEPATH} -c precon.c -o precon.o

schurDEPS=schur.c linalg.o linalg_mpi.o linalg_dense_mpi.o benlib.o fileio.o ben_mpi.o
schur.o : ${schurDEPS}
	$(CC) $(CFLAGS) ${INCLUDEPATH} -c schur.c -o schur.o

ILUDEPS=ILU.c linalg.o benlib.o
ILU.o : ${ILUDEPS}
	$(CC) $(CFLAGS) ${INCLUDEPATH} -c ILU.c -o ILU.o

MRDEPS=MR.c linalg.o linalg_mpi.o linalg_dense_mpi.o
MR.o : ${MRDEPS}
	$(CC) $(CFLAGS) ${INCLUDEPATH} -c MR.c -o MR.o

linalg_mpiDEPS=linalg_mpi.c ben_mpi.o
linalg_mpi.o : ${linalg_mpiDEPS}
	$(CC) $(CFLAGS) ${INCLUDEPATH} -c linalg_mpi.c -o linalg_mpi.o

linalg_dense_mpiDEPS=linalg_dense_mpi.c
linalg_dense_mpi.o : ${linalg_dense_mpiDEPS}
	$(CC) $(CFLAGS) ${INCLUDEPATH} -c linalg_dense_mpi.c -o linalg_dense_mpi.o

fileioDEPS=fileio.c linalg.o linalg_dense.o linalg_sparse.o
fileio.o : ${fileioDEPS}
	$(CC) $(CFLAGS) ${INCLUDEPATH} -c fileio.c -o fileio.o


ben_mpiDEPS=ben_mpi.c
ben_mpi.o : ${ben_mpiDEPS}
	$(CC) $(CFLAGS) ${INCLUDEPATH} -c ben_mpi.c -o ben_mpi.o

linalg.a: $(LINALGOBJ)
	$(AR) linalg.a $(LINALGOBJ)

linalgDEPS=linalg.c linalg_sparse.o linalg_dense.o benlib.o
linalg.o : ${linalgDEPS}
	$(CC) $(CFLAGS) ${INCLUDEPATH} -c linalg.c -o linalg.o

linalg_sparseDEPS=linalg_sparse.c  benlib.o indices.o
linalg_sparse.o : ${linalg_sparseDEPS}
	$(CC) $(CFLAGS) ${INCLUDEPATH} -c linalg_sparse.c -o linalg_sparse.o

linalg_denseDEPS=linalg_dense.c  benlib.o
linalg_dense.o : ${linalg_denseDEPS}
	$(CC) $(CFLAGS) ${INCLUDEPATH} -c linalg_dense.c -o linalg_dense.o

indicesDEPS=indices.c
indices.o : ${indecesDEPS} 
	$(CC) $(CFLAGS) ${INCLUDEPATH} -c indices.c -o indices.o

benlibDEPS=benlib.c
benlib.o : ${benlibDEPS}
	$(CC) $(CFLAGS) ${INCLUDEPATH} -c benlib.c -o benlib.o

clean :
	$(RM) *.o;
	$(RM) linalg.a;
	$(RM) ./decomp
	$(RM) ./split
	$(RM) ./solver
	$(RM) ./bin/*

	touch *.c;

all :   clean $(FINAL_OBJ) solver split decomp

lib : clean $(FINAL_OBJ)
	rm -rf ../lib
	mkdir ../lib
	cp *.o ../lib;
	cp *.h ../lib;
	ar rc ../lib/libSalt.a ../lib/*.o;
	rm -f ../lib/*.o;
