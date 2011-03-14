cc=icc
CC=icpc
#CC=g++ -Wall -Wno-sign-compare -Wno-unused
localOPTS=-DMPICH_IGNORE_CXX_SEEK -DUSE_MINLIN -DUSE_MKL
debugOPTS=-g -O3 -DUSE_CUDA
#debugOPTS=-g -O3 -DLIN_VERBOSE
#debugOPTS=-g -O3
#debugOPTS=-g -O3 -openmp
#debugOPTS=-g -O0 -DLIN_DEBUG
#debugOPTS=-DMPICH_IGNORE_CXX_SEEK -DFVM_DEBUG -DMESH_DEBUG -D_GLIBCXX_DEBUG_PEDANTIC -DVECTOR_DEBUG -DLIN_DEBUG -g -O0 -fno-inline -DLIN_DEBUG -DLIN_VERBOSE
parmsOPTS=-DUSE_MPI -DREAL=double -DDBL -DHAS_BLAS -DFORTRAN_UNDERSCORE -DVOID_POINTER_SIZE_8
RM=rm -f
IS64BIT=defined

ifdef IS64BIT
INCLUDE=-I../../include -I$(HOME)/include -I/opt/parmetis -I/opt/sundials2.4.0/include -I/home/cummingb/projects/minlin -I/usr/local/cuda/include
MKL=-lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5
#LIBS=-L/pkg/intel_ia64/Compiler/11.1/046/mkl/lib/64 -L/pkg/intel_ia64/Compiler/11.1/046/lib/ia64 -L$(HOME)/lib
LIBS=-L/opt/parmetis -L/opt/sundials2.4.0/lib -L/usr/local/cuda/lib64
MINLIN=/home/cummingb/projects/minlin
else
INCLUDE=-I../../include -I/opt/intel/impi/3.2/include -I/opt/intel/Compiler/11.1/069/mkl/include -I/home/cummingb/include
MKL=-lmkl_intel_threa
LIBS=-L/opt/intel/impi/3.2/lib -L/opt/intel/Compiler/11.1/069/mkl/lib/32 -L/opt/intel/Compiler/11.1/069/lib/ia32 -L/home/cummingb/lib
endif

#SUNDIALS=-lsundials_idas -lsundials_nvecparallel
#SUNDIALS=/opt/sundials2.4.0/lib/libsundials_idas.a /opt/sundials2.4.0/lib/libsundials_nvecparallel.a
SUNDIALS=/opt/sundials2.4.0/lib/libsundials_idas.a /opt/sundials2.4.0/lib/libsundials_nvecparallel_cuda.a

LIB=$(MKL) $(SUNDIALS) -lmpi -lpthread -lcublas -lcusparse -lcuda
MESH=../../mesh.o
UTIL=$(MINLIN)/cuda.o

#preconOPTS=-DPRECON_PARMS -DPRECON
#PRECON=parms_wrapper.o preconditioner_parms.o
#PRECON_DAE=parms_wrapper.o preconditioner_parms_DAE.o

preconOPTS=-DPRECON_DSS -DPRECON
#PRECON=preconditioner_dss.o
PRECON=preconditioner_ilu0.o
PRECON_DAE=preconditioner_dss_DAE.o

#preconOPTS=
#PRECON_DAE=
#PRECON=

IMPLEMENTATIONDEPS_ODE=fvmpor_ODE.o fvmpor.o shape.o $(MESH) $(UTIL) $(PRECON)
IMPLEMENTATIONDEPS_DAE=fvmpor_DAE.o fvmpor.o shape.o $(MESH) $(UTIL) $(PRECON_DAE)

OPTS=$(localOPTS) $(debugOPTS) $(preconOPTS) -openmp

# ...............
# all
# ...............
all: cassion vs cassionM vsM

# ................
# compile
# ................
cassion: cassion.h fvmpor_ODE_impl.cpp $(IMPLEMENTATIONDEPS_ODE)
	$(CC) $(OPTS) $(INCLUDE) $(LIBS) -o cassion fvmpor_ODE_impl.cpp -DPROBLEM_CASSION $(IMPLEMENTATIONDEPS_ODE) $(LIB)

test: test.cpp
	$(CC) $(OPTS) $(INCLUDE) $(LIBS) -o test test.cpp $(UTIL) $(LIB) $(MESH)

cassionM: cassion.h fvmpor_DAE_impl.cpp $(IMPLEMENTATIONDEPS_DAE)
	$(CC) $(OPTS) $(INCLUDE) $(LIBS) -o cassionM fvmpor_DAE_impl.cpp -DPROBLEM_CASSION $(IMPLEMENTATIONDEPS_DAE) $(LIB)

vs: vs.h fvmpor_ODE_impl.cpp $(IMPLEMENTATIONDEPS_ODE)
	$(CC) $(OPTS) $(INCLUDE) $(LIBS) -o vs fvmpor_ODE_impl.cpp -DPROBLEM_VS $(IMPLEMENTATIONDEPS_ODE) $(LIB)

vsM: vs.h fvmpor_DAE_impl.cpp $(IMPLEMENTATIONDEPS_DAE)
	$(CC) $(OPTS) $(INCLUDE) $(LIBS) -o vsM fvmpor_DAE_impl.cpp -DPROBLEM_VS $(IMPLEMENTATIONDEPS_DAE) $(LIB)
# ............
# Object files
# ............
cassion.o : cassion.cpp fvmpor.h
	$(CC) $(OPTS) $(INCLUDE) -c cassion.cpp

#fvmpor_ODE.o : fvmpor_ODE.h preconditioner_dss.h fvmpor_ODE.cpp
fvmpor_ODE.o : fvmpor_ODE.h preconditioner_ilu0.h fvmpor_ODE.cpp
	$(CC) $(OPTS) $(INCLUDE) -c fvmpor_ODE.cpp -o fvmpor_ODE.o

fvmpor_ODE_impl.o : fvmpor_ODE.h fvmpor_ODE_impl.cpp fvmpor.h
	$(CC) $(OPTS) -DPROBLEM_CASSION $(INCLUDE) -c fvmpor_ODE_impl.cpp -o fvmpor_ODE_impl.o

fvmpor.o : fvmpor.h fvmpor.cpp
	$(CC) $(OPTS) $(INCLUDE) -c fvmpor.cpp -o fvmpor.o

fvmpor_DAE.o : fvmpor_DAE.h preconditioner_dss.h fvmpor_DAE.cpp
	$(CC) $(OPTS) $(INCLUDE) -c fvmpor_DAE.cpp

#fvmpor_DAE_impl.o : fvmpor_DAE.h fvmpor_DAE_impl.cpp fvmpor.h
#	$(CC) $(OPTS) -DPROBLEM_CASSION $(INCLUDE) -c fvmpor_DAE_impl.cpp

preconditioner_parms.o : preconditioner_parms.h preconditioner_parms.cpp
	$(CC) $(OPTS) $(INCLUDE) -c preconditioner_parms.cpp

preconditioner_parms_DAE.o : preconditioner_parms_DAE.h preconditioner_parms_DAE.cpp
	$(CC) $(OPTS) $(INCLUDE) -c preconditioner_parms_DAE.cpp

preconditioner_ilu0.o : preconditioner_ilu0.h preconditioner_ilu0.cpp
	$(CC) $(OPTS) $(INCLUDE) -c preconditioner_ilu0.cpp -o preconditioner_ilu0.o

preconditioner_dss.o : preconditioner_dss.h preconditioner_dss.cpp
	$(CC) $(OPTS) $(INCLUDE) -c preconditioner_dss.cpp -o preconditioner_dss.o

preconditioner_dss_DAE.o : preconditioner_dss_DAE.h preconditioner_dss_DAE.cpp
	$(CC) $(OPTS) $(INCLUDE) -c preconditioner_dss_DAE.cpp

shape.o : shape.h shape.cpp
	$(CC) $(OPTS) $(INCLUDE) -c shape.cpp -o shape.o

parms_wrapper.o : parms_wrapper.c
	$(cc) $(OPTS) $(parmsOPTS) $(INCLUDE) -c parms_wrapper.c


# ............
# clean
# ............
clean:
	$(RM) cassion
	$(RM) cassionM
	$(RM) vs
	$(RM) vsM
	$(RM) test
	$(RM) *.o
