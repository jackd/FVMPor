cc=icc
CC=icpc
localOPTS=-DMPICH_IGNORE_CXX_SEEK -DUSE_MINLIN -DUSE_MKL
debugOPTS=-g -O0
parmsOPTS=-DUSE_MPI -DREAL=double -DDBL -DHAS_BLAS -DFORTRAN_UNDERSCORE -DVOID_POINTER_SIZE_8
#debugOPTS=-DMPICH_IGNORE_CXX_SEEK -DFVM_DEBUG -DMESH_DEBUG -D_GLIBCXX_DEBUG_PEDANTIC -DVECTOR_DEBUG -g -O0 -fno-inline
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
MKL=-lmkl_intel_thread -lmkl_intel -lmkl_core -liomp5
LIBS=-L/opt/intel/impi/3.2/lib -L/opt/intel/Compiler/11.1/069/mkl/lib/32 -L/opt/intel/Compiler/11.1/069/lib/ia32 -L/home/cummingb/lib
endif

#SUNDIALS=-lsundials_idas -lsundials_nvecparallel
SUNDIALS=/opt/sundials2.4.0/lib/libsundials_idas.a /opt/sundials2.4.0/lib/libsundials_nvecparallel.a

LIB=$(MKL) $(SUNDIALS) -lmpi -lpthread -lcublas -lcusparse
MESH=../../mesh.o
UTIL=$(MINLIN)/cuda.o

#preconOPTS=-DPRECON_PARMS -DPRECON
#PRECON=parms_wrapper.o preconditioner_parms.o
#PRECON_DAE=parms_wrapper.o preconditioner_parms_DAE.o

preconOPTS=-DPRECON_DSS -DPRECON
PRECON=preconditioner_dss.o
PRECON_DAE=preconditioner_dss_DAE.o

#preconOPTS=
#PRECON_DAE=
#PRECON=

IMPLEMENTATIONDEPS_ODE=fvmpor_ODE.o fvmpor.o fvmpor_ODE_impl.o  shape.o $(MESH) $(UTIL) $(PRECON)
IMPLEMENTATIONDEPS_DAE=fvmpor_DAE.o fvmpor.o fvmpor_DAE_impl.o  shape.o $(MESH) $(UTIL) $(PRECON_DAE)

OPTS=$(localOPTS) $(debugOPTS) $(preconOPTS)

# ...............
# all
# ...............
all: henryDAE

# ................
# compile
# ................
henryODE: henry.h $(IMPLEMENTATIONDEPS_ODE)
	$(CC) $(OPTS) -DPROBLEM_HENRY $(INCLUDE) $(LIBS) -o henryODE $(IMPLEMENTATIONDEPS_ODE) $(LIB)

henryDAE: henry.h $(IMPLEMENTATIONDEPS_DAE)
	$(CC) $(OPTS) $(INCLUDE) $(LIBS) -o henryDAE $(IMPLEMENTATIONDEPS_DAE) $(LIB)

# ............
# Object files
# ............
fvmpor_ODE.o : fvmpor_ODE.h preconditioner_dss.h fvmpor_ODE.cpp
	$(CC) $(OPTS) $(INCLUDE) -c fvmpor_ODE.cpp

fvmpor_ODE_impl.o : fvmpor_ODE.h fvmpor_ODE_impl.cpp fvmpor.h
	$(CC) $(OPTS) -DPROBLEM_HENRY $(INCLUDE) -c fvmpor_ODE_impl.cpp

fvmpor.o : fvmpor.h fvmpor.cpp
	$(CC) $(OPTS) $(INCLUDE) -c fvmpor.cpp

fvmpor_DAE.o : fvmpor_DAE.h fvmpor.h preconditioner_dss_DAE.h fvmpor_DAE.cpp
	$(CC) $(OPTS) $(INCLUDE) -c fvmpor_DAE.cpp

fvmpor_DAE_impl.o : henry.h fvmpor_DAE.h fvmpor_DAE_impl.cpp fvmpor.h
	$(CC) $(OPTS) -DPROBLEM_HENRY $(INCLUDE) -c fvmpor_DAE_impl.cpp

preconditioner_parms.o : preconditioner_parms.h preconditioner_parms.cpp
	$(CC) $(OPTS) $(INCLUDE) -c preconditioner_parms.cpp

preconditioner_parms_DAE.o : preconditioner_parms_DAE.h preconditioner_parms_DAE.cpp
	$(CC) $(OPTS) $(INCLUDE) -c preconditioner_parms_DAE.cpp

preconditioner_dss.o : preconditioner_dss.h preconditioner_dss.cpp
	$(CC) $(OPTS) $(INCLUDE) -c preconditioner_dss.cpp

preconditioner_dss_DAE.o : preconditioner_dss_DAE.h preconditioner_dss_DAE.cpp
	$(CC) $(OPTS) $(INCLUDE) -c preconditioner_dss_DAE.cpp

shape.o : shape.h shape.cpp
	$(CC) $(OPTS) $(INCLUDE) -c shape.cpp

parms_wrapper.o : parms_wrapper.c
	$(cc) $(OPTS) $(parmsOPTS) $(INCLUDE) -c parms_wrapper.c


# ............
# clean
# ............
clean:
	$(RM) cassion
	$(RM) cassionM
	$(RM) *.o
