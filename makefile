IS64BIT=defined
PARMS_OPTS=-DUSE_MPI -DREAL=double -DDBL -DHAS_BLAS -DFORTRAN_UNDERSCORE -DVOID_POINTER_SIZE_8
#OPTS=-DMPICH_IGNORE_CXX_SEEK
#OPTS=-DMPICH_IGNORE_CXX_SEEK -DFVM_DEBUG -DMESH_DEBUG -D_GLIBCXX_DEBUG_PEDANTIC -DVECTOR_DEBUG -g -O0 -fno-inline
#OPTS=-DMPICH_IGNORE_CXX_SEEK -O3 -LNO
OPTS=-DMPICH_IGNORE_CXX_SEEK -O2

cc=icc
CC=icpc
RM=rm -f

ifdef IS64BIT
#INCLUDE=-I./include -I$(HOME)/include
#MKL=-lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5
#LIBS=-L/pkg/intel_ia64/Compiler/11.1/046/mkl/lib/64 -L/pkg/intel_ia64/Compiler/11.1/046/lib/ia64 -L$(HOME)/lib
INCLUDE=-I./include -I$(HOME)/include -I/opt/intel/impi/4.0.0.028/intel64/include -I/opt/parmetis
MKL=-lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5
LIBS=-L/opt/parmetis -L/opt/intel/impi/4.0.0.028/intel64/lib -L$(HOME)/lib
else
INCLUDE=-I./include -I/opt/intel/impi/3.2/include -I/opt/intel/Compiler/11.1/069/mkl/include -I/home/cummingb/include
MKL=-lmkl_intel_thread -lmkl_intel -lmkl_core -liomp5
LIBS=-L/opt/intel/impi/3.2/lib -L/opt/intel/Compiler/11.1/069/mkl/lib/32 -L/opt/intel/Compiler/11.1/069/lib/ia32 -L/home/cummingb/lib
endif

all : mesh.o doublevector_arithmetic.o doublevector_io.o

# ............
# library
# ............
mesh.o : src/fvm/mesh.cpp include/fvm/mesh.h include/fvm/impl/mesh/*.h
	$(CC) $(OPTS) $(INCLUDE) -c src/fvm/mesh.cpp

doublevector_arithmetic.o :  src/util/doublevector_arithmetic.cpp include/util/doublevector.h
	$(CC) $(OPTS) $(INCLUDE) -c src/util/doublevector_arithmetic.cpp

doublevector_io.o :  src/util/doublevector_io.cpp include/util/doublevector.h
	$(CC) $(OPTS) $(INCLUDE) -c src/util/doublevector_io.cpp

# ............
# clean
# ............
clean:
	$(RM) *.o
