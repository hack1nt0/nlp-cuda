############################################################################
# <summary> CMakeLists.txt for OpenCV and CUDA. </summary>
# <date>    2012-02-07          </date>
# <author>  Quan Tran Minh. edit by Johannes Kast, Michael Sarahan </author>
# <email>   quantm@unist.ac.kr  kast.jo@googlemail.com msarahan@gmail.com</email>
############################################################################

# collect source files

CUDA_ADD_EXECUTABLE(dist-tbb main.cpp DistMatrixTbb.cpp)
CUDA_ADD_EXECUTABLE(dist-cu main.cpp DistMatrix.cu)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -fopenmp ")
CUDA_ADD_EXECUTABLE(dist-omp main.cpp dist.cpp)
