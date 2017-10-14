############################################################################
# <summary> CMakeLists.txt for OpenCV and CUDA. </summary>
# <date>    2012-02-07          </date>
# <author>  Quan Tran Minh. edit by Johannes Kast, Michael Sarahan </author>
# <email>   quantm@unist.ac.kr  kast.jo@googlemail.com msarahan@gmail.com</email>
############################################################################

# collect source files

set(CMAKE_CXX_FLAGS "-mavx2 -std=c++11 -stdlib=libc++")

ADD_EXECUTABLE(eigen eigen.cpp)

ADD_EXECUTABLE(blaze blaze.cpp)

ADD_EXECUTABLE(test-cpp TestCPP.cpp)

CUDA_ADD_EXECUTABLE(test-tbb TestTBB.cpp)
#target_link_libraries(test-tbb tbb)

CUDA_ADD_EXECUTABLE(test-scan main.cpp TestScan.cu)
CUDA_ADD_EXECUTABLE(test-reduce main.cpp TestReduce.cu)
CUDA_ADD_EXECUTABLE(test-transpose main.cpp TestTranspose.cu)

CUDA_ADD_EXECUTABLE(test-matrix main.cpp TestMatrix.cu)

ADD_EXECUTABLE(kdtree kdtree.cpp)

ADD_EXECUTABLE(quadtree quadtree.cpp)
