############################################################################
# <summary> CMakeLists.txt for OpenCV and CUDA. </summary>
# <date>    2012-02-07          </date>
# <author>  Quan Tran Minh. edit by Johannes Kast, Michael Sarahan </author>
# <email>   quantm@unist.ac.kr  kast.jo@googlemail.com msarahan@gmail.com</email>
############################################################################

# collect source files

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")

ADD_EXECUTABLE(gmm main.cpp ../utils/utils.cpp)
target_link_libraries(gmm mkl_intel_lp64 mkl_tbb_thread mkl_core iomp5 pthread m dl)

