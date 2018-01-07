set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp ")

ADD_EXECUTABLE(matrix main.cpp)
target_link_libraries(matrix mkl_intel_lp64 mkl_sequential mkl_core iomp5 pthread m dl)

#ADD_EXECUTABLE(mkltest cspblas_ccsr.cpp)
