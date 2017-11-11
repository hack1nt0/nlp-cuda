set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

CUDA_ADD_EXECUTABLE(cu.matrix main.cpp main.cu)
