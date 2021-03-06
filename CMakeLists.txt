############################################################################
# <summary> CMakeLists.txt for OpenCV and CUDA. </summary>
# <date>    2012-02-07          </date>
# <author>  Quan Tran Minh. edit by Johannes Kast, Michael Sarahan </author>
# <email>   quantm@unist.ac.kr  kast.jo@googlemail.com msarahan@gmail.com</email>
############################################################################

cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(nlp-cuda CXX)

set(CMAKE_VERBOSE_MAKEFILE ON)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/modules/")

set(EXECUTABLE_OUTPUT_PATH "${CMAKE_SOURCE_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY  "${CMAKE_SOURCE_DIR}/lib")

set(CMAKE_CXX_FLAGS "-std=c++11")

find_package(CUDA)
#link_libraries(${CUDA_cusparse_LIBRARY})
#link_libraries(${CUDA_CUBLAS_LIBRARIES})

#find_package(TBB REQUIRED)
#include_directories(${TBB_INCLUDE_DIRS})
##link_libraries(${TBB_LIBRARIES} )
#message(${TBB_LIBRARIES} )

#find_package(BLAS REQUIRED)

if (APPLE)
    include_directories(/opt/intel/mkl/include /opt/intel/tbb/include)
    link_directories(/opt/intel/lib /opt/intel/mkl/lib /opt/intel/tbb/lib)
else() #Linux
    include_directories(/opt/intel/mkl/include /opt/intel/tbb/include)
    link_directories(/opt/intel/lib/intel64 /opt/intel/mkl/lib/intel64 /opt/intel/tbb/lib/intel64/gcc4.7)
endif()


if(CUDA_FOUND)
    if (APPLE)
        set(CUDA_NVCC_FLAGS "-ccbin /usr/bin/clang++")
    endif(APPLE)
    # add -Wextra compiler flag for gcc compilations
    if (UNIX)
        set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Xcompiler -Wextra -lcublas -lcusparse")
    endif (UNIX)
    # add debugging to CUDA NVCC flags.  For NVidia's NSight tools.
    # set(CUDA_NVCC_FLAGS_DEBUG ${CUDA_NVCC_FLAGS_DEBUG} "-G")

else(CUDA_FOUND)
    message("CUDA is not installed on this system.")
endif()

include_directories(src/utils)
include_directories(src)

add_subdirectory (src/optimizer)
#add_subdirectory (src/java)
add_subdirectory (src/matrix)
add_subdirectory (src/dist)
add_subdirectory (src/tsne)
add_subdirectory (src/knn)
add_subdirectory (src/R)
add_subdirectory (src/kmeans)
add_subdirectory (src/gmm)
add_subdirectory (src/ds)

