############################################################################
# <summary> CMakeLists.txt for OpenCV and CUDA. </summary>
# <date>    2012-02-07          </date>
# <author>  Quan Tran Minh. edit by Johannes Kast, Michael Sarahan </author>
# <email>   quantm@unist.ac.kr  kast.jo@googlemail.com msarahan@gmail.com</email>
############################################################################

cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(nlp-cuda)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/modules/")

set(EXECUTABLE_OUTPUT_PATH "${CMAKE_SOURCE_DIR}/bin/")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY  "${CMAKE_SOURCE_DIR}/bin/")
#set (CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin/")
#set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin")

#if (UNIX)
#    set (CMAKE_CXX_COMPILER "/usr/local/opt/llvm/bin/clang")
#    #link_directories(/usr/local/opt/llvm/include)
#    set(CMAKE_CXX_FLAGS "-I/usr/local/opt/llvm/include -L/usr/local/opt/llvm/util -Wl,-rpath,/usr/local/opt/llvm/util")
#endif()

#find_package(OpenCV REQUIRED)
#link_libraries(${OpenCV_LIBS} )

find_package(CUDA REQUIRED)
link_libraries(${CUDA_cusparse_LIBRARY})
link_libraries(${CUDA_CUBLAS_LIBRARIES})

#find_package(TBB REQUIRED)
#include_directories(${TBB_INCLUDE_DIRS})
#link_libraries(${TBB_LIBRARIES} )


include_directories(src/util)
#include_directories(src/cppjieba/include)
#include_directories(src/cppjieba/deps)

if(CUDA_FOUND)
    # compared to class settings, we let NVidia's FindCUDA CMake detect
    # whether to build x64.  We tell it to support most devices, though,
    # to make sure more people can easily run class code without knowing
    # about this compiler argument
    if (DARWIN)
        set(CUDA_NVCC_FLAGS "
  -ccbin /usr/bin/clang;
  -gencode;arch=compute_30,code=sm_30;
  -gencode;arch=compute_35,code=sm_35;
  -gencode;arch=compute_20,code=sm_20;")
    endif(DARWIN)
    if(WIN)
        set(CUDA_NVCC_FLAGS "
  -ccbin cl.exe;
  -gencode;arch=compute_30,code=sm_30;
  -gencode;arch=compute_35,code=sm_35;
  -gencode;arch=compute_20,code=sm_20;")
    endif(WIN)


    # add -Wextra compiler flag for gcc compilations
    if (UNIX)
        set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-Xcompiler -Wextra -lcublas -lcusparse")
        set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-stdlib=libstdc++")

    endif (UNIX)

    # add debugging to CUDA NVCC flags.  For NVidia's NSight tools.
    # set(CUDA_NVCC_FLAGS_DEBUG ${CUDA_NVCC_FLAGS_DEBUG} "-G")

    add_subdirectory (src/R)
    add_subdirectory (src/gmm)
    add_subdirectory (src/test)
else(CUDA_FOUND)
    message("CUDA is not installed on this system.")
endif()
