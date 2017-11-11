
find_package(JNI REQUIRED)
message("${JNI_INCLUDE_DIRS}")
#include_directories(${JAVA_INCLUDE_DIRS})
include_directories(${JAVA_INCLUDE_PATH})
include_directories(${JAVA_INCLUDE_PATH2})
message("${JNI_LIBRARIES}")
#link_libraries(${JNI_LIBRARIES})
#link_libraries(${JAVA_JVM_LIBRARY})

set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fPIC")

CUDA_ADD_LIBRARY(jni_cuda_matrix SHARED main.cpp jni_cuda_CuMatrix.cu)

CUDA_ADD_EXECUTABLE(java-main main.cpp main.cu jni_cuda_CuMatrix.cu)


