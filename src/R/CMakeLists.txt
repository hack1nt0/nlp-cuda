
find_package(R REQUIRED)
include_directories(${R_INCLUDE_DIR})
link_directories(${R_LIB_DIRS})
include_directories("../gmm")
message(${R_LIBS})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${RCPP_CXX_FLAGS} -DNDEBUG")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -stdlib=libstdc++ -Wl,-headerpad_max_install_names -undefined dynamic_lookup -single_module -multiply_defined suppress -F/Library/Frameworks/R.framework/.. -framework R -Wl,-framework -Wl,CoreFoundation")

message(${CMAKE_CXX_FLAGS})

CUDA_ADD_LIBRARY(gmmR SHARED gmmR.cpp ../gmm/gmm.cu)
#CUDA_ADD_EXECUTABLE(gmmR gmmR.cpp)

