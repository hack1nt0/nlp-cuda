
#find_package(R REQUIRED)
#include_directories(${R_INCLUDE_DIR})
#link_directories(${R_LIB_DIRS})
#include_directories("../gmm")
#
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${RCPP_CXX_FLAGS} -DNDEBUG")
##set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -stdlib=libstdc++ -Wl,-headerpad_max_install_names -undefined dynamic_lookup -single_module -multiply_defined suppress -F/Library/Frameworks/R.framework/.. -framework R -Wl,-framework -Wl,CoreFoundation")
#if (APPLE)
#    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -stdlib=libstdc++ -Wl,-headerpad_max_install_names -undefined dynamic_lookup -single_module -multiply_defined suppress -F/Library/Frameworks/R.framework/.. -framework R -Wl,-framework -Wl,CoreFoundation")
#endif(APPLE)
##if(CYGWIN)
##C:/RBuildTools/3.4/mingw_64/bin/g++  -I"C:/PROGRA~1/R/R-34~1.1/include" -DNDEBUG   -I"C:/Users/dy/Documents/R/win-library/3.4/Rcpp/include" -I"C:/Users/dy/Documents"   -I"d:/Compiler/gcc-4.9.3/local330/include"     -O2 -Wall  -mtune=core2 -c times2.cpp -o times2.o
##C:/RBuildTools/3.4/mingw_64/bin/g++ -shared -s -static-libgcc -o sourceCpp_3.dll tmp.def times2.o -Ld:/Compiler/gcc-4.9.3/local330/lib/x64 -Ld:/Compiler/gcc-4.9.3/local330/lib -LC:/PROGRA~1/R/R-34~1.1/bin/x64 -lR
##    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -s -static-libgcc tmp.def -lR")
##endif(CYGWIN)
#if(UNIX AND NOT APPLE)
#    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-Bsymbolic-functions -Wl,-z,relro -lR")
#endif(UNIX AND NOT APPLE)
#
#CUDA_ADD_LIBRARY(gmm SHARED gmmR.cpp ../gmm/gmm.cu)

