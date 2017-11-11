
find_package(R REQUIRED)
include_directories(${R_INCLUDE_DIR})
link_directories(${R_LIB_DIRS})

set(CMAKE_CXX_FLAGS "-std=c++11 ${RCPP_CXX_FLAGS} -fopenmp")
#set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -stdlib=libstdc++ -Wl,-headerpad_max_install_names -undefined dynamic_lookup -single_module -multiply_defined suppress -F/Library/Frameworks/R.framework/.. -framework R -Wl,-framework -Wl,CoreFoundation")
if (APPLE)
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -stdlib=libstdc++ -Wl,-headerpad_max_install_names -undefined dynamic_lookup -single_module -multiply_defined suppress -F/Library/Frameworks/R.framework/.. -framework R -Wl,-framework -Wl,CoreFoundation")
endif(APPLE)
#if(CYGWIN)
#C:/RBuildTools/3.4/mingw_64/bin/g++  -I"C:/PROGRA~1/R/R-34~1.1/include" -DNDEBUG   -I"C:/Users/dy/Documents/R/win-library/3.4/Rcpp/include" -I"C:/Users/dy/Documents"   -I"d:/Compiler/gcc-4.9.3/local330/include"     -O2 -Wall  -mtune=core2 -c times2.cpp -o times2.o
#C:/RBuildTools/3.4/mingw_64/bin/g++ -shared -s -static-libgcc -o sourceCpp_3.dll tmp.def times2.o -Ld:/Compiler/gcc-4.9.3/local330/lib/x64 -Ld:/Compiler/gcc-4.9.3/local330/lib -LC:/PROGRA~1/R/R-34~1.1/bin/x64 -lR
#    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -s -static-libgcc tmp.def -lR")
#endif(CYGWIN)
if(UNIX AND NOT APPLE)
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-Bsymbolic-functions -Wl,-z,relro -lR")
endif(UNIX AND NOT APPLE)

set(CMAKE_SHARED_LIBRARY_SUFFIX ".so")
#CUDA_ADD_LIBRARY(rcudanlp SHARED gmmR.cpp kmeansR.cpp distR.cpp tsneR.cpp quadtreeR.cpp ../gmm/gmm.cu ../kmeans/kmeans.cu ../tsne/tsne.cu )
CUDA_ADD_LIBRARY(rcudanlp SHARED tsneR.cpp distR.cpp knnR.cpp ../matrix/dist/DistMatrix.cpp dtmR.cpp)
TARGET_LINK_LIBRARIES(rcudanlp tbb)

message(STATUS ${TBB_LIBRARIES})
INSTALL(TARGETS rcudanlp DESTINATION lib)
