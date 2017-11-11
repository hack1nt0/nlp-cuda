export HOME=/Users/dy

# added by far2l
# alias far2l='sudo $HOME/far2l/cmake-build-debug/install/far2l'

# ???
bind '"\e[A": history-search-backward'
bind '"\e[B": history-search-forward'

# added by new clang
export CLANG_HOME=/usr/local/Cellar/llvm/4.0.0_1
export PATH=$CLANG_HOME/bin:$PATH
# for openmp
export LIBRARY_PATH=$CLANG_HOME/lib:$LIBRARY_PATH

# added by Anaconda2 4.0.0 installer
export PATH=$PATH:/Users/dy/anaconda2/bin

#/usr/local/bin:/usr/local/sbin:/usr/bin:/bin:/sbin:/usr/sbin

# java
export JAVA_HOME=`/usr/libexec/java_home -v 1.8`
export PATH=$JAVA_HOME/bin:$PATH

# R
export R_HOME=/Library/Frameworks/R.framework/Resources
export PATH=$R_HOME/bin:$PATH
#export http_proxy=http://127.0.0.1:9743
#export HTTP_PROXY=http://127.0.0.1:9743

# scala
export SCALA_HOME=/Users/dy/scala-2.12.1
export PATH=$SCALA_HOME/bin:$PATH

#added by clang-omp
#export OPENMP_HOME=/usr/local/homebrew/Cellar/libiomp/20150701
#export CLANGOMP_HOME=/usr/local/homebrew/Cellar/clang-omp/2015-04-01
#export PATH=$CLANGOMP_HOME/bin:$PATH
#export C_INCLUDE_PATH=$CLANGOMP_HOME/libexec/include/clang-c:$OPENMP_HOME/include:$C_INCLUDE_PATH
#export CPLUS_INCLUDE_PATH=$CLANGOMP_HOME/libexec/include/c++/v1:$OPENMP_HOME/include:$CPLUS_INCLUDE_PATH
#export LIBRARY_PATH=$CLANGOMP_HOME/libexec/lib:$OPENMP_HOME/include:$LIBRARY_PATH
#export LD_LIBRARY_PATH=$CLANGOMP_HOME/libexec/lib:$OPENMP_HOME/include:$LD_LIBRARY_PATH

#added by cuda
export CUDA_HOME=/Developer/NVIDIA/CUDA-7.5
#export LD_LIBRARY_PATH=$CUDA_HOME/lib:$LD_LIBRARY_PATH
#export DYLD_LIBRARY_PATH=$CUDA_HOME/lib:$DYLD_LIBRARY_PATH
export LIBRARY_PATH=$CUDA_HOME/lib:$LIBRARY_PATH
export PATH=$CUDA_HOME/bin:$PATH

#added by opencv
export OPENCV_HOME=/usr/local/Cellar/opencv/2.4.11_2
export OPENCV_LIBS=$OPENCV_HOME/lib;
export LIBRARY_PATH=$OPENCV_HOME/lib:$LIBRARY_PATH

#added by OpenBlas
#export OpenBLAS_HOME=/usr/local/opt/openblas
