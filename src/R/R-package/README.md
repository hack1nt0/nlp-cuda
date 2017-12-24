# rcudanlp
R with CUDA on NLP

NOT Windows System...as In Windows, CUDA(nvcc) using cl.exe as host compiler, but Rcpp program cannot work with cl.exe.

If something changed, PLEASE let me(jealousing@gmail.com) know.

# Usage

### Compile the share library

https://github.com/hack1nt0/nlp-cuda

Make all targets WITHOUT suffix ".o"

### Setup

```R
library(devtools)
install_github('hack1nt0/rcudanlp')
```

### Read Document-Term-Matrix from csr(Compressed Sparse Row) file 

```R
dtm <- read_csr('path/to/csr')
```

### GMM on Sparse Matrix

```R
modelParams <- gmm(dtm, k = 10, max_itr = 10, seed = 17, alpha = 1e-5, beta = 1e-5)
```

### KMeans on Sparse Matrix

```R
modelParams <- kmeans(dtm, k = 10, max_itr = 10, seed = 17)
```
