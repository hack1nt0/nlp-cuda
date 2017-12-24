
gmm <- function(dtm, k, max_itr=10, seed=17, alpha=1e-5, beta=1e-5, topics=10) {
    .Call('gmm', PACKAGE = 'librcudanlp', dtm, k, max_itr, seed, alpha, beta, topics);
}

xmeans <- function(x, ...) UseMethod('xmeans', x)
xmeans.default <- function(x, centers, iter.max = 10L, nstart = 1L, algorithm = c("Elkan", "Cuda"), seed = 1L, tol = 1e-6, verbose = F) {
    .Call(.xmeans.matrix, as.matrix(x), centers, iter.max, nstart, algorithm[1], seed, tol, verbose);
}
xmeans.dgCMatrix <- function(x, centers, iter.max = 10L, nstart = 1L, algorithm = c("Elkan", "Cuda"), seed = 1L, tol = 1e-6, verbose = F) {
    .Call(.xmeans.dtm, x, centers, iter.max, nstart, algorithm[1], seed, tol, verbose);
}

