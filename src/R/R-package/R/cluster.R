
xmeans <- function(x, ...) UseMethod('xmeans', x)
xmeans.default <- function(x, centers, iter.max = 10L, algorithm = c("Elkan", "Cuda"), seed = 1L, tol = 1e-6, verbose = F) {
    .Call(.xmeans.matrix, as.matrix(x), centers, iter.max, algorithm[1], seed, tol, verbose);
}
xmeans.dgCMatrix <- function(x, centers, iter.max = 10L, algorithm = c("Elkan", "Cuda"), seed = 1L, tol = 1e-6, verbose = F) {
    .Call(.xmeans.dtm, x, centers, iter.max, algorithm[1], seed, tol, verbose);
}

plot.xmeanss <- function(x, ...) {
    if (length(x) > 1) {
        tot.withinss <- as.numeric(Map(function(x) with(x, loss[itr]), x))
        K <- as.numeric(Map(function(x) with(x, nrow(centers)), x))
        plot(K, tot.withinss, type = 'o')
    } else {
        stop("NOT IMPLMT YET.")
    }
}

abstract.matrix <- function(x, dict = NULL, ntop = 10L, decreasing = F, ...) {
    ntop <- min(ntop, ncol(x))
    topids <- apply(x, 1, function(x) order(x, decreasing = decreasing))[1:ntop,]
    if (!is.null(dict)) {
        as.data.frame(matrix(dict[topids], nrow = ntop))
    } else topids
}

summary.xmeans <- function(x, dict = NULL, ntop = 10L, ...) {
    abstract.matrix(x = x$centers, dict = dict, ntop = ntop, decreasing = T, ...)
}

xmm <- function(x, m = c("guassian"), k, max.itr = 10L, tol = 1e-5, ...) UseMethod('xmm', x)
xmm.dgCMatrix <- function(x, m = c("guassian"), k = as.integer(sqrt(nrow(x))), max.itr = 10L, tol = 1e-5,
                          lb.prior = 1 / k / 10, lb.var = 1e-5,
                          prior = rep(1 / k, k), mean = matrix(rnorm(k * cols), k), 
                          var = matrix(runif(k * cols, max = 10), k), verbose = F) {
    rows <- nrow(x)
    cols <- ncol(x)
    k <- min(k, rows)
    stopifnot(1 < k && k < rows)
    if (m[1] == 'guassian') {
        .Call(.xmm.gaussian.dtm, x, k, max.itr, tol, lb.prior, lb.var, prior, mean, var, verbose);
    } else {
        stop('NOT IMPL YET')
    }
}

summary.xmm <- function(x, dict = NULL, ntop = 10L, ...) {
    abstract.matrix(x = x$mean, dict = dict, ntop = ntop, ...)
}

plot.xmm <- function(x, ...) {
    plot(x$avg.log.L, type = 'o')
}
