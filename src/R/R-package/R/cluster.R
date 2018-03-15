
xmeans <- function(x, ...) UseMethod('xmeans', x)
xmeans.default <- function(x, centers, iter.max = 10L, algorithm = c("Elkan", "Cuda"), seeds = 1:1, tol = 1e-6, verbose = F) {
    .Call(.xmeans.matrix, as.matrix(x), centers, iter.max, algorithm[1], seeds, tol, verbose);
}
xmeans.dgCMatrix <- function(x, centers, iter.max = 10L, algorithm = c("Elkan", "Cuda"), seeds = 1:1, tol = 1e-6, verbose = F) {
    .Call(.xmeans.dtm, x, centers, iter.max, algorithm[1], seeds, tol, verbose);
}


tune.xmeans <- function(x, centers = 2:sqrt(nrow(x)), seeds = list(c(1:3))) {
    stopifnot(nrow(x) > 1 && all(centers > 1 & centers < nrow(x)))
    grids = make.grids(x = x, centers = centers, seeds = seeds)
    r <- list()
    for (i in 1:nrow(grids)) r <- append(r, list(do.call(xmeans, args = unlist(grids[i,]))))
    r
}

choose.K <- function(x, ...) {
    tot.withinss <- as.numeric(Map(function(x) with(x, loss[itr]), x))
    K <- as.numeric(Map(function(x) with(x, nrow(centers)), x))
    plot(K, tot.withinss, type = 'o')
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

xmm <- function(x, m = c("guassian"), k, iter.max = 10L, tol = 1e-5, ...) UseMethod('xmm', x)
xmm.dgCMatrix <- function(x, m = c("guassian"), k = as.integer(sqrt(nrow(x))), iter.max = 10L, tol = 1e-5,
                          lb.prior = 1 / k / 10, lb.var = 1e-5,
                          prior = rep(1 / k, k), mean = matrix(rnorm(k * cols), k), 
                          var = matrix(runif(k * cols, max = 10), k), verbose = F) {
    rows <- nrow(x)
    cols <- ncol(x)
    k <- min(k, rows)
    stopifnot(1 < k && k < rows)
    if (m[1] == 'guassian') {
        .Call(.xmm.gaussian.dtm, x, k, iter.max, tol, lb.prior, lb.var, prior, mean, var, verbose);
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

sort.cluster <- function(x) {
    stopifnot(is.integer(x))
    map <- rep(-1L, max(x))
    cnt <- 0L
    for (i in 1:length(x))
        if (map[x[i]] == -1L) {
            cnt <- cnt + 1
            map[x[i]] <- cnt
            x[i] <- cnt
        } else x[i] = map[x[i]]
    x
}
