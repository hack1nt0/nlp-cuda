
# setGeneric("knn", def = function(.o) standardGeneric('knn'))
# setMethod("knn", "dgCMatrix", function(.o) str(.o))

knn <- function(x, ...) UseMethod('knn', x)
knn.default <- class::knn
knn.dgCMatrix <- function(train, test = train[1L,,drop=F], cl = NULL, k = 1L, 
                          dtype = c("euclidean", "cosine", "edit"),
                          l = 0L, prob = FALSE, use.all = TRUE,
                          method = c('brute', 'vptree', 'kdtree'),
                          leafsize = 2L,
                          seed = 1L, verbose = F) {
    if (is.numeric(test)) {
        stopifnot(length(test) == ncol(train))
        test <- Matrix(matrix(test, nrow = 1L), sparse = T)
    } else if (is.matrix(test)) {
        test <- Matrix(test, sparse = T)
    }
    stopifnot(ncol(train) == ncol(test))
    # print(str(test))
    # print("hi")
    .Call(.knn.dtm, train, test, cl, k, dtype, l, prob, use.all, method, leafsize, seed, verbose)
}
