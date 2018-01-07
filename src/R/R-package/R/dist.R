# # distance for dtm
# setClass("Dist", representation( pointer = "externalptr"))
# 
# # helper
# .dist_method <- function(name) {
#     paste("Dist", name, sep = "_")
# }
# 
# # ONLY CAN BE NEW from DTM dist()
# setMethod("initialize", "DIST",
#           function(.Object) {
#               .Object
#           })
# 
# setMethod("$", "Dist", function(x, name) {
#     function(...)
#         .Call(.dist_method(name), PACKAGE='librcudanlp', x@pointer, ...)
# })
# 
# 
# setMethod("[", "Dist", function(x, i, j) {
#         .Call(.dist_method("at"), PACKAGE='librcudanlp', x@pointer, i, j)
# })
# 
# summary.Dist <- function(dist, exact=F, mi=0, ma=-1) {
#     dist$summary(exact, mi, ma)
# }
# 
# read.Dist <- function(path) {
#     d <- new('Dist')
#     d@pointer <- .Call(.dist_method("read"), PACKAGE='librcudanlp', path)
#     return(d)
# }
# 
# dist <- function(x, method = "euclidean", diag = FALSE, upper = FALSE, p = 2, verbose=F) {
#     if (class(x) != 'dtm') return(stats::dist(x, method, diag, upper, p));
#     r <- x$dist(verbose)
#     d <- new('Dist')
#     d@pointer <- r
#     return(d)
# }

dist <- function(x, ...) UseMethod('dist', x)
dist.default <- stats::dist
dist.dgCMatrix <- function(x, y = NULL, metric = "euclidean", verbose = F) {
    .Call(.dist.dtm, x, y, metric, verbose)
}
