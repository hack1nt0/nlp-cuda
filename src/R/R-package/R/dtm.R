require(Matrix)
require(irlba)

# setClass("dtm", representation(pointer = "externalptr"))
setClass("dtm", representation(idf = "numeric", d.len = "integer"), contains = "dgCMatrix")

# # helper
# .dtm_method <- function(name) {
#     paste0(name, '_dtm')
# }
#
# # syntactic sugar to allow object$method( ... )
# setMethod("$", "dtm", function(x, name) {
#     function(...)
#     .Call(.dtm_method(name), PACKAGE = 'libcorn', x@pointer, ...)
# })

# setMethod("initialize", "dtm",
#           function(.Object) {
#               .Object
#           })

dtm <- function(train.wll = NULL, test.wll = NULL, par = 0, dgCMatrix = NULL, idf = NULL) {
    stopifnot(!is.null(train.wll) || !is.null(dgCMatrix))
    if (!is.null(train.wll))
        .Call(C_new_dtm, train.wll, test.wll, par)
    else {
        r <- as(dgCMatrix, 'dtm')
        r@idf = idf
        r
    }
}

setMethod("[", signature(x = "dtm", i = "index", j = "index", drop = "logical"),
          function(x, i, j, ..., drop = TRUE) {
              super <- as(x, 'dgCMatrix')[i,j,drop = drop]
              if (drop) super else dtm(dgCMatrix = super, idf = x@idf)
          })
setMethod("[", signature(x = "dtm", i = "index", j = "missing", drop = "logical"),
          function(x, i, j, ..., drop = TRUE) { #todo
              super <- as(x, 'dgCMatrix')[i,,drop = drop]
              if (drop) super else dtm(dgCMatrix = super, idf = x@idf)
          })
setMethod("[", signature(x = "dtm", i = "missing", j = "index", drop = "logical"),
          function(x, i, j, ..., drop = TRUE) {
              super <- as(x, 'dgCMatrix')[,j,drop = drop]
              if (drop) super else dtm(dgCMatrix = super, idf = x@idf)
          })

setMethod("[[", "dtm", function(x, i) x[i,,drop=F])

setMethod("summary", signature(object="dtm"), function(object, ...) {
    x <- object
    r <- list()
    dict <- x@Dimnames[[2L]]
    idf  <- x@idf
    for (i in 1:nrow(x)) {
        di <- Matrix(x[i, ], sparse = T)
        row_ind <- di@i + 1
        ri <- data.frame(w.id = row_ind, local = di@x, global = idf[row_ind], word = dict[row_ind])
        r <- append(r, list(ri))
    }
    if (nrow(x) == 1L) r[[1L]] else r
})

topwords. <- function(x) {
    topwords <- data.frame(i = 1:ncol(x), idf = x@idf, word = colnames(x))
    topwords[order(topwords$idf, topwords$word, decreasing = T),]
}

scale.dtm <- function(x, L = 2L, byrow = T) {
    .Call(.scale.dtm, x, L, byrow)
}

is.dtm <- function(x) { return(class(x) == "dtm") }

test.dtm <- function(x, y = NULL) {
    z = 1:10
    .Call(.test.dtm, x, y)
}

tk <- function(texts,
               type = "mix",
               dict = jiebaR::DICTPATH,
               hmm = jiebaR::HMMPATH,
               user = jiebaR::USERPATH,
               idf = jiebaR::IDFPATH,
               stop_word = jiebaR::STOPPATH,
               write = T,
               qmax = 20,
               topn = 5,
               encoding = "UTF-8",
               detect = F,
               symbol = T,
               lines = 1e+05,
               output = NULL,
               bylines = T) {
    wk <- jiebaR::worker(type, dict, hmm, user, idf, stop_word, write, qmax, topn, encoding, detect,
                         symbol, lines, output, bylines)
    return(jiebaR::segment(texts, wk))
}

colVar <- function(x, ...) UseMethod("colVar", x)
colVar.dtm <- function(x, ...) {
    x = as(x, "dgCMatrix")
    Mean = colMeans(x, ...) 
    (colSums(x ^ 2, ...) - 2 * Mean * colSums(x, ...) + Mean * Mean * nrow(x)) / (nrow(x) - 1)
}
colSd <- function(x, ...) UseMethod("colSd", x)
colSd.dtm <- function(x, ...) {
    sqrt(colVar(x, ...))
}

rdtm <- function(..., colNames = NULL, rowNames = NULL) {
    res = rsparsematrix(...)
    dimnames(res) = list(rowNames, colNames)
    as(res, "dtm")
}

prcomp.dtm <- function(x, n = 2L, ...) {
    Mean = colMeans(x)
    Scale = pmax(colSd(x), .Machine$double.eps)
    ir.out = irlba(as(x, "dgCMatrix"), nv = n, nu = 0L, right_only = T, center = Mean, scale = Scale, ...)
    pr.out = structure(list(sdev = sqrt(ir.out$d ^ 2 / (nrow(x) - 1L)),
                            rotation = ir.out$v,
                            center = Mean,
                            scale = Scale,
                            # x = ir.out$u %*% diag(ir.out$d)
                            x = as.matrix(x %*% ir.out$v)
    ), class = c("prcomp.dtm", "prcomp"))
    # print(all.equal(pr.out$x, as.matrix(x %*% pr.out$rotation)))
    
    names.obs = rownames(x); if (is.null(names.obs)) names.obs = as.character(1:nrow(x))
    names.var = colnames(x); if (is.null(names.var)) names.var = paste("Var", as.character(1:ncol(x)))
    rownames(pr.out$rotation) = names(pr.out$center) = 
        names(pr.out$scale) = names.var
    colnames(pr.out$rotation) = names(pr.out$sdev) =
        colnames(pr.out$x) = paste0("PC", 1:n)
    rownames(pr.out$x) = names.obs
    pr.out
}

biplot.prcomp.dtm <- function(x, ...) {
    outliers = union(as.integer(names(boxplot.stats(x$x[,1L])$out)),
                     as.integer(names(boxplot.stats(x$x[,2L])$out)))
    if (is.vector(outliers)) x$x = x$x[-outliers,]
    require(ggbiplot)
    ggbiplot(x, ...)
    # plot(x$x)
}

# {
#     x = rdtm(10, 10, 0.3)
#     n = 3
#     ir.out = irlba(as(x, "dgCMatrix"), n, center = colMeans(x), 
#                    scale = colSd(x))
#     ir.scores = ir.out$u %*% diag(ir.out$d)
#     ir2.out = prcomp_irlba(x, n, center = T, scale. = )
#     pr.out = prcomp(x, n, center = T, scale. = T)
#     pr.scores = pr.out$x
#     all.equal(ir.scores, pr.scores[,1:n])
#     # range(ir.scores - pr.scores[,1:n])
# }
