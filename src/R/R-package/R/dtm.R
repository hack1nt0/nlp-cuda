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
    if (!is.null(train.wll)) {
        .Call(.new.dtm, train.wll, test.wll, par)
    }else {
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

scale.dtm <- function(x, byrow = T, L = 0L) {
    .Call(.scale.dtm, x, byrow, L)
}

is.dtm <- function(x) { return(class(x) == "dtm") }

test.dtm <- function(xs) {
    .Call(.test.dtm, xs)
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
