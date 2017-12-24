tsne <- function(X, dimY = 2, maxItr = 100, perplexity = 30, theta = 0.3, seed = 1, verbose = F) {
    .Call("tsne", PACKAGE='librcudanlp', X@pointer, dimY, maxItr, perplexity, theta, seed, verbose)
}

plot.Tsne <- function(Y, cl = NULL) {
    library(ggplot2)
    if (is.null(cl)) cl = rep(0, nrow(Y))
    ggplot() + geom_point(mapping = aes(x = Y[,1] , y = Y[,2], col = cl))
}


test.tsne <- function(dimY = 2, maxItr = 10, perplexity = 30, theta = 0.3, seed = 1) {
    dtm <- read.dtm('/Users/dy/TextUtils/data/train/spamsms.dtm')
    Y <- tsne(dtm, dimY = dimY, maxItr = maxItr, perplexity = perplexity, theta = theta, seed = seed, verbose = T)
    plot(Y)
}
