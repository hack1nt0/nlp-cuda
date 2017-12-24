# library(Rtsne) # Load package
# iris_unique <- unique(iris) # Remove duplicates
# set.seed(42) # Sets seed for reproducibility
# tsne_out <- Rtsne(as.matrix(iris_unique[,1:4])) # Run TSNE
# plot(tsne_out$Y,col=iris_unique$Species) # Plot the result
#
# m <- read.csv('~/Downloads/mnist-train.csv')
# mm <- Rtsne(as.matrix(m[,1:dim(m)[2]]))
# plot(mm$Y, col=as.character(m$label)) # Plot the result
