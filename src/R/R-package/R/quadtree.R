setClass("QuadTree",
         representation( pointer = "externalptr"))
# helper
quadtree_method <- function(name) {
    paste("QuadTree", name, sep = "__")
}
# syntactic sugar to allow object$method( ... )
setMethod("$", "QuadTree", function(x, name) {
    function(...)
        .Call(quadtree_method(name), PACKAGE=rcudanlp.path(),
              x@pointer, ...)
})

# syntactic sugar to allow new( "Uniform", ... )
setMethod("initialize", "QuadTree",
          function(.Object, ...) {
              .Object@pointer <-
                  .Call(quadtree_method("new"), PACKAGE=rcudanlp.path(), ...)
              .Object
          })


test.quadtree <- function(n=5L, seed=1L, theta=0.5, plt=T) {
    set.seed(seed)
    points <- matrix(rnorm(n * 2), ncol=2)
    qt <- new('QuadTree', points)
    nodes <- qt$allNodes(F)
    point <- rnorm(2)
    nearbies <- qt$nearbyNodes(point, theta)
    if (plt) plot.quadtree(nodes, points, nearbies, point)
}
