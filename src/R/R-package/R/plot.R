
# d(ata) : matrix of dim (#doc, #principal component)
# params : gmm model params (mean, cov, class_weight...)
plot.gmm.dy <- function(params, d, ellipse.dots=100, ...) {
    library(ggplot2)
    library(irlba)
    dd <- d
    mean <- params$mean
    cov  <- params$cov
    resp <- params$resp
    nclass      <- ncol(resp)
    which.class <- apply(resp, 1, which.max)
    which.color <- rainbow(nclass)[which.class]
    if (ncol(d) > 2) {
        #  SLOW
        # dd   <- matrix(0, nrow = nrow(d), ncol = 2)
        # mean <- matrix(nrow = nclass, ncol = 2)
        # cov  <- matrix(nrow = nclass, ncol = 2)
        # which.two <- matrix(nrow = nclass, ncol = 2)
        # for (i in 1:nclass) {
        #     # PCA
        #     pc <- order(-params$mean[i,], params$cov[i,])[1:2]
        #     mean[i,] <- params$mean[i, pc]
        #     cov[i,] <- params$cov[i, pc]
        #     which.two[i,] <- pc
        # }
        # str(which.two)
        # for (i in 1:nrow(d)) {
        #     dd[i,] <- d[i, which.two[which.class[i],]]
        #     print(dd[i,])
        # }
        dd <- d %*% irlba(d, nv=2, center=colMeans(d), right_only=T)$v
        print(summary(dd@x))
        gg <- ggplot(data=data.frame(X=dd[,1], Y=dd[,2])) +
            geom_point(aes(x=X, y=Y, color=as.factor(which.class))) +
            # stat_ellipse(aes(x=Y, y=Y, color=which.color, group=which.class), type='euclid')
            stat_(aes(x=Y, y=Y, color=which.color, group=which.class), type='euclid')
        print(gg)
        # for (i in 1:nclass) {
        #     ps <- ellipse.points(mean[i, 1], mean[i, 2], cov[i, 1], cov[i, 2], dots=ellipse.dots)
        #     gg <- gg + geom_point(data=ps, color=as.character(i)) + stat_ellipse(data=ps)
        # }
        # print(gg)
    }
}


ellipse.points <- function(cx, cy, xr, yr, dots=100, rot=0) {
    angle <- seq(0, 2*pi, length.out=dots)
    a     <- xr
    b     <- yr
    r     <- a * b / sqrt(a ^ 2 * sin(angle) ^ 2 + b ^ 2 * cos(angle) ^ 2)
    if (rot == 0) {
        x     <- cx + r * cos(angle)
        y     <- cy + r * sin(angle)
        # lines(x, y, ...)
    } else {
        rotateM <- matrix(c(cos(rot), -sin(rot), sin(rot), cos(rot)), nrow = 2, byrow = T)
        print(rotateM %*% solve(rotateM))
        x     <- r * cos(angle)
        y     <- r * sin(angle)
        m     <- matrix(c(x, y), nrow = 2, byrow = T)
        m     <- rotateM %*% m
        # lines(x = m[1,] + cx, y = m[2,] + cy, ...)
    }
    # x <- cx + a*cos(angle)*cos(rot) - b*sin(angle)*sin(rot)
    # y <- cy + a*cos(angle)*cos(rot) + b*sin(angle)*cos(rot)
    # lines(x, y, ...)
    return(data.frame(x=x, y=y))
}

plot.test <- function() {
    p <- ggplot(data=data.frame(X=ts$Y[,1], Y=ts$Y[,2])) +
        geom_point(aes(x=X, y=Y, color=as.factor(kp$belong))) +
        scale_color_hue(name='Topic', labels=apply(kp$topics, 2, paste0, collapse=',')) +
        theme(legend.text=element_text(family = 'Songti SC Light'))
}

plot.quadtree <- function(nodes, points, nearby=NULL, point=NULL) {
    require(ggplot2)
    gg <- ggplot() + geom_point(mapping=aes(x=points[,1], y=points[,2])) +
        geom_rect(data=nodes, mapping=aes(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax), color='black', alpha=0)
    if (!is.null(nearby)) {
        gg <- gg + geom_point(mapping=aes(x=point[1], y=point[2]), color='red') +
            geom_rect(data=nearby, mapping=aes(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax, fill=as.factor(capacity)), color='black', alpha=0.5)
    }
    print(gg)
}

