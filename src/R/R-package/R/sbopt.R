make.grids <- function(xs) {
    stopifnot(is.list(xs))
    xnames <- names(xs)
    if (is.null(xnames) || any(xnames == '')) stop("Arguments must be named")
    xs <- lapply(xs, function(x) if (is.vector(x)) x else list(x))
    nx <- length(xs)
    each <- Reduce(`*`, sapply(xs, length), init = 1L, accumulate = T)
    nrow <- each[nx + 1L]
    require(tibble)
    r <- tibble(... = 1:nrow)
    for (i in 1:nx) {
        r[[xnames[i]]] <- rep(xs[[i]], each = each[i], len = nrow)
    }
    r[-1L]
}

as.matrix.cv.grids <- function(x, ...) 
    cbind(lapply(x, 
                 function(xx) if (is.character(xx)) as.double(as.factor(xx)) else as.double(xx)
    ))

kernel.matrix <- function(A, B = A, type = c("SE", "M52", "SE.ARD", "M52.ARD"), alpha, theta) {
    type = match.arg(type)
    kernel.func = switch(type, SE = k.se, M52 = k.m52, SE.ARD = k.seard, M52.ARD = k.m52ard)
    stopifnot(ncol(A) == ncol(B))
    n <- nrow(A); m <- nrow(B); r <- matrix(NA, n, m)
    for (i in 1:n) for (j in 1:m) r[i, j] <- kernel.func(A[i,], B[j,], alpha, theta)
    r
}
k.se <- function(a, b, alpha, theta) alpha * exp(-sum((a - b) ^ 2))
k.seard <- function(a, b, alpha, theta) alpha * exp(-sum(((a - b) / theta) ^ 2))
k.m52ard <- function(a, b, alpha, theta) {
    rsq = sum(((a - b) / theta) ^ 2)
    alpha * (1 + sqrt(5 * rsq) + 5 * rsq / 3) * exp(-sqrt(5 * rsq))
}

ginv.sym <- function(x) {
    res = tryCatch({
        u = chol(x)
        list(det = prod(diag(u)) ^ 2, inv = chol2inv(u))
    }, error = function(e) {
        print("ginv&pdet")
        e = eigen(x, symmetric = T)
        val = e$values
        vec = e$vectors
        if (prod(val) <= 0) {
            print(x)
            print(val)
            stop()
        }
        # val[val == 0] = 1
        list(det = prod(val), inv = vec %*% (1 / val * t(vec)))
        # xsvd = svd(x)
        # d = xsvd$d; d[d == 0] = 1
        # list(det = 1 / prod(d), inv = xsvd$v %*% (1 )
    })
}

# # gp <- function(x, y, ...) UseMethod('gp', x) 
mle.gp <- function(x, y, kernel = c("SE.ARD", "M52.ARD", "M52", "SE"), par, ...) {
    stopifnot(nrow(x) == nrow(y) && ncol(y) == 1L)
    kernel = match.arg(kernel)
    kernel.func = switch(kernel, SE = k.se, M52 = k.m52, SE.ARD = k.seard, M52.ARD = k.m52ard)
    t.y = t(y)
    K = function(A, B, alpha, theta) {
        n <- nrow(A); m <- nrow(B); r <- matrix(NA, n, m)
        for (i in 1:n) for (j in 1:m) r[i, j] <- kernel.func(A[i,], B[j,], alpha, theta)
        r
    }
    fn = function(par) {
        alpha = par[1L]
        sigma = par[2L]
        theta = par[-(1:2)]
        Sigma = K(x, x, alpha, theta) + diag(sigma ^ 2, nrow(x))
        # Sigma = K(x, x, par)
        r = ginv.sym(Sigma)
        log(r$det) + t.y %*% r$inv %*% y
    }
    opt.obj <- optim(par, fn, ...);
    par = opt.obj$par
    res = function(nx) {
        alpha = par[1L]
        sigma = par[2L]
        theta = par[-(1:2)]
        Kxx = K(x, x, alpha, theta) + diag(sigma ^ 2, nrow(x))
        inv.Kxx = ginv.sym(Kxx)$inv
        Knx = K(nx, x, alpha, theta)
        Knn = K(nx, nx, alpha, theta) + diag(sigma ^ 2, nrow(nx))
        mean <- Knx %*% inv.Kxx %*% y
        corr <- Knn + diag(sigma ^ 2, nrow(nx)) - Knx %*% inv.Kxx %*% t(Knx)
        structure(list(mean = mean, Sigma = corr), class = "mle.gp.pred")
    }
    class(res) = c(class(res), "mle.gp")
    res
}

fix.gp <- function(x, y, kernel = c("SE.ARD", "M52.ARD", "M52", "SE"), par, ...) {
    stopifnot(nrow(x) == nrow(y) && ncol(y) == 1L)
    kernel = match.arg(kernel)
    kernel.func = switch(kernel, SE = k.se, M52 = k.m52, SE.ARD = k.seard, M52.ARD = k.m52ard)
    t.y = t(y)
    K = function(A, B, alpha, theta) {
        n <- nrow(A); m <- nrow(B); r <- matrix(NA, n, m)
        for (i in 1:n) for (j in 1:m) r[i, j] <- kernel.func(A[i,], B[j,], alpha, theta)
        r
    }
    res = function(nx) {
        alpha = par[1L]
        sigma = par[2L]
        theta = par[-(1:2)]
        Kxx = K(x, x, alpha, theta) + diag(sigma ^ 2, nrow(x))
        inv.Kxx = ginv.sym(Kxx)$inv
        Knx = K(nx, x, alpha, theta)
        Knn = K(nx, nx, alpha, theta) + diag(sigma ^ 2, nrow(nx))
        mean <- Knx %*% inv.Kxx %*% y
        corr <- Knn + diag(sigma ^ 2, nrow(nx)) - Knx %*% inv.Kxx %*% t(Knx)
        structure(list(mean = mean, Sigma = corr), class = "fix.gp.pred")
    }
    class(res) = c(class(res), "fix.gp")
    res
}

predict.mle.gp <- function(obj, newX) obj(newX)
predict.fix.gp <- predict.mle.gp

sbopt <- function(f, hp.space, n.init = 1L, n.iter = 10L, scale = F, 
                   gp.func = mle.gp, ..., train.x = NULL, train.y = NULL) {
    partial <- function(f, ...) {
        l <- list(...)
        function(xs) {
            do.call(f, c(l, xs))
        }
    }
    if (is.null(train.x)) cv <- function(hp) do.call(f, as.list(hp)) 
    else cv <- function(hp) {
        require(e1071)
        tune(f, train.x, train.y, ranges = as.list(hp))$best.performance
    }
    grids <- make.grids(hp.space)
    x <- as.matrix(grids)
    if (scale) x <- scale(x)
    n = nrow(x)
    n.iter = min(n.iter, n)
    xtrain <- sample.int(n, n.init)
    y = matrix(Inf, n, 1L)
    ei <- function(best.y, mean.x, sd.x) {
        rho = (best.y - mean.x) / sd.x
        sd.x * (rho * pnorm(rho) + dnorm(rho))
    }
    eis = rep(0, n)
    y[xtrain,] = apply(grids[xtrain,], 1L, cv)
    best.y = min(y)
    best.xi = xtrain[which.min(y)]
    gp.prd <- NULL
    eps = .Machine$double.eps
    require(foreach)
    N = 1:n
    for (i in 1:n.iter) {
        gp.obj = gp.func(x[xtrain,,drop=F], y[xtrain,,drop=F], ...)
        gp.prd = predict(gp.obj, x)
        foreach(j = N, me = as.vector(gp.prd$mean), se = sqrt(pmax(eps, diag(gp.prd$Sigma)))) %do% {
            eis[j] = ei(best.y, me, se)
        }
        # Why eis is of class 'matrix'??? todo
        nxi = which.max(eis)
        if (sum(xtrain == nxi) != 0L) {
            print(paste("iter =", i, "duplicated grid =", nxi))
            break
        }
        ny = cv(grids[nxi,])
        if (ny < best.y) { best.y = ny; best.xi = nxi; }
        xtrain = c(xtrain, nxi)
        y[nxi,1L] = ny
    }
    structure(list(best.x = grids[best.xi,,drop=T], best.y = best.y,
                   grids = grids, train.xi = xtrain,
                   mean = gp.prd$mean, se = sqrt(pmax(0, diag(gp.prd$Sigma))), acq = eis), class = "sbopt")
}

plot.sbopt <- function(x, true.f = NULL, ...) {
    op <- par(mfrow = c(1, 2))
    with(x, {
        n = nrow(mean)
        up <- mean + 2 * se; lo <- mean - 2 * se; 
        xlim <- c(1, n); ylim <- c(min(lo), max(up))
        plot(mean, xlim = xlim, ylim = ylim, 
             ylab = "mean(BLACK), true(RED), CI.95%(YELLOW)", xlab = "grid")
        polygon(c(1:n, n:1), c(lo, rev(up)), col = 'yellow', border = NA)
        lines(mean, type = 'b', xlim = xlim, ylim = ylim, col = 'blue')
        text(train.xi, mean[train.xi], labels = as.character(1:length(train.xi)), pos = 1, col = 'blue')
        if (!is.null(true.f)) {
            lines(apply(grids, 1, true.f), type = 'b', col = 'red')
        }
        names(acq) <- as.character(1:n)
        barplot(acq, ylab = 'ACQ', xlab = "grid")
    })
    par(op)
}

six.hump.camel <- function(x1, x2) (4 - 2.1 * x1^2 + x1^4 / 3) * x1^2 + x1 * x2 + (-4 + 4 * x2^2) * x2^2



{
    f <- function(x) {
        if(x < 5)  x ^ 2 + 3
        else if (x < 15) (x - 10) ^ 2
        else (x - 20) ^ 2 + 4
    }
    sbopt.obj <- sbopt(f, list(x = seq(-5, 25, 0.5)), n.init = 2L, n.iter = 10L, scale = F,
                       gp.func = mle.gp, kernel = "SE", method = "L-BFGS-B", par = c(1, 0, 1),
                       lower = c(1, 0, .Machine$double.eps), upper = c(10, 1, 1))
                       # gp.func = fix.gp, kernel = "SE", par = c(1, 0, 1))
    # print(sbopt.obj)
    plot(sbopt.obj, f)
}
