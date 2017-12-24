.onLoad <- function(libname, pkgname) {
    library.dynam(chname = 'libcorn', package = pkgname, lib.loc = libname, verbose = T)
}

.onUnload <- function(libpath) {
    library.dynam.unload(chname = 'libcorn', libpath = libpath, verbose = T)
}

