##
plot_tile <- function(x,y,y_pred,n,m,from=1L,space=c(2L,2L),res=dim(x)[c(2L,3L)]) {
    tile <- array(0,dim=c((res[1L]+space[1L])*n,(res[2L]+space[2L])*m,3L))
    for(i in 1L:n)
        for(j in 1L:m) {
            k <- m*(i-1L)+j+from-1L
            if(k <= dim(x)[1L])
                tile[(res[1L]+space[1L])*(i-1)+(1L:res[1L]),(res[2L]+space[2L])*(j-1)+(1L:res[2L]),] <-
                    x[k,,]
        }
    par(mar=c(1,1,1,1))
    plot(pixmapRGB(tile,dim(tile)[1L],dim(tile)[2L]),axes=FALSE)
    dev.hold()
    for(i in 1L:n)
        for(j in 1L:m)
            text(y=(n-i+1)*(res[1L]+space[1L])-2*space[1L],x=(j-1)*(res[2L]+space[2L])+2*space[2L],labels=y[m*(i-1L)+j+from-1L],xpd=TRUE,col="white",font=2L)
    dev.flush()
    if(!missing(y_pred)) {
        dev.hold()
        for(i in 1L:n)
            for(j in 1L:m) {
                k <- m*(i-1L)+j+from-1L
                if(k <= length(y_pred))
                    text(y=(n-i+1)*(res[1L]+space[1L])-6*space[1L],x=(j-1)*(res[2L]+space[2L])+2*space[2L],
                         labels=y_pred[k],xpd=TRUE,
                         col=if(y_pred[k]==y[k]) "green" else "red",font=2L)
            }
        dev.flush()
    }
    return(invisible(NULL))
}
##
plot_tile2 <- function(x,y,y_pred,n,m,from=1L,space=c(4L,4L),res=dim(x)[-1L]) {
    tile <- matrix(NA,(res[1L]+space[1L])*n,(res[2L]+space[2L])*m)
    for(i in 1L:n)
        for(j in 1L:m) {
            k <- m*(i-1L)+j+from-1L
            if(k <= dim(x)[1L])
                tile[(res[1L]+space[1L])*(i-1)+(1L:res[1L]),(res[2L]+space[2L])*(j-1)+(1L:res[2L])] <-
                    t(x[k,dim(x)[2L]:1L,])
        }
    par(mar=c(1,1,1,1))
    image(tile,zlim=c(0,255),col=grey(seq(1,0,length.out=256L)),axes=FALSE,xlab="",ylab="")
    dev.hold()
    for(i in 1L:n)
        for(j in 1L:m)
            text(x=i/n,y=j/m-0.025,labels=y[m*(i-1L)+j+from-1L],xpd=TRUE,col="blue")
    dev.flush()
    if(!missing(y_pred)) {
        dev.hold()
        for(i in 1L:n)
            for(j in 1L:m) {
                k <- m*(i-1L)+j+from-1L
                if(k <= length(y_pred))
                    text(x=i/n,y=(j-1L)/m+0.025,labels=y_pred[k],xpd=TRUE,
                         col=if(y_pred[k]==y[k]) "red" else "green")
            }
        dev.flush()
    }
    return(invisible(NULL))
}
##
