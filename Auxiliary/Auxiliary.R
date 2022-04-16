##
### Plots a tile of the MNIST handwritten digits data.
plot_tile <- function(x,y,y_pred,n,m,from=1L,space=c(4L,4L),res=dim(x)[-1L]) {
  tile <- matrix(NA,(res[1L]+space[1L])*n,(res[2L]+space[2L])*m)
  for(i in 1L:n)
    for(j in 1L:m) {
      k <- m*(i-1L)+j+from-1L
      if(k <= dim(x)[1L])
        tile[(res[1L]+space[1L])*(i-1)+(1L:res[1L]),
             (res[2L]+space[2L])*(j-1)+(1L:res[2L])] <-
          t(x[k,dim(x)[2L]:1L,])
    }
  par(mar=c(1,1,1,1))
  image(tile,zlim=c(0,255),col=grey(seq(1,0,length.out=256L)),axes=FALSE,xlab="",ylab="",asp=1)
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
decode_review <- function(x, rev_widx)
  sapply(
    x,
    function(index) {
      word <- if(index >= 3L) rev_widx[[as.character(index - 3L)]]
      if(!is.null(word)) word else "?"
    }
  )
##



