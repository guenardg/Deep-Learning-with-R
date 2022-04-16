#!/usr/bin/env Rscript
##
### Code d'analyse de séries temporelles.
### Guillaume Guénard - 2016
##






##
if(FALSE) {

##
## subset_ARWtNet <- function(object, subsets) {
##   ans <- list()
##   for(i in unique(subsets)) {
##     # i <- unique(subsets)[1L]
##     ans[[i]] <- object
##     mask <- subsets!=i
##     ans[[i]]$subset <- list(time=ans[[i]]$time[!mask],
##                             response=ans[[i]]$response[!mask],
##                             descriptors=ans[[i]]$descriptors[!mask,,drop=FALSE])
##     ans[[i]]$time <- ans[[i]]$time[mask]
##     ans[[i]]$response <- ans[[i]]$response[mask]
##     ans[[i]]$descriptors <- ans[[i]]$descriptors[mask,,drop=FALSE]
##   }
##   return(structure(ans,class="ARWtNet_cvset"))
## }
## #
## print.ARWtNet_cvset <- function(x, ...) {
##   cat("ARWtNet subset:")
##   cat("\nNumber of subsets:",length(x))
##   cat("\nOriginal time series spanning: ",paste(start(time(x[[1L]]$ts$y)),collapse=" "),"/",frequency(x[[1L]]$ts$y),", ",
##       paste(end(time(x[[1L]]$ts$y)),collapse=" "),"/",frequency(x[[1L]]$ts$y)," ; ",length(x[[1L]]$ts$y)," observations",sep="")
##   if(any(is.na(x[[1L]]$ts$y))) cat(", including",sum(is.na(x[[1L]]$ts$y)),"missing value(s)")
##   cat("\nWidth: ",x[[1L]]$width,", multi-resolution impulse filter: ",x[[1L]]$filter," wavelet, with\ncutoff point(s) set at: ",paste(x[[1L]]$cutoffs,collapse=" "),
##       " and\nsampling at points(s): ",paste(x[[1L]]$sample,collapse=" ")," before present","\n",sep="")
##   cat(if(is.null(x[[1L]]$ts$x)) "no auxiliary series" else paste(ncol(x[[1L]]$ts$x),"auxiliary series:",paste(colnames(x[[1L]]$ts$x),collapse=" ")))
##   cat("\nSample size: ",nrow(x[[1L]]$descriptors),", descriptors: ",paste(colnames(x[[1L]]$descriptors),collapse=", "),sep="")
##   cat("\nSubsets:\n")
##   ss <- data.frame(from=character(length(x)),to=character(length(x)),n=numeric(length(x)),stringsAsFactors=FALSE)
##   for(i in 1L:length(x)) {
##     tmp <- c(head(x[[i]]$subset$time,n=1L),tail(x[[i]]$subset$time,n=1L))
##     ss[i,c("from","to")] <- paste(floor(tmp), paste(round((tmp%%1)*frequency(x[[i]]$ts$y)),"/ 48"))
##     ss[i,"n"] <- length(x[[i]]$subset$time)
##   }
##   print(ss)
##   return(invisible(NULL))
## }
## #


#
DLARWrapper <- function(object,distribution,activation,hidden,l1=0,l2=0,epoch=100L,checkpoint=NULL) {    # Il faudrait pouvoir utiliser un checkpoint.
  if(!h2o::h2o.clusterIsUp())
      stop("h2o cluster must first be started!")
  res <- list()
  res$deeplearn <- h2o::h2o.deeplearning(y=1L,x=2L:ncol(object$h2oData),distribution=distribution,activation=activation,training_frame=object$h2oData,
                     hidden=hidden,l1=l1,l2=l2,epoch=epoch,export_weights_and_biases=TRUE,force_load_balance=TRUE,checkpoint=checkpoint)
  res$coefficients <- list()    # On peut réconstituer les tableau dans res$deeplearn@model$<weights,biases>
  for(i in 1L:(length(hidden)+1L)) {
    # i <- 1L
    biases <- h2o::h2o.biases(res$deeplearn,i)
    weights <- h2o::h2o.weights(res$deeplearn,i)
    res$coefficients[[i]] <- cbind(Bias=as.vector(biases),as.matrix(weights))
    h2o::h2o.rm(biases)
    h2o::h2o.rm(weights)
  }
  if(activation=="Rectifier")
    res$activation <- function(x) {x[x<0] <- 0 ; x}
  if(activation=="Tanh")
    res$activation <- tanh
  return(structure(res,class="DLARWrapper"))
}
#
### Sauvegarde les modèles de l'environnement h2o à l'intérieur même de l'object R (POJO lu comme un fichier binaire).
saveDLModel <- function(object) {
  tmpdir <- tempdir()
  h2o::h2o.saveModel(object$deeplearn,path=tmpdir,force=TRUE)
  object$h2oModel <- readBin(paste(tmpdir,object$deeplearn@model_id,sep="/"),what="raw",
                             n=file.size(paste(tmpdir,object$deeplearn@model_id,sep="/")),size=1L)  # Lit le modèle en format binaire.
  if(!file.remove(paste(tmpdir,object$deeplearn@model_id,sep="/")))
    warning("Temporary file \'",paste(tmpdir,object$deeplearn@model_id,sep="/"),"\' cannot be removed!")
  return(object)
}
#
### Prend les modèles préalablement sauvegardés dans l'environnement R (POJO stocké en fichier binaire) et les replace dans l'environnement h2o.
reloadDLModel <- function(object) {
  tmpdir <- tempdir()
  writeBin(object$h2oModel,paste(tmpdir,object$deeplearn@model_id,sep="/"),size=1L)
  object$deeplearn <- h2o::h2o.loadModel(paste(tmpdir,object$deeplearn@model_id,sep="/"))
  if(!file.remove(paste(tmpdir,object$deeplearn@model_id,sep="/")))
    warning("Temporary file \'",paste(tmpdir,object$deeplearn@model_id,sep="/"),"\' cannot be removed!")
  return(object)
}
#
fitDLARModels <- function(object,distribution,activation,hidden,l1=0,l2=0,epoch=c(100L,100L)) {
  if(!is.null(object$model)) {
    object$model$fit <- DLARWrapper(object=object$model,distribution=distribution,activation=activation,
                                    hidden=hidden,l1=l1,l2=l2,epoch=epoch[1L])
    if(!is.null(object$crval)) {
      for(i in 1L:length(object$crval)) {
        object$crval[[i]]$fit <- DLARWrapper(object=object$crval[[i]],distribution=distribution,activation=activation,
                                             hidden=hidden,l1=l1,l2=l2,epoch=epoch[1L]+epoch[2L],checkpoint=object$model$fit$deeplearn@model_id)
      }
    }
  }
  return(object)
}
#
### Stockage complet: appelle saveDLModel() pour le modèle principal et ceux les jeux de validation croisée (si présents).
saveDLARModels <- function(object) {
  if(!is.null(object$model)) {
    object$model$fit <- saveDLModel(object$model$fit)
    if(!is.null(object$crval)) {
      for(i in 1L:length(object$crval)) {
        object$crval[[i]]$fit <- saveDLModel(object$crval[[i]]$fit)
      }
    }
  }
  return(object)
}
#
### Chargement complet: appelle reloadDLModel() pour le modèle principal et ceux les jeux de validation croisée (si présents).
reloadDLARModels <- function(object) {
  if(!is.null(object$model)) {
    object$model$fit <- reloadDLModel(object$model$fit)
    if(!is.null(object$crval)) {
      for(i in 1L:length(object$crval)) {
        object$crval[[i]]$fit <- reloadDLModel(object$crval[[i]]$fit)
      }
    }
  }
  return(object)
}
#
clearDLARModels <- function(object) {
  if(!is.null(object$model)) {
    h2o::h2o.rm(object$model$fit$deeplearn)
    if(!is.null(object$crval)) {
      for(i in 1L:length(object$crval)) {
        h2o::h2o.rm(object$crval[[i]]$fit$deeplearn)
      }
    }
  }
  return(object)
}
#
cleanH2oFlow <- function(exclude) {
  Flow <- as.matrix(h2o::h2o.ls())[,1L]
  h2o::h2o.rm(Flow[-match(exclude,Flow)])
}
#
### Il n'est pas nécessaire d'utiliser l'environnement h2o pour calculer les prédictions 
predict.ARWtNet <- function(object,newdata) {
  if(is.null(dimnames(newdata)[[2L]]))
    if(ncol(newdata)!=ncol(object$descriptors))
      stop("Number of columns in 'newdata' (",ncol(newdata),") does not match the number of model descriptors (",ncol(object$descriptors),")!")
  else {
    mch <- match(colnames(object$descriptors),colnames(newdata))
    if(any(is.na(mch)))
      stop("Missing descriptor 'newdata' (",paste(colnames(object$descriptors)[is.na(mch)],collapse=" ,"),")!")
    newdata <- newdata[,mch,drop=FALSE]
  }
  tmp <- cbind(1,t((t(newdata)-object$scaled$center[-1L])/object$scaled$scale[-1L]))
  for(i in 1L:(length(object$fit$coefficients)-1L)) {
    # i <- 1L
    tmp <- tmp%*%t(object$fit$coefficients[[i]])
    tmp <- object$fit$activation(tmp)
    tmp <- cbind(1,tmp)
  }
  tmp <- tmp%*%t(object$fit$coefficients[[length(object$fit$coefficients)]])
  return(tmp*object$scaled$scale[1L]+object$scaled$center[1L])
}
#
forecast.ARWtNet <- function(X, from, newdata, nstep, return.descriptors=TRUE, predict.ARWtNet=predict.ARWtNet) {
  if(missing(nstep))
    nstep <- NROW(newdata)
  if(missing(newdata))
    newdata <- matrix(NA,nstep,0L)
  if(nstep>NROW(newdata))
    stop("The number of new data provided (",nrow(newdata),") is insufficient to forecast ",nstep," steps in the future")
  if(missing(from))
    from <- tail(X$time,n=1L)
  nc <- length(X$cutoffs)
  ans <- list(time=from+(1:nstep)/frequency(X$ts$y),
              response=numeric(nstep),
              descriptors=matrix(NA,nstep,ncol(X$descriptors),dimnames=list(NULL,colnames(X$descriptors))))
  if(nstep) {
    rwy <- X$ts$y[which.min(abs(from-time(X$ts$y)))-0L:(X$width-1L),]   # The series is already reversed.
    for(i in 1L:nstep) {
      dwt1 <- wavelets::dwt(X=rwy,filter=X$filter,n.levels=floor(log2(X$width)),boundary="periodic")
      for(j in 1L:nc)
        dwt1@W[[j]][-X$cutoffs[j]] <- 0
      ans$descriptors[i,] <- c(rev(wavelets::idwt(dwt1))[X$width-X$sample+1L],newdata[i,,drop=FALSE])
      ans$response[i] <- predict(X,ans$descriptors[i,,drop=FALSE])
      rwy <- c(ans$response[i],rwy[1L:(X$width-1L)])
    }
  }
  if(return.descriptors)
    return(ans)
  else
    return(ans$response)
}
#
getSSE <- function(X,forecast.ARWtNet=forecast.ARWtNet,predict.ARWtNet=predict.ARWtNet) {
  tmp <- forecast.ARWtNet(X=X,from=X$subset$time[1L],newdata=X$subset$descriptors[,-(1L:length(X$sample)),drop=FALSE],
                          return.descriptors=FALSE,predict.ARWtNet=predict.ARWtNet)
  return(c(sum((tmp-X$subset$response)^2),length(tmp)))
}
#
optMSRE_min <- function(par,object,distribution,activation,epoch,protect=as.matrix(h2o::h2o.ls())[,1L],cluster) {
  hidden <- round(par[(length(par)-2L)]*(exp(par[1L:(length(par)-3L)])/sum(exp(par[1L:(length(par)-3L)]))))
  hidden <- hidden[!!hidden]
  while(TRUE) {
    attempt <- try(fitDLARModels(object,distribution=distribution,activation=activation,hidden=hidden,
                                 l1=10^par[length(par)-1L],l2=10^par[length(par)],epoch=epoch))
    if(class(attempt)=="try-error") cleanH2oFlow(protect)
    else break
  }
  ans <- parLapplyLB(cl=cluster,attempt$crval,getSSE,forecast.ARWtNet=forecast.ARWtNet,predict.ARWtNet=predict.ARWtNet)
  ans <- colSums(matrix(unlist(ans),length(object$crval),2L,byrow=TRUE))
  MRSE <- sqrt(ans[1L]/ans[2L])
  cat("With nodes: (",paste(hidden,collapse=" ,"),"), l1:",10^par[length(par)-1L],", l2:",10^par[length(par)],",MRSE:",MRSE,"\n")
  cleanH2oFlow(protect)
  return(MRSE)
}
#
### Rendu ici.
#
#
#
#
### Les fonctions ci-dessous n'ont pas encore été mises à jour.
#
plot.ARWtNet_forecast <- function(X, trf, object, which, nstep, xlim, type="l", lwd=2, las=1L, col=c("black","green"), mar, xlab="", ylab="", format, forecaster=forecast.ARWtNet) {
  if(missing(mar)) mar <- par()$mar
  if(missing(xlim)) xlim <- c(-object$width/frequency(object$ts$y),1)
  if((X[["from"]]<min(object$time))||(X[["from"]]>max(object$time)))
    stop("Attempting to start prediction from ",X[["from"]],", outside the range of the model (",min(object$time),", ",max(object$time),")!")
  tmp <- trf(forecaster(X=object, which=which, from=X[["from"]], newdata=X[["newdata"]], nstep=nstep, return.descriptors=FALSE))
  if(!missing(format))
    png(width=480,height=480,file=sprintf(format,X[["i"]]))  # X11()
  plot(y=trf(as.numeric(object$ts$y)),x=time(object$ts$y),xlim=X[["from"]]+xlim,type=type,lwd=lwd,las=las,col=col[1L],xlab=xlab,ylab=ylab)
  lab <- as.character(ceiling(xlim[1L]):floor(xlim[2L]))
  lab[lab=="0"] <- paste(floor(X$from)," +",formatC((X$from%%1)*frequency(object$ts$y),2L),"/",frequency(object$ts$y),sep="")
  axis(side=3L,label=lab,at=X[["from"]]+ceiling(xlim[1L]):floor(xlim[2L]))
  lines(y=tmp,x=X[["from"]]+(1:length(tmp))/frequency(object$ts$y),col="red",lwd=lwd)
  abline(v=X$from,lty=3L)
  if(!missing(format))
    dev.off()
  return(invisible(NULL))
}
#
plot.ARWtNet_forecast_Multi <- function(X, trf, object, which, nstep, xlim, type="l", lwd=2, las=1L, col=c("black","green"), mar, xlab="", ylab="", format, forecaster=forecast.ARWtNet) {
  if(missing(mar)) mar <- par()$mar
  if(missing(xlim)) xlim <- c(-object[[1L]]$width/frequency(object[[1L]]$ts$y),1)
  pred <- numeric(nstep)
  obs <- numeric(length(object[[1L]]$ts$y))
  for(i in 1L:length(object)) {
    if((X[["from"]]<min(object[[i]]$time))||(X[["from"]]>max(object[[i]]$time)))
      stop("Attempting to start prediction from ",X[["from"]],", outside the range of the model (",min(object[[i]]$time),", ",max(object[[i]]$time),")!")
    pred <- pred + trf[[i]](forecaster(X=object[[i]], which=which[[i]], from=X[["from"]], newdata=X[["newdata"]][[i]], nstep=nstep, return.descriptors=FALSE))
    obs <- obs + trf[[i]](object[[i]]$ts$y)
  }
  if(!missing(format))
    png(width=480,height=480,file=sprintf(format,X[["i"]]))  # X11()
  plot(y=obs,x=time(object[[1L]]$ts$y),xlim=X[["from"]]+xlim,type=type,lwd=lwd,las=las,col=col[1L],xlab=xlab,ylab=ylab)
  lab <- as.character(ceiling(xlim[1L]):floor(xlim[2L]))
  lab[lab=="0"] <- paste(floor(X$from)," +",formatC((X$from%%1)*frequency(object[[1L]]$ts$y),2L),"/",frequency(object[[1L]]$ts$y),sep="")
  axis(side=3L,label=lab,at=X[["from"]]+ceiling(xlim[1L]):floor(xlim[2L]))
  lines(y=pred,x=X[["from"]]+(1:length(pred))/frequency(object[[1L]]$ts$y),col="red",lwd=lwd)
  abline(v=X$from,lty=3L)
  if(!missing(format))
    dev.off()
  return(invisible(NULL))
}
#













#
extract_descriptors <- function(ts_object, ARWtNet_object) {
  nts <- nrow(ts_object)
  nde <- length(ARWtNet_object$time)
  mde <- length(ARWtNet_object$sample)
  ans <- list()
  ans$time <- time(ts_object)
  ans$descriptors <- matrix(NA,nts,mde,dimnames=list(NULL,colnames(ARWtNet_object$descriptors)[1L:mde]))
  ans$descriptors[match(paste(floor(ARWtNet_object$time),round((ARWtNet_object$time%%1)*frequency(ts_object))),
                        paste(floor(ans[["time"]]),round((ans[["time"]]%%1)*frequency(ts_object)))),] <- ARWtNet_object$descriptors[,1L:mde]
  whna <- apply(ans$descriptors,1L,function(x) any(is.na(x)))
  if(whna[1L]) {
      ans$width <- which(!whna)[1L]-1L
  } else ans$width <- 0
  if(rev(whna)[1L]) {
    ans$forecast <- list()
    ans$forecast$nstep <- which(rev(!whna))[1L]-1L
    ans$forecast$from <- ans$time[length(whna)-ans$forecast$nstep]
    ans$forecast$time <- ans$time[(length(whna)-ans$forecast$nstep+1L):length(whna)]
  }
  return(ans)
}
#
getCVpreds <- function(X,trf,object,which,nstep,forecaster=forecast.ARWtNet,maxit) {
  if((X[["from"]]<min(object$time))||(X[["from"]]>max(object$time)))
    stop("Attempting to start prediction from ",X[["from"]],", outside the range of the model (",min(object$time),", ",max(object$time),")!")
  mask <- rep(FALSE,length(object$time))
  mask[-(which.min(abs(X$from-object$time))+(1L:nstep)-1L)] <- TRUE
  target <- list(time=object$time[!mask],
                 response=trf(object$response[!mask]))
  object$time <- object$time[mask]
  object$response <- object$response[mask]
  object$descriptors <- object$descriptors[mask,,drop=FALSE]
  object$fit$scaled$response <- scale(object$response,center=FALSE,scale=TRUE)
  object$fit$scaled$descriptors <- scale(object$descriptors,center=FALSE,scale=TRUE)
  object$fit$nnet[[which]] <- nnet::nnet(y=object$fit$scaled$response,x=object$fit$scaled$descriptors,size=object$fit$nnet[[which]]$n[2L],decay=object$fit$nnet[[which]]$decay,
                                         maxit=maxit,linout=TRUE,Wts=object$fit$nnet[[which]]$wts,trace=FALSE)
  target$forecast <- trf(forecaster(X=object, which=which, from=X[["from"]], newdata=X[["newdata"]], nstep=nstep, return.descriptors=FALSE))
  return(target)
}
#
getQMsynth <- function(object,nstep) {
  ans <- data.frame(Y0=numeric(),QM0=numeric(),matrix(NA,0L,nstep,dimnames=list(NULL,paste("t_",1L:nstep,"_",sep=""))),MRSE=numeric())
  for(i in 1L:length(object$CVpreds)) {
    t0 <- object$CVpreds[[i]]$time[1L]
    tmp <- object$CVpreds[[i]]$response-object$CVpreds[[i]]$forecast[1L:length(object$CVpreds[[i]]$response)]
    MRSE <- sqrt(mean(tmp^2,na.rm=TRUE))
    tmp <- c(tmp,rep(NA,nstep-length(tmp)))
    ans[i,] <- c(floor(t0),1+(t0%%1)*frequency(object$model$ts$y),tmp,MRSE)
  }
  return(ans)
}
#
getKernelDensity <- function(object,trf,nstep,res,width,prob=0.5) {
  krnl <- list(time=seq(object$CVpreds[[1L]]$time[1L],
                        object$CVpreds[[length(object$CVpreds)]]$time+
                          length(object$CVpreds[[length(object$CVpreds)]]$forecast)/
                          frequency(object$model$ts$y),
                        frequency(object$model$ts$y)^-1),
               density=matrix(NA,length(object$CVpreds)+nstep,res),
               quantile=matrix(NA,length(object$CVpreds)+nstep,length(prob)),
               y=seq(trf(min(object$model$ts$y)),trf(max(object$model$ts$y)),length.out=res))
  forecast <- matrix(NA,length(object$CVpreds)+nstep,nstep)
  #
  for(i in 1L:length(object$CVpreds)) {
    wh <- i+(1L:nstep)-1L
    forecast[cbind(wh,rowSums(!is.na(forecast[wh,]))+1L)] <- object$CVpreds[[i]]$forecast
  }
  #
  for(i in 1L:nrow(forecast)) {
    tmp <- forecast[i,]
    tmp <- tmp[!is.na(tmp)]
    if(length(tmp)>1) {
      krnl$density[i,] <- density(tmp,from=trf(min(object$model$ts$y)),to=trf(max(object$model$ts$y)),n=res,width=width)$y
      krnl$quantile[i,] <- quantile(tmp,prob=prob,na.rm=TRUE)
    }
  }
  return(krnl)
}
#
plotKernelDensity <- function(object,trf,xlim,ylim=object(object$krnl$y),colspace=grey(seq(1,0.2,length.out=1024)),xlab="",ylab="", lcl = "red", marker, ...) {
  if(missing(xlim))
    xlim <- c(min(time(object$model$ts$y)),max(object$krnl$time))
  par(...)
  image(z=object$krnl$density,x=object$krnl$time,y=object$krnl$y,col=colspace,xlab=xlab,ylab=ylab,las=1L,xlim=xlim)
  box()
  lines(y=trf(object$model$ts$y)[(time(object$model$ts$y)>xlim[1L])&(time(object$model$ts$y)<xlim[2L])],col=lcl,
        x=time(object$model$ts$y)[(time(object$model$ts$y)>xlim[1L])&(time(object$model$ts$y)<xlim[2L])])
  if(!missing(marker))
    for(i in 1L:length(marker$at))
      abline(v=round(xlim[1L]):round(xlim[2L])+marker$at[i],lty=marker$lty[i])
  return(invisible(NULL))
}
#
getPredMatrix <- function(object,nstep) {
  tms <- time(object$model$ts$y) ; tms <- c(tms,tail(tms,n=1L)+(1:nstep)/nstep)
  qms <- 1:nstep
  mat <- matrix(NA,length(tms),length(qms),dimnames=list(paste(floor(tms),"+",round(nstep*(tms%%1)+1),"/",nstep,sep=""),qms))
  t0 <- tms[(object$model$width+1L):length(object$model$ts$y)]
  for(i in 1L:length(object$CVpreds)) {
    object$CVpreds[[i]]$forecast
    rws <- cbind(which.min(abs(t0[i]-tms))+(1L:nstep)-1L,1:nstep)
    rws <- rws[rws[,1L]<=nrow(mat),,drop=FALSE]
    mat[rws] <- object$CVpreds[[i]]$forecast[1L:nrow(rws)]
  }
  return(list(tms=tms,qms=qms,mat=mat))
}
#
plotPredMatrix <- function(object,trf,xlim,ylim=object(object$krnl$y),which,cols=rainbow(length(which)),xlab="",ylab="", lcl = "black", marker, ...) {
  if(missing(xlim))
    xlim <- c(min(time(object$model$ts$y)),max(object$krnl$time))
  par(...)
  plot(x=trf(object$model$ts$y),col=lcl,xlab=xlab,ylab=ylab,las=1L,xlim=xlim)
  for(i in 1L:length(which)) {
    lines(y=trf(object$predmat$mat[,which[i]]),col=cols[i],
          x=object$predmat$tms)
  }
  if(!missing(marker))
    for(i in 1L:length(marker$at))
      abline(v=round(xlim[1L]):round(xlim[2L])+marker$at[i],lty=marker$lty[i])
  return(invisible(NULL))
}
#
MRSEpreds <- function(X,wh) return(sqrt(mean((X$model$ts$y-X$predmat$mat[1L:length(X$model$ts$y),wh])^2,na.rm=TRUE)))
#
PRSQpreds <- function(X,wh) return(1-sum((X$model$ts$y-X$predmat$mat[1L:length(X$model$ts$y),wh])^2,na.rm=TRUE)/sum((X$model$ts$y-mean(X$model$ts$y))^2))
#
}
