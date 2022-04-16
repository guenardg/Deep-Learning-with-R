#!/usr/bin/env Rscript
##
### Analyse spectrale de la cyclicité des apports en eau: système Namakan Rainy.
##
## rm(list=ls())
source("WaterPredict_ARWT-AUX.R")
init()   # closeall()  # pour désinitialiser.
##
load(file="../Data/WaterPeriodicity-dat.rda")
nona <- which(!apply(is.na(dat$model_data[,c("Namakan_Supply","TempIFL")]),1L,any))
mtseries <- tsmts(x=dat$model_data[nona,-1L],timestamp=dat$model_data[nona,1L])
rm(dat,nona)
## class(mtseries)
## frequency(mtseries)
## deltat(mtseries)
## start(mtseries)
## end(mtseries)
## plot(mtseries,main="Time series")
##
## tail(mtseries[,1L],n=1L)
## truc <- atime(mtseries,tref=tref)
## plot(x=truc@timestamp,y=truc@ts[,1L],type="l")
## lines(x=truc@timestamp,y=truc@ts[,2L],col="red")
## rm(truc)
##
modlst <- list()
##
### Temperature:
ds <- "TempIFL"
nstep <- 90L
modlst[[ds]]$x <- atime(mtseries,tref=tref)
## dim(modlst[[ds]]$x)
## length(modlst[[ds]]$x)
## ncol(modlst[[ds]]$x)
##
modlst[[ds]]$newx <- atime(tsmts(x=NA,timestamp=tail(attr(mtseries,"timestamp"),n=1L)+(1:nstep)*86400),tref=tref)
## modlst[[ds]]$model...




















### À partir d'ici, c'est du vieux code.
### La classe 'ARWtNet' est à scraper.
ARWtNet_list[[ds]]$model <- ARWtNet(y=mtseries[,ds,drop=FALSE],x=ARWtNet_list[[ds]]$x,width=512L)
## print(ARWtNet_list[[ds]]$model)
##
ARWtNet_list[[ds]]$model <- ARWtNet_add_cvfold(ARWtNet_list[[ds]]$model,3L)
## ARWtNet_list[[ds]]$model <- ARWtNet_rm_cvfold(ARWtNet_list[[ds]]$model)
## print(ARWtNet_list[[ds]]$model)
## head(ARWtNet_list[[ds]]$model)
##
## system("ssh -L 54321:127.0.0.1:54321 -N gguenard@ada.fil.umontreal.ca &")
## system("ps aux | grep ssh")
## system("ssh gguenard@ada.fil.umontreal.ca 'nice -n 10 java -jar ~/h2o/h2o.jar &>> ~/h2o/h2o.log &'")
## system("ssh gguenard@ada.fil.umontreal.ca 'ps aux | grep h2o'")
## localH2O <- h2o::h2o.init(ip='localhost',port=54321,max_mem_size='32G',nthreads=-1L)
##
ARWtNet_list[[ds]] <- initDLARdata(ARWtNet_list[[ds]])
## ARWtNet_list[[ds]] <- removeDLARdata(ARWtNet_list[[ds]])
## h2o::h2o.ls()
##
### Ok, là je suis vraiment écoeuré. Mes trop nombreuses tentatives de faire une recherche d'hyperparamètres avec
### DEoptim m'ont déjà coûté beaucoup de temps. h2o ne me semble pas assez stable pour ce genre d'exercice et il
### n'y a pas de fonctionalité permettant d'envoyer une batch de conditions d'exécution en utilisant h2o.grid().
### Je vais donc faire une "grid search" systématique sur un petit jeu de conditions:
### Seulement des réseau de profondeur 3
##
### Je ne peux pas vraiment utiliser la fonction gridsearch de h2o car les prévisions doivent être calculées
### récursivement sur plusieurs pas de temps.
### 



### de 3 à 15 noeuds cachés, toujours convergents:
node <- c(10L,15L)
hidden <- list()
for(i in node)
    for(j in node)
        for(k in node)
            if((i>=j)&&(j>=k))
                hidden[[length(hidden)+1L]] <- c(i,j,k)
rm(node,i,j,k)
### donc un grand total de 4 configurations. Il faudra aussi définir un ensemble raisonables de conditions des
### paramètres l1 et l2:
l1 <- c(1e-1,1e-3)
l2 <- 1e-7
### ce qui nous fera maintenant un total de 4 * 2 * 1 = 8 conditions. Il nous faut enfin les dropouts pour les
### entrées:
input_dropout_ratio <- c(0.1,0.05)
### (8 * 2 = 16 conditions) et ceux pour les cachés:
dropout <- c(0.2,0.1)
hidden_dropout_ratios <- list()
for(i in dropout)
    for(j in dropout)
        for(k in dropout)
            if((i>=j)&&(j>=k))
                hidden_dropout_ratios[[length(hidden_dropout_ratios)+1L]] <- c(i,j,k)
rm(dropout,i,j,k)
### ce qui nous fait un total de 4 groupes de valeurs, ce qui nous fait culminer à un total de 16 * 4 = 64
### conditions.
conditions <- list(hidden=hidden,l1=l1,l2=l2,input_dropout_ratio=input_dropout_ratio,
                   hidden_dropout_ratios=hidden_dropout_ratios)
rm(hidden,l1,l2,input_dropout_ratio,hidden_dropout_ratios)
##
## DLARGridSearch <- function(
x=ARWtNet_list[[ds]]
grid_id="DLARgrid"
distribution="gaussian"
activation="RectifierWithDropout"
resp=which(x$model@terms=="response")
desc=which(x$model@terms=="mri"|x$model@terms=="descriptor")
epochs=100L
cvfold=which(x$model@terms=="cvfold")[1L]
##
### À l'intérieure de la fonction objective:
if(!h2o::h2o.clusterIsUp())
    stop("h2o cluster must first be started!")
##
grid <- h2o::h2o.grid(algorithm="deeplearning",grid_id=grid_id,y=resp,x=desc,fold_column=colnames(x$model)[cvfold],
                      training_frame=x$h2oData,distribution=distribution,activation=activation,hyper_params=conditions,
                      epochs=epochs,export_weights_and_biases=FALSE,force_load_balance=TRUE,adaptive_rate=TRUE,
                      variable_importances=FALSE)
##
str(grid)
## grid@model_ids[[1L]]
grid@summary_table[1L,]
hidden <- as.numeric(strsplit(gsub("]","",gsub("[","",grid@summary_table[["hidden"]][1L],fixed=TRUE),fixed=TRUE),",")[[1L]])
hidden_dropout_ratios <- as.numeric(strsplit(gsub("]","",gsub("[","",grid@summary_table[["hidden_dropout_ratios"]][1L],fixed=TRUE),fixed=TRUE),",")[[1L]])
input_dropout_ratio <- grid@summary_table[["input_dropout_ratio"]][1L]




### J'ai considérablement réduit la taille de l'ensemble des conditions testées de >5000 à 160...
### et aussi le nombre de groupes de validation croisée (de >250 à 5).
### Sinon, le temps de calcul était beaucoup trop long.



    tmp <- try(h2o::h2o.deeplearning(y=resp,x=desc,fold_column=colnames(x$model)[cvfold],training_frame=x$h2oData,
                                     model_id="DL",distribution=distribution,activation=activation,
                                     hidden=hidden[nzh],l1=l1,l2=l2,input_dropout_ratio=input_dropout_ratio,
                                     hidden_dropout_ratios=hidden_dropout_ratios[nzh],epochs=epochs,
                                     export_weights_and_biases=FALSE,force_load_balance=TRUE,adaptive_rate=TRUE,
                                     variable_importances=FALSE),silent=TRUE)
    ans <- if(class(tmp)=="try-error") returnfailureas else -tmp@model$cross_validation_metrics@metrics$r2
    if(class(tmp)!="try-error")
        try(h2o::h2o.rm(tmp))
    ## try(h2o::h2o.rm(as.character(h2o::h2o.ls()[!(as.character(h2o::h2o.ls()[[1L]]) %in% ls0),1L])))
    return(ans)
} else
    return(returnfailureas)
##



list <- as.character(h2o::h2o.ls()[!(as.character(h2o::h2o.ls()[[1L]]) %in% ls0),1L])

h2o::h2o.rm(object = localH2O, keys="modelmetrics_DL_cv_9@5921211752046705646_on_DL_cv_9_valid@-1230174047390932727")


### Ce n'est pas très réaliste de précéder à une recherche globale dans l'espace de tous les paramêtres, il y en a trop.
##







    
}

    coefs <- list()
    for(j in 1L:(length(hidden)+1L)) {
        ## j=1L
        biases <- h2o::h2o.biases(tmp,j)
        weights <- h2o::h2o.weights(tmp,j)
        coefs[[j]] <- cbind(Bias=as.vector(biases),as.matrix(weights))
        h2o::h2o.rm(biases)
        h2o::h2o.rm(weights)
    }
    h2o::h2o.rm(tmp)
    ## h2o::h2o.ls()
    ## x$model@data[,resp]
    ## x$model@preliminary
    from <- which(x$model@data@ts[,cvfold]==i)[1L]
    pre_idx <- from-(x$model@settings$width:1)
    if(any(pre_idx<1)) {
        y <- numeric(x$model@settings$width)
        if(any(pre_idx>=1))
           y[pre_idx>=1] <- x$model@data@ts[min(pre_idx[pre_idx>=1]):max(pre_idx[pre_idx>=1]),resp]
        y[pre_idx<1] <- x$model@preliminary@ts[x$model@settings$width-(sum(pre_idx<1):1L)+1L]
        
    } else y <- x$model@data@ts[min(pre_idx):max(pre_idx),resp]
    ##
    ## Loop here...
### Il faudra ne pas passer la permière valeur...
    ## ARGGG!!! c(1,x$model@data@ts[from,desc])
    
    k=1L  ## 1L:nstep
    ry <- rev(y)
    dwt1 <- wavelets::dwt(X=ry,filter=x$model@settings$filter,n.levels=floor(log2(x$model@settings$width)),boundary="periodic")
    for(j in 1L:length(x$model@settings$cutoffs))
        dwt1@W[[j]][-x$model@settings$cutoffs[j]] <- 0
    ## rev(wavelets::idwt(dwt1))[x$model@settings$width-x$model@settings$sample+1L]
    x$model@data@ts[from,desc]
    c(1,rev(wavelets::idwt(dwt1))[x$model@settings$width-x$model@settings$sample+1L],
      x$model@data[k,x$model@terms=="descriptor"])
    
    
}










## as.integer(runif(1L,-2147483647,2147483647))  ## Pour le germe...




##

##
grid <- h2o::h2o.grid(algorithm="deeplearning",grid_id="DLARgrid",y=which(x$model@terms=="response"),
                      x=which(x$model@terms=="mri"|x$model@terms=="descriptor"),
                      distribution=distribution,activation=activation,
                      fold_column=colnames(x$model)[which(x$model@terms=="cvfold")[cvfold]],training_frame=x$h2oData,
                      hyper_params=hyper,search_criteria=list(strategy="RandomDiscrete",max_models=N[2L],seed=42152465L),
                      epochs=epochs,export_weights_and_biases=TRUE,force_load_balance=TRUE)

##


DL <- try(h2o::h2o.deeplearning(y=which(x$model@terms=="response"),x=which(x$model@terms=="mri"|x$model@terms=="descriptor"),
                                distribution=distribution,activation=activation,model_id="DL",
                                fold_column=colnames(x$model)[which(x$model@terms=="cvfold")[cvfold]],training_frame=x$h2oData,
                                hidden=c(3,3,3),l1=1e-5,l2=1e-7,input_dropout_ratio=0.05,hidden_dropout_ratios=c(0.2,0.1,0.2),
                                epochs=epochs,export_weights_and_biases=TRUE,force_load_balance=TRUE),silent=TRUE)
str(DL)
h2o::h2o.weights(DL,4L)
fold_column=colnames(x$model)[which(x$model@terms=="cvfold")[cvfold]]

x$h2oData[x$h2oData[,x$model@terms=="cvfold"]!=1L,]

x$model@data@ts[,x$model@terms=="cvfold"]


fold=1L  ## 1L:max(x$h2oData[,x$model@terms=="cvfold"])
DL <- list()


DL[[fold]] <- try(h2o::h2o.deeplearning(y=which(x$model@terms=="response"),x=which(x$model@terms=="mri"|x$model@terms=="descriptor"),
                                        distribution=distribution,activation=activation,model_id="DL",
                                        training_frame=x$h2oData[x$h2oData[,x$model@terms=="cvfold"]!=fold,],
                                        hidden=c(3,3,3),l1=1e-5,l2=1e-7,input_dropout_ratio=0.05,hidden_dropout_ratios=c(0.5,0.1,0.2),
                                        epochs=epochs,export_weights_and_biases=TRUE,force_load_balance=TRUE,
                                        variable_importances=FALSE),silent=TRUE)
h2o::h2o.weights(DL[[fold]],1L)
truc <- h2o::h2o.ls()






##

## 3 7 10 15 20
## 3 layers...
## 3 3 3
## 3 3 7
## 3 3 10
## 3 3 15
## 3 3 20
## 3 7 3
## 3 7 7
## 3 7 10
## 3 7 15
## 3 7 20
## 3 10 3
## 3 10 7
## 3 10 10
## 3 10 15
## 3 10 20...
5*5*5

## 5 10 5
## 3 7  3
## 






x$h2oData[,which(x$model@terms=="cvfold")[cvfold]]
str(x$h2oData)


str(x$h2oData)



return(x)
## }
### Rendu ici avec le trouble...


## DLARWrapper <- function(object,distribution,activation,hidden,l1=0,l2=0,epoch=100L,checkpoint=NULL) {    # Il faudrait pouvoir utiliser un checkpoint.


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














  


###

as.matrix(ARWtNet_list[[ds]]$model)



ARWtNet_list[[ds]]$grid


ARWtNet_list[[ds]] <- initDLARdata(ARWtNet_list[[ds]])




### L'idée dans cette version des travaux était d'utiliser un modèle à échelle journalière avec le même filtre
### à impulsion avec fenêtre exponentielle dyadique que avant mais en utilisant cette fois-ci des réseaux
### neuronaux profonds car c'est pas mal plus zeitgeist, plus avancé aussi; et ça va donc faire meilleure
### impression et être potentiellement mieux publié au bout du compte.
### J'ai passé des quarts de mois aux jour pour augmenter ma taille d'échantillon et pouvoir potentiellement
### mettre en évidence des patrons complexes dans les données. Il faut beaucoup de données pour faire ça.
### Pour abattre une partie du bruit sans trop coalescer les données, j'ai fait passer un filtre polynomial sur
### les séries de données et j'en ai profité pour calculer les première dérivées par rapport au temps.
### Peu importe comment ça va marcher, ça devrait faire un bon papier quand même.
##





# par(mar=c(5,5,4,2))
# plot.ARWtNet(ARWtNet_list[[ds]]$model,step=1L,lwd=3)
#
### Cette étape n'existera plus: ARWtNet_list[[ds]]$model <- RandomNetSeedNoDecay(ARWtNet_list[[ds]]$model,nhidden=1L:10L,nstart=25L,maxit=10000L)
ARWtNet_list[[ds]]$crval <- subset_ARWtNet(ARWtNet_list[[ds]]$model,
                                           subsets=rep(rep(1L:ceiling(length(ARWtNet_list[[ds]]$model$response)/nstep),each=nstep),
                                                       length.out=length(ARWtNet_list[[ds]]$model$response)))
ARWtNet_list[[ds]]$crval
#
# h2o::h2o.shutdown(prompt=FALSE)
# init()  # h2o::h2o.ls()
ARWtNet_list[[ds]] <- initDLARdata(ARWtNet_list[[ds]])
# ARWtNet_list[[ds]] <- removeDLARdata(ARWtNet_list[[ds]])
# if(FALSE) {
#   ARWtNet_list[[ds]] <- fitDLARModels(ARWtNet_list[[ds]],distribution="gaussian",activation="Rectifier",
#                                       hidden=c(9L,7L,5L),l1=1e-5,l2=1e-7,epoch=c(500L,100L))
#   ARWtNet_list[[ds]] <- saveDLARModels(ARWtNet_list[[ds]])
# }
# ARWtNet_list[[ds]] <- reloadDLARModels(ARWtNet_list[[ds]])
# h2o::h2o.ls()
#
# predict(ARWtNet_list[[ds]]$crval[[1L]],ARWtNet_list[[ds]]$crval[[1L]]$subset$descriptors[1L:5L,,drop=FALSE])
# h2o::h2o.predict(ARWtNet_list[[ds]]$crval[[1L]]$fit$deeplearn,h2o::as.h2o(ARWtNet_list[[ds]]$crval[[1L]]$subset$descriptors[1L:5L,,drop=FALSE]))
#
# forecast.ARWtNet(X=ARWtNet_list[[ds]]$model,from=tail(ARWtNet_list[[ds]]$model$time,n=1L),newdata=ARWtNet_list[[ds]]$newx,nstep=48L,return.descriptors=TRUE)
cluster <- makeCluster(32L) # detectCores(logical = FALSE))
# par <- c(runif(3L,-5,5),runif(1L,10,50),runif(2L,-7,-3))
# ans <- optMSRE_min(par=par,object=ARWtNet_list[[ds]],distribution="gaussian",activation="Rectifier",epoch=c(50L,25L),cluster=cluster)
#
if(FALSE)
  opt <- list()
else
  load(file="../Data/Optimization.rda")
#
opt[[ds]] <- DEoptim(optMSRE_min,control=DEoptim.control(NP = 60, itermax = 100, trace = FALSE),object=ARWtNet_list[[ds]],
                     distribution="gaussian",activation="Rectifier",epoch=c(50L,25L),protect=as.matrix(h2o::h2o.ls())[,1L],
                     cluster=cluster,lower=c(rep(-5,3L),10,rep(-7,2L)),upper=c(rep(5,3L),50,rep(-3,2L)))
### Ça passe sont temps à planter... il faudra sans doute procéder à un "grid search" car h2o n'est pas encore assez stable
### pour pouvoir utiliser un algorithme d'évolution dirigée...


### Rendu ici...










#
if(FALSE) {
  mmat <- list()
  opt <- opt_table <- opt_min <- list()
} else {
  load(file="../Data/Optimization.rda")
}
# cluster <- makeCluster(64L) # detectCores(logical = FALSE))
#
if(FALSE) {
  mmat[[ds]] <- matrix(NA,ncol(ARWtNet_list[[ds]]$model$descriptors),2L)
  mmat[[ds]][,1L] <- as.numeric(substr(colnames(ARWtNet_list[[ds]]$model$descriptors),1L,4L)=="Temp")
  mmat[[ds]][,2L] <- as.numeric(substr(colnames(ARWtNet_list[[ds]]$model$descriptors),1L,4L)=="SFRC")
  opt[[ds]] <- list()
  for(which in 1L:10L)
    opt[[ds]][[which]] <- DEoptim(optMSRE_min,control=DEoptim.control(NP = 20, itermax = 100, trace = FALSE),cvobject=ARWtNet_list[[ds]]$crval,
                                  which=which,mmat=mmat[[ds]],maxit=10000L,cluster=cluster,lower=rep(-10,2L),upper=rep(10,2L))
  rm(which)
  #
  for(i in 1L:length(opt[[ds]])) opt_table[[ds]] <- rbind(opt_table[[ds]],c(value=opt[[ds]][[i]]$optim$bestval,which=i,
                                                                            opt[[ds]][[i]]$optim$bestmem))
  rm(i) ; opt_min[[ds]] <- opt_table[[ds]][which.min(opt_table[[ds]][,"value"]),-1L]
}
#
### Peaufiner les modèles avec les meilleurs valeurs de nombre de noeuds.
for(i in 1L:nrow(opt_table[[ds]])) {
  ARWtNet_list[[ds]]$model <- RecomputeNetWithDecay(X=ARWtNet_list[[ds]]$model,which=opt_table[[ds]][i,"which"],hpar=opt_table[[ds]][i,3L:4L],
                                                    mmat=mmat[[ds]], maxit=10000L)
  ARWtNet_list[[ds]]$crval <- parLapplyLB(cl=cluster,ARWtNet_list[[ds]]$crval,RecomputeNetWithDecay,which=opt_table[[ds]][i,"which"],
                                          hpar=opt_table[[ds]][i,3L:4L],mmat=mmat[[ds]],maxit=10000L)
} ; rm(i)
#
tmp <- list()
for(i in 1L:length(ARWtNet_list[[ds]]$model$time)) {
  tmp[[i]] <- list(i=i,from=ARWtNet_list[[ds]]$model$time[i],
                   newdata=aforcing(ARWtNet_list[[ds]]$model$time[i]+(1:nstep)/frequency(ARWtNet_list[[ds]]$model$ts$y)))
} ; rm(i)
tmpdir <- tempdir()
#
# plot.ARWtNet_forecast(X=tmp[[1L]],trf=identity,object=ARWtNet_list[[ds]]$model,which=opt_min[[ds]]["which"],nstep=nstep,xlim=c(-256/48,1),
#                       type="l",lwd=2,las=1L,col=c("black","green"),mar=c(5,5,4,2),xlab="",
#                       ylab="",format=paste(tmpdir,"tmp%.4d.png",sep="/"),forecaster=forecast.ARWtNet)
#
tmp <- parLapplyLB(cl=cluster,tmp,plot.ARWtNet_forecast,trf=identity,object=ARWtNet_list[[ds]]$model,which=opt_min[[ds]]["which"],nstep=nstep,
                   xlim=c(-256/48,1),type="l",lwd=2,las=1L,col=c("black","green"),
                   mar=c(5,5,4,2),xlab="",ylab="",format=paste(tmpdir,"tmp%.4d.png",sep="/"),forecaster=forecast.ARWtNet)
# stopCluster(cluster)
system(sprintf("avconv -i %s/tmp%%04d.png -threads auto -qscale 1 FittedTS-%s.mp4",tmpdir,ds))
system(paste("rm ",tmpdir,"/*.png",sep=""))
rm(nstep,tmp,tmpdir)
# rm(cluster)
#
### Oscillation nord-Atlantique
ds <- "NAO" ; nstep <- 12L           # ARWtNet_list[[ds]] <- NULL
ARWtNet_list[[ds]]$x <- aforcing(time(mtseries))
ARWtNet_list[[ds]]$newx <- aforcing(tail(time(mtseries),n=1L)+(1:nstep)/frequency(mtseries))
ARWtNet_list[[ds]] <- list(model=ARWtNet(y=mtseries[,ds,drop=FALSE],x=ARWtNet_list[[ds]]$x,width=64L))
ARWtNet_list[[ds]]$model
# par(mar=c(5,5,4,2))
# plot.ARWtNet(ARWtNet_list[[ds]]$model,step=1L,lwd=3)
#
ARWtNet_list[[ds]]$model <- RandomNetSeedNoDecay(ARWtNet_list[[ds]]$model,nhidden=1L:10L,nstart=25L,maxit=10000L)
#
ARWtNet_list[[ds]]$crval <- subset_ARWtNet(ARWtNet_list[[ds]]$model,
                                           subsets=rep(rep(1L:ceiling(length(ARWtNet_list[[ds]]$model$response)/nstep),each=nstep),
                                                       length.out=length(ARWtNet_list[[ds]]$model$response)),
                                           recalculate=TRUE,maxit=10000L)
#
# cluster <- makeCluster(64L) # detectCores(logical = FALSE))
#
if(FALSE) {
  mmat[[ds]] <- matrix(NA,ncol(ARWtNet_list[[ds]]$model$descriptors),2L)
  mmat[[ds]][,1L] <- as.numeric(substr(colnames(ARWtNet_list[[ds]]$model$descriptors),1L,4L)=="NAO_")
  mmat[[ds]][,2L] <- as.numeric(substr(colnames(ARWtNet_list[[ds]]$model$descriptors),1L,4L)=="SFRC")
  opt[[ds]] <- list()
  for(which in 1L:10L)
    opt[[ds]][[which]] <- DEoptim(optMSRE_min,control=DEoptim.control(NP = 20, itermax = 100, trace = FALSE),cvobject=ARWtNet_list[[ds]]$crval,
                                  which=which,mmat=mmat[[ds]],maxit=10000L,cluster=cluster,lower=rep(-10,2L),upper=rep(10,2L))
  #
  rm(which)
  #
  for(i in 1L:length(opt[[ds]])) opt_table[[ds]] <- rbind(opt_table[[ds]],c(value=opt[[ds]][[i]]$optim$bestval,which=i,
                                                                            opt[[ds]][[i]]$optim$bestmem))
  rm(i) ; opt_min[[ds]] <- opt_table[[ds]][which.min(opt_table[[ds]][,"value"]),-1L]
}
#
### Peaufiner les modèles avec les meilleurs valeurs de nombre de noeuds.
for(i in 1L:nrow(opt_table[[ds]])) {
  ARWtNet_list[[ds]]$model <- RecomputeNetWithDecay(X=ARWtNet_list[[ds]]$model,which=opt_table[[ds]][i,"which"],
                                                    hpar=opt_table[[ds]][i,3L:4L],mmat=mmat[[ds]],maxit=10000L)
  ARWtNet_list[[ds]]$crval <- parLapplyLB(cl=cluster,ARWtNet_list[[ds]]$crval,RecomputeNetWithDecay,which=opt_table[[ds]][i,"which"],
                                          hpar=opt_table[[ds]][i,3L:4L],mmat=mmat[[ds]],maxit=10000L)
} ; rm(i)
#
tmp <- list()
for(i in 1L:length(ARWtNet_list[[ds]]$model$time)) {
  tmp[[i]] <- list(i=i,from=ARWtNet_list[[ds]]$model$time[i],
                   newdata=aforcing(ARWtNet_list[[ds]]$model$time[i]+(1:nstep)/frequency(ARWtNet_list[[ds]]$model$ts$y)))
} ; rm(i)
tmpdir <- tempdir()
#
tmp <- parLapplyLB(cl=cluster,tmp,plot.ARWtNet_forecast,trf=identity,object=ARWtNet_list[[ds]]$model,which=opt_min[[ds]]["which"],
                   nstep=nstep,xlim=c(-256/48,1),type="l",lwd=2,las=1L,col=c("black","green"),
                   mar=c(5,5,4,2),xlab="",ylab="",format=paste(tmpdir,"tmp%.4d.png",sep="/"),forecaster=forecast.ARWtNet)
# stopCluster(cluster)
system(sprintf("avconv -i %s/tmp%%04d.png -threads auto -qscale 1 FittedTS-%s.mp4",tmpdir,ds))
system(paste("rm ",tmpdir,"/*.png",sep=""))
rm(nstep,tmp,tmpdir)
# rm(cluster)
#
### Apports en eau dans le lac Namakan
ds <- "NBSNamakan" ; nstep <- 12L    # ARWtNet_list[[ds]] <- NULL
#### C'est un peu tricky parce que la série en x doit matcher celle en y. Cependant, en transférant les descripteurs entre modèles
#### les 'width' premiers elements de $descripteurs ne sont plus présents même quand les séries commençaient au même endroit (ce qui
#### n'est pas toujours le cas de toute façon.
#### Il faut une fonction pour prendre une nouvelle série à être prédite à partir d'une série ayant un modèle pré-existent et la contraindre
#### dans une fenêtre possible.
#
# ARWtNet_object <- ARWtNet_list[["NAO"]]$model
# ts_object <- mtseries
#
ARWtNet_list[[ds]] <- list()
tmp <- extract_descriptors(ts_object=mtseries[,ds,drop=FALSE], ARWtNet_object=ARWtNet_list[["NAO"]]$model)
ARWtNet_list[[ds]]$x <- tmp$descriptors[,-(1L:3L)]
fore <- forecast.ARWtNet(X=ARWtNet_list[["NAO"]]$model,which=opt_min[["NAO"]]["which"],from=tmp$forecast$from,
                         newdata=aforcing(c(tmp$forecast$time,tail(tmp$forecast$time,n=1L)+(1:nstep)/frequency(mtseries))),
                         nstep=tmp$forecast$nstep+nstep)
ARWtNet_list[[ds]]$x[nrow(ARWtNet_list[[ds]]$x)-(tmp$forecast$nstep:1L)+1L,] <- head(fore$descriptors,n=tmp$forecast$nstep)[,colnames(ARWtNet_list[[ds]]$x)]
ARWtNet_list[[ds]]$x <- cbind(ARWtNet_list[[ds]]$x,aforcing(time(mtseries)))
ARWtNet_list[[ds]]$newx <- tail(fore$descriptors,n=nstep)[,colnames(ARWtNet_list[[ds]]$x)]
tmp <- extract_descriptors(ts_object=mtseries[,ds,drop=FALSE], ARWtNet_object=ARWtNet_list[["TempIFL"]]$model)
fore <- forecast.ARWtNet(X=ARWtNet_list[["TempIFL"]]$model,which=opt_min[["TempIFL"]]["which"],
                         from=tail(ARWtNet_list[["TempIFL"]]$model$time,n=1L),
                         newdata=aforcing(tail(ARWtNet_list[["TempIFL"]]$model$time,n=1L)+(1:nstep)/frequency(mtseries)),nstep=nstep)
ARWtNet_list[[ds]]$x <- cbind(tmp$descriptors[,1L:4L],ARWtNet_list[[ds]]$x)
ARWtNet_list[[ds]]$newx <- cbind(tail(fore$descriptors,n=nstep)[,1L:4L],ARWtNet_list[[ds]]$newx)
rm(tmp,fore)
#
ARWtNet_list[[ds]]$model <- ARWtNet(y=mtseries[,ds,drop=FALSE],x=ARWtNet_list[[ds]]$x,width=64L)
ARWtNet_list[[ds]]$model
# par(mar=c(5,5,4,2))
# plot.ARWtNet(ARWtNet_list[[ds]]$model,step=1L,lwd=3)
#
ARWtNet_list[[ds]]$model <- RandomNetSeedNoDecay(ARWtNet_list[[ds]]$model,nhidden=1L:6L,nstart=25L,maxit=10000L)
# ARWtNet_list[[ds]]$model$fit$nnet <- ARWtNet_list[[ds]]$model$fit$nnet[1L:6L]
ARWtNet_list[[ds]]$crval <- subset_ARWtNet(ARWtNet_list[[ds]]$model,
                                           subsets=rep(rep(1L:ceiling(length(ARWtNet_list[[ds]]$model$response)/nstep),each=nstep),
                                                       length.out=length(ARWtNet_list[[ds]]$model$response)),
                                           recalculate=TRUE,maxit=10000L)
# for(i in 1L:length(ARWtNet_list[[ds]]$crval))
#   ARWtNet_list[[ds]]$crval[[i]]$fit$nnet <- ARWtNet_list[[ds]]$crval[[i]]$fit$nnet[1L:6L]
# rm(i)
#
# cluster <- makeCluster(64L) # detectCores(logical = FALSE))
#
if(TRUE) {
  mmat[[ds]] <- matrix(NA,ncol(ARWtNet_list[[ds]]$model$descriptors),4L)
  mmat[[ds]][,1L] <- as.numeric(substr(colnames(ARWtNet_list[[ds]]$model$descriptors),1L,4L)=="NAO_")
  mmat[[ds]][,2L] <- as.numeric(substr(colnames(ARWtNet_list[[ds]]$model$descriptors),1L,4L)=="Temp")
  mmat[[ds]][,3L] <- as.numeric(substr(colnames(ARWtNet_list[[ds]]$model$descriptors),1L,4L)=="NBSN")
  mmat[[ds]][,4L] <- as.numeric(substr(colnames(ARWtNet_list[[ds]]$model$descriptors),1L,4L)=="SFRC")
  opt[[ds]] <- list()
  for(which in 1L:6L)
    opt[[ds]][[which]] <- DEoptim(optMSRE_min,control=DEoptim.control(NP = 40, itermax = 100, trace = FALSE),cvobject=ARWtNet_list[[ds]]$crval,
                                  which=which,mmat=mmat[[ds]],maxit=10000L,cluster=cluster,lower=rep(-10,4L),upper=rep(10,4L))
  rm(which)
  #
  for(i in 1L:length(opt[[ds]]))
    opt_table[[ds]] <- rbind(opt_table[[ds]],c(value=opt[[ds]][[i]]$optim$bestval,which=i,par=opt[[ds]][[i]]$optim$bestmem))
  rm(i) ; opt_min[[ds]] <- opt_table[[ds]][which.min(opt_table[[ds]][,"value"]),-1L]
  save.image()
}
#
### Peaufiner les modèles avec les meilleurs valeurs de nombre de noeuds.
for(i in 1L:nrow(opt_table[[ds]])) {
  ARWtNet_list[[ds]]$model <- RecomputeNetWithDecay(X=ARWtNet_list[[ds]]$model,which=opt_table[[ds]][i,"which"],hpar=opt_table[[ds]][i,2L:5L],
                                                    mmat=mmat[[ds]], maxit=10000L)
  ARWtNet_list[[ds]]$crval <- parLapplyLB(cl=cluster,ARWtNet_list[[ds]]$crval,RecomputeNetWithDecay,which=opt_table[[ds]][i,"which"],
                                          hpar=opt_table[[ds]][i,2L:5L],mmat=mmat[[ds]],maxit=10000L)
} ; rm(i)
#
tmp <- list()
for(i in 1L:length(ARWtNet_list[[ds]]$model$time)) {
  # i <- 1L
  tmp[[i]] <- list(i=i,from=ARWtNet_list[[ds]]$model$time[i],
              newdata=rbind(ARWtNet_list[[ds]]$x,ARWtNet_list[[ds]]$newx)[which.min(abs(time(mtseries)-ARWtNet_list[[ds]]$model$time[i]))+(1L:nstep)-1L,])
} ; rm(i)
#
# nrow(tmp[[length(ARWtNet_list[[ds]]$model$time)]]$newdata)
tmpdir <- tempdir()
#
tmp <- parLapplyLB(cl=cluster,tmp,plot.ARWtNet_forecast,trf=identity,object=ARWtNet_list[[ds]]$model,
                   which=opt_min[[ds]]["which"],nstep=nstep,xlim=c(-256/48,1),type="l",lwd=2,las=1L,
                   col=c("black","green"),mar=c(5,5,4,2),xlab="",ylab="",format=paste(tmpdir,"tmp%.4d.png",sep="/"),
                   forecaster=forecast.ARWtNet)
# stopCluster(cluster)
system(sprintf("avconv -i %s/tmp%%04d.png -threads auto -qscale 1 FittedTS-%s.mp4",tmpdir,ds))
system(paste("rm ",tmpdir,"/*.png",sep=""))
rm(nstep,tmp,tmpdir)
# rm(cluster)
#
ds <- "NBSRainyNet" ; nstep <- 12L   # ARWtNet_list[[ds]] <- NULL
#
ARWtNet_list[[ds]] <- list()
tmp <- extract_descriptors(ts_object=mtseries[,ds,drop=FALSE], ARWtNet_object=ARWtNet_list[["NAO"]]$model)
ARWtNet_list[[ds]]$x <- tmp$descriptors[,-(1L:3L)]
fore <- forecast.ARWtNet(X=ARWtNet_list[["NAO"]]$model,which=opt_min[["NAO"]]["which"],from=tmp$forecast$from,
                         newdata=aforcing(c(tmp$forecast$time,tail(tmp$forecast$time,n=1L)+(1:nstep)/frequency(mtseries))),
                         nstep=tmp$forecast$nstep+nstep)
ARWtNet_list[[ds]]$x[nrow(ARWtNet_list[[ds]]$x)-(tmp$forecast$nstep:1L)+1L,] <- head(fore$descriptors,n=tmp$forecast$nstep)[,colnames(ARWtNet_list[[ds]]$x)]
ARWtNet_list[[ds]]$x <- cbind(ARWtNet_list[[ds]]$x,aforcing(time(mtseries)))
ARWtNet_list[[ds]]$newx <- tail(fore$descriptors,n=nstep)[,colnames(ARWtNet_list[[ds]]$x)]
tmp <- extract_descriptors(ts_object=mtseries[,ds,drop=FALSE], ARWtNet_object=ARWtNet_list[["TempIFL"]]$model)
fore <- forecast.ARWtNet(X=ARWtNet_list[["TempIFL"]]$model,which=opt_min[["TempIFL"]]["which"],from=tail(ARWtNet_list[["TempIFL"]]$model$time,n=1L),
                         newdata=aforcing(tail(ARWtNet_list[["TempIFL"]]$model$time,n=1L)+(1:nstep)/frequency(mtseries)),nstep=nstep)
ARWtNet_list[[ds]]$x <- cbind(tmp$descriptors[,1L:4L],ARWtNet_list[[ds]]$x)
ARWtNet_list[[ds]]$newx <- cbind(tail(fore$descriptors,n=nstep)[,1L:4L],ARWtNet_list[[ds]]$newx)
rm(tmp,fore)
#
ARWtNet_list[[ds]]$model <- ARWtNet(y=mtseries[,ds,drop=FALSE],x=ARWtNet_list[[ds]]$x,width=64L)
ARWtNet_list[[ds]]$model
# par(mar=c(5,5,4,2))
# plot.ARWtNet(ARWtNet_list[[ds]]$model,step=1L,lwd=3)
#
#
ARWtNet_list[[ds]]$model <- RandomNetSeedNoDecay(ARWtNet_list[[ds]]$model,nhidden=1L:6L,nstart=25L,maxit=10000L)
ARWtNet_list[[ds]]$crval <- subset_ARWtNet(ARWtNet_list[[ds]]$model,
                                           subsets=rep(rep(1L:ceiling(length(ARWtNet_list[[ds]]$model$response)/nstep),each=nstep),
                                                       length.out=length(ARWtNet_list[[ds]]$model$response)),
                                           recalculate=TRUE,maxit=10000L)
#
# cluster <- makeCluster(64L) # detectCores(logical = FALSE))
#
if(FALSE) {
  mmat[[ds]] <- matrix(NA,ncol(ARWtNet_list[[ds]]$model$descriptors),4L)
  mmat[[ds]][,1L] <- as.numeric(substr(colnames(ARWtNet_list[[ds]]$model$descriptors),1L,4L)=="NAO_")
  mmat[[ds]][,2L] <- as.numeric(substr(colnames(ARWtNet_list[[ds]]$model$descriptors),1L,4L)=="Temp")
  mmat[[ds]][,3L] <- as.numeric(substr(colnames(ARWtNet_list[[ds]]$model$descriptors),1L,4L)=="NBSR")
  mmat[[ds]][,4L] <- as.numeric(substr(colnames(ARWtNet_list[[ds]]$model$descriptors),1L,4L)=="SFRC")
  opt[[ds]] <- list()
  for(which in 1L:6L)
    opt[[ds]][[which]] <- DEoptim(optMSRE_min,control=DEoptim.control(NP = 40, itermax = 100, trace = FALSE),cvobject=ARWtNet_list[[ds]]$crval,
                                  which=which,mmat=mmat[[ds]],maxit=10000L,cluster=cluster,lower=rep(-10,4L),upper=rep(10,4L))
  rm(which)
  #
  for(i in 1L:length(opt[[ds]]))
    opt_table[[ds]] <- rbind(opt_table[[ds]],c(value=opt[[ds]][[i]]$optim$bestval,which=i,par=opt[[ds]][[i]]$optim$bestmem))
  rm(i) ; opt_min[[ds]] <- opt_table[[ds]][which.min(opt_table[[ds]][,"value"]),-1L]
  save.image()
}
#
### Peaufiner les modèles avec les meilleurs valeurs de nombre de noeuds.
for(i in 1L:nrow(opt_table[[ds]])) {
  ARWtNet_list[[ds]]$model <- RecomputeNetWithDecay(X=ARWtNet_list[[ds]]$model,which=opt_table[[ds]][i,"which"],
                                                    hpar=opt_table[[ds]][i,3L:6L],
                                                    mmat=mmat[[ds]], maxit=10000L)
  ARWtNet_list[[ds]]$crval <- parLapplyLB(cl=cluster,ARWtNet_list[[ds]]$crval,RecomputeNetWithDecay,which=opt_table[[ds]][i,"which"],
                                          hpar=opt_table[[ds]][i,3L:6L],mmat=mmat[[ds]],maxit=10000L)
} ; rm(i)
#
tmp <- list()
for(i in 1L:length(ARWtNet_list[[ds]]$model$time)) {
  # i <- 1L
  tmp[[i]] <- list(i=i,from=ARWtNet_list[[ds]]$model$time[i],
              newdata=rbind(ARWtNet_list[[ds]]$x,ARWtNet_list[[ds]]$newx)[which.min(abs(time(mtseries)-ARWtNet_list[[ds]]$model$time[i]))+(1L:nstep)-1L,])
} ; rm(i)
#
# nrow(tmp[[length(ARWtNet_list[[ds]]$model$time)]]$newdata)
tmpdir <- tempdir()
#
# plot.ARWtNet_forecast(tmp[[1L]],trf=idensity,object=ARWtNet_list[[ds]]$model,which=opt_min[[ds]]["which"],
#                       nstep=nstep,xlim=c(-256/48,1),type="l",lwd=2,las=1L,col=c("black","green"),
#                       mar=c(5,5,4,2),xlab="",ylab="",format=paste(tmpdir,"tmp%.4d.png",sep="/"),forecaster=forecast.ARWtNet)
#
tmp <- parLapplyLB(cl=cluster,tmp,plot.ARWtNet_forecast,trf=identity,object=ARWtNet_list[[ds]]$model,
                   which=opt_min[[ds]]["which"],nstep=nstep,xlim=c(-256/48,1),type="l",lwd=2,las=1L,
                   col=c("black","green"),mar=c(5,5,4,2),xlab="",ylab="",format=paste(tmpdir,"tmp%.4d.png",sep="/"),
                   forecaster=forecast.ARWtNet)
# stopCluster(cluster)
system(sprintf("avconv -i %s/tmp%%04d.png -threads auto -qscale 1 FittedTS-%s.mp4",tmpdir,ds))
system(paste("rm ",tmpdir,"/*.png",sep=""))
rm(nstep,tmp,tmpdir)
# rm(cluster)
#
### Modification à partir d'ici pour créer un troisième modèle: NBSRainy
ds <- "NBSRainy" ; nstep <- 12L    # ARWtNet_list[[ds]] <- NULL
#
ARWtNet_list[[ds]] <- list()
tmp <- extract_descriptors(ts_object=mtseries[,ds,drop=FALSE], ARWtNet_object=ARWtNet_list[["NAO"]]$model)
ARWtNet_list[[ds]]$x <- tmp$descriptors[,-(1L:3L)]
fore <- forecast.ARWtNet(X=ARWtNet_list[["NAO"]]$model,which=opt_min[["NAO"]]["which"],from=tmp$forecast$from,
                         newdata=aforcing(c(tmp$forecast$time,tail(tmp$forecast$time,n=1L)+(1:nstep)/frequency(mtseries))),
                         nstep=tmp$forecast$nstep+nstep)
ARWtNet_list[[ds]]$x[nrow(ARWtNet_list[[ds]]$x)-(tmp$forecast$nstep:1L)+1L,] <- head(fore$descriptors,n=tmp$forecast$nstep)[,colnames(ARWtNet_list[[ds]]$x)]
ARWtNet_list[[ds]]$x <- cbind(ARWtNet_list[[ds]]$x,aforcing(time(mtseries)))
ARWtNet_list[[ds]]$newx <- tail(fore$descriptors,n=nstep)[,colnames(ARWtNet_list[[ds]]$x)]
tmp <- extract_descriptors(ts_object=mtseries[,ds,drop=FALSE], ARWtNet_object=ARWtNet_list[["TempIFL"]]$model)
fore <- forecast.ARWtNet(X=ARWtNet_list[["TempIFL"]]$model,which=opt_min[["TempIFL"]]["which"],
                         from=tail(ARWtNet_list[["TempIFL"]]$model$time,n=1L),
                         newdata=aforcing(tail(ARWtNet_list[["TempIFL"]]$model$time,n=1L)+(1:nstep)/frequency(mtseries)),nstep=nstep)
ARWtNet_list[[ds]]$x <- cbind(tmp$descriptors[,1L:4L],ARWtNet_list[[ds]]$x)
ARWtNet_list[[ds]]$newx <- cbind(tail(fore$descriptors,n=nstep)[,1L:4L],ARWtNet_list[[ds]]$newx)
rm(tmp,fore)
#
ARWtNet_list[[ds]]$model <- ARWtNet(y=mtseries[,ds,drop=FALSE],x=ARWtNet_list[[ds]]$x,width=64L)
ARWtNet_list[[ds]]$model
# par(mar=c(5,5,4,2))
# plot.ARWtNet(ARWtNet_list[[ds]]$model,step=1L,lwd=3)
#
ARWtNet_list[[ds]]$model <- RandomNetSeedNoDecay(ARWtNet_list[[ds]]$model,nhidden=1L:6L,nstart=25L,maxit=10000L)
ARWtNet_list[[ds]]$crval <- subset_ARWtNet(ARWtNet_list[[ds]]$model,
                                           subsets=rep(rep(1L:ceiling(length(ARWtNet_list[[ds]]$model$response)/nstep),each=nstep),
                                                       length.out=length(ARWtNet_list[[ds]]$model$response)),
                                           recalculate=TRUE,maxit=10000L)
#
# cluster <- makeCluster(64L) # detectCores(logical = FALSE))
#
if(TRUE) {
  mmat[[ds]] <- matrix(NA,ncol(ARWtNet_list[[ds]]$model$descriptors),4L)
  mmat[[ds]][,1L] <- as.numeric(substr(colnames(ARWtNet_list[[ds]]$model$descriptors),1L,4L)=="NAO_")
  mmat[[ds]][,2L] <- as.numeric(substr(colnames(ARWtNet_list[[ds]]$model$descriptors),1L,4L)=="Temp")
  mmat[[ds]][,3L] <- as.numeric(substr(colnames(ARWtNet_list[[ds]]$model$descriptors),1L,4L)=="NBSN")
  mmat[[ds]][,4L] <- as.numeric(substr(colnames(ARWtNet_list[[ds]]$model$descriptors),1L,4L)=="SFRC")
  opt[[ds]] <- list()
  for(which in 1L:6L)
    opt[[ds]][[which]] <- DEoptim(optMSRE_min,control=DEoptim.control(NP = 40, itermax = 100, trace = FALSE),cvobject=ARWtNet_list[[ds]]$crval,
                                  which=which,mmat=mmat[[ds]],maxit=10000L,cluster=cluster,lower=rep(-10,4L),upper=rep(10,4L))
  rm(which)
  #
  for(i in 1L:length(opt[[ds]]))
    opt_table[[ds]] <- rbind(opt_table[[ds]],c(value=opt[[ds]][[i]]$optim$bestval,which=i,par=opt[[ds]][[i]]$optim$bestmem))
  rm(i) ; opt_min[[ds]] <- opt_table[[ds]][which.min(opt_table[[ds]][,"value"]),-1L]
}
#
#
### Peaufiner les modèles avec les meilleurs valeurs de nombre de noeuds.
for(i in 1L:nrow(opt_table[[ds]])) {
  ARWtNet_list[[ds]]$model <- RecomputeNetWithDecay(X=ARWtNet_list[[ds]]$model,which=opt_table[[ds]][i,"which"],hpar=opt_table[[ds]][i,2L:5L],
                                                    mmat=mmat[[ds]], maxit=10000L)
  ARWtNet_list[[ds]]$crval <- parLapplyLB(cl=cluster,ARWtNet_list[[ds]]$crval,RecomputeNetWithDecay,which=opt_table[[ds]][i,"which"],
                                          hpar=opt_table[[ds]][i,2L:5L],mmat=mmat[[ds]],maxit=10000L)
} ; rm(i)
#
tmp <- list()
for(i in 1L:length(ARWtNet_list[[ds]]$model$time)) {
  # i <- 1L
  tmp[[i]] <- list(i=i,from=ARWtNet_list[[ds]]$model$time[i],
              newdata=rbind(ARWtNet_list[[ds]]$x,ARWtNet_list[[ds]]$newx)[which.min(abs(time(mtseries)-ARWtNet_list[[ds]]$model$time[i]))+(1L:nstep)-1L,])
} ; rm(i)
#
# nrow(tmp[[length(ARWtNet_list[[ds]]$model$time)]]$newdata)
tmpdir <- tempdir()
#
tmp <- parLapplyLB(cl=cluster,tmp,plot.ARWtNet_forecast,trf=identity,object=ARWtNet_list[[ds]]$model,
                   which=opt_min[[ds]]["which"],nstep=nstep,xlim=c(-256/48,1),type="l",lwd=2,las=1L,
                   col=c("black","green"),mar=c(5,5,4,2),xlab="",ylab="",format=paste(tmpdir,"tmp%.4d.png",sep="/"),
                   forecaster=forecast.ARWtNet)
# stopCluster(cluster)
system(sprintf("avconv -i %s/tmp%%04d.png -threads auto -qscale 1 FittedTS-%s.mp4",tmpdir,ds))
system(paste("rm ",tmpdir,"/*.png",sep=""))
rm(ds,nstep,tmp,tmpdir)
# rm(cluster)
#
if(TRUE) save(opt,mmat,opt_table,opt_min,file="../Data/Optimization.rda")
#
### Séquence avec l'addition des deux séries:
# length(ARWtNet_list[["NBSNamakan"]]$model$time)==length(ARWtNet_list[["NBSRainyNet"]]$model$time)
# all(ARWtNet_list[["NBSNamakan"]]$model$time==ARWtNet_list[["NBSRainyNet"]]$model$time)
nstep <- 12L ; tmp <- list()
for(i in 1L:length(ARWtNet_list[["NBSNamakan"]]$model$time)) {
  # i <- 1L
  tmp[[i]] <- list(i=i,from=ARWtNet_list[["NBSNamakan"]]$model$time[i],
  newdata=list(NBSNamakan=rbind(ARWtNet_list[["NBSNamakan"]]$x,ARWtNet_list[["NBSNamakan"]]$newx)[which.min(abs(time(mtseries)-ARWtNet_list[["NBSNamakan"]]$model$time[i]))+(1L:nstep)-1L,],
               NBSRainyNet=rbind(ARWtNet_list[["NBSRainyNet"]]$x,ARWtNet_list[["NBSRainyNet"]]$newx)[which.min(abs(time(mtseries)-ARWtNet_list[["NBSRainyNet"]]$model$time[i]))+(1L:nstep)-1L,]))
} ; rm(i)
#
tmpdir <- tempdir()
# plot.ARWtNet_forecast_Multi(X=tmp[[1L]],trf=list(identity,identity),
#                             object=list(NBSNamakan=ARWtNet_list[["NBSNamakan"]]$model,NBSRainyNet=ARWtNet_list[["NBSRainyNet"]]$model),
#                             which=list(opt_min[["NBSNamakan"]]["which"],opt_min[["NBSRainyNet"]]["which"]),nstep=nstep,xlim=c(-256/48,1),
#                             type="l",lwd=2,las=1L,col=c("black","green"),mar=c(5,5,4,2),xlab="",ylab="",
#                             format=paste(tmpdir,"tmp%.4d.png",sep="/"),forecaster=forecast.ARWtNet)
#
# cluster <- makeCluster(64L) # detectCores(logical = FALSE))
tmp <- parLapplyLB(cl=cluster,tmp,plot.ARWtNet_forecast_Multi,trf=list(identity,identity),
                   object=list(NBSNamakan=ARWtNet_list[["NBSNamakan"]]$model,NBSRainyNet=ARWtNet_list[["NBSRainyNet"]]$model),
                   which=list(opt_min[["NBSNamakan"]]["which"],opt_min[["NBSRainyNet"]]["which"]),nstep=nstep,xlim=c(-256/48,1),type="l",lwd=2,
                   las=1L,col=c("black","green"),mar=c(5,5,4,2),xlab="",ylab="",format=paste(tmpdir,"tmp%.4d.png",sep="/"),
                   forecaster=forecast.ARWtNet)
# stopCluster(cluster)
system(sprintf("avconv -i %s/tmp%%04d.png -threads auto -qscale 1 FittedTS-%s.mp4",tmpdir,"NBSRainy+Namakan"))
system(paste("rm ",tmpdir,"/*.png",sep=""))
rm(nstep,tmp,tmpdir)
# rm(cluster)
#
### Calculer toutes les prédictions au mois
# cluster <- makeCluster(64L) # detectCores(logical = FALSE))
ds <- "TempIFL" ; nstep <- 12L ; tmp <- list()
for(i in 1L:length(ARWtNet_list[[ds]]$model$time)) {
  tmp[[i]] <- list(i=i,from=ARWtNet_list[[ds]]$model$time[i],
                   newdata=aforcing(ARWtNet_list[[ds]]$model$time[i]+(1:nstep)/frequency(ARWtNet_list[[ds]]$model$ts$y)))
} ; rm(i)
ARWtNet_list[[ds]]$CVpreds <- parLapplyLB(cl=cluster,tmp,getCVpreds,trf=identity,object=ARWtNet_list[[ds]]$model,which=opt_min[[ds]]["which"],
                                          nstep,forecaster=forecast.ARWtNet,maxit=10000L)
#
ds <- "NAO" ; nstep <- 12L ; tmp <- list()
for(i in 1L:length(ARWtNet_list[[ds]]$model$time)) {
  tmp[[i]] <- list(i=i,from=ARWtNet_list[[ds]]$model$time[i],
                   newdata=aforcing(ARWtNet_list[[ds]]$model$time[i]+(1:nstep)/frequency(ARWtNet_list[[ds]]$model$ts$y)))
} ; rm(i)
ARWtNet_list[[ds]]$CVpreds <- parLapplyLB(cl=cluster,tmp,getCVpreds,trf=identity,object=ARWtNet_list[[ds]]$model,which=opt_min[[ds]]["which"],
                                          nstep,forecaster=forecast.ARWtNet,maxit=10000L)
#
ds <- "NBSNamakan" ; nstep <- 12L ; tmp <- list()
for(i in 1L:length(ARWtNet_list[[ds]]$model$time)) {
  tmp[[i]] <- list(i=i,from=ARWtNet_list[[ds]]$model$time[i],
              newdata=rbind(ARWtNet_list[[ds]]$x,ARWtNet_list[[ds]]$newx)[which.min(abs(time(mtseries)-ARWtNet_list[[ds]]$model$time[i]))+(1L:nstep)-1L,])
} ; rm(i)
ARWtNet_list[[ds]]$CVpreds <- parLapplyLB(cl=cluster,tmp,getCVpreds,trf=identity,object=ARWtNet_list[[ds]]$model,which=opt_min[[ds]]["which"],
                                          nstep,forecaster=forecast.ARWtNet,maxit=10000L)
#
ds <- "NBSRainyNet" ; nstep <- 12L ; tmp <- list()
for(i in 1L:length(ARWtNet_list[[ds]]$model$time)) {
  tmp[[i]] <- list(i=i,from=ARWtNet_list[[ds]]$model$time[i],
              newdata=rbind(ARWtNet_list[[ds]]$x,ARWtNet_list[[ds]]$newx)[which.min(abs(time(mtseries)-ARWtNet_list[[ds]]$model$time[i]))+(1L:nstep)-1L,])
} ; rm(i)
#
ARWtNet_list[[ds]]$CVpreds <- parLapplyLB(cl=cluster,tmp,getCVpreds,trf=identity,object=ARWtNet_list[[ds]]$model,which=opt_min[[ds]]["which"],
                                          nstep,forecaster=forecast.ARWtNet,maxit=10000L)
#
ds <- "NBSRainy" ; nstep <- 12L ; tmp <- list()
for(i in 1L:length(ARWtNet_list[[ds]]$model$time)) {
  tmp[[i]] <- list(i=i,from=ARWtNet_list[[ds]]$model$time[i],
              newdata=rbind(ARWtNet_list[[ds]]$x,ARWtNet_list[[ds]]$newx)[which.min(abs(time(mtseries)-ARWtNet_list[[ds]]$model$time[i]))+(1L:nstep)-1L,])
} ; rm(i)
ARWtNet_list[[ds]]$CVpreds <- parLapplyLB(cl=cluster,tmp,getCVpreds,trf=identity,object=ARWtNet_list[[ds]]$model,which=opt_min[[ds]]["which"],
                                          nstep,forecaster=forecast.ARWtNet,maxit=10000L)
# stopCluster(cluster) ; rm(cluster)
rm(ds,nstep,tmp)
#
# plot.ARWtNet_forecast(X=tmp[[1L]],object=ARWtNet_list[[ds]]$model,which=opt_min[[ds]]["which"],nstep=nstep,xlim=c(-256/48,1),type="l",
#                       lwd=2,las=1L,col=c("black","green"),mar=c(5,5,4,2),xlab="",ylab="",format="./figures/FittedTS/FittedTS%.4d.png",
#                       forecaster=forecast.ARWtNet)
# getCVpreds(X=tmp[[1L]],object=ARWtNet_list[[ds]]$model,which=opt_min[[ds]]["which"],nstep,forecaster=forecast.ARWtNet,maxit=10000L)
#
### Calcul des QMSynths:
ARWtNet_list[["NBSNamakan"]]$QMsynth <- getQMsynth(object=ARWtNet_list[["NBSNamakan"]],nstep=12L)
ARWtNet_list[["NBSRainyNet"]]$QMsynth <- getQMsynth(object=ARWtNet_list[["NBSRainyNet"]],nstep=12L)
ARWtNet_list[["NBSRainy"]]$QMsynth <- getQMsynth(object=ARWtNet_list[["NBSRainy"]],nstep=12L)
#
### Cas spécial de Rainy Lake:
ds <- "NBSRainy+Namakan" ; nstep <- 12L
ARWtNet_list[[ds]] <- list()
tmp <- extract_descriptors(ts_object=mtseries[,"NBSRainy",drop=FALSE], ARWtNet_object=ARWtNet_list[["NAO"]]$model)
ARWtNet_list[[ds]]$x <- tmp$descriptors[,-(1L:3L)]
fore <- forecast.ARWtNet(X=ARWtNet_list[["NAO"]]$model,which=opt_min[["NAO"]]["which"],from=tmp$forecast$from,
                         newdata=aforcing(c(tmp$forecast$time,tail(tmp$forecast$time,n=1L)+(1:nstep)/frequency(mtseries))),
                         nstep=tmp$forecast$nstep+nstep)
ARWtNet_list[[ds]]$x[nrow(ARWtNet_list[[ds]]$x)-(tmp$forecast$nstep:1L)+1L,] <- head(fore$descriptors,n=tmp$forecast$nstep)[,colnames(ARWtNet_list[[ds]]$x)]
ARWtNet_list[[ds]]$x <- cbind(ARWtNet_list[[ds]]$x,aforcing(time(mtseries)))
ARWtNet_list[[ds]]$newx <- tail(fore$descriptors,n=nstep)[,colnames(ARWtNet_list[[ds]]$x)]
tmp <- extract_descriptors(ts_object=mtseries[,"NBSRainy",drop=FALSE], ARWtNet_object=ARWtNet_list[["TempIFL"]]$model)
fore <- forecast.ARWtNet(X=ARWtNet_list[["TempIFL"]]$model,which=opt_min[["TempIFL"]]["which"],
                         from=tail(ARWtNet_list[["TempIFL"]]$model$time,n=1L),
                         newdata=aforcing(tail(ARWtNet_list[["TempIFL"]]$model$time,n=1L)+(1:nstep)/frequency(mtseries)),nstep=nstep)
ARWtNet_list[[ds]]$x <- cbind(tmp$descriptors[,1L:4L],ARWtNet_list[[ds]]$x)
ARWtNet_list[[ds]]$newx <- cbind(tail(fore$descriptors,n=nstep)[,1L:4L],ARWtNet_list[[ds]]$newx)
rm(tmp,fore)
#
ARWtNet_list[[ds]]$model <- ARWtNet(y=mtseries[,"NBSRainy",drop=FALSE],x=ARWtNet_list[[ds]]$x,width=64L)
ARWtNet_list[[ds]]$model
ARWtNet_list[[ds]]$CVpreds <- list()
for(i in 1L:length(ARWtNet_list[["NBSNamakan"]]$CVpreds)) {
  ARWtNet_list[[ds]]$CVpreds[[i]] <- list(time=ARWtNet_list[["NBSNamakan"]]$CVpreds[[i]]$time,
                               response=ARWtNet_list[["NBSNamakan"]]$CVpreds[[i]]$response+ARWtNet_list[["NBSRainyNet"]]$CVpreds[[i]]$response,
                               forecast=ARWtNet_list[["NBSNamakan"]]$CVpreds[[i]]$forecast+ARWtNet_list[["NBSRainyNet"]]$CVpreds[[i]]$forecast)
} ; rm(i)
ARWtNet_list[[ds]]$QMsynth <- getQMsynth(object=ARWtNet_list[[ds]],nstep=nstep)
#
plot(t_1_~QM0,data=ARWtNet_list[[ds]]$QMsynth,subset=which(ARWtNet_list[[ds]]$QMsynth$MRSE<200))
plot(t_2_~QM0,data=ARWtNet_list[[ds]]$QMsynth,subset=which(ARWtNet_list[[ds]]$QMsynth$MRSE<300))
plot(t_3_~QM0,data=ARWtNet_list[[ds]]$QMsynth,subset=which(ARWtNet_list[[ds]]$QMsynth$MRSE<300))
plot(t_4_~QM0,data=ARWtNet_list[[ds]]$QMsynth,subset=which(ARWtNet_list[[ds]]$QMsynth$MRSE<300))
plot(t_6_~QM0,data=ARWtNet_list[[ds]]$QMsynth,subset=which(ARWtNet_list[[ds]]$QMsynth$MRSE<300))
plot(t_12_~QM0,data=ARWtNet_list[[ds]]$QMsynth,subset=which(ARWtNet_list[[ds]]$QMsynth$MRSE<300))
plot(t_24_~QM0,data=ARWtNet_list[[ds]]$QMsynth,subset=which(ARWtNet_list[[ds]]$QMsynth$MRSE<300))
plot(t_36_~QM0,data=ARWtNet_list[[ds]]$QMsynth,subset=which(ARWtNet_list[[ds]]$QMsynth$MRSE<300))
plot(t_48_~QM0,data=ARWtNet_list[[ds]]$QMsynth,subset=which(ARWtNet_list[[ds]]$QMsynth$MRSE<300))
rm(ds)
#
ds <- "NBSNamakan" ; nstep <- 12L
cols <- rainbow(floor(1.2*nstep))[1L:nstep] ; i <- 0
plot(y=c(0,apply(ARWtNet_list[[ds]]$QMsynth[((ARWtNet_list[[ds]]$QMsynth[,"QM0"]-i-1)<.Machine$double.eps^0.5)&(ARWtNet_list[[ds]]$QMsynth[,"MRSE"]<1000),-c(1L,2L,nstep+3L)],2L,
                 function(x) sqrt(mean(x^2,na.rm=TRUE)))),
     x=0:nstep,ylab="Mean deviation from model forecast", xlab="Prediction time (month)", type="l",col=cols[i+1L],las=1L,ylim=c(-15,150),xaxt="n",main="Namakan Lake")
axis(1L,at=seq(0,12,1))
for(i in 1:(nstep-1))
    lines(y=c(0,apply(ARWtNet_list[[ds]]$QMsynth[((ARWtNet_list[[ds]]$QMsynth[,"QM0"]-i-1)<.Machine$double.eps^0.5)&(ARWtNet_list[[ds]]$QMsynth[,"MRSE"]<1000),-c(1L,2L,nstep+3L)],2L,
                      function(x) sqrt(mean(x^2,na.rm=TRUE)))),
          x=0:nstep,col=cols[i+1L])
legend(x=5,y=50,legend=c("Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"),col=cols,ncol=3L,lwd=3,cex=1)
rm(cols,i)
dev.copy2pdf(file=sprintf("Forecasting Error-%s.pdf",ds)) ; dev.off()
#
ds <- "NBSRainyNet" ; nstep <- 12L
cols <- rainbow(floor(1.2*nstep))[1L:nstep] ; i <- 0
plot(y=c(0,apply(ARWtNet_list[[ds]]$QMsynth[((ARWtNet_list[[ds]]$QMsynth[,"QM0"]-i-1)<.Machine$double.eps^0.5)&(ARWtNet_list[[ds]]$QMsynth[,"MRSE"]<1000),-c(1L,2L,nstep+3L)],2L,
                 function(x) sqrt(mean(x^2,na.rm=TRUE)))),
     x=0:nstep,ylab="Mean deviation from model forecast", xlab="Prediction time (future)", type="l",col=cols[i+1L],las=1L,ylim=c(-18,180),xaxt="n",main="Rainy Lake (net)")
axis(1L,at=seq(0,12,1))
for(i in 1:(nstep-1))
    lines(y=c(0,apply(ARWtNet_list[[ds]]$QMsynth[((ARWtNet_list[[ds]]$QMsynth[,"QM0"]-i-1)<.Machine$double.eps^0.5)&(ARWtNet_list[[ds]]$QMsynth[,"MRSE"]<1000),-c(1L,2L,nstep+3L)],2L,
                      function(x) sqrt(mean(x^2,na.rm=TRUE)))),
          x=0:nstep,col=cols[i+1L])
legend(x=5,y=45,legend=c("Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"),col=cols,ncol=3L,lwd=3,cex=1)
rm(cols,i)
dev.copy2pdf(file=sprintf("Forecasting Error-%s.pdf",ds)) ; dev.off()
#
ds <- "NBSRainy" ; nstep <- 12L
cols <- rainbow(floor(1.2*nstep))[1L:nstep] ; i <- 0
plot(y=c(0,apply(ARWtNet_list[[ds]]$QMsynth[((ARWtNet_list[[ds]]$QMsynth[,"QM0"]-i-1)<.Machine$double.eps^0.5)&(ARWtNet_list[[ds]]$QMsynth[,"MRSE"]<1000),-c(1L,2L,nstep+3L)],2L,
                 function(x) sqrt(mean(x^2,na.rm=TRUE)))),
     x=0:nstep,ylab="Mean deviation from model forecast", xlab="Prediction time (future)", type="l",col=cols[i+1L],las=1L,ylim=c(-30,300),xaxt="n",main="Rainy Lake (total)")
axis(1L,at=seq(0,12,4))
for(i in 1:(nstep-1))
    lines(y=c(0,apply(ARWtNet_list[[ds]]$QMsynth[((ARWtNet_list[[ds]]$QMsynth[,"QM0"]-i-1)<.Machine$double.eps^0.5)&(ARWtNet_list[[ds]]$QMsynth[,"MRSE"]<1000),-c(1L,2L,nstep+3L)],2L,
                      function(x) sqrt(mean(x^2,na.rm=TRUE)))),
          x=0:nstep,col=cols[i+1L])
legend(x=5,y=80,legend=c("Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"),col=cols,ncol=3L,lwd=3,cex=1)
rm(cols,i)
dev.copy2pdf(file=sprintf("Forecasting Error-%s.pdf",ds)) ; dev.off()
#
ds <- "NBSRainy+Namakan" ; nstep <- 12L
cols <- rainbow(floor(1.2*nstep))[1L:nstep] ; i <- 0
plot(y=c(0,apply(ARWtNet_list[[ds]]$QMsynth[((ARWtNet_list[[ds]]$QMsynth[,"QM0"]-i-1)<.Machine$double.eps^0.5)&(ARWtNet_list[[ds]]$QMsynth[,"MRSE"]<1000),-c(1L,2L,nstep+3L)],2L,
                 function(x) sqrt(mean(x^2,na.rm=TRUE)))),
     x=0:nstep,ylab="Mean deviation from model forecast", xlab="Prediction time (future)", type="l",col=cols[i+1L],las=1L,ylim=c(-30,300),xaxt="n",main="Rainy Lake (net) + Namakan Lake")
axis(1L,at=seq(0,12,4))
for(i in 1:(nstep-1))
    lines(y=c(0,apply(ARWtNet_list[[ds]]$QMsynth[((ARWtNet_list[[ds]]$QMsynth[,"QM0"]-i-1)<.Machine$double.eps^0.5)&(ARWtNet_list[[ds]]$QMsynth[,"MRSE"]<1000),-c(1L,2L,nstep+3L)],2L,
                      function(x) sqrt(mean(x^2,na.rm=TRUE)))),
          x=0:nstep,col=cols[i+1L])
legend(x=5,y=80,legend=c("Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"),col=cols,ncol=3L,lwd=3,cex=1)
rm(cols,i)
dev.copy2pdf(file=sprintf("Forecasting Error-%s.pdf",ds)) ; dev.off()
#
### Arrête ici pour l'instant car le code commence à devenir très compliqué, signe qu'il faudrait retravailler plus haut afin de le peaufiner car il manque des fonctionnalités
### (méthodes génératrices, automation du paramètrage) pour permettre sa génératisation à un ensemble de problèmes.
#
### Calcul des densité de noyau (kernel densities):
ds <- "NBSNamakan" ; nstep <- 12L
ARWtNet_list[[ds]]$krnl <- getKernelDensity(object=ARWtNet_list[[ds]],trf=identity,nstep=nstep,res=2^8,width=150)
rng <- (max(ARWtNet_list[[ds]]$krnl$time)-min(time(ARWtNet_list[[ds]]$model$ts$y)))/6
X11(width=7.25,height=6)
plotKernelDensity(object=ARWtNet_list[[ds]],trf=identity,xlim=min(time(ARWtNet_list[[ds]]$model$ts$y))+c(0,rng),
                  colspace=grey(seq(1,0.2,length.out=1024)),xlab="",ylab="",mar=c(3,5,2,2),fig=c(0,1,0.67,1),
                  marker=list(at=c(10,21,33)/48,lty=c(2L,3L,4L)))
plotKernelDensity(object=ARWtNet_list[[ds]],trf=identity,xlim=min(time(ARWtNet_list[[ds]]$model$ts$y))+c(rng,2*rng),
                  colspace=grey(seq(1,0.2,length.out=1024)),xlab="",ylab="",mar=c(4,5,1,2),fig=c(0,1,0.33,0.67),new=TRUE,
                  marker=list(at=c(10,21,33)/48,lty=c(2L,3L,4L)))
plotKernelDensity(object=ARWtNet_list[[ds]],trf=identity,xlim=min(time(ARWtNet_list[[ds]]$model$ts$y))+c(2*rng,3*rng),
                  colspace=grey(seq(1,0.2,length.out=1024)),xlab="Time (Year AD)",ylab="",mar=c(5,5,0,2),fig=c(0,1,0,0.33),new=TRUE,
                  marker=list(at=c(10,21,33)/48,lty=c(2L,3L,4L)))
mtext(side=2L,expression(Bassin~supply~(m^3/s)),outer=TRUE,line=-1.5,at=0.55)
dev.copy2pdf(file=sprintf("./figures/Kernel_densities%s_1948-1982.pdf",ds))
#
plotKernelDensity(object=ARWtNet_list[[ds]],trf=identity,xlim=min(time(ARWtNet_list[[ds]]$model$ts$y))+c(3*rng,4*rng),
                  colspace=grey(seq(1,0.2,length.out=1024)),xlab="",
                  ylab="",mar=c(3,5,2,2),fig=c(0,1,0.67,1),marker=list(at=c(10,21,33)/48,lty=c(2L,3L,4L)))
plotKernelDensity(object=ARWtNet_list[[ds]],trf=identity,xlim=min(time(ARWtNet_list[[ds]]$model$ts$y))+c(4*rng,5*rng),
                  colspace=grey(seq(1,0.2,length.out=1024)),xlab="",
                  ylab="",mar=c(4,5,1,2),fig=c(0,1,0.33,0.67),new=TRUE,marker=list(at=c(10,21,33)/48,lty=c(2L,3L,4L)))
plotKernelDensity(object=ARWtNet_list[[ds]],trf=identity,xlim=min(time(ARWtNet_list[[ds]]$model$ts$y))+c(5*rng,6*rng),
                  colspace=grey(seq(1,0.2,length.out=1024)),
                  xlab="Time (Year AD)",ylab="",mar=c(5,5,0,2),fig=c(0,1,0,0.33),new=TRUE,marker=list(at=c(10,21,33)/48,lty=c(2L,3L,4L)))
mtext(side=2L,expression(Bassin~supply~(m^3/s)),outer=TRUE,line=-1.5,at=0.55)
dev.copy2pdf(file=sprintf("./figures/Kernel_densities%s_1982-2016.pdf",ds))
dev.off()
#
ds <- "NBSRainyNet" ; nstep <- 12L
ARWtNet_list[[ds]]$krnl <- getKernelDensity(object=ARWtNet_list[[ds]],trf=identity,nstep=nstep,res=2^8,width=150)
rng <- (max(ARWtNet_list[[ds]]$krnl$time)-min(time(ARWtNet_list[[ds]]$model$ts$y)))/6
plotKernelDensity(object=ARWtNet_list[[ds]],trf=identity,xlim=min(time(ARWtNet_list[[ds]]$model$ts$y))+c(0,rng),
                  colspace=grey(seq(1,0.2,length.out=1024)),xlab="",
                  ylab="",mar=c(3,5,2,2),fig=c(0,1,0.67,1),marker=list(at=c(10,21,33)/48,lty=c(2L,3L,4L)))
plotKernelDensity(object=ARWtNet_list[[ds]],trf=identity,xlim=min(time(ARWtNet_list[[ds]]$model$ts$y))+c(rng,2*rng),
                  colspace=grey(seq(1,0.2,length.out=1024)),xlab="",
                  ylab="",mar=c(4,5,1,2),fig=c(0,1,0.33,0.67),new=TRUE,marker=list(at=c(10,21,33)/48,lty=c(2L,3L,4L)))
plotKernelDensity(object=ARWtNet_list[[ds]],trf=identity,xlim=min(time(ARWtNet_list[[ds]]$model$ts$y))+c(2*rng,3*rng),
                  colspace=grey(seq(1,0.2,length.out=1024)),
                  xlab="Time (Year AD)",ylab="",mar=c(5,5,0,2),fig=c(0,1,0,0.33),new=TRUE,marker=list(at=c(10,21,33)/48,lty=c(2L,3L,4L)))
mtext(side=2L,expression(Bassin~supply~(m^3/s)),outer=TRUE,line=-1.5,at=0.55)
dev.copy2pdf(file=sprintf("./figures/Kernel_densities%s_1948-1982.pdf",ds))
#
plotKernelDensity(object=ARWtNet_list[[ds]],trf=identity,xlim=min(time(ARWtNet_list[[ds]]$model$ts$y))+c(3*rng,4*rng),
                  colspace=grey(seq(1,0.2,length.out=1024)),xlab="",
                  ylab="",mar=c(3,5,2,2),fig=c(0,1,0.67,1),marker=list(at=c(10,21,33)/48,lty=c(2L,3L,4L)))
plotKernelDensity(object=ARWtNet_list[[ds]],trf=identity,xlim=min(time(ARWtNet_list[[ds]]$model$ts$y))+c(4*rng,5*rng),
                  colspace=grey(seq(1,0.2,length.out=1024)),xlab="",
                  ylab="",mar=c(4,5,1,2),fig=c(0,1,0.33,0.67),new=TRUE,marker=list(at=c(10,21,33)/48,lty=c(2L,3L,4L)))
plotKernelDensity(object=ARWtNet_list[[ds]],trf=identity,xlim=min(time(ARWtNet_list[[ds]]$model$ts$y))+c(5*rng,6*rng),
                  colspace=grey(seq(1,0.2,length.out=1024)),
                  xlab="Time (Year AD)",ylab="",mar=c(5,5,0,2),fig=c(0,1,0,0.33),new=TRUE,marker=list(at=c(10,21,33)/48,lty=c(2L,3L,4L)))
mtext(side=2L,expression(Bassin~supply~(m^3/s)),outer=TRUE,line=-1.5,at=0.55)
dev.copy2pdf(file=sprintf("./figures/Kernel_densities%s_1982-2016.pdf",ds))
dev.off()
#
ds <- "NBSRainy" ; nstep <- 12L
ARWtNet_list[[ds]]$krnl <- getKernelDensity(object=ARWtNet_list[[ds]],trf=identity,nstep=nstep,res=2^8,width=150)
rng <- (max(ARWtNet_list[[ds]]$krnl$time)-min(time(ARWtNet_list[[ds]]$model$ts$y)))/6
plotKernelDensity(object=ARWtNet_list[[ds]],trf=identity,xlim=min(time(ARWtNet_list[[ds]]$model$ts$y))+c(0,rng),
                  colspace=grey(seq(1,0.2,length.out=1024)),xlab="",
                  ylab="",mar=c(3,5,2,2),fig=c(0,1,0.67,1),marker=list(at=c(10,21,33)/48,lty=c(2L,3L,4L)))
plotKernelDensity(object=ARWtNet_list[[ds]],trf=identity,xlim=min(time(ARWtNet_list[[ds]]$model$ts$y))+c(rng,2*rng),
                  colspace=grey(seq(1,0.2,length.out=1024)),xlab="",
                  ylab="",mar=c(4,5,1,2),fig=c(0,1,0.33,0.67),new=TRUE,marker=list(at=c(10,21,33)/48,lty=c(2L,3L,4L)))
plotKernelDensity(object=ARWtNet_list[[ds]],trf=identity,xlim=min(time(ARWtNet_list[[ds]]$model$ts$y))+c(2*rng,3*rng),
                  colspace=grey(seq(1,0.2,length.out=1024)),
                  xlab="Time (Year AD)",ylab="",mar=c(5,5,0,2),fig=c(0,1,0,0.33),new=TRUE,marker=list(at=c(10,21,33)/48,lty=c(2L,3L,4L)))
mtext(side=2L,expression(Bassin~supply~(m^3/s)),outer=TRUE,line=-1.5,at=0.55)
dev.copy2pdf(file=sprintf("./figures/Kernel_densities%s_1948-1982.pdf",ds))
#
plotKernelDensity(object=ARWtNet_list[[ds]],trf=identity,xlim=min(time(ARWtNet_list[[ds]]$model$ts$y))+c(3*rng,4*rng),
                  colspace=grey(seq(1,0.2,length.out=1024)),xlab="",
                  ylab="",mar=c(3,5,2,2),fig=c(0,1,0.67,1),marker=list(at=c(10,21,33)/48,lty=c(2L,3L,4L)))
plotKernelDensity(object=ARWtNet_list[[ds]],trf=identity,xlim=min(time(ARWtNet_list[[ds]]$model$ts$y))+c(4*rng,5*rng),
                  colspace=grey(seq(1,0.2,length.out=1024)),xlab="",
                  ylab="",mar=c(4,5,1,2),fig=c(0,1,0.33,0.67),new=TRUE,marker=list(at=c(10,21,33)/48,lty=c(2L,3L,4L)))
plotKernelDensity(object=ARWtNet_list[[ds]],trf=identity,xlim=min(time(ARWtNet_list[[ds]]$model$ts$y))+c(5*rng,6*rng),
                  colspace=grey(seq(1,0.2,length.out=1024)),
                  xlab="Time (Year AD)",ylab="",mar=c(5,5,0,2),fig=c(0,1,0,0.33),new=TRUE,marker=list(at=c(10,21,33)/48,lty=c(2L,3L,4L)))
mtext(side=2L,expression(Bassin~supply~(m^3/s)),outer=TRUE,line=-1.5,at=0.55)
dev.copy2pdf(file=sprintf("./figures/Kernel_densities%s_1982-2016.pdf",ds))
dev.off()
#
ds <- "NBSRainy+Namakan" ; nstep <- 12L
ARWtNet_list[[ds]]$krnl <- getKernelDensity(object=ARWtNet_list[[ds]],trf=identity,nstep=nstep,res=2^8,width=150)
rng <- (max(ARWtNet_list[[ds]]$krnl$time)-min(time(ARWtNet_list[[ds]]$model$ts$y)))/6
plotKernelDensity(object=ARWtNet_list[[ds]],trf=identity,xlim=min(time(ARWtNet_list[[ds]]$model$ts$y))+c(0,rng),
                  colspace=grey(seq(1,0.2,length.out=1024)),xlab="",
                  ylab="",mar=c(3,5,2,2),fig=c(0,1,0.67,1),marker=list(at=c(10,21,33)/48,lty=c(2L,3L,4L)))
plotKernelDensity(object=ARWtNet_list[[ds]],trf=identity,xlim=min(time(ARWtNet_list[[ds]]$model$ts$y))+c(rng,2*rng),
                  colspace=grey(seq(1,0.2,length.out=1024)),xlab="",
                  ylab="",mar=c(4,5,1,2),fig=c(0,1,0.33,0.67),new=TRUE,marker=list(at=c(10,21,33)/48,lty=c(2L,3L,4L)))
plotKernelDensity(object=ARWtNet_list[[ds]],trf=identity,xlim=min(time(ARWtNet_list[[ds]]$model$ts$y))+c(2*rng,3*rng),
                  colspace=grey(seq(1,0.2,length.out=1024)),
                  xlab="Time (Year AD)",ylab="",mar=c(5,5,0,2),fig=c(0,1,0,0.33),new=TRUE,marker=list(at=c(10,21,33)/48,lty=c(2L,3L,4L)))
mtext(side=2L,expression(Bassin~supply~(m^3/s)),outer=TRUE,line=-1.5,at=0.55)
dev.copy2pdf(file=sprintf("./figures/Kernel_densities%s_1948-1984.pdf",ds))
#
plotKernelDensity(object=ARWtNet_list[[ds]],trf=identity,xlim=min(time(ARWtNet_list[[ds]]$model$ts$y))+c(3*rng,4*rng),
                  colspace=grey(seq(1,0.2,length.out=1024)),xlab="",
                  ylab="",mar=c(3,5,2,2),fig=c(0,1,0.67,1),marker=list(at=c(10,21,33)/48,lty=c(2L,3L,4L)))
plotKernelDensity(object=ARWtNet_list[[ds]],trf=identity,xlim=min(time(ARWtNet_list[[ds]]$model$ts$y))+c(4*rng,5*rng),
                  colspace=grey(seq(1,0.2,length.out=1024)),xlab="",
                  ylab="",mar=c(4,5,1,2),fig=c(0,1,0.33,0.67),new=TRUE,marker=list(at=c(10,21,33)/48,lty=c(2L,3L,4L)))
plotKernelDensity(object=ARWtNet_list[[ds]],trf=identity,xlim=min(time(ARWtNet_list[[ds]]$model$ts$y))+c(5*rng,6*rng),
                  colspace=grey(seq(1,0.2,length.out=1024)),
                  xlab="Time (Year AD)",ylab="",mar=c(5,5,0,2),fig=c(0,1,0,0.33),new=TRUE,marker=list(at=c(10,21,33)/48,lty=c(2L,3L,4L)))
mtext(side=2L,expression(Bassin~supply~(m^3/s)),outer=TRUE,line=-1.5,at=0.55)
dev.copy2pdf(file=sprintf("./figures/Kernel_densities%s_1984-2016.pdf",ds))
dev.off()
rm(ds,rng,nstep)
#
## organisation des prévisions:
ARWtNet_list[["NBSNamakan"]]$predmat <- getPredMatrix(ARWtNet_list[["NBSNamakan"]],12L)
ARWtNet_list[["NBSRainyNet"]]$predmat <- getPredMatrix(ARWtNet_list[["NBSRainyNet"]],12L)
ARWtNet_list[["NBSRainy"]]$predmat <- getPredMatrix(ARWtNet_list[["NBSRainy"]],12L)
ARWtNet_list[["NBSRainy+Namakan"]]$predmat <- getPredMatrix(ARWtNet_list[["NBSRainy+Namakan"]],12L)
if(FALSE) save(ARWtNet_list,file="../Data/ARWtNet.rda")
else load(file="../Data/ARWtNet.rda")
#
# ds <- "NBSNamakan" ; wh <- c(1L,3L,6L,9L,12L)
# ds <- "NBSRainyNet" ; wh <- c(1L,3L,6L,9L,12L)
# ds <- "NBSRainy" ; wh <- c(1L,3L,6L,9L,12L)
# ds <- "NBSRainy+Namakan" ; wh <- c(1L,3L,6L,9L,12L)
rng <- (max(ARWtNet_list[[ds]]$krnl$time)-min(time(ARWtNet_list[[ds]]$model$ts$y)))/6
plotPredMatrix(object=ARWtNet_list[[ds]],trf=identity,xlim=min(time(ARWtNet_list[[ds]]$model$ts$y))+c(0,rng),which=wh,
               xlab="",ylab="",mar=c(3,5,2,2),fig=c(0,1,0.67,1),marker=list(at=c(10,21,33)/48,lty=c(2L,3L,4L)))
plotPredMatrix(object=ARWtNet_list[[ds]],trf=identity,xlim=min(time(ARWtNet_list[[ds]]$model$ts$y))+c(rng,2*rng),which=wh,
               xlab="",ylab="",mar=c(4,5,1,2),fig=c(0,1,0.33,0.67),new=TRUE,marker=list(at=c(10,21,33)/48,lty=c(2L,3L,4L)))
plotPredMatrix(object=ARWtNet_list[[ds]],trf=identity,xlim=min(time(ARWtNet_list[[ds]]$model$ts$y))+c(2*rng,3*rng),which=wh,
               xlab="Time (Year AD)",ylab="",mar=c(5,5,0,2),fig=c(0,1,0,0.33),new=TRUE,marker=list(at=c(10,21,33)/48,lty=c(2L,3L,4L)))
mtext(side=2L,expression(Bassin~supply~(m^3/s)),outer=TRUE,line=-1.5,at=0.55)
dev.copy2pdf(file=sprintf("./figures/Monthly_forecasts%s_1948-1982.pdf",ds))
#
plotPredMatrix(object=ARWtNet_list[[ds]],trf=identity,xlim=min(time(ARWtNet_list[[ds]]$model$ts$y))+c(3*rng,4*rng),which=wh,
               xlab="",ylab="",mar=c(3,5,2,2),fig=c(0,1,0.67,1),marker=list(at=c(10,21,33)/48,lty=c(2L,3L,4L)))
plotPredMatrix(object=ARWtNet_list[[ds]],trf=identity,xlim=min(time(ARWtNet_list[[ds]]$model$ts$y))+c(4*rng,5*rng),which=wh,
               xlab="",ylab="",mar=c(4,5,1,2),fig=c(0,1,0.33,0.67),new=TRUE,marker=list(at=c(10,21,33)/48,lty=c(2L,3L,4L)))
plotPredMatrix(object=ARWtNet_list[[ds]],trf=identity,xlim=min(time(ARWtNet_list[[ds]]$model$ts$y))+c(5*rng,6*rng),which=wh,
               xlab="Time (Year AD)",ylab="",mar=c(5,5,0,2),fig=c(0,1,0,0.33),new=TRUE,marker=list(at=c(10,21,33)/48,lty=c(2L,3L,4L)))
mtext(side=2L,expression(Bassin~supply~(m^3/s)),outer=TRUE,line=-1.5,at=0.55)
dev.copy2pdf(file=sprintf("./figures/Monthly_forecasts%s_1982-2016.pdf",ds))
dev.off()
#
# names(ARWtNet_list)
# names(ARWtNet_list[["NBSIFL"]])
ARWtNet_plot <- list(TempIFL=ARWtNet_list[["TempIFL"]][1L],
                     NAO=ARWtNet_list[["NAO"]][1L],
                     NBSNamakan=ARWtNet_list[["NBSNamakan"]][-(4L:5L)],
                     NBSRainyNet=ARWtNet_list[["NBSRainyNet"]][-(4L:5L)],
                     NBSRainy=ARWtNet_list[["NBSRainy"]][-(4L:5L)],
                     `NBSRainy+Namakan`=ARWtNet_list[["NBSRainy+Namakan"]][-4L])
save(ARWtNet_plot,file="../Data/ARWtNet_plot.rda")
# floor(min(time(ARWtNet_plot[["NBSNamakan"]]$model$ts$y)))
# floor(max(time(ARWtNet_plot[["NBSNamakan"]]$model$ts$y)))
# dim(ARWtNet_plot[["NBSIFL"]]$QMsynth)
#
### Pour affichage avec sweave/knitr (R s'exécutant dans du LaTeX)
ds <- "NBSRainy"
wh <- c(1L,3L,6L,9L,12L)
X <- ARWtNet_plot[[ds]]
T <- time(X$model$ts$y)
rng <- (max(T)-min(T))/9
plotPredMatrix(object=X,trf=identity,xlim=min(T)+c(0,rng),xlab="",which=wh,cols=c("red","yellow","green","blue","purple"),
               ylab="",mar=c(3,5,2,2),fig=c(0,1,0.67,1),marker=list(at=c(10,21,33)/48,lty=c(2L,3L,4L)))
plotPredMatrix(object=X,trf=identity,xlim=min(T)+c(rng,2*rng),xlab="",which=wh,cols=c("red","yellow","green","blue","purple"),
               ylab="",mar=c(4,5,1,2),fig=c(0,1,0.33,0.67),new=TRUE,marker=list(at=c(10,21,33)/48,lty=c(2L,3L,4L)))
plotPredMatrix(object=X,trf=identity,xlim=min(T)+c(2*rng,3*rng),xlab="Time (Year AD)",which=wh,
               cols=c("red","yellow","green","blue","purple"),ylab="",mar=c(5,5,0,2),fig=c(0,1,0,0.33),new=TRUE,
               marker=list(at=c(10,21,33)/48,lty=c(2L,3L,4L)))
mtext(side=2L,expression(Bassin~supply~(m^3/s)),outer=TRUE,line=-1.5,at=0.55)
rm(ds,X,rng,wh)
#
## Créer un tableau de qualité de prédiction
foreQualTable <- NULL ; nstep <- 12L
for(i in 1L:nstep) {
  foreQualTable <- rbind(foreQualTable,c(MRSEpreds(ARWtNet_plot[["NBSNamakan"]],i),PRSQpreds(ARWtNet_plot[["NBSNamakan"]],i),
                                         MRSEpreds(ARWtNet_plot[["NBSRainyNet"]],i),PRSQpreds(ARWtNet_plot[["NBSRainyNet"]],i),
                                         MRSEpreds(ARWtNet_plot[["NBSRainy"]],i),PRSQpreds(ARWtNet_plot[["NBSRainy"]],i),
                                         MRSEpreds(ARWtNet_plot[["NBSRainy+Namakan"]],i),PRSQpreds(ARWtNet_plot[["NBSRainy+Namakan"]],i)))
}
#
## Namakan
sprintf("$RMSE=%.1f\\,R^{2}=%.2f$",MRSEpreds(ARWtNet_plot[["NBSNamakan"]],1L),PRSQpreds(ARWtNet_plot[["NBSNamakan"]],1L))
sprintf("$RMSE=%.1f\\,R^{2}=%.2f$",MRSEpreds(ARWtNet_plot[["NBSNamakan"]],3L),PRSQpreds(ARWtNet_plot[["NBSNamakan"]],3L))
sprintf("$RMSE=%.1f\\,R^{2}=%.2f$",MRSEpreds(ARWtNet_plot[["NBSNamakan"]],5L),PRSQpreds(ARWtNet_plot[["NBSNamakan"]],5L))
#
sprintf("$RMSE=%.1f\\,R^{2}=%.2f$",MRSEpreds(ARWtNet_plot[["NBSNamakan"]],6L),PRSQpreds(ARWtNet_plot[["NBSNamakan"]],6L))
sprintf("$RMSE=%.1f\\,R^{2}=%.2f$",MRSEpreds(ARWtNet_plot[["NBSNamakan"]],12L),PRSQpreds(ARWtNet_plot[["NBSNamakan"]],12L))
#
## Rainy net
sprintf("$RMSE=%.1f\\,R^{2}=%.2f$",MRSEpreds(ARWtNet_plot[["NBSRainyNet"]],1L),PRSQpreds(ARWtNet_plot[["NBSRainyNet"]],1L))
sprintf("$RMSE=%.1f\\,R^{2}=%.2f$",MRSEpreds(ARWtNet_plot[["NBSRainyNet"]],3L),PRSQpreds(ARWtNet_plot[["NBSRainyNet"]],3L))
sprintf("$RMSE=%.1f\\,R^{2}=%.2f$",MRSEpreds(ARWtNet_plot[["NBSRainyNet"]],5L),PRSQpreds(ARWtNet_plot[["NBSRainyNet"]],6L))
#
sprintf("$RMSE=%.1f\\,R^{2}=%.2f$",MRSEpreds(ARWtNet_plot[["NBSRainyNet"]],6L),PRSQpreds(ARWtNet_plot[["NBSRainyNet"]],7L))
sprintf("$RMSE=%.1f\\,R^{2}=%.2f$",MRSEpreds(ARWtNet_plot[["NBSRainyNet"]],12L),PRSQpreds(ARWtNet_plot[["NBSRainyNet"]],12L))
#
## Rainy total
sprintf("$RMSE=%.1f\\,R^{2}=%.2f$",MRSEpreds(ARWtNet_plot[["NBSRainy"]],1L),PRSQpreds(ARWtNet_plot[["NBSRainy"]],1L))
sprintf("$RMSE=%.1f\\,R^{2}=%.2f$",MRSEpreds(ARWtNet_plot[["NBSRainy"]],3L),PRSQpreds(ARWtNet_plot[["NBSRainy"]],3L))
sprintf("$RMSE=%.1f\\,R^{2}=%.2f$",MRSEpreds(ARWtNet_plot[["NBSRainy"]],5L),PRSQpreds(ARWtNet_plot[["NBSRainy"]],6L))
#
sprintf("$RMSE=%.1f\\,R^{2}=%.2f$",MRSEpreds(ARWtNet_plot[["NBSRainy"]],6L),PRSQpreds(ARWtNet_plot[["NBSRainy"]],7L))
sprintf("$RMSE=%.1f\\,R^{2}=%.2f$",MRSEpreds(ARWtNet_plot[["NBSRainy"]],12L),PRSQpreds(ARWtNet_plot[["NBSRainy"]],12L))
#
## Rainy + Namakan
sprintf("$RMSE=%.1f\\,R^{2}=%.2f$",MRSEpreds(ARWtNet_plot[["NBSRainy+Namakan"]],1L),PRSQpreds(ARWtNet_plot[["NBSRainy+Namakan"]],1L))
sprintf("$RMSE=%.1f\\,R^{2}=%.2f$",MRSEpreds(ARWtNet_plot[["NBSRainy+Namakan"]],3L),PRSQpreds(ARWtNet_plot[["NBSRainy+Namakan"]],3L))
sprintf("$RMSE=%.1f\\,R^{2}=%.2f$",MRSEpreds(ARWtNet_plot[["NBSRainy+Namakan"]],5L),PRSQpreds(ARWtNet_plot[["NBSRainy+Namakan"]],6L))
#
sprintf("$RMSE=%.1f\\,R^{2}=%.2f$",MRSEpreds(ARWtNet_plot[["NBSRainy+Namakan"]],6L),PRSQpreds(ARWtNet_plot[["NBSRainy+Namakan"]],7L))
sprintf("$RMSE=%.1f\\,R^{2}=%.2f$",MRSEpreds(ARWtNet_plot[["NBSRainy+Namakan"]],12L),PRSQpreds(ARWtNet_plot[["NBSRainy+Namakan"]],12L))
#
### Prévision par crues.
#
# ds <- "NBSNamakan" ; trf <- identity ; focus <- c(3,9)
# ds <- "NBSRainyNet" ; trf <- identity ; focus <- c(3,9)
# ds <- "NBSRainy" ; trf <- identity ; focus <- c(3,9)
# ds <- "NBSRainy+Namakan" ; trf <- identity ; focus <- c(3,9)
#
tmp <- as.integer(1+frequency(ARWtNet_list[[ds]]$model$ts$y)*ARWtNet_list[[ds]]$krnl$time%%1)
tmp <- (tmp>=focus[1L])&(tmp<=focus[2L])
tmp2 <- floor(ARWtNet_list[[ds]]$krnl$time)
fore <- tapply(trf(ARWtNet_list[[ds]]$krnl$quantile[1L:length(ARWtNet_list[[ds]]$model$response)][tmp[1L:length(ARWtNet_list[[ds]]$model$response)]]),
                   tmp2[1L:length(ARWtNet_list[[ds]]$model$response)][tmp[1L:length(ARWtNet_list[[ds]]$model$response)]],mean,na.rm=TRUE)
tmp <- as.integer(1+frequency(ARWtNet_list[[ds]]$model$ts$y)*ARWtNet_list[[ds]]$model$time%%1)
tmp <- (tmp>=focus[1L])&(tmp<=focus[2L])
tmp2 <- floor(ARWtNet_list[[ds]]$model$time)
obs <- tapply(ARWtNet_list[[ds]]$model$response[tmp],tmp2[tmp],mean,na.rm=TRUE)
ans <- cbind(observed=obs,forecast=fore[match(names(fore),names(obs))])
par(mar=c(5,5,2,2))
plot(observed~forecast,data=ans,asp=1,xlim=range(ans[,"observed"]),ylim=range(ans[,"observed"])) ; abline(0,1)
1-sum((ans[,"observed"]-ans[,"forecast"])^2)/sum((ans[,"observed"]-mean(ans[,"observed"]))^2)
#
nqtl <- 3L ; qtl <- quantile(ans[,"observed"],prob=seq(0,1,nqtl^-1)) ; qtl[nqtl+1L] <- Inf
abline(v=qtl) ; abline(h=qtl)
classif <- matrix(0,nqtl,nqtl)
for(i in 1L:nrow(classif)) for(j in 1L:ncol(classif))
  classif[i,j] <- sum(((ans[,"observed"]>=qtl[i])&(ans[,"observed"]<qtl[i+1L]))&((ans[,"forecast"]>=qtl[j])&(ans[,"forecast"]<qtl[j+1L])))
100*sum(diag(classif))/sum(classif)
rm(ds,trf,focus,tmp,tmp2,fore,ans,nqtl,qtl,classif,i,j)
#
### NBSNamakan:       43.75   -  0.06612266
### NBSRainyNet:      39.0625 -  0.1357559
### NBSRainy:         35.9375 -  0.1130831
### NBSRainy+Namakan: 37.5    -  0.08680067
#
### La performance pour une saison entière est somme toute assez marginale.
### Conclusion: le phénomène est très difficile à prévoir, si ce n'est à brève échéance.
### On peut bien voir venir l'évolution des coups d'eau une fois qu'ils ont commencé
### mais on ne peut par le voir venir avant qu'ils ne commence à survenir.
### Ce n'est pas évident de prévoir des saisons entières avant même qu'elles aient débuté mais
### il est sans doute possible de voir venir le reste d'une saison qui a déjà débutée.
### Reste à voir dans quelle mesure et avec quelle précision.
#
