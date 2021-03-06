#
### Analyse spectrale de la cyclicit� des apports en eau: syst�me Namakan Rainy.
#
## rm(list=ls())
dat <- list()
dat$series <- read.table(file = "../Data/DataRainyNamakanLevelDischarge.csv", sep=";", dec=".", header = TRUE, skip = 3L)
## � l'origine, j'utilisais le fichier au quarts de mois. �a impliquait beaucoup de manipulation alors on va y aller
## au jour puis on smoothera �a proc�dera au quart de mois par la suite.
dat$series[["Date"]] <- as.POSIXlt(sprintf("%s-%s-%s 00:00:00",
                                           substr(as.character(dat$series[["Date"]]),1L,4L),
                                           substr(as.character(dat$series[["Date"]]),6L,7L),
                                           substr(as.character(dat$series[["Date"]]),9L,10L)),tz="GMT")
##
## summary(dat$series)
## tail(dat$series)
dat$Namakan <- list(LV=read.table(file = "../Data/Namakan-levels.csv", sep=";", dec=",", header = TRUE))
dat$Rainy <- list(LV=read.table(file = "../Data/Rainy-levels.csv", sep=";", dec=",", header = TRUE))
## source("NBScalc-AUX.R")
## library(glmnet)
for(i in c("Namakan","Rainy")) {
    ## i <- "Namakan"
    dat[[i]]$poly <- poly(dat[[i]]$LV$Level,degree=5)
    dat[[i]]$glmnet <- glmnet::cv.glmnet(y=dat[[i]]$LV$Volume,alpha=0.01,grouped=FALSE,x=dat[[i]]$poly)
    coef(dat[[i]]$glmnet,s="lambda.min")
    plot(y=dat[[i]]$LV$Volume,x=dat[[i]]$LV$Level)
    ## predict(object=dat[[i]]$poly,newdata=seq(min(dat[[i]]$LV$Level),max(dat[[i]]$LV$Level),0.01))
    lines(y=predict(dat[[i]]$glmnet,newx=predict(object=dat[[i]]$poly,newdata=seq(min(dat[[i]]$LV$Level),max(dat[[i]]$LV$Level),0.01)),s="lambda.min"),
          x=seq(min(dat[[i]]$LV$Level),max(dat[[i]]$LV$Level),0.01),col=1L)
    if(is.null(locator(1L))) break
} ; rm(i)
dat[["Namakan"]]$Volume <- predict(object=dat[["Namakan"]]$glmnet,newx=predict(object=dat[["Namakan"]]$poly,newdata=dat$series[["Stn.188"]]),s="lambda.min")
dat[["Rainy"]]$Volume <- predict(object=dat[["Rainy"]]$glmnet,newx=predict(object=dat[["Rainy"]]$poly,newdata=dat$series[["Stn.132"]]),s="lambda.min")
##
dat$NBS <- list()
### Calcul des NBS par saut inter-journaliers
i <- "Namakan"
dat$NBS[[i]] <- data.frame(Supply=(dat[[i]]$Volume[2L:length(dat[[i]]$Volume)]-dat[[i]]$Volume[1L:(length(dat[[i]]$Volume)-1L)])/86400 +
                               0.5*(dat$series[2L:length(dat[[i]]$Volume),"Stn.47"]+dat$series[1L:(length(dat[[i]]$Volume)-1L),"Stn.47"]),
                           Date=dat$series$Date[2L:length(dat[[i]]$Volume)])
plot(y=dat$NBS[[i]][,"Supply"],x=dat$NBS[[i]][,"Date"],type="l",xlab="Time",ylab=expression(Bassin~supply~(m^3~s^{-1})))
## plot(y=dat$NBS[[i]][,"Supply"],x=dat$NBS[[i]][,"Date"],type="l",xlab="Time",ylab=expression(Bassin~supply~(m^3~s^{-1})),
##      xlim=as.numeric(as.POSIXlt(c("1942-01-01 00:00:00","1952-01-01 00:00:00"),tz="GMT")))
i <- "Rainy"
dat$NBS[[i]] <- data.frame(Supply=(dat[[i]]$Volume[2L:length(dat[[i]]$Volume)]-dat[[i]]$Volume[1L:(length(dat[[i]]$Volume)-1L)])/86400 +
                               0.5*(dat$series[2L:length(dat[[i]]$Volume),"Stn.58"]+dat$series[1L:(length(dat[[i]]$Volume)-1L),"Stn.58"]),
                           Date=dat$series$Date[2L:length(dat[[i]]$Volume)])
plot(y=dat$NBS[[i]][,"Supply"],x=dat$NBS[[i]][,"Date"],type="l",xlab="Time",ylab=expression(Bassin~supply~(m^3~s^{-1})))
## plot(y=dat$NBS[[i]][,"flow"],x=dat$NBS[[i]][,"Date"],type="l",xlab="Time",ylab=expression(Bassin~supply~(m^3~s^{-1})),
##      xlim=as.numeric(as.POSIXlt(c("1942-01-01 00:00:00","1943-01-01 00:00:00"),tz="GMT")))
dat$NBS[["RainyNet"]] <- data.frame(Supply=(dat[[i]]$Volume[2L:length(dat[[i]]$Volume)]-dat[[i]]$Volume[1L:(length(dat[[i]]$Volume)-1L)])/86400 +
                                        0.5*(dat$series[2L:length(dat[[i]]$Volume),"Stn.58"]+dat$series[1L:(length(dat[[i]]$Volume)-1L),"Stn.58"]) -
                                        0.5*(dat$series[2L:length(dat[[i]]$Volume),"Stn.47"]+dat$series[1L:(length(dat[[i]]$Volume)-1L),"Stn.47"]),
                                    Date=dat$series$Date[2L:length(dat[[i]]$Volume)])
plot(y=dat$NBS[["RainyNet"]][,"Supply"],x=dat$NBS[["RainyNet"]][,"Date"],type="l",xlab="Time (y)",ylab=expression(Bassin~supply~(m^3~s^{-1})))
rm(i)
##
plot(y=dat$NBS[["Namakan"]]$Supply,x=dat$NBS[["Namakan"]]$Date,type="l") ; abline(h=0)
lines(y=dat$series[,"Stn.172"],dat$series[,"Date"],col=2)
plot(y=dat$NBS[["Rainy"]]$Supply,x=dat$NBS[["Rainy"]]$Date,type="l") ; abline(h=0)
lines(y=dat$series[,"Stn.176"],dat$series[,"Date"],col=2)
plot(y=dat$series[,"TMED"],dat$series[,"Date"],type="l") ; abline(h=0)
##
### On laisse tomber les calculs au quart de mois: c'est de la bouette, on scrap nos ddl.
### Je propose tout simplement de passer un filtre passe bas pour att�nuer le bruit sur la variable
### r�ponse... encore que le mod�le s'en chargera sans doute de toute fa�on.
##
width <- 7L
sigma <- 0.02
weights <- exp(-sigma*(-width:width)^2)
degree <- 5L
##
## all(dat$series[-1,"Date"]==dat$NBS[["Namakan"]][,"Date"])
dat$model_data <- data.frame(Date=dat$series[-1,"Date"])
## dat$model_data[,"Date"]
for(i in c("Namakan","Rainy","RainyNet")) {
    ## i <- "Namakan"
    dat$model_data[[paste(i,"Supply",sep="_")]] <- rep(NA,nrow(dat$NBS[[i]]))
    dat$model_data[[paste(i,"Rate",sep="_")]] <- rep(NA,nrow(dat$NBS[[i]]))
    cols <- paste(i,c("Supply","Rate"),sep="_")
    for(j in which(!is.na(dat$NBS[[i]][,1L]))) {
        ## j <- head(which(!is.na(dat$NBS[[i]][,1L])),1L)
        ## j <- tail(which(!is.na(dat$NBS[[i]][,1L])),1L)
        sel <- 1L:(2*width+1L)
        win <- (j-width):(j+width)
        sel <- sel[win>1]
        win <- win[win>1]
        sel <- sel[win<nrow(dat$model_data)]
        win <- win[win<nrow(dat$model_data)]
        sel <- sel[!is.na(dat$NBS[[i]][win,"Supply"])]
        win <- win[!is.na(dat$NBS[[i]][win,"Supply"])]
        X <- poly(x=(-width:width)[sel],degree)
        lm1 <- lm.wfit(x=cbind(1,X),y=dat$NBS[[i]][win,"Supply"],w=weights[sel])
        dat$model_data[j,cols] <- cbind(1,predict(X,newdata=0))%*%cbind(lm1$coefficients,c((1:degree)*lm1$coefficients[2L:(degree+1L)],0))
        if(FALSE) {
            plot(y=dat$NBS[[i]][["Supply"]],x=dat$NBS[[i]][,"Date"],type="l",xlab="Time",ylab=expression(Bassin~supply~(m^3~s^{-1})),
                 xlim=dat$NBS[[i]][j+c(-10,10),"Date"])
            lines(y=(cbind(1,predict(X,newdata=-width:width))%*%lm1$coefficients)[sel],
                  x=dat$NBS[[i]][win,"Date"])
            points(y=dat$model_data[j,cols[1L]],x=dat$NBS[[i]][j,"Date"],pch=21,bg="red")
            ## if(is.null(locator(1L))) break
        }
    }
    par(mar=c(5,5,2,5))
    plot(y=dat$NBS[[i]][,"Supply"],x=dat$NBS[[i]][,"Date"],type="l",xlab="Time",ylab=expression(Water~supply~(m^3~s^{-1})))
    lines(y=dat$model_data[,cols[1L]],x=dat$model_data[,"Date"],col="red")
    par(new=TRUE)
    plot(y=dat$model_data[,cols[2L]],x=dat$model_data[,"Date"],type="l",xlab="Time",ylab="",col="green",axes=FALSE)
    axis(4L)
    mtext(text=expression(Supply~change~rate~(m^3~s^{-1}~d^{-1})),side=4L,line=-2,outer=TRUE)
}
rm(i,j,sel,win,X,lm1,cols)
##
par(mar=c(5,5,2,5))
plot(y=dat$NBS[["Namakan"]][,"Supply"],x=dat$NBS[["Namakan"]][,"Date"],type="l",xlab="Time",ylab=expression(Water~supply~(m^3~s^{-1})),
     xlim=as.numeric(as.POSIXlt(c("1942-01-01 00:00:00","1952-01-01 00:00:00"),tz="GMT")))
lines(y=dat$model_data[["Namakan_Supply"]],x=dat$model_data[["Date"]],col="red")
par(new=TRUE)
plot(y=dat$model_data[["Namakan_Rate"]],x=dat$model_data[["Date"]],type="l",xlab="Time",ylab="",col="green",axes=FALSE,
     xlim=as.numeric(as.POSIXlt(c("1942-01-01 00:00:00","1952-01-01 00:00:00"),tz="GMT")))
axis(4L)
mtext(text=expression(Supply~change~rate~(m^3~s^{-1}~d^{-1})),side=4L,line=-2,outer=TRUE)
##
### Int�gration de variables climatologiques.
##
### La temp�rature moyenne � IFL + son taux d'augmentation calul�s par la m�thode ci dessus:
dat$model_data[,"TempIFL"] <- NA
dat$model_data[,"TempRate"] <- NA
cols <- paste("Temp",c("IFL","Rate"),sep="")
for(j in which(!is.na(dat$series[,"TMED"]))) {
    ## j <- head(which(!is.na(dat$series[,"TMED"])),1L)+100
    ## j <- tail(which(!is.na(dat$series[,"TMED"])),1L)
    sel <- 1L:(2*width+1L)
    win <- (j-width):(j+width)
    sel <- sel[win>1]
    win <- win[win>1]
    sel <- sel[win<nrow(dat$model_data)]
    win <- win[win<nrow(dat$model_data)]
    sel <- sel[!is.na(dat$series[win,"TMED"])]
    win <- win[!is.na(dat$series[win,"TMED"])]
    X <- poly(x=(-width:width)[sel],degree)
    lm1 <- lm.wfit(x=cbind(1,X),y=dat$series[win,"TMED"],w=weights[sel])
    wh <- which(dat$model_data[,"Date"]==dat$series[j,"Date"])
    dat$model_data[wh,cols] <- cbind(1,predict(X,newdata=0))%*%cbind(lm1$coefficients,c((1:degree)*lm1$coefficients[2L:(degree+1L)],0))
    if(FALSE) {
        plot(y=dat$series[["TMED"]],x=dat$series[["Date"]],type="l",xlab="Time",ylab=expression(Bassin~supply~(m^3~s^{-1})),
             xlim=as.numeric(dat$series[j+c(-10,10),"Date"]))
        lines(y=(cbind(1,predict(X,newdata=-width:width))%*%lm1$coefficients)[sel],
              x=dat$series[win,"Date"])
        points(y=dat$model_data[wh,cols[1L]],x=dat$model_data[wh,"Date"],pch=21,bg="red")
        ## if(is.null(locator(1L))) break
    }
}
rm(j,sel,win,X,lm1,wh)
par(mar=c(5,5,2,5))
plot(y=dat$series[,"TMED"],x=dat$series[,"Date"],type="l",xlab="Time",ylab=expression(Temperature~(degree~C)))
lines(y=dat$model_data[,cols[1L]],x=dat$model_data[,"Date"],col="red")
par(new=TRUE)
plot(y=dat$model_data[,cols[2L]],x=dat$model_data[,"Date"],type="l",xlab="Time",ylab="",col="green",axes=FALSE)
axis(4L)
mtext(text=expression(Temperature~change~rate~(degree~C~d^{-1})),side=4L,line=-2,outer=TRUE)
rm(width,sigma,weights,degree)
rm(cols)
##
### Pourrait �tre obtenu automatiquement, voir:
### http://www.cpc.noaa.gov/products/precip/CWlink/pna/norm.nao.monthly.b5001.current.ascii.table
NAO <- read.table(file = "../Data/NAO.csv", sep=";", dec=".", header = FALSE)
tmp <- NAO[,1L] ; NAO <- NAO[,-1L]
NAO <- data.frame(Date=as.POSIXlt(sprintf("%d-%02d-01 00:00:00",rep(tmp,each=12L),rep(1:12,length(tmp))),tz="GMT"),
                  NAO=as.numeric(t(NAO)))
NAO <- NAO[!is.na(NAO[,"NAO"]),]
NAO[NAO[,"NAO"]==-999.90,"NAO"] <- NA  # NAO[is.na(NAO[,"NAO"]),]
NAO <- NAO[-(1L:tail(which(is.na(NAO[,"NAO"])),n=1L)),] ; rm(tmp)
## plot(y=NAO[,"NAO"],x=NAO[,"Date"],type="l",las=1L,xlim=as.numeric(as.POSIXlt(sprintf("%d-01-01 00:00:00",c(2000,2020)),tz="GMT")))
## plot(y=NAO[,"NAO"],x=NAO[,"Date"],type="l",las=1L)
##
### Il faudra faire un lissage de ces donn�es pour les amener au m�me niveau de r�solution que les autres.
### Je ne vais pas investir de temps dans cette entreprise car c'est ENSO qui est utilis� en fin de compte.
##
### ENSO: Estimation Historiques (� fiable)
tmp <- read.table("../Data/MRI.ext",skip=5L)
ENSO.ext <- tmp[,-1L]
ENSO.ext <- data.frame(Date=as.POSIXlt(sprintf("%d-%02d-01 00:00:00",rep(tmp[,1L],each=12L),rep(1:12,length(tmp[,1L]))),tz="GMT"),
                       ENSO=as.numeric(t(ENSO.ext)))
##
### ENSO: depuis NOAA (d'apr�s Oli, il serait pr�f�rable d'utiliser leur web service. Ils sont bons... m'a-t-on dit.)
if(FALSE) {
    tmp <- XML::readHTMLTable(doc="http://origin.cpc.ncep.noaa.gov/products/analysis_monitoring/ensostuff/ONI_v5.php")[[10L]]
    tmp <- tmp[tmp[1L]!="Year",]
    tmp[,1L] <- as.numeric(as.character(tmp[,1L]))
    ENSO.recent <- tmp[,-1L]
    ENSO.recent <- data.frame(Date=as.POSIXlt(sprintf("%d-%02d-01 00:00:00",rep(tmp[,1L],each=12L),rep(1:12,length(tmp[,1L]))),tz="GMT"),
                              ENSO=as.numeric(t(ENSO.recent)))
    rm(tmp)
    write.csv(ENSO.recent,file="../Data/ENSO.recent.csv")
    save(ENSO.recent,file="../Data/ENSO.recent.rda")
} else load(file="../Data/ENSO.recent.rda")
##
ENSO <- rbind(ENSO.ext[is.na(match(ENSO.ext[,"Date"],ENSO.recent[,"Date"])),],ENSO.recent)
##
### Encore une fois de plus, il faudra faire un lissage de ces donn�es pour les amener au m�me niveau de r�solution que les autres.
### Je vais devoir investir du temps dans cette entreprise...
##
## str(dat)
n <- 5L
degree <- 3L
sigma <- 0.0005
##
dat$model_data[,"ENSO"] <- NA
for(i in 1L:nrow(dat$model_data)) {
    ## i <- 1L
    wh <- order(abs(ENSO[,"Date"]-dat$model_data[i,"Date"]))[1L:n]
    wh <- sort(wh)
    smpl <- (as.numeric(ENSO[wh,"Date"])-as.numeric(dat$model_data[i,"Date"]))/84600
    w <- exp(-sigma*(smpl)^2)
    X <- poly(x=smpl,degree=degree)
    lm1 <- lm.wfit(x=cbind(1,X),y=ENSO[wh,"ENSO"],w=w)
    dat$model_data[i,"ENSO"] <- cbind(1,predict(X,newdata=0))%*%lm1$coefficients
    if(FALSE) {
        plot(y=ENSO[,"ENSO"],x=ENSO[,"Date"],type="l",xlim=ENSO[range(wh),"Date"])
        lines(y=cbind(1,predict(X,newdata=seq(min(smpl),max(smpl),0.1)))%*%lm1$coefficients,
              x=dat$model_data[i,"Date"]+84600*seq(min(smpl),max(smpl),0.1),col="red")
        points(y=dat$model_data[i,"ENSO"],dat$model_data[i,"Date"],pch=21,bg="red")
    }
}
rm(n,degree,sigma,i,wh,smpl,w,X,lm1)
##
par(mar=c(5,5,2,5))
plot(y=ENSO[,"ENSO"],x=ENSO[,"Date"],type="l",xlab="Time",ylab=expression(Temperature~(degree~C)))
lines(y=dat$model_data[,"ENSO"],x=dat$model_data[,"Date"],col="red")
##
### Les variables climatologiques (TempIFL, TempRate et ENSO) sont entr�es.
##
### Retirer les lignes inutiles de model_data:
from <- which(apply(dat$model_data[,paste(c("Namakan","Rainy","RainyNet"),"Rate",sep="_")],1L,
                    function(x) any(!is.na(x))) & !is.na(dat$model_data[,"TempIFL"]))[1L]
## from <- which(apply(dat$model_data[,paste(c("Namakan","Rainy","RainyNet"),"Rate",sep="_")],1L,
##                     function(x) any(!is.na(x))))[1L]
safe <- dat$model_data
dat$model_data <- dat$model_data[from:nrow(dat$model_data),]
rownames(dat$model_data) <- NULL
## summary(dat$model_data)
## dim(dat$model_data)
## tail(dat$model_data,20L)
##
### R�capitulatif:
##
par(mar=c(5,5,4,5))
plot(Namakan_Supply~Date,data=dat$model_data,type="l") ; par(new=TRUE)
plot(ENSO~Date,data=dat$model_data,type="l",col="red",lty=3L,yaxt="n",ylab="")
axis(4L) ; mtext("ENSO",side=4L,line=2)
##
par(mar=c(5,5,4,5))
plot(Namakan_Supply~Date,data=dat$model_data,type="l") ; par(new=TRUE)
plot(TempIFL~Date,data=dat$model_data,type="l",col="red",lty=3L,yaxt="n",ylab="")
axis(4L) ; mtext("Temperature",side=4L,line=2)
##
par(mar=c(5,5,4,5))
plot(Namakan_Supply~Date,data=dat$model_data,type="l") ; par(new=TRUE)
plot(TempRate~Date,data=dat$model_data,type="l",col="red",lty=3L,yaxt="n",ylab="")
axis(4L) ; mtext("Temperature rate",side=4L,line=2)
##
save(dat,file="../Data/WaterPeriodicity-dat.rda")
##
