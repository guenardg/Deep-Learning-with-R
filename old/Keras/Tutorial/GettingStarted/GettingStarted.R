##
### Didacticiel de package Keras
### Une interface pour Google's tensorflow
### D'après: https://tensorflow.rstudio.com/keras/
### Tel qu'accédé le 19 juillet 2018 à 13h06
##
### R interface to Keras
##
## Keras is a high-level neural networks API developed with a focus on
## enabling fast experimentation. Being able to go from idea to result
## with the least possible delay is key to doing good research. Keras
## has the following key features:
##
##   Allows the same code to run on CPU or on GPU, seamlessly.
##
##   User-friendly API which makes it easy to quickly prototype deep
##   learning models.
##
##   Built-in support for convolutional networks (for computer
##   vision), recurrent networks (for sequence processing), and any
##   combination of both.
##
##   Supports arbitrary network architectures: multi-input or
##   multi-output models, layer sharing, model sharing, etc. This
##   means that Keras is appropriate for building essentially any deep
##   learning model, from a memory network to a neural Turing machine.
##
##   Is capable of running on top of multiple back-ends including
##   TensorFlow, CNTK, or Theano.
##
## For additional details on why you might consider using Keras for
## your deep learning projects, see the Why Use Keras? article.
##
## This website provides documentation for the R interface to
## Keras. See the main Keras website at https://keras.io for additional
## information on the project.
##
### Getting Started
##
## Installation
##
## First, install the keras R package from CRAN as follows:
##
install.packages("keras")
##
## The Keras R interface uses the TensorFlow backend engine by
## default. To install both the core Keras library as well as the
## TensorFlow backend use the install_keras() function:
##
library(keras)
library(pixmap)
## install_keras()
##
## This will provide you with default CPU-based installations of Keras
## and TensorFlow. If you want a more customized installation, e.g. if
## you want to take advantage of NVIDIA GPUs, see the documentation
## for install_keras().
##
### Learning Keras
##
## Below we walk through a simple example of using Keras to recognize
## handwritten digits from the MNIST dataset. After getting familiar
## with the basics, check out the tutorials and additional learning
## resources available on this website.
##
## The Deep Learning with R book by François Chollet (the creator of
## Keras) provides a more comprehensive introduction to both Keras and
## the concepts and practice of deep learning.
##
## You may also find it convenient to download the Deep Learning with
## Keras cheat sheet, a quick high-level reference to all of the
## capabilities of Keras.
##
source("../../R/AuxCode.R")
##
if(FALSE) {
    mnist <- dataset_mnist()
    x_train <- mnist$train$x
    y_train <- mnist$train$y
    x_test <- mnist$test$x
    y_test <- mnist$test$y
    save(mnist,x_train,y_train,x_test,y_test,file="../../Data/mnist.rda")
} else
    load(file="../../Data/mnist.rda")
##
{
    X11(width=6.4,height=4.8)
    plot_tile2(x=mnist$train$x,y=mnist$train$y,n=16L,m=12L)
}
##
### Explications pour Keras::array_reshape()
## The x data is a 3-d array (images,width,height) of grayscale values.
## To prepare the data for training we convert the 3-d arrays into matrices by reshaping width and height
## into a single dimension (28x28 images are flattened into length 784 vectors). Then, we convert the
## grayscale values from integers ranging between 0 to 255 into floating point values ranging between 0 and 1:
##
## str(x_train)
x_train <- array_reshape(x_train, c(nrow(x_train), 784))
## str(x_train)
x_test <- array_reshape(x_test, c(nrow(x_test), 784))
# rescale
x_train <- x_train / 255
x_test <- x_test / 255
##
### Je crois que c'est la manière standard de stocker les valeurs en python. Ce n'est pas le cas en R car nous
### sommes souvent face à des tableaux de données où les éléments des lignes ne sont pas de types hétérogène.
### Il n'est pas rare, par exemple que des colonnes soient de type numérique alors que d'autres sont des facteurs
### (variables qualitatives multi-classes). Dans ce cas, la sémantique 'column-major' est plus pratique.
##
### Codage multiclasse passe en codage binaire sur plusieurs colonnes:
y_train <- to_categorical(y_train, 10)
y_test <- to_categorical(y_test, 10)
##
hidden <- data.frame(node=c(128,65,32),dropout=c(0.3,0.2,0.1))
##
### On définie la structure du modèle comme suit:
model <- keras_model_sequential()  ## Modèle séquentiel (vierge)
##
model %>%
    layer_dense(units = 256, activation = 'relu', input_shape = c(784)) %>% 
    layer_dropout(rate = 0.4)
##
for(i in 1L:nrow(hidden))
    model %>%
        layer_dense(units = hidden[i,"node"], activation = 'relu') %>%
        layer_dropout(rate = hidden[i,"dropout"])
##
model %>%
    layer_dense(units = 10, activation = 'softmax')
##
model
## summary(model)    ## Ça revient au même...
##
### Le modèle une fois défini, doit être compiler (pour plus de rapidité, évidemment)
### Next, compile the model with appropriate loss function, optimizer, and metrics:
##
model %>% compile(
    loss = 'categorical_crossentropy',
    optimizer = optimizer_rmsprop(),
    metrics = c('accuracy')
)
##
### Il faut ensuite ajuster:
### Use the fit() function to train the model for 30 epochs using batches of 128 images:
history <- model %>%
    fit(x_train,
        y_train,
        epochs = 30,
        batch_size = 128,
        validation_split = 0.2)
##
### Affiche l'évolution des valeurs de la fonction objective en fonction des époques du
### processus d'ajustement.
plot(history)
##
### Pour évaluer la performance du modèle:
model %>% evaluate(x_test, y_test)
##
### Pour faire les prédictions
y_pred <- model %>% predict_classes(x_test)
##
## plot_tile(x=mnist$test$x,y_pred=y_pred,y=mnist$test$y,n=16L,m=12L,from=1L)
## plot_tile(x=mnist$test$x,y_pred=y_pred,y=mnist$test$y,n=16L,m=12L,from=1L*(16L*12L))
## plot_tile(x=mnist$test$x,y_pred=y_pred,y=mnist$test$y,n=16L,m=12L,from=2L*(16L*12L))
##
for(from in seq(1L,length(y_pred),16L*12L)) {
    plot_tile2(x=mnist$test$x,y_pred=y_pred,y=mnist$test$y,n=16L,m=12L,from=from)
    Sys.sleep(1/4)
}
##
### Récapitulation:
##
## On produite un modèle séquentiel par empilage de couches. Ces dernières sont autant
## d'opérateurs sur des tenseurs. Les fonctions modificatrices sont intercallées entre
## les couches:
seqModDense <- function(shape,dropout,activation,l1,l2,loss,optimizer,metrics) {
    model <- keras_model_sequential()
    model %>%
        layer_dense(
            units = shape[2L],
            activation = activation[1L],
            input_shape = shape[1L],
            kernel_regularizer = regularizer_l1_l2(l1 = l1[1L], l2 = l2[1L])
        ) %>%
        layer_dropout(rate = dropout[1L])
    if(length(shape)>3L)
        for(i in 3L:(length(shape)-1L)) {
            model %>%
                layer_dense(
                    units = shape[i],
                    activation = activation[i-1L],
                    kernel_regularizer = regularizer_l1_l2(l1 = l1[i-1L], l2 = l2[i-1L])
                ) %>%
                layer_dropout(rate = dropout[i-1L])
        }
    model %>%
        layer_dense(units = shape[length(shape)], activation = activation[length(shape)])
    model %>%
        compile(
            loss = loss,
            optimizer = optimizer,
            metrics = metrics
        )
    return(model)
}
##
model <- seqModDense(shape=c(784,20,20,20,10),dropout=c(0.2,0.5,0,0.3),activation=c('relu','relu','relu','relu','softmax'),
                     l1=c(0.001,0.002,0.02,0.01),l2=c(0.001,0,0.002,0.005),loss='categorical_crossentropy',optimizer=optimizer_rmsprop(),
                     metrics=c('accuracy'))
## summary(model)
## unlink(checkpoint_dir, recursive = TRUE)
checkpoint_dir <- "checkpoints"
dir.create(checkpoint_dir, showWarnings = FALSE)
filepath <- file.path(checkpoint_dir, "weights.{epoch:02d}-{val_loss:.2f}.hdf5")
##
## Création d'une fonction de checkpoint callback
## 
cp_callback <- callback_model_checkpoint(
    filepath = filepath,
    save_weights_only = TRUE,
    verbose = 1
)
## period = 5   ## Sauvegarde par intervalle (ici de 5)
## save_best_only = TRUE ## Pour ne sauvegarder que quand un amélioration est trouvée
##
### Fonction de stoppage précoce:
early_stop <- callback_early_stopping(monitor = "val_loss", patience = 5)
##
##
history <- model %>%
    fit(
        x_train,
        y_train,
        epochs = 50,
        validation_data = list(x_test, y_test),
        batch_size = 128,
        callbacks = list(cp_callback,early_stop),
        verbose = 1
    )
## validation_split = 0.2  ## Pour la validation croisée
plot(history)
##
## list.files(checkpoint_dir)
fresh_model <- seqModDense(shape=c(784,20,20,20,10),dropout=c(0.2,0.5,0,0.3),activation=c('relu','relu','relu','relu','softmax'),
                           l1=c(0.001,0.002,0.002,0.01),l2=c(0.01,0,0.02,0.05),loss='categorical_crossentropy',
                           optimizer=optimizer_rmsprop(),metrics=c('accuracy'))
fresh_model %>%
    load_model_weights_hdf5(file.path(checkpoint_dir, list.files("./checkpoints")[24L]))
##
{
    score <- fresh_model %>% evaluate(x_test, y_test)
    cat('Test loss:', score$loss, '\n')
    cat('Test accuracy:', score$acc, '\n')
}
##
##
### Exemples classiques:
data(iris)
##
iris.Y <- diag(3L)[iris[,5L],]
colnames(iris.Y) <- levels(iris[,5L])
iris.X <- scale(iris[,-5L])
##
model.iris <- keras_model_sequential()
model.iris %>%
    layer_dense(units = 10,
                input_shape = c(4),
                kernel_regularizer = regularizer_l1_l2(l1 = 0.001, l2 = 0.001)) %>%
    layer_activation('relu') %>%
    layer_dense(units = 3) %>%
    layer_activation('softmax')
model.iris %>%
    compile(
        optimizer = 'rmsprop',
        loss = 'categorical_crossentropy',
        metrics = c('accuracy')
    )
##
early_stop <- callback_early_stopping(monitor = "val_loss", patience = 5)
##
history.iris <-
    model.iris %>%
    fit(
        iris.X,
        iris.Y,
        epochs = 150,
        batch_size = 25,
        validation_split = 0.1,
        callbacks = list(early_stop),
        verbose = 1
    )
##
plot(history.iris)
##
save_model_hdf5(model.iris,"iris.hdf5")
## Liste avec les coefficients ([[impaires]]) + bias
iris.weights <- get_weights(model.iris)
## set_weights(object, weights)
##
data(state)
resp <- which(colnames(state.x77)=="Murder")
x77.Y <- state.x77[,resp]
x77.X <- t(t(state.x77[,-resp])/colSums(state.x77[,-resp]^2)^0.5)
## colSums(x77.X^2)
##
model.x77 <- keras_model_sequential()
model.x77 %>%
    layer_dense(units = 10,
                input_shape = c(7),
                kernel_regularizer = regularizer_l1_l2(l1 = 0.001, l2 = 0.001)) %>%
    layer_activation('relu') %>%
    layer_dense(units = 1)
model.x77 %>%
    compile(
        optimizer = 'rmsprop',
        loss = 'mse',
        metrics = c('mean_absolute_error')
    )
##
early_stop <- callback_early_stopping(monitor = "val_loss", patience = 5)
##
history.x77 <-
    model.x77 %>%
    fit(
        x77.X,
        x77.Y,
        epochs = 400,
        batch_size = 25,
        validation_split = 0.1,
        callbacks = list(early_stop),
        verbose = 1
    )
##
plot(history.x77)
##
save_model_hdf5(model.iris,"x77.hdf5")
## Liste avec les coefficients ([[impaires]]) + bias
x77.weights <- get_weights(model.x77)
##
{
    score <- model.x77 %>% evaluate(x77.X, x77.Y)
    cat('Training loss:', score$loss, '\n')
    cat('Training mse:', score$mean_absolute_error, '\n')
}
##
### C'est là sans doute le mieux que l'on puisse espérer.
##
### J'avais fais un didacticiel sur les réseaux récursifs (LSTM) et je
### voulais appliquer cet méthode aux séries temporelles de météo (ENSO,
### température IFL) et d'apports en eau mais je dois d'abord faire
### davantage de lecture car je ne comprends pas encore assez ce concept.
##
## list.files("../../Data")
load(file="../../Data/WaterPeriodicity-dat.rda")
source("../../Data/tsmts.R")
##
nona <- which(!apply(is.na(dat$model_data[,c("Namakan_Supply","TempIFL")]),1L,any))
mtseries <- tsmts(x=dat$model_data[nona,-1L],timestamp=dat$model_data[nona,1L])
rm(dat,nona)
##
width <- 360
resp <- mtseries[(width+1L):nrow(mtseries),"ENSO"]
expl <- atime(resp,tref=tref)
### Rendu ici...



