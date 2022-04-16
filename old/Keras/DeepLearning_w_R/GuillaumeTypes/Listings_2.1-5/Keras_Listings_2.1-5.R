##
### Cholet et Allaire 2018 - Listings 2.1 - 2.5
##
## rm(list=ls())
library(keras)
if(FALSE) {
    mnist <- dataset_mnist()
    c(c(train_images,
        train_labels),
      c(test_images,
        test_labels)) %<-% mnist
    rm(mnist)
    save(train_images, train_labels, test_images, test_labels,
         file="../../Data/mnist.rda")
} else
    load(file="../../Data/mnist.rda")
##
## str(train_images)
## str(train_labels)
##
### Définition de la topologie du réseau:
## network <- keras_model_sequential() %>% 
##   layer_dense(units = 512, activation = "relu", input_shape = c(28 * 28)) %>% 
##   layer_dense(units = 10, activation = "softmax")
##
network <-
    keras_model_sequential() %>%
    layer_dense(
        units=512L,
        activation="relu",
        input_shape=c(28L*28L)
    ) %>%
    layer_dense(
        units=10L,
        activation="softmax"
    )
##
### Compilation du réseau:
network %>%
    compile(
        optimizer="rmsprop",
        loss="categorical_crossentropy",
        metrics=c("accuracy")
    )
## summary(network)
##
train_images <-
    array_reshape(
        x=train_images / 255,
        dim=c(
            dim(train_images)[1L],
            prod(dim(train_images)[-1L])
        )
    )
## str(train_images)
##
test_images <-
    array_reshape(
        x=test_images / 255,
        dim=c(
            dim(test_images)[1L],
            prod(dim(test_images)[-1L])
        )
    )
## str(test_images)
##
train_labels <-
    to_categorical(train_labels)
test_labels <-
    to_categorical(test_labels)
##
network %>%
    fit(
        x=train_images,
        y=train_labels,
        epoch=5L,
        batch_size=128L
    )
##
metrics <-
    network %>%
    evaluate(
        test_images,
        test_labels
    )
##
metrics
##
preds <-
    network %>%
    predict_classes(
        x=test_images[1L:10L,]
    )
##
par(mar=c(0,0,0,0))
array_reshape(
    255 * (1-train_images[1L,]),
    dim=c(28L, 28L)
) %>%
    as.raster(
        max=255
    ) %>%
    plot
##





