##
### Cholet et Allaire 2018 - Listings 2.1 - 2.5
##
## rm(list=ls())
library(keras)
load(file="../../Data/mnist.rda")
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
### Utilisation de l'API fonctionnel
input_tensor <-
    layer_input(
        shape=c(dim(train_images)[2L])
    )
##
output_tensor <-
    input_tensor %>%
    layer_dense(
        units=32L,
        activation="relu"
    ) %>%
    layer_dense(
        units=10L,
        activation="softmax"
    )
##
model <-
    keras_model(
        inputs=input_tensor,
        outputs=output_tensor
    )
##
model %>%
    compile(
        optimizer=optimizer_rmsprop(lr=1e-4),
        loss="mse",
        metrics = c("accuracy")
    )
##
model %>%
    fit(
        train_images,
        train_labels
    )
##
rm(list=ls())
##
### On recommence à neuf à partir d'ici...
library(keras)
if(FALSE) {
    imdb <-
        dataset_imdb(num_words=10000L)
    c(c(train_data,
        train_labels),
      c(test_data,
        test_labels)) %<-%
        imdb
    rm(imdb)
    word_index <- dataset_imdb_word_index()
    save(train_data,train_labels,test_data,
         test_labels,word_index,
         file="../../Data/imdb_10k_words.rda")
} else
    load(file="../../Data/imdb_10k_words.rda")
##
## str(train_data[[1L]])
## train_labels[[1L]]
## max(sapply(train_data, max))
## min(sapply(train_data, min))
##
reverse_word_index <- names(word_index)
names(reverse_word_index) <- word_index
review_decoder <-
    function(review, rev_idx) {
        review %>%
            sapply(
                function(index, rev_idx) {
                    word <-
                        if(index >= 3L)
                            rev_idx[[as.character(index-3L)]]
                    if(!is.null(word)) word else "?"
                }
            ) %>%
            paste(collapse=" ")
    }
##
review_decoder(train_data[[1L]],reverse_word_index)
##
vectorize_sequences <-
    function(sequences, dimension=10000L) {
        results <-
            matrix(0, nrow=length(sequences), ncol=dimension)
        for(i in 1L:length(sequences))
            results[i, sequences[[i]]] <- i
        return(results)
    }
##
x_train <-
    train_data %>%
    vectorize_sequences
##
x_test <-
    test_data %>%
    vectorize_sequences
## str(x_train[1L,])
##
y_train <-
    train_labels %>%
    as.numeric
##
y_test <-
    test_labels %>%
    as.numeric
## head(y_train)
##
model <-
    keras_model_sequential() %>%
    layer_dense(
        units=16L,
        activation="relu",
        input_shape=c(10000L)
    ) %>%
    layer_dense(
        units=16L,
        activation="relu"
    ) %>%
    layer_dense(
        units=1L,
        activation="sigmoid"
    )
##
model %>%
    compile(
        optimizer="rmsprop",
        loss="binary_crossentropy",
        metrics=c("accuracy")
    )
## ou:
model %>%
    compile(
        optimizer=optimizer_rmsprop(lr=1e-3),
        loss="binary_crossentropy",
        metrics = c("accuracy")
    )
## ou encore:
model %>%
    compile(
        optimizer=optimizer_rmsprop(lr=1e-3),
        loss=loss_binary_crossentropy,
        metrics=metric_binary_accuracy
    )
##
val_indices <- 1L:10000L
x_val <-
    x_train[val_indices,]
partial_x_train <-
    x_train[-val_indices,]
y_val <-
    y_train[val_indices]
partial_y_train <-
    y_train[-val_indices]
##
model %>%
    compile(
        optimizer="rmsprop",
        loss="binary_crossentropy",
        metrics=c("accuracy")
    )
##
history <-
    model %>%
    fit(
        partial_x_train,
        partial_y_train,
        epochs=20L,
        batch_size=512L,
        validation_data=list(
            x_val,
            y_val
        )
    )
##

        














