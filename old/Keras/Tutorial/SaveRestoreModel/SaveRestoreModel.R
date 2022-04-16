##
### Tutorial: Save and Restore Models
##
## Model progress can be saved after as well as during training. This
## means a model can resume where it left off and avoid long training
## times. Saving also means you can share your model and others can
## recreate your work. When publishing research models and techniques,
## most machine learning practitioners share:
##
##  > code to create the model, and
##  > the trained weights, or parameters, for the model
##
## Sharing this data helps others understand how the model works and
## try it themselves with new data.
##
### Setup
##
## We’ll use the MNIST dataset to train our model to demonstrate
## saving weights. To speed up these demonstration runs, only use the
## first 1000 examples:
##
library(keras)
##
if(FALSE) {
    mnist <- dataset_mnist()
    save(mnist,file="../../Data/mnist.rda")
} else
    load(file="../../Data/mnist.rda")
##
c(train_images, train_labels) %<-% mnist$train
c(test_images, test_labels) %<-% mnist$test
##
train_labels <- train_labels[1:1000]
test_labels <- test_labels[1:1000]
##
train_images <- train_images[1:1000, , ] %>%
    array_reshape(c(1000, 28 * 28))
train_images <- train_images / 255
##
test_images <- test_images[1:1000, , ] %>%
    array_reshape(c(1000, 28 * 28))
test_images <- test_images / 255
##
### Define a model
##
## Let’s build a simple model we’ll use to demonstrate saving and
## loading weights.
##
## Returns a short sequential model
create_model <- function() {
  model <- keras_model_sequential() %>%
    layer_dense(units = 512, activation = "relu", input_shape = 784) %>%
    layer_dropout(0.2) %>%
    layer_dense(units = 10, activation = "softmax")
  model %>% compile(
    optimizer = "adam",
    loss = "sparse_categorical_crossentropy",
    metrics = list("accuracy")
  )
  model
}
##
model <- create_model()
model %>% summary()
##
### Save the entire model
##
## The habitual form of saving a Keras model is saving to the HDF5
## format.
##
## The resulting file contains the weight values, the model’s
## configuration, and even the optimizer’s configuration. This allows
## you to save a model and resume training later — from the exact same
## state — without access to the original code.
##
model <- create_model()
##
model %>%
    fit(train_images, train_labels, epochs = 5)
##
model %>%
    save_model_hdf5("my_model.h5")
##
## If you only wanted to save the weights, you could replace that last
## line by
##
model %>%
    save_model_weights_hdf5("my_model_weights.h5")
##
## Now recreate the model from that file:
new_model <- load_model_hdf5("my_model.h5")
new_model %>% summary()
##
### Save checkpoints during training
##
## It is useful to automatically save checkpoints during and at the
## end of training. This way you can use a trained model without
## having to retrain it, or pick-up training where you left of, in
## case the training process was interrupted.
##
## callback_model_checkpoint is a callback that performs this task.
##
## The callback takes a couple of arguments to configure
## checkpointing. By default, save_weights_only is set to false, which
## means the complete model is being saved - including architecture
## and configuration. You can then restore the model as outlined in
## the previous paragraph.
##
## Now here, let’s focus on just saving and restoring weights. In the
## following code snippet, we are setting save_weights_only to true,
## so we will need the model definition on restore.
##
## The filepath argument can contain named formatting options, for
## example: if filepath is weights.{epoch:02d}-{val_loss:.2f}.hdf5,
## then the model checkpoints will be saved with the epoch number and
## the validation loss in the filename.
##
## The saved model weights again will be in HDF5 format.
##
### Checkpoint callback usage
##
## Train the model and pass it the callback_model_checkpoint:
checkpoint_dir <- "checkpoints"
dir.create(checkpoint_dir, showWarnings = FALSE)
filepath <- file.path(checkpoint_dir, "weights.{epoch:02d}-{val_loss:.2f}.hdf5")
##
## Create checkpoint callback
cp_callback <- callback_model_checkpoint(
    filepath = filepath,
    save_weights_only = TRUE,
    verbose = 1
)
##
model <- create_model()
##
model %>%
    fit(
        train_images,
        train_labels,
        epochs = 10, 
        validation_data = list(test_images, test_labels),
        callbacks = list(cp_callback)  # pass callback to training
    )
##
## Inspect the files that were created:
list.files(checkpoint_dir)
##
## Create a new, untrained model. When restoring a model from only
## weights, you must have a model with the same architecture as the
## original model. Since it’s the same model architecture, we can
## share weights despite that it’s a different instance of the model.
##
## Now rebuild a fresh, untrained model, and evaluate it on the test
## set. An untrained model will perform at chance levels (~10%
## accuracy):
{
    fresh_model <- create_model()
    score <- fresh_model %>% evaluate(test_images, test_labels)
    cat('Test loss:', score$loss, '\n')
    cat('Test accuracy:', score$acc, '\n')
}
##
## list.files("./checkpoints")[10L]
fresh_model %>%
    load_model_weights_hdf5(
        file.path(checkpoint_dir, list.files("./checkpoints")[10L])
    )
##
{
    score <- fresh_model %>% evaluate(test_images, test_labels)
    cat('Test loss:', score$loss, '\n')
    cat('Test accuracy:', score$acc, '\n')
}
##
## To reduce the number of files, you can also save model weights only
## once every n^th epoch. E.g.,
##
checkpoint_dir <- "checkpoints"
unlink(checkpoint_dir, recursive = TRUE)
dir.create(checkpoint_dir)
filepath <- file.path(checkpoint_dir, "weights.{epoch:02d}-{val_loss:.2f}.hdf5")
##
## Create checkpoint callback
cp_callback <- callback_model_checkpoint(
    filepath = filepath,
    save_weights_only = TRUE,
    period = 5,
    verbose = 1
)
##
model <- create_model()
##
model %>% fit(
    train_images,
    train_labels,
    epochs = 10, 
    validation_data = list(test_images, test_labels),
    callbacks = list(cp_callback)  # pass callback to training
)
##
list.files(checkpoint_dir)
##
## Alternatively, you can also decide to save only the best model,
## where best by default is defined as validation loss. See the
## documentation for callback_model_checkpoint for further
## information.
## https://tensorflow.rstudio.com/keras/reference/callback_model_checkpoint.html
##
checkpoint_dir <- "checkpoints"
unlink(checkpoint_dir, recursive = TRUE)
dir.create(checkpoint_dir)
filepath <- file.path(checkpoint_dir, "weights.{epoch:02d}-{val_loss:.2f}.hdf5")
##
## Create checkpoint callback
cp_callback <- callback_model_checkpoint(
    filepath = filepath,
    save_weights_only = TRUE,
    save_best_only = TRUE,
    verbose = 1
)
##
model <- create_model()
##
model %>% fit(
    train_images,
    train_labels,
    epochs = 10, 
    validation_data = list(test_images, test_labels),
    callbacks = list(cp_callback)  # pass callback to training
)
##
list.files(checkpoint_dir)
##
rm(mnist,test_images,test_labels,train_images,train_labels)
if(FALSE) {
    load(file="../../Data/mnist.rda")
    c(train_images, train_labels) %<-% mnist$train
    c(test_images, test_labels) %<-% mnist$test
    train_labels <- train_labels[1:1000]
    test_labels <- test_labels[1:1000]
    train_images <- train_images[1:1000, , ] %>%
        array_reshape(c(1000, 28 * 28))
    train_images <- train_images / 255
    test_images <- test_images[1:1000, , ] %>%
        array_reshape(c(1000, 28 * 28))
    test_images <- test_images / 255
}
##
