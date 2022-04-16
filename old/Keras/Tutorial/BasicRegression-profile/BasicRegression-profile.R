##
### Tutorial: Basic Regression
##
## In a regression problem, we aim to predict the output of a
## continuous value, like a price or a probability. Contrast this with
## a classification problem, where we aim to predict a discrete label
## (for example, where a picture contains an apple or an orange).
##
## This notebook builds a model to predict the median price of homes
## in a Boston suburb during the mid-1970s. To do this, we’ll provide
## the model with some data points about the suburb, such as the crime
## rate and the local property tax rate. 
##
library(keras)
##
### The Boston Housing Prices dataset
##
## The Boston Housing Prices dataset is accessible directly from keras.
if(FALSE) {
    boston_housing <- dataset_boston_housing()
    save(boston_housing,file="../../Data/boston_housing.rda")
} else
    load(file="../../Data/boston_housing.rda")
##
c(train_data, train_response) %<-% boston_housing$train
c(test_data, test_response) %<-% boston_housing$test
##
## This dataset is much smaller than the others we’ve worked with so
## far: it has 506 total examples that are split between 404 training
## examples and 102 test examples:
paste0("Training entries: ", length(train_data), ", response: ", length(train_response))
##
## The dataset contains 13 different features:
##    Per capita crime rate.
##    The proportion of residential land zoned for lots over 25,000 square feet.
##    The proportion of non-retail business acres per town.
##    Charles River dummy variable (= 1 if tract bounds river; 0 otherwise).
##    Nitric oxides concentration (parts per 10 million).
##    The average number of rooms per dwelling.
##    The proportion of owner-occupied units built before 1940.
##    Weighted distances to five Boston employment centers.
##    Index of accessibility to radial highways.
##    Full-value property-tax rate per $10,000.
##    Pupil-teacher ratio by town.
##    1000 * (Bk - 0.63) ** 2 where Bk is the proportion of Black people by town.
##    Percentage lower status of the population.
##
## Each one of these input data features is stored using a different
## scale. Some features are represented by a proportion between 0 and
## 1, other features are ranges between 1 and 12, some are ranges
## between 0 and 100, and so on.
##
colnames(train_data) <-
    colnames(test_data) <-
    c('CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
      'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT')
##
## mean(rep(1,nrow(train_data))^2)
##
### Normalize features
##
std <- sqrt(colMeans(train_data^2))
## colMeans(scale(train_data,center=FALSE,scale=sqrt(colMeans(train_data^2)))^2)
train_data <- scale(train_data,center=FALSE,scale=sqrt(colMeans(train_data^2)))
## colMeans(train_data^2)
test_data <- scale(test_data,center=FALSE,scale=attr(train_data,"scaled:scale"))
## colMeans(test_data^2)
##
train_data[1L,] ## Display sample features, notice they are now on the same scale
test_data[1L,]  ## Idem for the test data
##
## Let’s add column names for better data inspection.
## 
### Response variable
##
## The response variables are the house prices in thousands of dollars. (You may
## notice the mid-1970s prices.)
##
train_response[1L:10L] ## Display first 10 entries
##
### Create the model
##
## Let’s build our model. Here, we’ll use a sequential model with two
## densely connected hidden layers, and an output layer that returns a
## single, continuous value. The model building steps are wrapped in a
## function, build_model, since we’ll create a second model, later on.
##
get_layer_stack <- function(input_shape,output_shape,output_link,hidden,activation,dropout,l1,l2) {
    out <- keras_model_sequential()
    out %>%
        layer_dense(units = hidden[1L],
                    activation = activation[1L],
                    input_shape = input_shape,
                    kernel_regularizer = regularizer_l1_l2(l1=l1[1L],l2=l2[1L])) %>%
        layer_dropout(rate=dropout[1L])
    if(length(hidden)>1L)
        for(i in 2L:length(hidden)) {
            out %>%
                layer_dense(units = hidden[i],
                            activation = activation[i],
                            kernel_regularizer = regularizer_l1_l2(l1=l1[i],l2=l2[i])) %>%
                layer_dropout(rate=dropout[i])
        }
    if(missing(output_link)) {
        out %>%
            layer_dense(units = output_shape)
    } else {
        out %>%
            layer_dense(units = output_shape,
                        activation=output_link)
    }
    return(out)
}
##
build_model <- function(input_shape,output_shape,output_link,hidden,activation,dropout,l1,l2) {
    model <- get_layer_stack(input_shape,output_shape,output_link,hidden,activation,dropout,l1,l2)
    ##
    model %>%
        compile(
            loss = "mse",
            optimizer = optimizer_rmsprop(),
            metrics = list("mean_absolute_error")
        )
    ##
    return(model)
}
##
### Doesn't work that way: you got to remake the stack each time, the R object is only a reference to the stored model
### You can do as many copy of the reference, it will remain bound to the same object internally.
### Calling separate compile commands won't change anything, as they do not affect weights.
##
stack0 <- get_layer_stack(input_shape=dim(train_data)[2],output_shape=1L,hidden=c(10L,7L,4L),activation=c("relu","relu","relu"),
                          dropout=c(0.1,0.2,0.2),l1=c(0.001,0.001,0.001),l2=c(0.0001,0.0001,0.0001))
stack1 <- stack0
stack0 %>%
    compile(
        loss = "mse",
        optimizer = optimizer_rmsprop(),
        metrics = list("mean_absolute_error")
    )
stack1 %>%
    compile(
        loss = "mse",
        optimizer = optimizer_rmsprop(),
        metrics = list("mean_absolute_error")
    )
stack0
##
### Train the model
##
## The model is trained for 500 epochs, recording training and
## validation accuracy in a keras_training_history object. We also
## show how to use a custom callback, replacing the default training
## output by a single dot per epoch.
##
## Display training progress by printing a single dot for each completed epoch.
print_dot_callback <- callback_lambda(
    on_epoch_end = function(epoch, logs) {
        if (epoch %% 80 == 0) cat("\n")
        cat(".")
    }
)
##
epochs <- 500L
##
## Fit the model and store training stats
history0 <- stack0 %>%
    fit(
        train_data,
        train_response,
        epochs = epochs,
        validation_split = 0.2,
        verbose = 0,
        callbacks = list(print_dot_callback)
    )
##
## Now, we visualize the model’s training progress using the metrics
## stored in the history variable. We want to use this data to
## determine how long to train before the model stops making progress.
##
library(ggplot2)
##
plot(history0, metrics = "mean_absolute_error", smooth = FALSE) +
    coord_cartesian(ylim = c(0, 25))
##
## This graph shows little improvement in the model after about 200
## epochs. Let’s update the fit method to automatically stop training
## when the validation score doesn’t improve. We’ll use a callback
## that tests a training condition for every epoch. If a set amount of
## epochs elapses without showing improvement, it automatically stops
## the training.
##
# The patience parameter is the amount of epochs to check for improvement.
early_stop <- callback_early_stopping(monitor = "val_loss", patience = 20)
## early_stop <- callback_early_stopping(monitor = "loss", patience = 20)  ## Only training loss if not val_ before
##
### Not even that would help...
##
## stack1 %>%
##     compile(
##         loss = "mse",
##         optimizer = optimizer_rmsprop(),
##         metrics = list("mean_absolute_error")
##     )
##
history1 <- stack1 %>%
    fit(
        train_data,
        train_response,
        epochs = epochs,
        validation_split = 0.2,
        verbose = 0,
        callbacks = list(early_stop, print_dot_callback)
    )
plot(history1, metrics = "mean_absolute_error", smooth = FALSE) +
    coord_cartesian(xlim = c(0, 250), ylim = c(0, 25))
##
## The graph shows the average error is about $2,500 dollars. Is this
## good? Well, $2,500 is not an insignificant amount when some of the
## labels are only $15,000.
##
## Let’s see how did the model performs on the test set:
c(loss, mae) %<-% (model %>% evaluate(test_data, test_response, verbose = 0))
##
paste0("Mean absolute error on test set: $", sprintf("%.2f", mae * 1000))
##
### Predict
##
## Finally, predict some housing prices using data in the testing set:
test_predictions <- model %>% predict(test_data)
test_predictions[,1L]
##
### Conclusion
##
## This notebook introduced a few techniques to handle a regression
## problem.
## > Mean Squared Error (MSE) is a common loss function used for
##   regression problems (different than classification problems).
## > Similarly, evaluation metrics used for regression differ from
##   classification. A common regression metric is Mean Absolute Error
##   (MAE).
## > When input data features have values with different ranges, each
##   feature should be scaled independently.
## > If there is not much training data, prefer a small network with
##   few hidden layers to avoid overfitting.
## > Early stopping is a useful technique to prevent overfitting.
##
## loss
## mae
par(mar=c(4.5,4.5,2,2))
plot(y=test_response,x=test_predictions,asp=1,
     xlim=range(test_response,test_predictions),
     ylim=range(test_response,test_predictions),
     pch=21,bg="blue")
points(y=test_response,
       x=predict(lm(y~.,data=data.frame(y=train_response,train_data)),newdata=data.frame(test_data)),
       pch=21,bg="red")
##
mean(abs(test_response-predict(lm(y~.,data=data.frame(y=train_response,train_data)),newdata=data.frame(test_data))))
##
rm(boston_housing,train_data,train_labels,test_data,test_labels,col_means_train,col_stddevs_train)
if(FALSE) {
    load(file="../../Data/boston_housing.rda")
    c(train_data, train_labels) %<-% boston_housing$train
    c(test_data, test_labels) %<-% boston_housing$test
    train_data <- scale(train_data)
    col_means_train <- attr(train_data, "scaled:center") 
    col_stddevs_train <- attr(train_data, "scaled:scale")
    test_data <- scale(test_data, center = col_means_train, scale = col_stddevs_train)
}
##
