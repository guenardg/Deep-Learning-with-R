##
### Example script showing how to use stateful RNNs to model long sequences efficiently
##
library(keras)
##
## since we are using stateful rnn tsteps can be set to 1
tsteps <- 1
batch_size <- 25
epochs <- 25
## number of elements ahead that are used to make the prediction
lahead <- 1
##
## Generates an absolute cosine time series with the amplitude exponentially decreasing
## Arguments:
##   amp: amplitude of the cosine function
##   period: period of the cosine function
##   x0: initial x of the time series
##   xn: final x of the time series
##   step: step of the time series discretization
##   k: exponential rate
gen_cosine_amp <- function(amp = 100, period = 1000, x0 = 0, xn = 50000, step = 1, k = 0.0001) {
    n <- (xn-x0) * step
    cos <- array(data = numeric(n), dim = c(n, 1, 1))
    for (i in 1:length(cos)) {
        idx <- x0 + i * step
        cos[[i, 1, 1]] <- amp * cos(2 * pi * idx / period)
        cos[[i, 1, 1]] <- cos[[i, 1, 1]] * exp(-k * idx)
    }
    cos
}
##
{
    cat('Generating Data...\n')
    cos <- gen_cosine_amp()
    cat('Input shape:', dim(cos), '\n')
    plot(ts(cos))
}
##
expected_output <- array(data = numeric(length(cos)), dim = c(length(cos), 1))
for (i in 1:(length(cos) - lahead))
    expected_output[[i, 1]] <- mean(cos[(i + 1):(i + lahead)])
##
cat('Output shape:', dim(expected_output), '\n')
{function(rng) {
    if(missing(rng)) plot(ts(cos)) else plot(ts(cos),xlim=rng)
    lines(ts(expected_output),xlim=rng,col="red")
}}()## (c(1000,1200))
head(cos,n=12L)
head(expected_output,n=12L)
##
cat('Creating model:\n')
model <- keras_model_sequential()
model %>%
    layer_lstm(
        units = 50L,
        input_shape = c(tsteps, 1L),
        batch_size = batch_size,
        return_sequences = TRUE,
        stateful = TRUE
    ) %>%
    layer_lstm(
        units = 50L,
        return_sequences = FALSE,
        stateful = TRUE) %>%
    layer_dense(units = 1)
##
model %>%
    compile(
        loss = 'mse',
        optimizer = 'rmsprop'
    )
##
{
    cat('Training\n')
    for (i in 1:epochs) {
        model %>%
            fit(
                cos,
                expected_output,
                batch_size = batch_size,
                epochs = 1,
                verbose = 1,
                shuffle = FALSE
                )
        model %>%
            reset_states()
    }
}
##
{
    cat('Predicting\n')
    predicted_output <-
        model %>%
        predict(
            cos,
            batch_size = batch_size
        )
}
##
{
    cat('Plotting Results\n')
    op <- par(mfrow=c(2,1))
    plot(ts(expected_output), xlab = '')
    title("Expected")
    plot(ts(predicted_output), xlab = '')
    title("Predicted")
    par(op)
}
##
plot(ts(expected_output), xlab = '')
lines(ts(predicted_output), col="red")
##
### On pourra aussi utiliser une étape de convnet sur un grand jeu de données précédentes.
