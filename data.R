library(keras)

mnist <- dataset_mnist()
mnist$train$x <- (mnist$train$x - 127.5)/127.5
mnist$test$x <- (mnist$test$x - 127.5)/127.5
mnist$train$x <- array_reshape(mnist$train$x, c(60000, 1, 28, 28))
mnist$test$x <- array_reshape(mnist$test$x, c(10000, 1, 28, 28))

num_train <- dim(mnist$train$x)[1]
num_test <- dim(mnist$test$x)[1]