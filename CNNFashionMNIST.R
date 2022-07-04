#Libraries for training model
library(knitr)
library(tensorflow)
library(ggplot2)
library(tidyr)
library(tidyverse)
library(readr)
library(keras)
library(caret)

#loading dataset

Train_Data <- read_csv("fashion-mnist_train.csv",
                       col_types = cols(.default = "i"))
Test_Data <- read_csv("fashion-mnist_test.csv",
                      col_types = cols(.default = "i"))

# Fashion MNIST Image Data is 28*28 pixels
ImgRows <- 28
ImgCols <- 28

# Data Preparation

Test_X <- as.matrix(Test_Data[, 2:dim(Train_Data)[2]])
Test_Y <- as.matrix(Test_Data[, 1])
Train_X <- as.matrix(Train_Data[, 2:dim(Train_Data)[2]])
Train_Y <- as.matrix(Train_Data[, 1])


# Unflattening the data.

dim(Train_X) <- c(nrow(Train_X), ImgRows, ImgCols, 1) 
dim(Test_X) <- c(nrow(Test_X), ImgRows, ImgCols, 1) 


Fashion_Labels<-c( "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                  "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot")


# Function to rotate matrices
rotate <- function(x) t(apply(x, 2, rev))


# Function to plot image from a matrix x
plot_image <- function(x, title = "", title.color = "black") {
  dim(x) <- c(ImgRows, ImgCols)
  image(rotate(rotate(x)), axes = FALSE,
        col = grey(seq(0, 1, length = 256)),
        main = list(title, col = title.color))
}


# Plot images from the training set
par(mfrow=c(4, 4), mar=c(0, 0.2, 1, 0.2))
for (i in 1:16) {
  n_row <- i * 10
  plot_image(Train_X[n_row, , , 1],
            Fashion_Labels[as.numeric(Train_Data[n_row, 1] + 1)])

}



#scalling x

Train_X <- Train_X / 255
Test_X <- Test_X / 255

#applying one hot encoding

Train_Y <- to_categorical(Train_Y, 10)
Test_Y <- to_categorical(Test_Y, 10)


# Hyperparameters Setting
batch_size <- 256
epochs <- 20
num_classes = 10
input_shape <- c(ImgRows, ImgCols, 1)


# Convolutional Nerual Network Model using sgd optimizer........................

model <- keras_model_sequential()
model %>% 
  layer_conv_2d(filter = 32, kernel_size = c(3,3), input_shape = c(28, 28, 1), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_dropout(0.25) %>%
  layer_conv_2d(filter = 64, kernel_size = c(3,3), activation ="relu") %>%
  layer_dropout(0.25) %>%
  layer_conv_2d(filter = 128, kernel_size = c(3,3), activation ="relu") %>%
  layer_dropout(0.4) %>%
  layer_flatten() %>%
  layer_dense(units = 128, activation = "relu") %>% 
  layer_dropout(0.4) %>%
  layer_dense(units = 10, activation = "softmax")

summary(model)



# compile model
model %>% compile(
  loss = loss_categorical_crossentropy,
  optimizer = optimizer_sgd(),
  metrics = c('accuracy')
)

# Fit the model
history = model %>% fit(
  Train_X, Train_Y,
  batch_size = batch_size,
  epochs = epochs,
  verbose = 1,
  validation_data = list(Test_X, Test_Y)
)



#Model evaluation
model %>% evaluate(Test_X, Test_Y, batch_size = 256, verbose = 1)



#applying different learning rate than default

# compile model
model_2 %>% compile(
  loss = loss_categorical_crossentropy,
  optimizer = optimizer_sdg(0.00001),
  metrics = c('accuracy')
)

# fit the model
history_2 = model %>% fit(
  Train_X, Train_Y,
  batch_size = batch_size,
  epochs = epochs,
  verbose = 1,
  validation_data = list(Test_X, Test_Y)
)

#CNN using Adam optimizer.......................................................

#Setting hyperparameters

batch_size <- 256
num_classes <- 10
epochs <- 20



model_adam <- keras_model_sequential()
model_adam %>%
  layer_conv_2d(filters = 32, kernel_size = c(5,5), activation = 'relu',
                input_shape = input_shape) %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_dropout(rate = 0.25) %>%
  layer_conv_2d(filters = 64, kernel_size = c(3,3), activation = 'relu') %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_dropout(rate = 0.25) %>%
  layer_conv_2d(filters = 128, kernel_size = c(3,3), activation = 'relu') %>%
  layer_dropout(rate = 0.4) %>%
  layer_flatten() %>%
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = num_classes, activation = 'softmax')

summary(model_adam)

# compile model
model_adam %>% compile(
  loss = loss_categorical_crossentropy,
  optimizer = optimizer_adam(),
  metrics = c('accuracy')
)

# train and evaluate
history_adam = model_adam %>% fit(
  Train_X, Train_Y,
  batch_size = batch_size,
  epochs = epochs,
  verbose = 1,
  validation_data = list(Test_X, Test_Y)
)


#Model evaluation
model_adam %>% evaluate(Test_X, Test_Y, batch_size = 256, verbose = 1)

#Applying diffferent lr than default

#compile model
model_adam_2 %>% compile(
  loss = loss_categorical_crossentropy,
  optimizer = optimizer_adam(0.00001),
  metrics = c('accuracy')
)

# fit the model
history_adam_2 = model_adam %>% fit(
  Train_X, Train_Y,
  batch_size = batch_size,
  epochs = epochs,
  verbose = 1,
  validation_data = list(Test_X, Test_Y)
)

#compare performance of models with default learning rates

compare <- data.frame(
  SGD_val = history$metrics$val_loss,
  SGD_train = history$metrics$loss,
  adam_val = history_adam$metrics$val_loss,
  adam_train =history_adam$metrics$loss
) %>%
  rownames_to_column() %>%
  mutate(rowname = as.integer(rowname)) %>%
  gather(key = "type", value = "value", -rowname)

#plot the comparison

ggplot(compare, aes(x = rowname, y = value, color = type)) +
  geom_line() +
  xlab("epoch") +
  ylab("loss")




# Visulization the model predictions
for (i in 1:32) {
  n_row <- i * 10
  T_Tensor <- Train_X[n_row, , , 1]
  dim(T_Tensor) <- c(1, ImgRows, ImgCols, 1)
  pred <- model_adam %>% predict(T_Tensor)
  plot_image(Train_X[n_row, , , 1],
             Fashion_Labels[which.max(pred)],
             "green")
}

