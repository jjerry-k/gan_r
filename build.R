library(keras)
k_set_image_data_format('channels_first')

build_generator <- function(latent_size){

  cnn <- keras_model_sequential()
  
  cnn %>%
    layer_dense(1024, input_shape = latent_size, activation = "relu") %>%
    layer_dense(128*7*7, activation = "relu") %>%
    layer_reshape(c(128, 7, 7)) %>%
    layer_upsampling_2d(size = c(2, 2)) %>%
    layer_conv_2d(
      256, c(5,5), padding = "same", activation = "relu",
      kernel_initializer = "glorot_normal"
    ) %>%
    layer_upsampling_2d(size = c(2, 2)) %>%
    layer_conv_2d(
      128, c(5,5), padding = "same", activation = "tanh",
      kernel_initializer = "glorot_normal"
    ) %>%
    layer_conv_2d(
      1, c(2,2), padding = "same", activation = "tanh",
      kernel_initializer = "glorot_normal"
    )
  

  latent <- layer_input(shape = list(latent_size))
  
  image_class <- layer_input(shape = list(1))
  
  cls <-  image_class %>%
    layer_embedding(
      input_dim = 10, output_dim = latent_size, 
      embeddings_initializer='glorot_normal'
    ) %>%
    layer_flatten()
  
  

  h <- layer_multiply(list(latent, cls))
  
  fake_image <- cnn(h)
  
  keras_model(list(latent, image_class), fake_image)
}

build_discriminator <- function(){
  
  cnn <- keras_model_sequential()
  
  cnn %>%
    layer_conv_2d(
      32, c(3,3), padding = "same", strides = c(2,2),
      input_shape = c(1, 28, 28)
    ) %>%
    layer_activation_leaky_relu() %>%
    layer_dropout(0.3) %>%
    
    layer_conv_2d(64, c(3, 3), padding = "same", strides = c(1,1)) %>%
    layer_activation_leaky_relu() %>%
    layer_dropout(0.3) %>%  
    
    layer_conv_2d(128, c(3, 3), padding = "same", strides = c(2,2)) %>%
    layer_activation_leaky_relu() %>%
    layer_dropout(0.3) %>%  
    
    layer_conv_2d(256, c(3, 3), padding = "same", strides = c(1,1)) %>%
    layer_activation_leaky_relu() %>%
    layer_dropout(0.3) %>%  
    
    layer_flatten()

  image <- layer_input(shape = c(1, 28, 28))
  features <- cnn(image)
  
  fake <- features %>% 
    layer_dense(1, activation = "sigmoid", name = "generation")
  
  aux <- features %>%
    layer_dense(10, activation = "softmax", name = "auxiliary")
  
  keras_model(image, list(fake, aux))
}