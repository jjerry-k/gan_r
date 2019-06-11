# Load Package
library(keras)
library(abind)
k_set_image_data_format('channels_first')

generate_image <- function(){
    noise <- runif(10*latent_size, min = -1, max = 1) %>% matrix(nrow = 10, ncol = latent_size)

    sampled_labels <- 0:9 %>% matrix(ncol = 1)

    # Get a batch to display
    generated_images <- predict(generator, list(noise, sampled_labels))

    img <- NULL

    for(i in 1:10){
    img <- cbind(img, generated_images[i,,,])
    }

    ((img + 1)/2) %>% as.raster() %>%
    plot()
}

latent_size <- 100

source("build.R")

generator <- load_model_hdf5('gen_model.h5')

generate_image()

#plot(noise[4,], col='red', type='l')