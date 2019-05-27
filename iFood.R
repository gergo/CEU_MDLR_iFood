# devtools::install_github("rstudio/reticulate")
# devtools::install_github("rstudio/tensorflow")
# devtools::install_github("rstudio/keras")
# tensorflow::install_tensorflow()

library(data.table)  
library(tidyverse)
library(keras)
library(magick)
library(grid)

rm(list = ls())
default_seed <- 1337

# load data
setwd("/Users/szekelygergo/kaggle/ifood-2019-fgvc6")
class_list <- read.table("class_list.txt", sep=" ", header=F) %>%
  rename(id = V1, category = V2)

# not used
# sample_submission <- read.table("ifood2019_sample_submission.csv", sep=",", header=T)
# train_labels <- read.table("train_labels.csv", sep=",", header=T)
# val_labels <- read.table("val_labels.csv", sep=",", header=T)

example_image_path <- file.path("train_set/63/train_063915.jpg")
image_read(example_image_path)

img <- image_load(example_image_path, target_size = c(150, 150))
x <- image_to_array(img) / 255
grid::grid.raster(x)

# Use data augmentation:
train_datagen = image_data_generator(
  rescale = 1/255,
  rotation_range = 40,
  width_shift_range = 0.2,
  height_shift_range = 0.2,
  shear_range = 0.2,
  zoom_range = 0.2,
  horizontal_flip = TRUE,
  fill_mode = "nearest"
)

validation_datagen <- image_data_generator(rescale = 1/255)  
test_datagen <- image_data_generator(rescale = 1/255)  

# take the previous image as base, multiplication is 
# only to conform with the image generator's rescale parameter
xx <- flow_images_from_data(
  array_reshape(x * 255, c(1, dim(x))),  
  generator = train_datagen
)

augmented_versions <- lapply(1:10, function(ix) generator_next(xx) %>%  {.[1, , , ]})

# see examples by running in console:
grid::grid.raster(augmented_versions[[4]])

image_size <- c(150, 150)
batch_size <- 50

train_generator <- flow_images_from_directory(
  file.path("train_set"), # Target directory  
  train_datagen,             # Data generator
  target_size = image_size,  # Resizes all images to 150 × 150
  batch_size = batch_size,
  class_mode = "categorical"       # binary_crossentropy loss for binary labels
)

validation_generator <- flow_images_from_directory(
  file.path("val_set"),   
  validation_datagen,
  target_size = image_size,
  batch_size = batch_size,
  class_mode = "categorical"
)

simple_model <- keras_model_sequential() 
simple_model %>% 
  layer_conv_2d(filters = 32,
                kernel_size = c(3, 3), 
                activation = 'relu',
                input_shape = c(150, 150, 3)) %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_conv_2d(filters = 16,
                kernel_size = c(3, 3), 
                activation = 'relu') %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_conv_2d(filters = 16,
                kernel_size = c(3, 3), 
                activation = 'relu') %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_dropout(rate = 0.25) %>% 
  layer_flatten() %>% 
  layer_dense(units = 36, activation = 'relu') %>% 
  layer_dense(units = 251, activation = "sigmoid")

simple_model %>% compile(
  loss = "categorical_crossentropy",
  optimizer = optimizer_rmsprop(lr = 2e-5),
  metrics = c("accuracy")
)

# TODO add early stopping callback
history <- simple_model %>% fit_generator(
  train_generator,
  steps_per_epoch = 2000 / batch_size,
  epochs = 1, # TODO increase when building the final model
  validation_data = validation_generator,
  validation_steps = 50
)

test_generator <- flow_images_from_directory(
  file.path("test_set"),
  test_datagen,
  target_size = image_size,
  batch_size = 1,
  class_mode = NULL
)

# TODO categories -1?
predictions <- predict_generator(simple_model, test_generator, steps = 20, verbose = 1)
predictions

predictions1 <- predictions
predictions1
first <- cbind(max.col(predictions1, 'first'))
first <- apply(predictions1,1,which.max)

# replace max probability category value with -Infinity
# so in the next iteration it will be ignored
predictions2 <- t(apply(predictions1, 1, function(x) replace(x, x== max(x), -Inf)))
second <- cbind(max.col(predictions2, 'first'))
second <- apply(predictions2,1,which.max)

predictions3 <- t(apply(predictions2, 1, function(x) replace(x, x== max(x), -Inf)))
third <- cbind(max.col(predictions3, 'first'))
third <- apply(predictions3,1,which.max)

res_df <- tibble(filenames = test_generator$filenames,
                 cat_1 = first,
                 cat_2 = second,
                 cat_3 = third)

res <- res_df %>%
  transform(img_name=str_replace(filenames,"unknown/","")) %>%
  transform(label = str_c(as.character(cat_1),
                          as.character(cat_2),
                          as.character(cat_3),
                          sep = " ")) %>%
  select(img_name, label)

write.csv(res, file = "to_submit.csv", row.names = TRUE)


