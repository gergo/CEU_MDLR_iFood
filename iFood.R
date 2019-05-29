# devtools::install_github("rstudio/reticulate")
# devtools::install_github("rstudio/tensorflow")
# devtools::install_github("rstudio/keras")
# tensorflow::install_tensorflow()
# install.packages('data.table')
# install.packages('magick')
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

# not used in R
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

#################################################
# Simple model replaced with a more complex model
#################################################
# image_size <- c(150, 150)
# batch_size <- 50
# 
# train_generator <- flow_images_from_directory(
#   file.path("train_set"), # Target directory  
#   train_datagen,             # Data generator
#   target_size = image_size,  # Resizes all images to 150 × 150
#   batch_size = batch_size,
#   class_mode = "categorical"       # binary_crossentropy loss for binary labels
# )
# 
# validation_generator <- flow_images_from_directory(
#   file.path("val_set"),   
#   validation_datagen,
#   target_size = image_size,
#   batch_size = batch_size,
#   class_mode = "categorical"
# )
# 
# simple_model <- keras_model_sequential() 
# simple_model %>% 
#   layer_conv_2d(filters = 32,
#                 kernel_size = c(3, 3), 
#                 activation = 'relu',
#                 input_shape = c(150, 150, 3)) %>%
#   layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
#   layer_conv_2d(filters = 16,
#                 kernel_size = c(3, 3), 
#                 activation = 'relu') %>%
#   layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
#   layer_conv_2d(filters = 16,
#                 kernel_size = c(3, 3), 
#                 activation = 'relu') %>%
#   layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
#   layer_dropout(rate = 0.25) %>% 
#   layer_flatten() %>% 
#   layer_dense(units = 36, activation = 'relu') %>% 
#   layer_dense(units = 251, activation = "sigmoid")
# 
# simple_model %>% compile(
#   loss = "categorical_crossentropy",
#   optimizer = optimizer_rmsprop(lr = 2e-5),
#   metrics = c("accuracy")
# )
# 
# early_stopping <- callback_early_stopping(monitor = 'val_loss', patience = 3)
# 
# history <- simple_model %>% fit_generator(
#   train_generator,
#   steps_per_epoch = 2000 / batch_size,
#   epochs = 50, # TODO increase when building the final model
#   validation_data = validation_generator,
#   validation_steps = 50
# )
# 
# test_generator <- flow_images_from_directory(
#   file.path("test_set"),
#   test_datagen,
#   target_size = image_size,
#   batch_size = 1,
#   class_mode = NULL
# )
# 
# predictions <- predict_generator(simple_model, test_generator, steps = 20, verbose = 1)
# predictions
# 
# predictions1 <- predictions
# predictions1
# # first <- cbind(max.col(predictions1, 'first'))
# first <- apply(predictions1,1,which.max)
# 
# # replace max probability category value with -Infinity
# # so in the next iteration it will be ignored
# predictions2 <- t(apply(predictions1, 1, function(x) replace(x, x== max(x), -Inf)))
# second <- apply(predictions2,1,which.max)
# 
# predictions3 <- t(apply(predictions2, 1, function(x) replace(x, x== max(x), -Inf)))
# third <- apply(predictions3,1,which.max)
# 
# # categories are indexed from 0, column names are from 1 so subtract 1
# res_df <- tibble(filenames = test_generator$filenames,
#                  cat_1 = first-1,
#                  cat_2 = second-1,
#                  cat_3 = third-1)
# 
# res <- res_df %>%
#   transform(img_name=str_replace(filenames,"unknown/","")) %>%
#   transform(label = str_c(as.character(cat_1),
#                           as.character(cat_2),
#                           as.character(cat_3),
#                           sep = " ")) %>%
#   select(img_name, label)
# 
# write.csv(res, file = "to_submit.csv", row.names = FALSE)

###################################
# Imagenet pre-trained model
###################################
## Transfer learning: use pre-trained models as base
model_imagenet <- application_mobilenet(weights = "imagenet")

# 224: to conform with pre-trained network's inputs
image_net_dimensions <- c(224, 224)
imgnet_img <- image_load(example_image_path, target_size = image_net_dimensions)  
x <- image_to_array(imgnet_img)

# ensure we have a 4d tensor with single element in the batch dimension,
# the preprocess the input for prediction using mobilenet
x <- array_reshape(x, c(1, dim(x)))
x <- mobilenet_preprocess_input(x)

# make predictions then decode and print them
preds <- model_imagenet %>% predict(x)
mobilenet_decode_predictions(preds, top = 3)[[1]]

# train_datagen = image_data_generator(
#   rescale = 1/255,
#   rotation_range = 40,
#   width_shift_range = 0.2,
#   height_shift_range = 0.2,
#   shear_range = 0.2,
#   zoom_range = 0.2,
#   horizontal_flip = TRUE,
#   fill_mode = "nearest"
# )

train_datagen <- image_data_generator(rescale = 1/255)  
validation_datagen <- image_data_generator(rescale = 1/255)  
test_datagen <- image_data_generator(rescale = 1/255)  

image_size <- c(128, 128)
batch_size <- 100  # for speed up

train_generator <- flow_images_from_directory(
  file.path("train_set"),  
  train_datagen,              # Data generator
  target_size = image_size,  # Resizes all images 
  batch_size = batch_size,
  class_mode = "categorical"
)

validation_generator <- flow_images_from_directory(
  file.path("val_set"),  
  validation_datagen,
  target_size = image_size,
  batch_size = batch_size,
  class_mode = "categorical"
)

test_generator <- flow_images_from_directory(
  file.path("test_set"),
  test_datagen,
  target_size = image_size,
  batch_size = batch_size,
  class_mode = "categorical"
)

# create the base pre-trained model
base_model <- application_mobilenet(weights = 'imagenet',
                                    include_top = FALSE,
                                    input_shape = c(image_size, 3))
# freeze all convolutional mobilenet layers
# freeze_weights(base_model)

# add our custom layers
imagenet_predictions <- base_model$output %>% 
  layer_global_average_pooling_2d() %>% 
  layer_dense(units = 1024, activation = 'relu') %>%
  layer_dropout(rate = 0.25) %>%
  layer_dense(units = 1024, activation = 'relu') %>%
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 251, activation = 'sigmoid')

# this is the model we will train
imagenet_model <- keras_model(inputs = base_model$input,
                              outputs = imagenet_predictions)

imagenet_model %>% compile(
  loss = "categorical_crossentropy",
  optimizer = optimizer_rmsprop(lr = 2e-5),
  metrics = c("accuracy")
)

early_stopping <- callback_early_stopping(monitor = 'val_loss',
                                          patience = 3)

# train the model
imagenet_model %>% fit_generator(
  train_generator,
  steps_per_epoch = 2000 / batch_size,
  epochs = 150,
  validation_data = validation_generator,
  validation_steps = 50,
#  initial_epoch=50, # only when continuing a previous build
  callbacks = early_stopping
)

# save trained model
imagenet_model %>% save_model_hdf5("imagenet_model_01.h5")
imagenet_model %>% save_model_weights_hdf5("imagenet_model_weights_01.h5")

# reload model
# imagenet_model <- load_model_hdf5("/Users/szekelygergo/Dropbox/CEU/MasteringDeepLearningInR/CEU_MDLR_iFood/imagenet_model.h5")

predictions <- predict_generator(imagenet_model, test_generator, steps = 20, verbose = 1)
dim(predictions)

predictions1 <- predictions
predictions1
# first <- cbind(max.col(predictions1, 'first'))
first <- apply(predictions1,1,which.max)

# replace max probability category value with -Infinity
# so in the next iteration it will be ignored
predictions2 <- t(apply(predictions1, 1, function(x) replace(x, x== max(x), -Inf)))
second <- apply(predictions2,1,which.max)

predictions3 <- t(apply(predictions2, 1, function(x) replace(x, x== max(x), -Inf)))
third <- apply(predictions3,1,which.max)

# categories are indexed from 0, column names are from 1 so subtract 1
res_df <- tibble(filenames = test_generator$filenames,
                 cat_1 = first-1,
                 cat_2 = second-1,
                 cat_3 = third-1)

res <- res_df %>%
  transform(img_name=str_replace(filenames,"unknown/","")) %>%
  transform(label = str_c(as.character(cat_1),
                          as.character(cat_2),
                          as.character(cat_3),
                          sep = " ")) %>%
  select(img_name, label)

write.csv(res, file = "to_submit.csv", row.names = FALSE)

