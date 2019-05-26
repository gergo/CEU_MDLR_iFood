library(data.table)  

rm(list = ls())
default_seed <- 1337

setwd("/Users/szekelygergo/kaggle/ifood-2019-fgvc6")
data_raw <- fread("online-news-popularity/train.csv")
data_evaluate <- fread("online-news-popularity/test.csv")

# transform target variable to factor
data_raw[, is_popular := factor(is_popular)]

training_ratio <- 0.7
set.seed(default_seed)
train_indices <- createDataPartition(y = data_raw[["is_popular"]],
                                     times = 1,
                                     p = training_ratio,
                                     list = FALSE)
data_train <- data_raw[train_indices, ]
data_test <- data_raw[-train_indices, ]

# remove article_id from training set
data_train[, c("article_id") := NULL]

# train + target variables
y <- "is_popular"
X <- names(data_train)

h2o.init()
data_train <- as.h2o(data_train)
data_test <- as.h2o(data_test)
data_evaluate <- as.h2o(data_evaluate)

################
# BASE MODELS  #
################
gbm_model <- h2o.gbm(
  X, y,
  training_frame = data_train,
  ntrees = 200,
  max_depth = 8,
  learn_rate = 0.1,
  seed = default_seed,
  nfolds = 5,
  keep_cross_validation_predictions = TRUE
)

gbm_model_2 <- h2o.gbm(
  X, y,
  training_frame = data_train,
  ntrees = 250,
  max_depth = 9,
  learn_rate = 0.09,
  seed = default_seed,
  nfolds = 5,
  keep_cross_validation_predictions = TRUE
)

rf_model <- h2o.randomForest(
  x = X, y = y,
  training_frame = data_train,
  seed = default_seed,
  nfolds = 5,
  keep_cross_validation_predictions = TRUE
)

rf_model_2 <- h2o.randomForest(
  x = X, y = y,
  training_frame = data_train,
  ntrees = 800,
  max_depth = 9,
  mtries = 8,
  stopping_metric = 'AUC',
  seed = default_seed,
  nfolds = 5,
  keep_cross_validation_predictions = TRUE
)

cart_model = h2o.gbm(X, y,
                     training_frame = data_train, 
                     ntrees = 1, min_rows = 1,
                     sample_rate = 1, col_sample_rate = 1,
                     max_depth = 5,
                     nfolds = 5,
                     stopping_rounds = 3, stopping_tolerance = 0.01, 
                     stopping_metric = "AUC", 
                     seed = default_seed,
                     keep_cross_validation_predictions = TRUE
)

deeplearning_model_1 <- h2o.deeplearning(
  X, y,
  training_frame = data_train,
  hidden = c(60,30),
  seed = default_seed,
  nfolds = 5,
  keep_cross_validation_predictions = TRUE
)

deeplearning_model_2 <- h2o.deeplearning(
  X, y,
  training_frame = data_train,
  hidden = c(60,60,30),
  epochs=50,
  stopping_rounds=3,
  stopping_metric="misclassification",
  seed = default_seed,
  nfolds = 5,
  keep_cross_validation_predictions = TRUE
)

glm_model <- h2o.glm(
  X, y,
  training_frame = data_train,
  family = "binomial",
  alpha = 1,
  lambda_search = TRUE,
  seed = default_seed,
  nfolds = 5,
  keep_cross_validation_predictions = TRUE
)

glm_model_2 <- h2o.glm(
  X, y,
  training_frame = data_train,
  family = "binomial",
  alpha = 0.5,
  lambda_search = TRUE,
  seed = default_seed,
  nfolds = 5,
  keep_cross_validation_predictions = TRUE
)

xgb_model <- h2o.xgboost(X, y,
                         training_frame = data_train,
                         distribution = "bernoulli",
                         ntrees = 200,
                         max_depth = 9,
                         min_rows = 1,
                         learn_rate = 0.1,
                         sample_rate = 0.7,
                         col_sample_rate = 0.9,
                         nfolds = 5,
                         keep_cross_validation_predictions = TRUE,
                         seed = default_seed
)

xgb_model_2 <- h2o.xgboost(X, y,
                           training_frame = data_train,
                           distribution = "multinomial",
                           ntrees = 300,
                           max_depth = 9,
                           min_rows = 1,
                           learn_rate = 0.05,
                           sample_rate = 0.6,
                           col_sample_rate = 0.9,
                           nfolds = 5,
                           keep_cross_validation_predictions = TRUE,
                           seed = default_seed
)

#############
# Auto ML   #
#############
aml_model <- h2o.automl(X, y,
                        training_frame = data_train,
                        max_models = 20,
                        nfolds = 5,
                        keep_cross_validation_predictions = TRUE,
                        seed = default_seed
)
aml_best <- aml_model@leader

#####################
# Model Performance #
#####################
h2o.auc(h2o.performance(cart_model, newdata = data_test))
h2o.auc(h2o.performance(gbm_model, newdata = data_test))
h2o.auc(h2o.performance(gbm_model_2, newdata = data_test))
h2o.auc(h2o.performance(rf_model, newdata = data_test))
h2o.auc(h2o.performance(rf_model_2, newdata = data_test))
h2o.auc(h2o.performance(deeplearning_model_1, newdata = data_test))
h2o.auc(h2o.performance(deeplearning_model_2, newdata = data_test))
h2o.auc(h2o.performance(glm_model, newdata = data_test))
h2o.auc(h2o.performance(glm_model_2, newdata = data_test))
h2o.auc(h2o.performance(xgb_model, newdata = data_test))
h2o.auc(h2o.performance(xgb_model_2, newdata = data_test))
h2o.auc(h2o.performance(aml_best, newdata = data_test))

############
# ENSEMBLE #
############
ensemble_model_deeplearning <- h2o.stackedEnsemble(
  X, y,
  training_frame = data_train,
  metalearner_algorithm = "deeplearning",
  base_models = list(cart_model,
                     gbm_model,
                     gbm_model_2,
                     rf_model,
                     rf_model_2,
                     deeplearning_model_1,
                     deeplearning_model_2,
                     glm_model,
                     glm_model_2,
                     xgb_model,
                     xgb_model_2),
  keep_levelone_frame = TRUE)

h2o.auc(h2o.performance(ensemble_model_deeplearning, newdata = data_test))


# data_evaluate
results <- as.data.frame(h2o.predict(ensemble_model_deeplearning, newdata = data_evaluate))
data_valid_for_submission <- as.data.frame(data_evaluate)
final <- cbind(data_valid_for_submission, results)
to_submit <- data.frame(article_id = final$article_id, score = final$p1)

write.csv(to_submit, file = "to_submit.csv", row.names = FALSE)

# write aml_best
results_aml <- as.data.frame(h2o.predict(aml_best, newdata = data_evaluate))
data_valid_for_submission_aml <- as.data.frame(data_evaluate)
final_aml <- cbind(data_valid_for_submission_aml, results_aml)
to_submit_aml <- data.frame(article_id = final_aml$article_id, score = final_aml$p1)
write.csv(to_submit_aml, file = "to_submit_aml_best.csv", row.names = FALSE)

h2o.shutdown()
