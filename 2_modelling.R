# remotes::install_github("curso-r/treesnip@catboost")
# remotes::install_github("tidymodels/stacks", ref = "main")
# devtools::install_github('catboost/catboost', subdir = 'catboost/R-package')

# install.packages("doParallel")
# install.packages("tidymodels")
# install.packages("OneR")
# install.packages("timetk")
# install.packages("caret")
# install.packages("mlr")
# install.packages("randomForest")
# install.packages("kknn")
# install.packages(klaR)
# install.packages(discrim)

library(tidyverse)
library(doParallel)
library(tidymodels)
library(readr)
library(OneR)
library(timetk)
library(caret)
library(mlr)
library(TTR)
library(foreach)
library(magrittr)
library(quantmod)
library(treesnip)
library(stacks)
library(catboost)
library(klaR)
library(discrim)

# Load dataset here
load("engineered_data.RData") # 34 variables

## Preparation for Model Training
# splits----------------------------------------------------------------------------
splits = asset_select %>%
  time_series_split(
    date_var = date,
    assess  = "52 weeks",  # test set
    skip    = "9 weeks",   # skip most recent n period data
    slice = 2,             # returns second slice which excludes the skipped data
    cumulative = TRUE      # take all the rest data as training
  )

data_train = training(splits) %>% mutate(Ranking = droplevels(Ranking))
data_test = testing(splits) %>% mutate(Ranking = droplevels(Ranking))
data_pred = testing(asset_select %>% time_series_split(date_var = date, cumulative = T, assess  = "9 weeks"))

## Recipe -------------------------------------------------
recipe_stacks <- recipe(Ranking ~ ., data = data_train) %>%
  update_role(c('date','id'), new_role = "id") %>%
  update_role(all_of('Ranking'), new_role = "outcome")

resamples_stacks <- recipe_stacks %>% 
  prep() %>% 
  juice() %>%
  time_series_cv(date_var = date,
                 assess = '52 weeks',
                 skip = '9 weeks',
                 cumulative = TRUE)
  

stack_control <- control_grid(save_pred = TRUE, save_workflow = TRUE)  #save_workflow = TRUE
stack_metrics <- metric_set(roc_auc)

# Create a new recipe to deselect time. This is bc date is alr taken into account in cross validation step
recipe_stacks_time <- recipe(Ranking ~ ., data = data_train %>% dplyr::select(-date)) %>%
  update_role(c('id'), new_role = "id") %>%
  update_role(all_of('Ranking'), new_role = "outcome")

#---------------------------------------------------------------------------------------------------
## PLEASE READ:
## All model outputs have already been saved below from line 441.
## Hence, there is no need to run the tuning process again.
## Details of how each model is trained can be found in individual collapsible chunks.
#---------------------------------------------------------------------------------------------------

## CatBoost model-------------------------------------------------
set_dependency("boost_tree", eng = "catboost", "catboost")
set_dependency("boost_tree", eng = "catboost", "treesnip")

cat_model <- boost_tree(
  learn_rate = 0.01,
  tree_depth = tune(),
  min_n = 1,
  mtry = tune(),
  trees = tune(),
  stop_iter = 50
) %>%
  set_engine('catboost') %>%
  set_mode('classification')

cat_wflw_stack <- workflow() %>% 
  add_model(cat_model) %>% 
  add_recipe(recipe_stacks_time)

# define grid
cat_params <- grid_regular(
  trees(),
  tree_depth(range = c(2, 8)),
  mtry(range = c(10, 20)), 
  levels = 3, 
  filter = c(trees > 1)
)

# Tuning Model with Resamples
cat_res <- tune_grid(
  cat_wflw_stack, 
  grid = cat_params,
  metrics = stack_metrics,
  resamples = resamples_stacks,
  control = stack_control)

# evaluation
cat_best_params <- cat_res %>% select_best("roc_auc", maximise = F)

cat_best_model <- cat_model %>% finalize_model(cat_best_params)

cat_best_wflw <- workflow() %>%
  add_recipe(recipe_stacks_time) %>%
  add_model(cat_best_model) %>% 
  fit(data = data_train)

train_prediction_cat <- bind_cols(
  predict(cat_best_wflw, new_data = data_train), 
  predict(cat_best_wflw, new_data = data_train, type = "prob"),
  data_train 
) %>% dplyr::select(Ranking, contains(c(".pred", "id", "date")))

metrics_list <- metric_set(roc_auc,accuracy)

train_score_cat <- train_prediction_cat %>%
  metrics_list(truth=Ranking, c(.pred_rank_1,.pred_rank_2, .pred_rank_3, .pred_rank_4, .pred_rank_5), estimate = .pred_class)

test_prediction_cat <- bind_cols(
  predict(cat_best_wflw, new_data = data_test), 
  predict(cat_best_wflw, new_data = data_test, type = "prob"),
  data_test
) %>% dplyr::select(Ranking, contains(c(".pred", "id", "date")))

test_score_cat <- test_prediction_cat %>%
  metrics_list(truth=Ranking, c(.pred_rank_1,.pred_rank_2, .pred_rank_3, .pred_rank_4, .pred_rank_5), estimate = .pred_class)

#save(cat_res, train_score_cat, test_score_cat, file = "cat_res.RData")#

## LightGBM model-------------------------------------------------
lgbm_model <- boost_tree(learn_rate = 0.01,
                         tree_depth = tune(),
                         min_n = 1,
                         mtry = tune(),
                         trees = 1000,
                         stop_iter = 50) %>% set_engine('lightgbm') %>% set_mode('classification')

lgbm_wflw_stack <- workflow() %>% 
  add_model(lgbm_model) %>% 
  add_recipe(recipe_stacks_time)

# define grid
lgbm_params <- grid_regular(tree_depth(range = c(2, 5)), 
                            mtry(range = c(10, 20)), 
                            levels = 3
)


# Tuning Model with Resamples
lgbm_res <- tune_grid(
  lgbm_wflw_stack, 
  grid = lgbm_params,
  metrics = stack_metrics,
  resamples = resamples_stacks,
  control = stack_control)

# evaluation
lgbm_best_params <- lgbm_res %>% select_best("roc_auc", maximise = F)

lgbm_best_model <- lgbm_model %>% finalize_model(lgbm_best_params)

lgbm_best_wflw <- workflow() %>%
  add_recipe(recipe_stacks_time) %>%
  add_model(lgbm_best_model) %>% 
  fit(data = data_train)

train_prediction_lgbm <- bind_cols(
  predict(lgbm_best_wflw, new_data = data_train), 
  predict(lgbm_best_wflw, new_data = data_train, type = "prob"),
  data_train 
) %>% dplyr::select(Ranking, contains(c(".pred", "id", "date")))

metrics_list <- metric_set(roc_auc,accuracy)

train_score_lgbm <- train_prediction_lgbm %>%
  metrics_list(truth=Ranking, c(.pred_rank_1,.pred_rank_2, .pred_rank_3, .pred_rank_4, .pred_rank_5), estimate = .pred_class)

test_prediction_lgbm <- bind_cols(
  predict(lgbm_best_wflw, new_data = data_test), 
  predict(lgbm_best_wflw, new_data = data_test, type = "prob"),
  data_test
) %>% dplyr::select(Ranking, contains(c(".pred", "id", "date")))

test_score_lgbm <- test_prediction_lgbm %>%
  metrics_list(truth=Ranking, c(.pred_rank_1,.pred_rank_2, .pred_rank_3, .pred_rank_4, .pred_rank_5), estimate = .pred_class)

#save(lgbm_res, train_score_lgbm, test_score_lgbm, file = "lgbm_res.RData")#

## K Nearest Neighbour model -------------------------------------------------

knn_model <- nearest_neighbor(
  neighbors = tune(), 
  weight_func = tune() 
) %>%  
  set_engine("kknn") %>% 
  set_mode("classification")

knn_wflw_stack <- workflow() %>% 
  add_model(knn_model) %>% 
  add_recipe(recipe_stacks_time)

# define grid
knn_params <- grid_regular(
  hardhat::extract_parameter_set_dials(knn_model),  # replace parameter() bc it's "deprecated" according to R
  levels = 6, 
  filter = c(neighbors <15))

## Tuning Model with Resamples
knn_res <- tune_grid(
  knn_wflw_stack, 
  grid = knn_params,
  metrics = stack_metrics,
  resamples = resamples_stacks,
  control = stack_control)

knn_best_params <- knn_res %>% select_best("roc_auc", maximise = F)

knn_best_model <- knn_model %>% finalize_model(knn_best_params)

knn_best_wflw <- workflow() %>%
  add_recipe(recipe_stacks_time) %>%
  add_model(knn_best_model) %>% 
  fit(data = data_train)

train_prediction_knn <- bind_cols(
  predict(knn_best_wflw, new_data = data_train), 
  predict(knn_best_wflw, new_data = data_train, type = "prob"),
  data_train 
) %>% dplyr::select(Ranking, contains(c(".pred", "id", "date")))

metrics_list <- metric_set(roc_auc,accuracy)

train_score_knn <- train_prediction_knn %>%
  metrics_list(truth=Ranking, c(.pred_rank_1,.pred_rank_2, .pred_rank_3, .pred_rank_4, .pred_rank_5), estimate = .pred_class)

test_prediction_knn <- bind_cols(
  predict(knn_best_wflw, new_data = data_test), 
  predict(knn_best_wflw, new_data = data_test, type = "prob"),
  data_test
) %>% dplyr::select(Ranking, contains(c(".pred", "id", "date")))

test_score_knn <- test_prediction_knn %>%
  metrics_list(truth=Ranking, c(.pred_rank_1,.pred_rank_2, .pred_rank_3, .pred_rank_4, .pred_rank_5), estimate = .pred_class)

#save(knn_res, train_score_knn, test_score_knn, file = "knn_res.RData")

## Naive Bayes Regression model-------------------------------------------------
nb_model <- naive_Bayes(
  smoothness = tune(),
  Laplace = tune(),
) %>% set_mode("classification") %>%
  set_engine("klaR")

nb_wflw_stack <- workflow() %>%
  add_model(nb_model) %>%
  add_recipe(recipe_stacks_time)

nb_params <- grid_regular(
  hardhat::extract_parameter_set_dials(nb_model),  # replace parameter() bc it's "deprecated" according to R
  levels = 3
)

# Tuning Model with Resamples: took ard 30mins to run
nb_res <- tune_grid(
  nb_wflw_stack,
  resamples = resamples_stacks,
  grid = nb_params,
  metrics = stack_metrics,
  control = stack_control
)

nb_best_params <- nb_res %>% select_best("roc_auc", maximise = F)

nb_best_model <- nb_model %>% finalize_model(nb_best_params)

nb_best_wflw <- workflow() %>%
  add_recipe(recipe_stacks_time) %>%  # use the one for time series
  add_model(nb_best_model) %>% 
  fit(data = data_train)

train_prediction_nb <- bind_cols(
  predict(nb_best_wflw, new_data = data_train), 
  predict(nb_best_wflw, new_data = data_train, type = "prob"),
  data_train 
) %>% dplyr::select(Ranking, contains(c(".pred", "id", "date")))

metrics_list <- metric_set(roc_auc,accuracy)

train_score_nb <- train_prediction_nb %>% 
  metrics_list(truth=Ranking, c(.pred_rank_1,.pred_rank_2, .pred_rank_3, .pred_rank_4, .pred_rank_5), estimate = .pred_class)

test_prediction_nb <- bind_cols(
  predict(nb_best_wflw, new_data = data_test), 
  predict(nb_best_wflw, new_data = data_test, type = "prob"),
  data_test
) %>% dplyr::select(Ranking, contains(c(".pred", "id", "date")))

test_score_nb <- test_prediction_nb %>%
  metrics_list(truth=Ranking, c(.pred_rank_1,.pred_rank_2, .pred_rank_3, .pred_rank_4, .pred_rank_5), estimate = .pred_class)

# save(nb_res, train_score_nb, test_score_nb, file = "nb_res.RData")


## Multinomial Regression model-------------------------------------------------
log_model <- multinom_reg(penalty = tune(), mixture = tune()) %>% 
  set_engine("glmnet")

log_wflw_stack <- workflow() %>% 
  add_model(log_model) %>% 
  add_recipe(recipe_stacks_time)  # use the one for time series

log_params <- grid_regular(
  penalty(range = c(-5, -1)),
  mixture(range = c(0, 1)),
  levels = 5
)

# tuning took around a few mins to run
log_res <- tune_grid(
  log_wflw_stack,
  resamples = resamples_stacks,
  grid = log_params,
  metrics = stack_metrics,
  control = stack_control
)

# evaluation
log_best_params <- log_res %>% select_best("roc_auc", maximise = F)

log_best_model <- log_model %>% finalize_model(log_best_params)

log_best_wflw <- workflow() %>%
  add_recipe(recipe_stacks_time) %>%
  add_model(log_best_model) %>% 
  fit(data = data_train)

train_prediction_log <- bind_cols(
  predict(log_best_wflw, new_data = data_train), 
  predict(log_best_wflw, new_data = data_train, type = "prob"),
  data_train 
) %>% dplyr::select(Ranking, contains(c(".pred", "id", "date")))

metrics_list <- metric_set(roc_auc,accuracy)

train_score_log <- train_prediction_log %>%
  metrics_list(truth=Ranking, c(.pred_rank_1,.pred_rank_2, .pred_rank_3, .pred_rank_4, .pred_rank_5), estimate = .pred_class)

test_prediction_log <- bind_cols(
  predict(log_best_wflw, new_data = data_test), 
  predict(log_best_wflw, new_data = data_test, type = "prob"),
  data_test
) %>% dplyr::select(Ranking, contains(c(".pred", "id", "date")))

test_score_log <-test_prediction_log %>%
  metrics_list(truth=Ranking, c(.pred_rank_1,.pred_rank_2, .pred_rank_3, .pred_rank_4, .pred_rank_5), estimate = .pred_class)

# save(log_res, train_score_log, test_score_log, file = "log_res.RData")
load("log_res.RData")

## Random Forest model-------------------------------------------------
rf_model <- rand_forest(
  mtry = tune(),
  min_n = tune(),
  trees = tune()
) %>%
  set_engine("ranger") %>%
  set_mode("classification")

rf_wflow_stack <- workflow() %>%
  add_model(rf_model) %>%
  add_recipe(recipe_stacks_time)

# define grid
rf_params <- grid_regular(     
  trees(range = c(5, 35)),       # Tried to reduce number of trees as too many trees will cause model to overfit
  mtry(range = c(2, 10)),        # number of variables randomly sampled as candidates at split
  min_n(range = c(500, 900)),    # Should set higher to prevent overfitting: min number of data points in a node that is required for the node to be split further
  levels = 4
)

## Tuning Model with Resamples: took a few mins to run
rf_res <- tune_grid(
  rf_wflow_stack,
  resamples = resamples_stacks,
  grid = rf_params,
  metrics = stack_metrics,
  control = stack_control)

rf_best_params <- rf_res %>% select_best("roc_auc", maximise = F)

rf_best_model <- rf_model %>% finalize_model(rf_best_params)

rf_best_wflw <- workflow() %>%
  add_recipe(recipe_stacks_time) %>%
  add_model(rf_best_model) %>% 
  fit(data = data_train)

train_prediction_rf <- bind_cols(
  predict(rf_best_wflw, new_data = data_train), 
  predict(rf_best_wflw, new_data = data_train, type = "prob"),
  data_train 
) %>% dplyr::select(Ranking, contains(c(".pred", "id", "date")))

metrics_list <- metric_set(roc_auc,accuracy)

train_score_rf <- train_prediction_rf %>%
  metrics_list(truth=Ranking, c(.pred_rank_1,.pred_rank_2, .pred_rank_3, .pred_rank_4, .pred_rank_5), estimate = .pred_class)

test_prediction_rf <- bind_cols(
  predict(rf_best_wflw, new_data = data_test), 
  predict(rf_best_wflw, new_data = data_test, type = "prob"),
  data_test
) %>% dplyr::select(Ranking, contains(c(".pred", "id", "date")))

test_score_rf <- test_prediction_rf %>%
  metrics_list(truth=Ranking, c(.pred_rank_1,.pred_rank_2, .pred_rank_3, .pred_rank_4, .pred_rank_5), estimate = .pred_class)

# save(rf_res, train_score_rf, test_score_rf, file = "rf_res.RData")



## Load tuning results and scores from all models -----
load("cat_res.RData")       # Catboost
load("lgbm_res.RData")      # LightGBM
load("knn_res.RData")       # KNN
load("nb_res.RData")        # Naive Bayes
load("log_res.RData")       # Multinomial Regression
load("rf_res.RData")        # Random Forest

## Scores compilation: Select the best models with roc_auc
scores <- bind_rows(
  test_score_cat %>% mutate(Model = "Catboost", Data = "Test"),
  train_score_cat %>% mutate(Model = "Catboost", Data = "Train"),
  test_score_lgbm %>% mutate(Model = "LightGBM", Data = "Test"),
  train_score_lgbm %>% mutate(Model = "LightGBM", Data = "Train"),
  test_score_knn %>% mutate(Model = "KNN", Data = "Test"),
  train_score_knn %>% mutate(Model = "KNN", Data = "Train"),
  test_score_log %>% mutate(Model = "Multinomial", Data = "Test"),
  train_score_log %>% mutate(Model = "Multinomial", Data = "Train"),
  test_score_nb %>% mutate(Model = "Naive Bayes", Data = "Test"),
  train_score_nb %>% mutate(Model = "Naive Bayes", Data = "Train"),
  test_score_rf %>% mutate(Model = "Random Forest", Data = "Test"),
  train_score_rf %>% mutate(Model = "Random Forest", Data = "Train"),
) %>% dplyr::select("Model", ".metric", Data, .estimate) %>%
  arrange(Data,.metric, desc(.estimate))
scores

## Save for stacking -----
save(
  cat_model, cat_params,
  lgbm_model, lgbm_params,
  rf_model, rf_params,
  file="all_models.RData")
