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

# Load dataset here
load("engineered_data.RData") # 34 variables

load("all_models.RData")

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

# recipe -----
recipe_stacks <- recipe(Ranking ~ ., data = data_train) %>%
  update_role(c('date','id'), new_role = "id") %>%
  update_role(all_of('Ranking'), new_role = "outcome")

  # set seed
set.seed(123)

resamples_stacks <- recipe_stacks %>% 
  prep() %>% 
  juice() %>% 
  vfold_cv(v = 5)

stack_control <- control_grid(save_pred = TRUE, save_workflow = TRUE)  #save_workflow = TRUE
stack_metrics <- metric_set(roc_auc)

#---------------------------------------------------------------------------------------------------
## PLEASE READ:
## All model outputs have already been saved below from line 60.
## Hence, there is no need to run the tuning process again.
## Details of how each model is trained can be found in individual collapsible chunks.
#---------------------------------------------------------------------------------------------------

## CatBoost model-------------------------------------------------
# define workflow
cat_wflw_stack <- workflow() %>% 
  add_model(cat_model) %>% 
  add_recipe(recipe_stacks)

# tune model
cat_res <- tune_grid(
  cat_wflw_stack, 
  grid = cat_params,
  metrics = stack_metrics,
  resamples = resamples_stacks,
  control = stack_control)

# save(cat_res, file = "cat_res_5cv.RData")

## LightGBM model-------------------------------------------------
# define workflow
lgbm_wflw_stack <- workflow() %>% 
  add_model(lgbm_model) %>% 
  add_recipe(recipe_stacks)

# tune model
lgbm_res <- tune_grid(
  lgbm_wflw_stack, 
  grid = lgbm_params,
  metrics = stack_metrics,
  resamples = resamples_stacks,
  control = stack_control)

# save(lgbm_res, file = "lgbm_res_5cv.RData")

## Random Forest model-------------------------------------------------
# define workflow
rf_wflow_stack <- workflow() %>%
  add_model(rf_model) %>%
  add_recipe(recipe_stacks)

# tune model
rf_res <- tune_grid(
  rf_wflow_stack,
  resamples = resamples_stacks,
  grid = rf_params, # import from modelling file
  metrics = stack_metrics,
  control = stack_control)

# save(rf_res, file = "rf_res_5cv.RData")

#----------------------------------------------------------------------------
## Stacking Models
#----------------------------------------------------------------------------
  # load all tuning results
load("cat_res_5cv.RData")
load("lgbm_res_5cv.RData")
load("rf_res_5cv.RData")

stack_model <- stacks() %>% 
  add_candidates(cat_res) %>% 
  add_candidates(lgbm_res) %>%
  add_candidates(rf_res) 

stack_weights <- stack_model %>% 
  blend_predictions(non_negative = FALSE, penalty = 0.00001)  # assign weights to each model

final_stack <- stack_weights %>% fit_members() # drops the zero coefficient weights

#----------------------------------------------------------------------------
## Model Prediction
#----------------------------------------------------------------------------

# stack_pred <- predict(final_stack, new_data = data_pred, type = 'prob') %>%
#   bind_cols(data_test) %>% select(contains(c(".pred","id","date")))

stack_pred <- bind_cols(
  predict(final_stack, new_data = data_pred, type = 'prob'),
  data_pred
  ) %>% dplyr::select(contains(c(".pred", "id", "date")))

final_preds <- stack_pred %>% subset(date == "2022-09-16")

#----------------------------------------------------------------------------
## Investment Decision
#----------------------------------------------------------------------------

invest <- foreach(i=1:nrow(final_preds),.packages = c("dplyr","zoo","tidyr","purrr"),.combine = rbind) %dopar% {
  final_preds[i,1] %>% pluck(1) * - 0.0075 + final_preds[i,2] %>% pluck(1) * - 0.0025 + final_preds[i,4] %>% pluck(1) * 0.0025 + final_preds[i,5] %>% pluck(1) * 0.0075  # 0.0025 + 0.0075 = 0.01
}

final_prediction <- final_preds %>% 
  mutate(Decision = invest[,1]) %>% 
  dplyr::select(c('id',contains(".pred"),'Decision')) %>%
  set_names(c('ID','Rank1','Rank2','Rank3','Rank4','Rank5','Decision'))

sum(abs(final_prediction$Decision))

#----------------------------------------------------------------------------
## Saving it as submission
#----------------------------------------------------------------------------

write.csv(final_prediction,"investment_decision.csv", row.names = FALSE)


