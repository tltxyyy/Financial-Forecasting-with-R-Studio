# Summary of Project
- Data given is from the M6 Financial Forecasting Competition
- 

# Project Details
## 1. General
Specs: 8 Gb Ram

As shown in the Figure below, the source code is saved into three R files: (1) 1_feature_engineering, (2) 2_modelling and (3) 3_stacking. For each of the R files, the final output is saved for use in the next step.

Workflow Steps | Output
--- | ---
Feature Engineering (1_feature_engineering.R) | “engineered_data.RData”: the cleaned and feature engineered data
Modelling (2_modelling.R)	 | 30“all_models.RData”: the chosen models to be stacked1 
Stacking (3_stacking.R)	 | “investment_decision.csv”: the output csv containing the investment decision. A seed was set for the stacking file at 123 to reproduce the final results. 

To reproduce the stacking results, it is only necessary to run the stacking file.

## 2. Data Preparation
### 2.1 Loading the Data
- The datasets etfs_prices.csv, stock_prices.csv, newsArticles.csv and M6_Universe.csv were loaded.
- These four datasets were then combined into one data frame.

### 2.2 Feature Engineering
- Converted prices to daily scale 
- Added date features - day, week, month
- Calculated basic features related to changes in stock price
- Added market capitalisation
- Added more technical indicators from TTR package
- Addition of time-based features - holidays
- Used Box-cox transformation to normalise the price 

### 2.3 Train-test split
- 80-20 split for training and testing data respectively.
- The last 9 weeks of the dataset were used as prediction dataset.
- Since this is a time series problem, validation was done through time series cross validation in creating the predictive modelling to account for the temporal nature of the dataset. When stacking, 5-fold cross validation was used instead since time series cross validation does not work when stacking.

## 3. Modelling approach
To select the models to be stacked, we first fitted models using CatBoost, LightGBM, KNN, Random Forest, Naïve Bayes Regression and Multinomial Regression. Roc_auc score was used to gauge the performance of the models fitted. 

Eventually, the models CatBoost, LightGBM and Random Forest were selected to be part of the final stack as they have the best performance. The modelling approach of these three methods will be discussed in Sections 4.1 to 4.3. The trained models and tuned parameters were saved in an output folder (RData format) to be conveniently loaded for stacking. The modelling approach for the other models that were dropped from the final stack will be discussed in the Appendix. 

### 3.1 CatBoost 
- We first tuned the parameters of the CatBoost model over a wide range for about 3 hours to get a sense of the range of inputs that generally produced better performance. 
- Upon analysing the performances of the model, we observed that learn_rate=0.01 generally performed the best and hence was set as fixed values for subsequent tunings. 
- To reduce overfitting of trees, only tree_depth was tuned and min_n was set a fixed value to control model complexity.
- For the other parameters, the range of the inputs were reduced based on the results obtained to optimise subsequent tuning.
- We then reran the tuning to find the best parameters for CatBoost.

### 3.2 LightGBM
- We first tuned the parameters of the LightGBM model over a wide range for about 6 hours to get a sense of the range of inputs that generally produced better performance. 
- Upon analysing the performance of the model, we observed that learn_rate=0.01 and trees=1000 generally performed the best and hence they were set as fixed values for subsequent tunings. 
- For the other parameters, the range of the inputs were reduced based on the results obtained to optimise subsequent tuning.
- We then reran the tuning to find the best parameters for LightGBM. 

### 3.3 Random Forest
- We first tuned all the three parameters of the random forest model over a wide range for 4 hours, but the outcome was only producing good results for the train dataset but not for the test dataset, suggesting that the model was overfitted.
- Hence, trial and error with the parameters was to get a sense of what range to input before setting the tuning range again. From there, we found out that having too many trees and a low min_n will cause overfitting of the model. Hence, the range of trees was set between 5 to 35 and min_n between 500 to 900.
- We then tune the model again to find the best parameters.
 

### 3.4 Output Generation Time
Model	| Time 
--- | ---
CatBoost | 1 hour
LightGBM | 1 hour
Random Forest	| 5 minutes

## 4. Final Investment Decision Generation
- Run 3_stacking.R.  Note that the model tuning step does not need to be run again as results can be loaded in line 111 from cat_res_5cv.RData, lgbm_res_5cv.RData and rf_res_5cv.RData. 
- We used the final output model on the prediction set to predict the ranking probabilities for each ranking bucket.
- Submission file will be generated in csv format and filtered to “2022-09-16” to obtain the predicted ranking probabilities for 18th Nov (returns 9 weeks later).
- Using the predicted ranking probabilities, we generated the investment decision by assigning investment weights to each ranking bucket and calculating the investment decision for each stock accordingly. Rank bucket 5 was assigned the highest weight at 0.0075 while Rank bucket 1 was assigned the lowest weight at -0.0075 since if a stock has a high probability of having the lowest returns amongst all stocks, the stock should be shorted, or amount investment should be low. 
- The final submission file is name as investment_decision.csv

## 5. Appendix

This section contains the other modelling approaches which are not selected for the final stacked model.

### 5.1 K Nearest Neighbours & Naïve Bayes
- The grid function uses a parameters object created from a model to tune both parameters of the model.
- The range of the parameters were reduced based on initial models generated. 

### 5.2 Multinomial Regression
- Trial and error with both the parameters were conducted to get a sense of what range to input before setting the tuning range again.
