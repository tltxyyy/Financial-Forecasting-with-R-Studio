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
library(randomForest)
library(kknn)
library(lightgbm)


all_cores <- parallel::detectCores(logical = F)
registerDoParallel(cores = all_cores)

rm(list = ls())
#### DATA IMPORT ----
path_root = "."
path_data=file.path(path_root, "data")

#Load files into global environment with name of file as object name
output_links=file.path(path_data, list.files(path=path_data, pattern="*.csv"))

for(file in output_links){
  name <- stringr::str_extract_all(file, "[^/]+") %>%
    pluck(1) %>% last() %>%
    stringr::str_remove(".csv")
  
  suppressWarnings(
    assign(name, readr::read_csv(file, col_types = cols()),envir=.GlobalEnv)
  )
}


etfs_prices %>% glimpse()
stock_prices %>% glimpse()

asset_prices <- rbind(etfs_prices,stock_prices)

#----------------------------------------------------------------------------
## Data Preparation
#----------------------------------------------------------------------------
prep_price = asset_prices %>%
  select(
    ticker,
    ref.date,
    price.open,
    price.high,
    price.low,
    price.close,
    price.adjusted,
    volume
  ) %>%
  set_names(c("id",
              "date",
              "open",
              "high",
              "low",
              "close",
              "price.adj",
              "volume")
  )

lag_period = 9
period = "week"

prep_price_ret = prep_price %>%
  group_by(id) %>%
  # Convert to daily scale
  timetk::summarize_by_time(
    .date_var   = date,
    .by         = period,
    value       = last(close),
    .type       = "ceiling"
  ) %>%
  # Log returns over n lag period
  mutate(return = log(dplyr::lead((value / dplyr::lag(value, n = lag_period)),n = lag_period)), #% return if invest today and sell n periods later  
         boxcox = dplyr::lead(timetk::box_cox_vec(value / timetk::lag_vec(value, lag = lag_period), lambda = "auto", silent = TRUE), n = lag_period)) %>% #normalize data
  ungroup() %>%
  select(-value)

x <- case_when(
  period == 'day' ~ 1,
  period == 'week' ~ 2,
  period == 'month' ~ 3
)

prep_price_ret <- switch(x,prep_price_ret,
                         prep_price_ret %>%
                           mutate(date = date - 2)) #Adjust to Fridays,
out <- prep_price_ret %>%
  dplyr::left_join(prep_price, by = c('id','date'))

#Data cleaning for sentiment score 
newsArticles$date <- as.Date(newsArticles$publishedDate,
                             format = "%d/%m/%Y")

newsArticles <- newsArticles %>% group_by(sector, date) %>% summarise(average_sentiment=mean(sentiment))
#----------------------------------------------------------------------------
## Industry Mapping
#----------------------------------------------------------------------------

asset_universe <- M6_Universe %>%
  set_names(c('id','class','ticker','name','sector_etftype','industry_etfsubtype','sector')) %>%
  select(c('ticker','sector'))

out <- left_join(out,asset_universe,by = c("id" = "ticker"))

#### DATA ENGINEERING ----

asset_features <- function(data, n = 10, na.rm = FALSE, timebased.feat = TRUE){
  
  # Function to add technical indicators
  In<-function(data, p = n){
    # replace NA with 0
    data <- data %>%
      mutate(open = na.locf0(open),
             high = na.locf0(high),
             low = na.locf0(low),
             close = na.locf0(close),
             price.adj = na.locf0(price.adj),
             volume = na.locf0(volume),
             med = na.locf0(med))
    
    # calculate input matrix
    adx  <- ADX(HLC(data), n = p) #Directional Movement Index - percentage of the true range that is up/down
    ar   <- aroon(data[ ,c('high', 'low')], n = p)[ ,'oscillator'] #identify starting trends - how long it has been since the highest high/lowest low has occurred in the last n periods
    cci  <- CCI(HLC(data), n = p) #identify starting and ending trends - relates the current price and the average of price over n periods
    chv  <- chaikinVolatility(HLC(data), n = p) #measures the rate of change of the security's trading range
    cmo  <- CMO(data[ ,'med'], n = p) #ratio of total movement to the net movement
    macd <- MACD(data[ ,'med'], 12, 26, 9)[ ,'macd'] #finds the rate of change between the fast MA and the slow MA
    rsi  <- RSI(data[ ,'med'], n = p) #ratio of the recent upward price movements to the absolute price movement
    stoh <- stoch(HLC(data),14, 3, 3) #relates the location of each day's close relative to the high/low range over the past n periods
    vol  <- volatility(OHLC(data), n = p, calc = "yang.zhang", N = 96) 
    vwap <- VWAP(Cl(data),data[,c('volume')],p) #volume-weighted moving average price - MA takes into account lagged price 
    tdi  <- TDI(data[ ,'close'], n = p, multiple = 2) #identify starting and ending trends
    kst <- KST(data[ ,'close']) # smooth, summed, rate of change indicator
    cti <- CTI(data[ ,'close'], n = p) #correlation of the price with the ideal trend line
    clv <- CLV(HLC(data)) #relates the day's close to its trading range.
    bbands.HLC <- BBands(HLC(data) ) #compare a security's volatility and price levels over a period of time
    cvol <- chaikinVolatility(data[ ,c('high', 'low')], n = p) #rate of change of the security's trading range
    vhf <- VHF(data[ ,'close'], n = p) #identify starting and ending trends
    snr <- SNR(HLC(data), n = p) #taking the absolute price change over an n-day period and dividing it by the average n-day volatility
    cmf <- CMF(HLC(data), data[ ,'volume']) #total volume over the last n time periods to total volume times the Close Location Value (CLV) over the last n time periods
    
    In   <- cbind(adx, ar, cci, chv, cmo, macd, rsi, stoh, vol, vwap,tdi, kst, cti, clv, bbands.HLC, cvol, vhf, snr, cmf)
    
    
    return(In)
  }
  
  # 1. Addition of basic features
  data_eng_tbl = data %>%
    group_by(id) %>%
    mutate(
      CO    = close - open,              #' CO: Difference of Close and Open (Close−Open)
      HO    = high - open,               #' HO: Difference of High and Open (High−Open)
      LO    = low - open,                #' LO: Difference of Low and Open (Low−Open)
      HL    = high - low,                #' HL: Difference of High and Low (High−Low)
      dH    = c(NA, diff(high)),         #' dH: High of the previous 15min candle (Lag(High))
      dL    = c(NA, diff(low)),          #' dL: Low of the previous 15min candle (Lag(Low))
      dC    = c(NA, diff(close)),        #' dC: Close of the previous 15min candle (Lag(Close))
      med   = (high + close)/2,          #' dC: Close of the previous 15min candle (High + Close)
      HL_2  = (high + low)/2,            #' HL_2: Average of the High and Low (High+Low)/2
      HLC_3 = (high + low + close)/3,    #' HLC_3: Average of the High, Low and Close (High+Low+Close)/3
      Wg    = (high + low + 2 * close)/4,  #' Wg: Weighted Average of the High, Low, Close by 0.25,0.25,0.5 (High+Low+2(Close))/4
      mCap  = volume * price.adj         # market capitalisation
    ) %>%
    ungroup()
  
  # 2. Addition of technical indicators (Refer to the functions.R code)
  # Apply function across all asset ids
  asset_name = unique(data_eng_tbl$id)
  
  ind <- foreach(i = 1:length(asset_name),.packages = "dplyr",.combine = "rbind")%dopar%{
    In(data = data_eng_tbl %>% filter(id == asset_name[i]))}
  
  final_data_eng_tbl <- data_eng_tbl %>%
    cbind(ind)
  
  if(na.rm){final_data_eng_tbl %<>% drop_na()}
  
  # 3. Addition of time based features
  if(timebased.feat){final_data_eng_tbl %<>% tk_augment_timeseries_signature(.date_var = date) %>%
      tk_augment_holiday_signature(.date_var = date,.holiday_pattern = "^$", .locale_set = "all", .exchange_set = "all") %>%
      select(-c(index.num,diff,hour,minute,second,hour12,am.pm,wday.lbl,month.lbl,mday,ends_with(".iso"),ends_with(".xts"))) %>% arrange(id,date)}
  
  return(final_data_eng_tbl)
}


out <- asset_features(out)



#Addition of sentiment score
out <- out %>% left_join(newsArticles, by = c("sector", "date"))

#### TARGET VARIABLE ----

create_target <- function(data, bin = 5, nlabel = c("rank_1","rank_2","rank_3","rank_4","rank_5"), method = "content"){
  data$return = data$return %>% replace(is.na(.), 0)
  target_df <- data %>% group_by(date) %>%
    mutate(Ranking = OneR::bin(return, nbins = bin,
                               labels = nlabel,
                               method = method)) %>%
    ungroup()
  
  return(target_df)
}

asset_tbl <- create_target(out, bin = 5, 
                           nlabel = c("rank_1","rank_2","rank_3","rank_4","rank_5"),
                           method = "content") #assign ranking according to returns (no ranking for last 9 weeks bc returns cannot be calculated)

#### DATA TRAINING ----

asset_dataset <- asset_tbl %>%
  select(-c('return','boxcox')) #remove returns col

#----------------------------------------------------------------------------
## One Hot Encoding
#----------------------------------------------------------------------------
asset_numeric <- recipe(Ranking ~ ., data = asset_dataset) %>%
  update_role(c('date','id'), new_role = "id") %>%
  update_role(all_of('Ranking'), new_role = "outcome") %>%
  step_dummy(all_nominal_predictors(), one_hot = T) %>%
  prep() %>% juice() %>% arrange(id, date)


#----------------------------------------------------------------------------
## Feature Selection
#----------------------------------------------------------------------------

remove_corr <- function(data, target,corr_cut = 0.7){
  x    <- data %>% dplyr::select(-c("id","date",target))
  y_id <- data[,c("id","date",target)]
  
  descCor  <- suppressWarnings(cor(x))
  descCor[is.na(descCor)] = 0.99
  highCor  <- caret::findCorrelation(descCor, cutoff = corr_cut)
  x.f      <- x[ ,-highCor]
  data_out <- cbind(y_id,x.f) %>% as_tibble()
  num = length(data) - length(data_out)
  message(paste('Remove corr:',num,'variables removed'))
  return(data_out)
}

remove_constants <- function(data,target){
  x    <- data %>% dplyr::select(-c("id","date",target))
  y_id <- data[,c("id","date",target)]
  
  x.f <- suppressMessages(mlr::removeConstantFeatures(x, perc=.10, na.ignore = TRUE))
  data_out <- cbind(y_id, x.f) %>% as_tibble()
  num = length(data) - length(data_out)
  message(paste('Remove constant:',num,'variables removed'))
  return(data_out)
}

remove_duplicates <- function(data, target){
  x    <- data %>% dplyr::select(-c("id","date",target))
  y_id <- data[,c("id","date",target)]
  
  x.f <- x[!duplicated(as.list(x))]
  data_out <- cbind(y_id, x.f) %>% as_tibble()
  
  num = length(data) - length(data_out)
  message(paste('Remove duplicates:',num,'variables removed'))
  return(data_out)
}

feature_selection <- function(data, target, corr = 0.7){
  id_col <- data %>% select(c("id", "date", target))
  x <- data %>% select(-c("id", "date", target))
  x_rm <- foreach(i=1:ncol(x),.packages = c("dplyr","zoo","tidyr","purrr"),.combine = cbind) %dopar% {
    x[i] %>% pluck(1) %>% na.locf0() %>% replace_na(0)
  }
  colnames(x_rm) <- colnames(x) 
  data <- cbind(id_col, x_rm)
  
  out <- data %>%
    remove_corr(target = target,corr_cut = corr) %>%
    remove_constants(target = target) %>%
    remove_duplicates(target = target)
  return(out)
}

asset_select <- feature_selection(asset_numeric, target = 'Ranking', corr = 0.7)

save(asset_select, file="engineered_data.RData")
