library(dplyr)
library(bestglm)
library(caret)
library(ggplot2)

path <- "data/processed_data/processed_train.csv"
train <- read.csv(path)

train_df <- train |>
  select(date_missing, year, month, day, hour, minute, 
         second, weekday, week, quarter, day_of_year, week_of_year, weekend, 
         hour_sin, hour_cos, weekday_num, weekday_sin, weekday_cos, month_sin, 
         month_cos, author_freq, has_url, class)

# Use model.matrix to create dummy variables
predictors <- model.matrix(~ . - 1, data = train_df |> select(-class))

# Combine predictors and response
train_bestglm <- data.frame(predictors, data=train_df$class)

train_bestglm <- train_bestglm |>
  select(-weekdayMon, -weekdayTue, -weekdayWed, -weekdayThu, -weekdayFri,
         -weekdaySat, -weekdaySun, -second, -minute, -hour, -week, -quarter,
         -weekend)

# Perform Best Subset Selection
set.seed(123)  # For reproducibility

best_subset <- bestglm(
  Xy = train_bestglm,
  family = binomial,
  IC = "BIC",
  method = "exhaustive"
)

# View the best model
print(best_subset$BestModel)

# View the selected variables
selected_variables <- names(coef(best_subset$BestModel))[-1]  # Exclude intercept
print("Selected Variables:")
print(selected_variables)

