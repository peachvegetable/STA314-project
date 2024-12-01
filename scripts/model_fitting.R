library(dplyr)
library(caret)
library(randomForest)
library(pROC)

path <- "data/processed_data/processed_train.csv"
train <- read.csv(path)

comment_embedding <- readRDS("data/embeddings/train_content_embeddings.rds")
video_embedding <- readRDS("data/embeddings/train_video_embeddings.rds")

train_df <- train |>
  select(date_missing, year, month, day, hour, minute, 
         second, weekday, week, quarter, day_of_year, week_of_year, weekend, 
         hour_sin, hour_cos, weekday_num, weekday_sin, weekday_cos, month_sin, 
         month_cos, author_freq, has_url, class)

train_df$comment <- comment_embedding
train_df$video_name <- video_embedding

# Convert 'class' to a factor with prefixed labels to make them valid R variable names
train_df$class <- factor(train_df$class, levels = c(0, 1), labels = c("C0", "C1"))

# Define custom summary function for F1-score
f1Summary <- function(data, lev = NULL, model = NULL) {
  if(length(lev) != 2) {
    stop("The summary function requires exactly two levels.")
  }
  
  # Calculate Precision and Recall
  precision <- posPredValue(data$pred, data$obs, positive = "C1")  # 'C1' corresponds to spam
  recall <- sensitivity(data$pred, data$obs, positive = "C1")
  
  # Handle division by zero
  if((precision + recall) == 0){
    F1 <- 0
  } else {
    F1 <- (2 * precision * recall) / (precision + recall)
  }
  
  # Return F1-score
  out <- c(F1 = F1)
  return(out)
}

# Set a seed for reproducibility
set.seed(123)

# Define training control with 10-fold cross-validation
train_control <- trainControl(
  method = "cv",
  number = 10, 
  classProbs = TRUE,
  summaryFunction = f1Summary,
  savePredictions = "final",
  sampling = "smote"
)

# Train Logistic Regression with cross-validation
logistic_model <- train(
  class ~ ., 
  data = train_df,
  method = "glm",
  family = "binomial",
  trControl = train_control,
  metric = "F1"
)

# View model summary
print(logistic_model)
saveRDS(logistic_model, "models/logistic.rds")

# Define a reasonable mtry grid based on the number of predictors
rf_grid <- expand.grid(
  mtry = c(2, 5, 10)
)

rf_grid_expanded <- expand.grid(
  mtry = seq(2, ncol(train_df) - 1, by = 1)
)

# Train Random Forest with cross-validation
set.seed(123)
rf_model <- train(
  class ~ .,
  data = train_df,
  method = "rf",
  trControl = train_control,
  metric = "F1",
  tuneGrid = rf_grid,
  ntree = 1000
)

print(rf_model)
saveRDS(rf_model, "models/rf.rds")

# Define a tuning grid for Lasso (alpha = 1)
tune_grid_lasso <- expand.grid(
  alpha = 1,  # Lasso
  lambda = 10^seq(-4, 1, length = 100)  # Range of lambda values
)

# Define a tuning grid for Ridge (alpha = 0)
tune_grid_ridge <- expand.grid(
  alpha = 0,  # Ridge
  lambda = 10^seq(-4, 1, length = 100)  # Range of lambda values
)

# Train Lasso Logistic Regression with cross-validation
set.seed(123)
lasso_model <- train(
  class ~ ., 
  data = train_df,
  method = "glmnet",
  family = "binomial",
  trControl = train_control,
  metric = "F1",
  tuneGrid = tune_grid_lasso
)

# View the Lasso model summary
print(lasso_model)

# Save the Lasso model
saveRDS(lasso_model, "models/lasso.rds")

# Train Ridge Logistic Regression with cross-validation
set.seed(123)
ridge_model <- train(
  class ~ ., 
  data = train_df,
  method = "glmnet",
  family = "binomial",
  trControl = train_control,
  metric = "F1",
  tuneGrid = tune_grid_ridge
)

# View the Ridge model summary
print(ridge_model)

# Save the Ridge model
saveRDS(ridge_model, "models/ridge.rds")
