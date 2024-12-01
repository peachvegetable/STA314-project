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
  savePredictions = "final"
)

train_df_1 <- train_df |>
  select(-weekday)

# Use lasso to perform feature selection
set.seed(123)
lasso_model <- train(
  class ~ ., 
  data = train_df_1,
  method = "glmnet",
  family = "binomial",
  trControl = train_control,
  metric = "F1",
  tuneGrid = expand.grid(
    alpha = 1,  # Lasso
    lambda = 10^seq(-4, 1, length = 100)
  )
)
# Extract selected features
coef_lasso <- coef(lasso_model$finalModel, s = lasso_model$bestTune$lambda)
selected_features_lasso <- rownames(coef_lasso)[which(coef_lasso != 0)]
selected_features_lasso <- selected_features_lasso[selected_features_lasso != "(Intercept)"]

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