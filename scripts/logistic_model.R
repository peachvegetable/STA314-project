library(dplyr)
library(caret)
library(randomForest)
library(pROC)

path <- "data/processed_data/processed_train.csv"
train <- read.csv(path)

comment_embedding <- readRDS("train_embeddings.rds")
video_embedding <- readRDS("train_video_embeddings.rds")

train_df <- train |>
  select(comment_id, date_missing, year, month, day, hour, minute, 
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
  method = "cv",                        # Cross-validation
  number = 10,                          # Number of folds
  classProbs = TRUE,                    # Enable class probabilities
  summaryFunction = f1Summary,          # Use custom F1 summary
  savePredictions = "final"             # Save final predictions
)

# Train Logistic Regression with cross-validation
logistic_model <- train(
  class ~ .,                             # Formula: predict 'class' using all other variables
  data = train_df,                     # Training data
  method = "glm",                         # Model type: Generalized Linear Model
  family = "binomial",                    # Specify binomial family for logistic regression
  trControl = train_control,              # Training control
  metric = "F1"                            # Performance metric to optimize
)

# View model summary
print(logistic_model)

# Train Random Forest with cross-validation
set.seed(123)
rf_model <- train(
  class ~ .,                             # Formula
  data = train_df,                    # Training data
  method = "rf",                         # Model type: Random Forest
  trControl = train_control,             # Training control
  metric = "F1",                         # Performance metric to optimize
  tuneLength = 5                         # Number of tuning parameters to try
)

# View model details
print(rf_model)

# Plot feature importance
varImpPlot(rf_model$finalModel)
