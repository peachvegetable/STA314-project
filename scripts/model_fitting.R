library(dplyr)
library(caret)
library(randomForest)
library(pROC)

path <- "data/processed_data/processed_train.csv"
train <- read.csv(path)

comment_embedding <- readRDS("train_content_mean_embeddings.rds")

#pca_result <- prcomp(comment_embedding, center = TRUE, scale. = TRUE)
# Determine how many PCs to keep
#variance_explained <- summary(pca_result)$importance[2, ]
# Choose the number of PCs that explain 90% of variance
#n_pcs <- which(cumsum(variance_explained) >= 0.99)[1]
# Extract those principal components
#reduced_embeddings <- pca_result$x[, 1:n_pcs]

train$content <- comment_embedding

# Convert 'class' to a factor with prefixed labels to make them valid R variable names
train$class <- factor(train$class, levels = c(0, 1), labels = c("C0", "C1"))

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

# Define a reasonable mtry grid based on the number of predictors
rf_grid_expanded <- expand.grid(
  mtry = seq(2, ncol(train) - 1, by = 1)
)

# Train Random Forest with cross-validation
set.seed(123)
rf_model <- train(
  class ~ .,
  data = train,
  method = "rf",
  trControl = train_control,
  metric = "F1",
  tuneGrid = rf_grid_expanded,
  ntree = 1000
)

print(rf_model)
saveRDS(rf_model, "models/rf_1000.rds")