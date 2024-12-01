Sys.setenv(TOKENIZERS_PARALLELISM = "false")
library(reticulate)
# Use the Conda environment
use_condaenv("r", required = TRUE)
library(torch)
library(text)
library(dplyr)

transformers <- import("transformers")
torch <- import("torch")

train_path <- "data/processed_data/processed_train.csv"
test_path <- "data/raw_data/test.csv"
train <- read.csv(train_path)
test <- read.csv(test_path)

train_content <- train$content
test_content <- test$content
train_label <- train$class
test_label <- test$class

# Load the tokenizer and model
tokenizer <- transformers$AutoTokenizer$from_pretrained("bert-base-uncased")
model <- transformers$AutoModel$from_pretrained("bert-base-uncased")

# Tokenize the training content
train_content_inputs <- tokenizer(
  train_content,
  return_tensors = "pt",  # Return PyTorch tensors
  padding = TRUE,         # Pad all sentences to the same length
  truncation = TRUE       # Truncate sentences longer than BERT's max length
)

# Embedding train_inputs
with(torch$no_grad(), {  # Disable gradient computation for speed
  train_content_outputs <- model$forward(train_content_inputs$input_ids, attention_mask = train_content_inputs$attention_mask)
})

# Extract training CLS token embeddings (sentence-level embeddings)
train_content_embeddings <- train_content_outputs$last_hidden_state[, 0, ]
train_content_embeddings_matrix <- as.matrix(train_content_embeddings$detach()$numpy())
saveRDS(train_content_embeddings_matrix, "train_content_embeddings.rds")

# Convert to a data frame for easier handling
train_df <- as.data.frame(train_embeddings_matrix)

video_name <- train$video_name

train_video_inputs <- tokenizer(
  video_name,
  return_tensors = "pt",  # Return PyTorch tensors
  padding = TRUE,         # Pad all sentences to the same length
  truncation = TRUE       # Truncate sentences longer than BERT's max length
)

# Embedding train_inputs
with(torch$no_grad(), {  # Disable gradient computation for speed
  train_video_outputs <- model$forward(train_video_inputs$input_ids, attention_mask = train_video_inputs$attention_mask)
})

# Extract training CLS token embeddings (sentence-level embeddings)
train_video_embeddings <- train_video_outputs$last_hidden_state[, 0, ]
train_video_embeddings_matrix <- as.matrix(train_video_embeddings$detach()$numpy())
saveRDS(train_video_embeddings_matrix, "train_video_embeddings.rds")





# Tokenize the testing content
test_inputs <- tokenizer(
  test_content,
  return_tensors = "pt",  # Return PyTorch tensors
  padding = TRUE,         # Pad all sentences to the same length
  truncation = TRUE       # Truncate sentences longer than BERT's max length
)

# Embedding test_inputs
with(torch$no_grad(), {  # Disable gradient computation for speed
  test_outputs <- model$forward(test_inputs$input_ids, attention_mask = test_inputs$attention_mask)
})

# Extract testing CLS token embeddings (sentence-level embeddings)
test_embeddings <- test_outputs$last_hidden_state[, 0, ]
test_embeddings_matrix <- as.matrix(test_embeddings$detach()$numpy())
saveRDS(test_embeddings_matrix, "test_embeddings.rds")

# Convert to a data frame for easier handling
test_df <- as.data.frame(test_embeddings_matrix)

train_df$label <- train_label
test_df$label <- test_label

logistic_model <- glm(label ~ ., data=train_df, family=binomial)
saveRDS(logistic_model, "logistic_model.rds")

predictions <- predict(logistic_model, test_df, type="response")

predicted_labels <- ifelse(predictions > 0.5, 1, 0)
predicted_df <- data.frame(predicted_labels)
predicted_df$COMMENT_ID <- seq_len(nrow(predicted_df))
predicted_df <- rename(predicted_df, CLASS=predicted_labels)
predicted_df <- predicted_df |> select(COMMENT_ID, CLASS)
write.csv(predicted_df, "logistic_prediction", row.names=FALSE)
