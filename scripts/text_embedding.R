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
test_path <- "data/processed_data/processed_test.csv"
train <- read.csv(train_path)
test <- read.csv(test_path)

train_content <- train$content
test_content <- test$content
train_label <- train$class
train_video_name <- train$video_name
test_video_name <- test$video_name

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
train_content_mean_embeddings <- torch$mean(train_content_outputs$last_hidden_state, dim = as.integer(1))
train_content_mean_embeddings_matrix <- as.matrix(train_content_mean_embeddings$detach()$numpy())
saveRDS(train_content_mean_embeddings_matrix, "train_content_mean_embeddings.rds")
saveRDS(train_content_embeddings_matrix, "train_content_embeddings.rds")

# Tokenize the testing content
test_content_inputs <- tokenizer(
  test_content,
  return_tensors = "pt",  # Return PyTorch tensors
  padding = TRUE,         # Pad all sentences to the same length
  truncation = TRUE       # Truncate sentences longer than BERT's max length
)

# Embedding test_inputs
with(torch$no_grad(), {  # Disable gradient computation for speed
  test_content_outputs <- model$forward(test_content_inputs$input_ids, attention_mask = test_content_inputs$attention_mask)
})

# Extract testing CLS token embeddings (sentence-level embeddings)
test_content_embeddings <- test_content_outputs$last_hidden_state[, 0, ]
test_content_embeddings_matrix <- as.matrix(test_content_embeddings$detach()$numpy())
test_content_mean_embeddings <- torch$mean(test_content_outputs$last_hidden_state, dim = as.integer(1))
test_content_mean_embeddings_matrix <- as.matrix(test_content_mean_embeddings$detach()$numpy())
saveRDS(test_content_mean_embeddings_matrix, "test_content_mean_embeddings.rds")
saveRDS(test_content_embeddings_matrix, "test_content_embeddings.rds")

