library(dplyr)
library(lubridate)
library(stringr)
library(textclean)

rf_model_path <- "models/rf_1000.rds"
rf <- readRDS(rf_model_path)

lasso_model_path <- "models/lasso.rds"
lasso <- readRDS(lasso_model_path)

test_path <- "data/processed_data/processed_test.csv"
test <- read.csv(test_path)

comment_embedding <- readRDS("test_content_mean_embeddings.rds")

test$content <- comment_embedding

rf_test <- predict(rf, newdata = test)

lasso_test <- predict(lasso, newdata = test)

final_submission <- data.frame(COMMENT_ID = seq(1370, 1370 + nrow(test) - 1), CLASS = rf_test)

final_submission <- final_submission |>
  mutate(
    CLASS = recode(CLASS, "C0" = "0", "C1" = "1"),
    CLASS = as.integer(as.character(CLASS))
  )

write.csv(final_submission, "predictions/rf_prediction.csv", row.names=FALSE)
