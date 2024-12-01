library(dplyr)
library(lubridate)
library(stringr)
library(textclean)

rf_model_path <- "models/rf.rds"
rf <- readRDS(rf_model_path)

test_path <- "data/processed_data/processed_test.csv"
test <- read.csv(test_path)

comment_embedding <- readRDS("data/embeddings/test_embeddings.rds")
video_embedding <- readRDS("data/embeddings/test_video_embeddings.rds")

test_df <- test |>
  select(comment_id, date_missing, year, month, day, hour, minute, 
         second, weekday, week, quarter, day_of_year, week_of_year, weekend, 
         hour_sin, hour_cos, weekday_num, weekday_sin, weekday_cos, month_sin, 
         month_cos, author_freq, has_url)

test_df$comment <- comment_embedding
test_df$video_name <- video_embedding

rf_test <- predict(rf, newdata = test_df)

final_submission <- data.frame(COMMENT_ID = test$comment_id, CLASS = rf_test)

final_submission <- final_submission |>
  mutate(
    CLASS = recode(CLASS, "C0" = "0", "C1" = "1"),
    CLASS = as.integer(as.character(CLASS))
  )

write.csv(final_submission, "predictions/rf_prediction.csv", row.names=FALSE)
