library(dplyr)
library(lubridate)
library(stringr)
library(textclean)

data_path <- "data/raw_data/test.csv"
test <- read.csv(data_path)

# Create missingness indicator for testing data
test <- test |>
  mutate(DATE_missing = ifelse(is.na(DATE), 1, 0))

# Convert DATE to POSIXct format for accurate median calculation
test$DATE <- as.POSIXct(test$DATE, format="%Y-%m-%d %H:%M:%S", tz = "UTC")

# Calculate median DATE from testing data (excluding NAs)
median_date_test <- median(test$DATE, na.rm = TRUE)

# Impute missing DATE with median_date_test
test$DATE[is.na(test$DATE)] <- median_date_test

# Confirm no NAs remain
sum(is.na(test$DATE))

# Function to extract date features
extract_date_features <- function(df) {
  df <- df |>
    mutate(
      year = year(DATE),
      month = month(DATE),
      day = day(DATE),
      hour = hour(DATE),
      minute = minute(DATE),
      second = second(DATE),
      weekday = wday(DATE, label = TRUE),
      week = week(DATE),
      quarter = quarter(DATE),
      day_of_year = yday(DATE),
      week_of_year = isoweek(DATE),
      weekend = ifelse(wday(DATE) %in% c(1, 7), 1, 0)  # 1 = Weekend, 0 = Weekday
    )
  return(df)
}

# Function to create cyclical features
create_cyclical_features <- function(df) {
  df <- df %>%
    mutate(
      # Hour of Day
      hour_sin = sin(2 * pi * hour / 24),
      hour_cos = cos(2 * pi * hour / 24),
      
      # Day of Week
      weekday_num = as.numeric(wday(DATE)),  # Sunday = 1, Saturday = 7
      weekday_sin = sin(2 * pi * weekday_num / 7),
      weekday_cos = cos(2 * pi * weekday_num / 7),
      
      # Month
      month_sin = sin(2 * pi * month / 12),
      month_cos = cos(2 * pi * month / 12)
    )
  return(df)
}

# Apply the function to testing and testing data
test <- extract_date_features(test)

# Apply the function to testing and testing data
test <- create_cyclical_features(test)

# Frequency encoding for AUTHOR (if applicable)
author_freq <- test |>
  group_by(AUTHOR) |>
  summarise(freq = n())

test <- left_join(test, author_freq, by = "AUTHOR")

test <- test |>
  rename("author_freq" = "freq")

# Function to calculate capital letter ratio using stringr
capital_ratio <- function(text) {
  # Ensure the input is a character string
  if (!is.character(text)) {
    stop("Input must be a character string.")
  }
  
  # Extract all alphabetic characters
  letters_only <- str_extract_all(text, "[A-Za-z]")[[1]]
  
  # Total number of alphabetic characters
  total_letters <- length(letters_only)
  
  # If there are no letters, return 0
  if (total_letters == 0) {
    return(0)
  }
  
  # Number of uppercase letters
  capital_letters <- sum(str_detect(letters_only, "[A-Z]"))
  
  # Calculate ratio
  ratio <- capital_letters / total_letters
  
  return(ratio)
}

test$cpt_ratio <- sapply(test$CONTENT, capital_ratio)

clean_content <- function(content) {
  content <- iconv(content, from = "UTF-8", to = "ASCII", sub = "")  # converts the text from UTF-8 encoding to ASCII encoding
  
  # Decode HTML entities and remove HTML tags
  content <- replace_html(content)
  
  # Retain alphanumerics and selected punctuation (!, ?, .)
  # content <- str_replace_all(content, "[^[:alnum:]\\s!?.]", "")
  
  content <- str_squish(content)  # white space normalization 
  return(content)
}

test$CONTENT <- clean_content(test$CONTENT)

# Function to detect URLs
contains_url <- function(text) {
  str_detect(text, "https?://\\S+")
}

# Apply the function to create a new feature
test <- test |>
  mutate(has_url = ifelse(contains_url(CONTENT), 1, 0))

# Apply to testing data
test$VIDEO_NAME <- model.matrix(~ VIDEO_NAME - 1, data = test)

names(test) <- tolower(names(test))

# detect for common words, and add word_count
test <- test |>
  mutate(
    contains_out = as.integer(str_detect(content, "\\bout\\b")),
    contains_check = as.integer(str_detect(content, "\\bcheck\\b")),
    contains_please = as.integer(str_detect(content, "\\bplease\\b")),
    contains_youtube = as.integer(str_detect(content, "\\byoutube\\b")),
    contains_subscribe = as.integer(str_detect(content, "\\bsubscribe\\b")),
    contains_video = as.integer(str_detect(content, "\\bvideo\\b")),
    contains_39 = as.integer(str_detect(content, "\\b39\\b")),
    contains_channel = as.integer(str_detect(content, "\\bchannel\\b")),
    contains_money = as.integer(str_detect(content, "\\bmoney\\b")),
    contains_comment = as.integer(str_detect(content, "\\bcomment\\b")),
    contains_follow = as.integer(str_detect(content, "\\bfollow\\b")),
    word_count = str_count(content, "\\S+")
  )

# -hour, -second, -quarter, -hour_sin, -hour_cos
test <- test |>
  select(-comment_id, -author, -date, -weekday)

write.csv(test, "data/processed_data/processed_test.csv")


