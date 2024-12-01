library(dplyr)
library(lubridate)
library(stringr)
library(textclean)

data_path <- "data/raw_data/train.csv"
train <- read.csv(data_path)

# Create missingness indicator for training data
train <- train |>
  mutate(DATE_missing = ifelse(is.na(DATE), 1, 0))

# Convert DATE to POSIXct format for accurate median calculation
train$DATE <- as.POSIXct(train$DATE, format="%Y-%m-%d %H:%M:%S", tz = "UTC")

# Calculate median DATE from training data (excluding NAs)
median_date_train <- median(train$DATE, na.rm = TRUE)

# Impute missing DATE with median_date_train
train$DATE[is.na(train$DATE)] <- median_date_train

# Confirm no NAs remain
sum(is.na(train$DATE))

# Function to extract date features
extract_date_features <- function(df) {
  df <- df %>%
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

# Apply the function to training and testing data
train <- extract_date_features(train)

# Apply the function to training and testing data
train <- create_cyclical_features(train)

# Frequency encoding for AUTHOR (if applicable)
author_freq <- train |>
  group_by(AUTHOR) |>
  summarise(freq = n())

train <- left_join(train, author_freq, by = "AUTHOR")

train <- train |>
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
  
  # If there are no letters, return NA to avoid division by zero
  if (total_letters == 0) {
    return(NA)
  }
  
  # Number of uppercase letters
  capital_letters <- sum(str_detect(letters_only, "[A-Z]"))
  
  # Calculate ratio
  ratio <- capital_letters / total_letters
  
  return(ratio)
}

train$cpt_ratio <- sapply(train$CONTENT, capital_ratio)

clean_content <- function(content) {
  content <- iconv(content, from = "UTF-8", to = "ASCII", sub = "")  # converts the text from UTF-8 encoding to ASCII encoding
  
  # Decode HTML entities and remove HTML tags
  content <- replace_html(content)
  
  # Retain alphanumerics and selected punctuation (!, ?, .)
  # content <- str_replace_all(content, "[^[:alnum:]\\s!?.]", "")
  
  content <- str_squish(content)  # white space normalization 
  return(content)
}

train$CONTENT <- clean_content(train$CONTENT)

# Function to detect URLs
contains_url <- function(text) {
  str_detect(text, "https?://\\S+")
}

# Apply the function to create a new feature
train <- train |>
  mutate(has_url = ifelse(contains_url(CONTENT), 1, 0))

# Clean variable video_names
clean_video_name <- function(video_names) {
  # Remove multiple question marks
  video_names <- str_replace_all(video_names, "\\?+", "")
  
  # Remove empty parentheses resulting from removal
  video_names <- str_replace_all(video_names, "\\(\\)", "")
  
  # Trim any extra whitespace
  video_names <- str_squish(video_names)
  
  return(video_names)
}

# Apply to training data
train$VIDEO_NAME <- clean_video_name(train$VIDEO_NAME)

# Convert CLASS to numeric
train$CLASS <- as.numeric(as.character(train$CLASS))

names(train) <- tolower(names(train))

write.csv(train, "data/processed_data/processed_train.csv")
