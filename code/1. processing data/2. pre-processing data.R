
# Library

library(rhdf5)
library(data.table)
library(magrittr)

# Set

set.seed(18574)

basic_infor_file_path <- "data/raw_data/Sami-Trop.csv"
ecg_data_file_path <- "data/raw_data/Sami-Trop.hdf5"

train_file_path <- "data/processed_data/train_data.RData"
valid_file_path <- "data/processed_data/valid_data.RData"

# Read basic and ecg data

basic_data <- fread(input = basic_infor_file_path, data.table = FALSE)

ecg_h5_info <- h5ls(file = ecg_data_file_path)
ecg_raw_data <- h5read(file = ecg_data_file_path, name = ecg_h5_info[,'name'])
 
# Randomly split dataset

basic_data[["dataset"]] <- sample(x = c("train", "valid"), 
                                  size = nrow(basic_data), replace = TRUE, prob = c(0.80, 0.20))

# Declare data and label for training

ecg <- array(data = NA, dim = c(4096, 12, 1, 1631))
label <- data.frame(id = NA,
                    age	= NA,
                    sex = NA,
                    normal_ecg = NA,
                    event = NA,
                    time = NA,
                    weight = NA,
                    dataset = NA,
                    stringsAsFactors = FALSE)

# Create data and label for deep learning model

for (i in 1:nrow(basic_data)) {
  
  label[i, "id"] <- basic_data[i, "exam_id"]
  label[i, "age"] <- basic_data[i, "age"]
  label[i, "sex"] <- basic_data[i, "is_male"] + 0L
  label[i, "normal_ecg"] <- basic_data[i, "normal_ecg"] + 0L
  label[i, "event"] <- basic_data[i, "death"] + 0L
  label[i, "time"] <- basic_data[i, "timey"]
  label[i, "dataset"] <- basic_data[i, "dataset"]
  
  ecg[,,1,i] <- t(ecg_raw_data[,,i])
}

# Set weight for training

train_idx <- label$dataset == "train"

train_event <- label[train_idx, 'event']
event_tab <- table(factor(train_event, levels = 0:1))
event_weight <- sum(event_tab) / event_tab

train_time <- label[train_idx, 'time']
time_cuts <- quantile(train_time, probs = seq(0, 1, by = 0.1))
time_cuts[1] <- -Inf
time_cuts[length(time_cuts)] <- Inf
time_cuts <- time_cuts %>% unique()
train_time_w <- cut(x = train_time, breaks = time_cuts, labels = 1:(length(time_cuts) - 1)) %>% as.integer()
train_time_w[train_event == 1] <- length(time_cuts) - 1

train_w <- train_time_w * event_weight[as.character(train_event)]
train_w <- train_w * length(train_idx) / sum(train_w)

label[train_idx, 'weight'] <- train_w

# Save out data

train_ecg <- ecg[,,,label[, "dataset"] == "train", drop = FALSE]
train_label <- label[label[, "dataset"] == "train", ]
save(train_ecg, train_label, file = train_file_path)

valid_ecg <- ecg[,,,label[, "dataset"] == "valid", drop = FALSE]
valid_label <- label[label[, "dataset"] == "valid", ]
save(valid_ecg, valid_label, file = valid_file_path)


