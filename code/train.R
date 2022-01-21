
# library

source('code/2. training process/X. library custom function.R')

# Load data: train_data, train_ecg, valid_data, and valid_ecg

load("data/processed_data/train_data.RData")

# Set Parameter

model_name <- "ECGSurvNet"

my_ctx <- mx.cpu() # you may use mx.gpu() if gpu version of mxnet is installed.
verbose_FREQ <- 10
start_epoch <- 1
end_epoch <- 20
early_stop_epoch <- 9
batch_size <- 32
learning_rate <- c(5e-5, 5e-6)

set.seed(0)
mx.set.seed(0)

# Iterator

train_iter <- deepcox_iterator_func(iter = NULL, 
                                    train_data = train_ecg, train_label = train_label,
                                    my_ctx = my_ctx, batch_size = batch_size)

# Model

var_list <- declare_variable(var_names = c("data", "label", "mask"))

model_symbol <- ECGSurvNet(indata = var_list[["data"]], start_filter = 32, inverted_coef = 4,
                           num_filters = c(32, 64, 64, 128), num_unit = c(3, 3, 6, 4), end_filters = c(512))

loss_symbol <- cox_loss(indata = model_symbol, inlabel = var_list[["label"]], inmask = var_list[["mask"]])


# Main loop

for (k in 1:length(learning_rate)) {
  
  # Stage setting
  
  my_optimizer <- optimizer_manager(name = "adam", train_stage = k, 
                                    learning_rate = learning_rate, beta1 = 0.9, beta2 = 0.999, 
                                    epsilon = 1e-07, wd = 1e-3, rescale.grad = 1)
  
  if (k == 1) {
    
    arg_params <- NULL 
    aux_params <- NULL
    current_round <- 1 
    logger <- NULL
    
  } else {
    
    arg_params <- model_list[["model"]][["arg.params"]]
    aux_params <- model_list[["model"]][["aux.params"]]
    current_round <- length(model_list[["logger"]][["epoch_loss"]][["cox_loss"]]) + 1
    logger <- model_list[["logger"]]
    
  }
  
  model_list <- model_feed_forward(optimizer = my_optimizer, logger = logger,
                                   pred_symbol = loss_symbol,
                                   arg_params = arg_params, aux_params = aux_params, fixed_params = NULL,
                                   model_name = model_name, verbose_FREQ = verbose_FREQ,
                                   my_ctx = my_ctx, early_stop_epoch = early_stop_epoch,
                                   start_epoch = current_round, end_epoch = end_epoch, stage = k)
  
}


