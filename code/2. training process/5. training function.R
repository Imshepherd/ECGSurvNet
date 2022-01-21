
# Training function for deep cox.

model_feed_forward <- function (optimizer, logger,
                                pred_symbol,
                                arg_params, aux_params, fixed_params,
                                model_name, verbose_FREQ,
                                my_ctx, early_stop_epoch,
                                start_epoch, end_epoch, stage){

  
  if (start_epoch > end_epoch){stop("End Training")}
  
  message(paste0("Start to training: ",  model_name))
  
  model_path <- paste0('model/', model_name)
  if (!dir.exists(model_path)){dir.create(model_path)}
  
  # Get shape of input and ouput data.
  
  data_dim <- get_data_dim(iterator = train_iter)
  
  # Main Feature Extracter Executor
  
  exec_list <- list(symbol = pred_symbol, fixed.param = fixed_params, ctx = my_ctx, grad.req = "write")
  exec_list <- append(exec_list, data_dim)
  my_executor <- do.call(mx.simple.bind, exec_list)
  
  # Initial parameters 
  
  init_list <- list(symbol = pred_symbol, ctx = my_ctx, input.shape = data_dim, output.shape = NULL)
  init_list <- append(init_list, list(initializer = mxnet:::mx.init.Xavier(rnd_type = "uniform", magnitude = 2.24)))
  my_arg <- do.call(mx.model.init.params, init_list)  
  
  # Update parameters
  
  mx.exec.update.arg.arrays(my_executor, my_arg[["arg.params"]], match.name = TRUE)
  mx.exec.update.aux.arrays(my_executor, my_arg[["aux.params"]], match.name = TRUE)  
  
  if (!is.null(arg_params) & !is.null(aux_params)) {
    
    # Update parameters from previous model.
    
    mx.exec.update.arg.arrays(my_executor, arg_params, match.name = TRUE)
    mx.exec.update.aux.arrays(my_executor, aux_params, match.name = TRUE)
  } 
  
  message(paste0('The number of total parameters = ', sum(sapply(my_arg[["arg.params"]], length))))
  my_updater <- mx.opt.get.updater(optimizer = optimizer, weights = my_executor[["ref.arg.arrays"]])
  
  # Forward/Backward
  
  for (epoch in start_epoch:end_epoch) {
    
    # Training
    
    train_iter$reset()
    batch_loss <- list()
    batch_t0 <- Sys.time()
    
    while (train_iter$iter.next()) {
      
      my_values <- train_iter$value()
      
      mx.exec.update.arg.arrays(my_executor, arg.arrays = my_values, match.name = TRUE)
      mx.exec.forward(my_executor, is.train = TRUE)
      mx.exec.backward(my_executor)
      
      update_args <- my_updater(weight = my_executor$ref.arg.arrays, grad = my_executor$ref.grad.arrays)
      mx.exec.update.arg.arrays(my_executor, update_args, skip.null = TRUE)
      
      # Logging loss
      
      batch_loss$cox_loss[length(batch_loss$cox_loss) + 1] <- as.array(my_executor$ref.outputs[[1]])
      
      if (length(batch_loss$cox_loss) %% verbose_FREQ == 0 | length(batch_loss$cox_loss) < 6) {
        batch_time <- as.double(difftime(Sys.time(), batch_t0, units = "secs"))
        batch_speed <- length(batch_loss$cox_loss) * tail(data_dim$data, 1) / batch_time
        batch_result <- lapply(batch_loss, function(x){mean(x) %>% formatC(., 6, format = "f")})
        message(paste0("Batch [", length(batch_loss$cox_loss),
                       "] Speed: ", formatC(batch_speed, 3, format = "f"),
                       " samples/sec Train-loss: ", paste0(names(batch_result), "=", batch_result, collapse = ", ")))
      }
      
    }
    
    epoch_loss <- lapply(batch_loss, function(x){mean(x)})
    message_loss <- lapply(epoch_loss, function(x){formatC(x, 6, format = "f")})
    message(paste0("Current training : ", model_name))
    message(paste0("Epoch [", epoch, "] Train-loss: ", paste0(names(message_loss), "=", message_loss, collapse = ", ")))
    
    # Get model and save
    
    my_model <- mxnet:::mx.model.extract.model(symbol = pred_symbol, train.execs = list(my_executor))
    my_model[['arg.params']] <- append(my_model[['arg.params']], my_executor$arg.arrays[fixed_params])
    my_model[['arg.params']] <- my_model[['arg.params']][!names(my_model[['arg.params']]) %in% names(data_dim)]
    mx.model.save(model = my_model, prefix = paste0(model_path, '/',model_name), iteration = epoch)
    
    # Update logger
    
    logger <- logger_updater(logger = logger, epoch = epoch, stage = stage, epoch_loss = epoch_loss)
    if (!is.null(model_path)) {save(logger, file = paste0(model_path, "/",model_name, "_logger.RData"))}
    
    # Visualize, close old plot
    
    if (dev.cur() > 1){dev.off()} 
    show_logger_plot(logger = logger)
    
    # Early stop
    
    if (early_stop_epoch > 0) {
      current_stage_loss <- logger$epoch_loss$cox_loss[logger$stage == stage]
      if ((length(current_stage_loss)) > early_stop_epoch) {break}
    }
    
  }
  
  return(list(model = my_model, logger = logger))
  
}

get_data_dim <- function (iterator){
  
  # The shape of input and output was decided by iterator function
  
  iterator$reset()
  iterator$iter.next()
  got_train_data <- iterator$value()
  input_dim <- lapply(got_train_data, dim)
  
  return(input_dim)
}

test_model_feed_forward <- function(){
  
  optimizer = my_optimizer; logger = logger;
  pred_symbol = loss_symbol;
  arg_params = arg_params; aux_params = aux_params; fixed_params = NULL;
  model_name = model_name; verbose_FREQ = verbose_FREQ;
  my_ctx = my_ctx; early_stop_epoch = early_stop_epoch;
  start_epoch = current_round; end_epoch = end_epoch; stage = k;
  evalu_type = "c_index"; tune_data = tune_ecg; tune_label = tune_label;
}


