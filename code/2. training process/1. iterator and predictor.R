
# Library

library(mxnet)
library(magrittr)
library(survival)

# Iterator

deepcox_iterator_core = function(train_data, train_label, my_ctx, batch_size) {
  
  batch <- 0
  
  sample_size <- nrow(train_label)
  batch_per_epoch <- ceiling(sample_size/(batch_size))
  
  reset = function() {
    batch <<- 0
    gc(reset = TRUE)
  }
  
  iter.next = function() {
    batch <<- batch + 1
    if (batch %% 100 == 0) {gc(reset = TRUE)}
    if (batch > batch_per_epoch) {
      return(FALSE)
    } else {
      return(TRUE)
    }
  }
  
  value = function() {
    
    sample_idx <- base:::sample(x = 1:nrow(train_label), size = batch_size, replace = FALSE, prob = train_label[, "weight"])
    
    batch_label <- train_label[sample_idx, ]
    batch_data <- train_data[,,,sample_idx, drop = FALSE]
    
    batch_order <- order(batch_label[, "time"], decreasing = FALSE)
    
    batch_label <- batch_label[batch_order, ]
    batch_data <- batch_data[,,,batch_order, drop = FALSE]
    
    label <- array(0, dim = rep(batch_size, 2))
    mask <- array(0, dim = rep(batch_size, 2))
    
    had_event <- batch_label[, "event"] == 1

    for (i in 1:nrow(batch_label)) {
      
      # Only back-propagation if sample had event.
      if (batch_label[i, "event"] == 1){

        same_time <- batch_label[, "time"] == batch_label[i, "time"]
        same_time_had_event <- same_time & had_event
        
        label[i, same_time_had_event] <- 1
        
        same_and_bigger_time <- batch_label[, "time"] >= batch_label[i, "time"]
        mask[i, same_and_bigger_time] <- 1
      }
    }

    label_nd_array <- mx.nd.array(label, ctx = my_ctx)
    mask_nd_array <- mx.nd.array(mask, ctx = my_ctx)
    
    # Crop the middle ECG data for a segment with length of 2800.
    batch_data <- batch_data[648:3447,,,,drop=FALSE]
    
    data_nd_array <- mx.nd.array(batch_data, ctx = my_ctx)
    
    return(list(data = data_nd_array, label = label_nd_array, mask = mask_nd_array))
  }
  
  return(list(reset = reset, iter.next = iter.next, value = value, batch_size = batch_size, batch = batch))
}


deepcox_iterator_func <- setRefClass("Custom_deepcox_Iter",
                                     fields = c("iter", "train_data", "train_label", "my_ctx", "batch_size"),
                                     contains = "Rcpp_MXArrayDataIter",
                                     methods = list(
                                       initialize = function(iter, train_data, train_label, my_ctx, batch_size = 16){
                                         .self$iter <- deepcox_iterator_core(train_data = train_data, train_label = train_label,
                                                                             my_ctx = my_ctx, batch_size = batch_size)
                                         .self
                                       },
                                       value = function(){
                                         .self$iter$value()
                                       },
                                       iter.next = function(){
                                         .self$iter$iter.next()
                                       },
                                       reset = function(){
                                         .self$iter$reset()
                                       },
                                       finalize=function(){
                                       }
                                     )
)


survival_risk_predict <- function(model, data, batch_size, my_ctx) {

  data_dim <- dim(data)
  
  # Crop the middle ECG data for a segment with length of 2800.
  x <- list()
  x[[1]] <- data[648:3447,,,,drop=FALSE]
  
  # build a exec
  
  all_layers <- model$symbol$get.internals()
  
  output_symbol <- which(grepl("final_pred_output", all_layers$outputs)) %>% all_layers$get.output()
  arg_list <- list(symbol = output_symbol, ctx = my_ctx, grad.req = 'null', data = c(data_dim[1:3], batch_size))
  
  pexec <- do.call(mx.simple.bind, arg_list)
  
  arg_name <- names(pexec$arg.arrays)[!names(pexec$arg.arrays) %in% "data"]
  aux_name <- names(pexec$aux.arrays)[!names(pexec$aux.arrays) %in% "data"]
  
  mx.exec.update.arg.arrays(pexec, model$arg.params[arg_name], match.name = TRUE)
  mx.exec.update.aux.arrays(pexec, model$aux.params[aux_name], match.name = TRUE) 
  
  # predict
  
  total_len <- ceiling(data_dim[4]/batch_size)
  
  predict_risk <- NULL
  
  pb <- txtProgressBar(max = total_len, style = 3)
  t0 <- Sys.time()
  
  for (i in 1:total_len) {
    
    idx <- (i - 1) * batch_size + 1:batch_size
    idx[idx > data_dim[4]] <- 1
    
    X_batch_predict_out <- array(0, dim = c(1, batch_size))
    for (j in 1:length(x)){
      sub_x <- x[[j]]
      mx_nd_x <- mx.nd.array(array(sub_x[,,,idx, drop = FALSE], dim = c(data_dim[1:3], batch_size)), ctx = my_ctx)
      mx.exec.update.arg.arrays(pexec, arg.arrays = list(data = mx_nd_x), match.name = TRUE)
      mx.exec.forward(pexec, is.train = FALSE)
      X_batch_predict_out <- X_batch_predict_out + as.array(pexec$ref.outputs[[1]])
    }
    
    predict_risk[idx] <- X_batch_predict_out/length(x) # as linear output of function
    
    setTxtProgressBar(pb, i)
  }
  
  close(pb)
  
  time <- as.double(difftime(Sys.time(), t0,  units = "secs"))
  speed <- total_len * batch_size / time
  
  message(paste0("Total sample = ", total_len * batch_size, "  Speed = ", formatC(speed, 3, format = "f"), " samples/sec"))
  
  return(list(predict_out = predict_risk))
}


