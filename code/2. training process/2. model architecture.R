
# Main architecture

ECGSurvNet <- function(indata, end_pooling = TRUE, bn = TRUE,
                       start_filter = 32, inverted_coef = 4, relu_slope = 0.2,
                       num_filters = c(64, 128, 128), num_unit = c(3, 4, 6), end_filters = c(512, 512),
                       end_no_act = TRUE){

  # Share net 
  res_net <- res_block(indata = indata, end_pooling = end_pooling, bn = bn,
                       start_filter = start_filter, inverted_coef = inverted_coef, 
                       num_filters = num_filters, num_unit = num_unit, relu_slope = relu_slope,
                       end_no_act = end_no_act)
  message("main_net out dim: ", mx.symbol.infer.shape(res_net, data = c(2800, 12, 1, 7))$out.shapes)
  
  end_net <- end_net(indata = res_net, end_filters = end_filters)
  message("end_net out dim: ", mx.symbol.infer.shape(end_net, data = c(2800, 12, 1, 7))$out.shapes)
  
  return(end_net)
}


# Sub unit

basic_conv <- function(indata, kernel_size = c(3, 1), stride = c(1, 1), pad = (kernel_size - 1)/2, 
                       no_bias = TRUE, no_act = FALSE, no_norm = FALSE, no_pad = FALSE, bn = TRUE,
                       relu_slope = 0.2, num_filters = 128, stage = "start_1"){

  # Standard conv block Conv-BN-Relu  
  
  if (no_pad){pad <- c(0, 0)}

  conv <- mx.symbol.Convolution(data = indata, 
                                kernel = kernel_size, stride = stride, pad = pad, 
                                no.bias = no_bias, num.filter = num_filters, 
                                name = paste0(stage, "_conv"))
  
  if (no_norm){
    
    norm <- conv
    
  } else {
    
    if (bn){
      
      norm <- mx.symbol.BatchNorm(data = conv, eps = 1e-04, fix_gamma = FALSE,
                                  momentum = 0.9, use_global_stats = FALSE,
                                  name = paste0(stage, "_bn"))
    } else {
      
      norm <- mx.symbol.InstanceNorm(data = conv, name = paste0(stage, "_insnorm"))
    }
  }
  
  if (no_act){
    
    return(norm)
    
  }
  
  relu <- mx.symbol.LeakyReLU(data = norm, act.type = 'leaky', slope = relu_slope, 
                              name = paste0(stage, "_relu"))
  return(relu)
}

residual_block <- function(indata, kernel_size = c(3, 1), relu_slope = 0.2, stride_reduce, bn = bn,
                           num_filters = num_filters, inverted_coef = 4, end_no_act = TRUE, stage) {
  
  # Reduce layer conv 1 * 1
  conv_reduce <- basic_conv(indata = indata, kernel_size = c(1, 1), stride = stride_reduce, 
                            num_filters = num_filters, no_pad = FALSE, relu_slope = relu_slope,
                            bn = bn, stage = paste0(stage, "_reduce"))
  
  # Bottleneck extract main feature conv layer 3 * 3
  conv_standard <- basic_conv(indata = conv_reduce, kernel_size = kernel_size, stride = c(1, 1), 
                              num_filters = num_filters, no_pad = FALSE, relu_slope = relu_slope,
                              bn = bn, stage = paste0(stage, "_standard"))
  
  # Restore filter number for element-wise plus
  conv_restore <- basic_conv(indata = conv_standard, kernel_size = c(1, 1), stride = c(1, 1), 
                             num_filters = num_filters * inverted_coef, no_pad = FALSE, relu_slope = relu_slope,
                             bn = bn, no_act = end_no_act, stage = paste0(stage, "_restore"))
  
  return(conv_restore)
}


# Sub module: ResNet block

res_block <- function(indata, end_pooling, bn,
                      start_filter, inverted_coef, 
                      num_filters, num_unit, relu_slope, end_no_act){
      
  if (!all.equal(length(num_filters), length(num_unit))){ #
    stop("Please declare same length with num_filters, trans_filters and num_unit")
  }
  num_block <- length(num_filters)
  
  # Start 
  start_1 <- basic_conv(indata = indata, kernel_size = c(5, 1), stride = c(2, 1), 
                        num_filters = start_filter, no_pad = FALSE, bn = bn, relu_slope = relu_slope,
                        stage = paste0("start_1"))

  res_in <- mx.symbol.Pooling(data = start_1, global_pool = FALSE, pool_type = "avg", pooling_convention = "valid",
                              kernel = c(2, 1), stride = c(2, 1), pad = c(0, 0), 
                              name = paste0("res_start", "_avg_pool"))
  
  # Main ResNet
  for (k in 1:num_block){
    for (m in 1:num_unit[k]){ 
      # message("m:", m, ", k:", k)
      if (m == 1){ # Down sample and branch with each start.
        stride_reduce <- c(2, 1) 
        branch <- TRUE
        if (k == 1){ # Only start no down sampling
          stride_reduce <- c(1, 1)
        }
      } else {
        stride_reduce <- c(1, 1)
        branch <- FALSE
      }
      
      if (branch){
        res_branch <- basic_conv(indata = res_in, kernel_size = c(1, 1), stride = stride_reduce, 
                                 num_filters = num_filters[k] * inverted_coef, no_pad = FALSE, no_act = end_no_act,
                                 bn = bn, relu_slope = relu_slope, stage = paste0("branch_", k, "_", m))
      } else {
        res_branch <- res_in
      }
      
      res_block_out <- residual_block(indata = res_in, kernel_size = c(3, 1), stride = stride_reduce,
                                      bn = bn, relu_slope = relu_slope, 
                                      num_filters = num_filters[k], inverted_coef = inverted_coef,
                                      end_no_act = end_no_act, stage = paste0("res_", k, "_", m))
      
      res_add <- mx.symbol.elemwise_add(lhs = res_block_out, rhs = res_branch,
                                        name = paste0(paste0("res_", k, "_", m), "_add"))
      
      res_relu <- mx.symbol.LeakyReLU(data = res_add, act.type = 'leaky', slope = relu_slope, 
                                      name = paste0(paste0("res_", k, "_", m), "_relu"))
      # To next block
      res_in <- res_relu
      
    }
  }

  # End with pooling for each lead
  if (end_pooling){
    res_end <- mx.symbol.mean(data = res_in, axis = 3, keepdims = TRUE, exclude = FALSE,
                              name = paste0("res_end", "_avg_pool"))
  } else {
    res_end <- res_in
  }
  
  return(res_end)
}

end_net <- function(indata, relu_slope = 0.2, end_filters = c(512, 512, 128)){
  
  num_end_conv <- length(end_filters)
  
  for (k in 1:num_end_conv){
    
    if (k == 1){feature <- indata} else {feature <- relu}
    
    end_conv <- mx.symbol.Convolution(data = feature, kernel = c(1, 1), stride = c(1, 1), pad = c(0, 0),
                                      num.filter = end_filters[k], no.bias = FALSE, 
                                      name = paste0("end_net_", k, "_conv"))
    
    if (k == num_end_conv){
      
      end_net_leadto1_conv <- mx.symbol.Convolution(data = end_conv, kernel = c(1, 1), stride = c(1, 1), pad = c(0, 0),
                                                    num.filter = 1, no.bias = FALSE,
                                                    name = paste0("end_net_leadto1_conv"))
      
      end_net_leadto1_relu <- mx.symbol.LeakyReLU(data = end_net_leadto1_conv, act.type = 'leaky', slope = relu_slope,
                                                  name = "end_net_leadto1_relu")
      
      fc_end <- mx.symbol.FullyConnected(data = end_net_leadto1_relu, num.hidden = 1, no.bias = TRUE, name = 'fc_end')
      final_pred <- mx.symbol.BatchNorm(data = fc_end, eps = 1e-04, fix_gamma = TRUE,
                                        momentum = 0.9, use_global_stats = FALSE, name = 'final_pred')
      
      return(final_pred)
      
    } else {
      
      relu <- mx.symbol.LeakyReLU(data = end_conv, act.type = 'leaky', slope = relu_slope,
                                  name = paste0('end_net_', k, "_relu"))
    }
  }
}


# Declare function

declare_variable <- function(var_names = c("data", "label", "mask")){
  
  # Declare input variable
  
  var_list <- list()
  for (i in 1:length(var_names)){
    var_list[[var_names[i]]] = mx.symbol.Variable(name = var_names[i])
  }
  
  return(var_list)
}



