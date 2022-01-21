
# Optimizer

optimizer_manager <- function(name = "adam", train_stage = 1, 
                              learning_rate = c(1e-4, 1e-5, 1e-6), beta1 = 0.9, beta2 = 0.999, 
                              epsilon = 1e-08, wd = 1e-4, rescale.grad = 1){
  
  if (name == "adam"){
    
    optimizer <- mx.opt.create(name = "adam", learning.rate = learning_rate[train_stage], 
                               beta1 = beta1, beta2 = beta2, epsilon = epsilon, wd = wd, 
                               rescale.grad = rescale.grad)
    
  } else if (name == "sgd"){
    
    optimizer <- mx.opt.create(name = "sgd", learning.rate = learning_rate[train_stage], 
                               momentum = beta1, wd = wd, 
                               rescale.grad = rescale.grad, 
                               clip_gradient = NULL, lr_scheduler = NULL)
    
    
  }
  
  return(optimizer)
}