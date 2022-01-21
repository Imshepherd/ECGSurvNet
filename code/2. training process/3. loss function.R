
# Loss function from cox proportional hazards model.

cox_loss <- function (indata, inlabel, inmask, eps = 1e-8, make_loss = TRUE) {
  
  exp_data <- mx.symbol.exp(data = indata, name = 'exp_data')
  
  part_event_y <-  mx.symbol.broadcast_mul(lhs = inlabel, rhs = inmask, name = 'part_event_y')
  part_event <-  mx.symbol.broadcast_mul(lhs = part_event_y, rhs = exp_data, name = 'part_event')
  sum_part_event <- mx.symbol.sum(data = part_event, axis = 0, keepdims = TRUE, name = 'sum_part_event')

  part_at_risk <-  mx.symbol.broadcast_mul(lhs = inmask, rhs = exp_data, name = 'part_at_risk')
  sum_part_at_risk <- mx.symbol.sum(data = part_at_risk, axis = 0, keepdims = TRUE, name = 'sum_part_at_risk')
  
  log_part_event <- mx.symbol.log(data = sum_part_event + eps, name = 'log_part_event')
  log_part_at_risk <- mx.symbol.log(data = sum_part_at_risk + eps, name = 'log_part_at_risk')
  
  diff_at_risk_event <- mx.symbol.broadcast_minus(lhs = log_part_event, rhs = log_part_at_risk, name = 'diff_at_risk_event')

  important_time <- mx.symbol.sum(data = part_event_y, axis = 0, keepdims = TRUE, name = 'important_time')
  event_loss <-  mx.symbol.broadcast_mul(lhs = diff_at_risk_event, rhs = important_time, name = 'event_loss')
  mean_event_loss <- mx.symbol.mean(data = event_loss, axis = 1, name = 'mean_event_loss')
  
  final_loss <- 0 - mean_event_loss
  
  if (make_loss) {final_loss <- mx.symbol.MakeLoss(data = final_loss, name = 'cox_loss')}
  
  return(final_loss)
  
}

