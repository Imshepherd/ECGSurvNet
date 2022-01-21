
# Library

library(ggplot2)
library(magrittr)

# Function

logger_updater <- function(logger, epoch, stage, epoch_loss){
  
  logger$stage[epoch] <- stage
  
  if (!"epoch_loss" %in% names(logger)){logger$epoch_loss <- list()}
  if (!"result" %in% names(logger)){logger$result <- list()}
  
  for (i in 1:length(epoch_loss)){
    names_loss <- names(epoch_loss)[i]
    logger$epoch_loss[[names_loss]][[epoch]] <- epoch_loss[[names_loss]]
  }
  
  return(logger)
}

show_logger_plot <- function(logger){
  
  # Create data frame for plotting      
  loss_data <- set_loss_data(logger)

  plot_data <- rbind(loss_data)
  plot_data$epoch <- as.integer(plot_data$epoch)
  plot_data$value <- as.numeric(plot_data$value)

  # Plot
  max_epoch <- max(plot_data$epoch)
  min_epoch <- min(plot_data$epoch)
  
  gplot <- ggplot(data = plot_data, aes(x = epoch, y = value, color = name)) + geom_line() + geom_point() +
    theme_light() + theme(legend.position = 'top') + ylab(" ") +
    scale_x_continuous(limits = c(min_epoch, max_epoch + 15))
  
  # Text
  anno_data <- set_anno_data(plot_data)
  
  gplot <- gplot + facet_grid(name ~ ., scales = "free_y") + 
    geom_text(data = anno_data, mapping = aes(x = max_epoch + 8, y = position, label = text), size=3)
  
  # Stage shadow
  gplot <- set_shadow_data(gplot, logger)
  
  print(gplot)
  
}

set_loss_data <- function(logger){
  
  plot_data <- data.frame()
  
  for (i in 1:length(logger$epoch_loss)){
    evalu_data <- cbind(epoch = 1:length(logger$epoch_loss[[i]]), 
                        value = logger$epoch_loss[[i]],
                        name = rep(names(logger$epoch_loss)[i], length(logger$epoch_loss[[i]]))
    )
    plot_data <- rbind(plot_data, evalu_data, stringsAsFactors = FALSE)
  }
  
  return(plot_data)
}

set_anno_data <- function(plot_data, text_evalu = "cox_loss"){
  
  anno_data <- NULL
  # Calculate best
  for (i in 1:length(text_evalu)){
    
    target_value <- as.numeric(plot_data$value[plot_data$name == text_evalu[i]])
    
    if (text_evalu[i] %in% c("roc", "c_index")){
      best_value <- max(target_value)
    } else if (text_evalu[i] %in% c("mae", "cox_loss")){
      best_value <- min(target_value)
    } 
    
    best_epoch <- which(target_value == best_value)
    sub_plot_data <- NULL
    sub_plot_data[1] <- range(target_value) %>% mean(.)
    sub_plot_data[2] <- paste0("Best ", text_evalu[i], ": \n", 
                               formatC(best_value, 6, format = "f"), 
                               "\n epoch:", best_epoch)
    sub_plot_data[3] <- text_evalu[i]
    anno_data <- rbind(anno_data, sub_plot_data)
  }
  anno_data <- data.frame(anno_data, row.names = NULL, stringsAsFactors=FALSE)
  colnames(anno_data) <- c("position", "text", "name")
  anno_data$position <- as.numeric(anno_data$position)
  
  return(anno_data)
}

set_shadow_data <- function(gplot, logger){
  
  total_stage <- unique(logger$stage)
  stage_colors <- c("pink", "green", "purple")
  
  color_data <- data.frame()
  for (i in 1:length(total_stage)){
    
    stage_pos <- which(logger$stage == total_stage[i])
    
    start_epoch <- stage_pos[1] - 0.5
    end_epoch <- stage_pos[length(stage_pos)] + 0.5
    
    if (i == 1){
      start_epoch <- stage_pos[1]
    }
    color_stage <- c(start_epoch, end_epoch, -Inf, Inf)
    color_data <- rbind(color_data, color_stage)
  }
  colnames(color_data) <- c("xmin", "xmax", "ymin", "ymax")
  
  for (i in 1:nrow(color_data)){
    gplot <- gplot + annotate("rect", fill = stage_colors[i], alpha = 0.20, 
                              xmin = color_data$xmin[i], xmax = color_data$xmax[i],
                              ymin = color_data$ymin[i], ymax = color_data$ymax[i])
  }
  
  return(gplot)
}
