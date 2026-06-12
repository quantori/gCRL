# Script analyzing the parameter sweep results across several settings

# set up
rm(list = ls())
library(tidyverse)
library(ggplot2)
library(patchwork)
 httpgd::hgd()
 
# control panel
# Joung2023 --> Joung2023_subsample
# Joung2023_random_subsample --> Joung2023
# Norman2019 --> Norman2019_before_2026_04_29
to_analyze <- c('Norman2019', 'Joung2023', 
                'Joung2023_add_controls',
                'Joung2023_random_subsample')

# looping
tmp <- NULL
for(index in 1:length(to_analyze)){

  # loading and adding columns
  to_plot <- read.csv(paste0('../../real/', to_analyze[index], '/sweep_50pct/sweep_results.csv'))
  to_plot$emode_std <- paste0(to_plot$eigengene_mode, '-',
                              to_plot$ae_standardize)
  to_plot$diff <- to_plot$mcc_real_mean - to_plot$mcc_perm_mean
  if(index == 1){
    tmp <- cbind(to_plot, setting = to_analyze[index])
  }else{
    tmp <- rbind(tmp, cbind(to_plot, setting = to_analyze[index]))
  }
  
  # correlation analysis
  print(to_analyze[index])
  suppressWarnings(res <- cor.test(to_plot$diff, to_plot$val_loss_final, method = 'spearman'))
  print(paste0('Corr with val loss: ', round(res$estimate, 3), ', pvalue: ', round(res$p.value, 3)))
  suppressWarnings(res <- cor.test(to_plot$diff, to_plot$train_loss_final, method = 'spearman'))
  print(paste0('Corr with train loss: ', round(res$estimate, 3), ', pvalue: ', round(res$p.value, 3)))
  print('')
  
  p1 <- ggplot(to_plot, aes(x = gamma, y = diff)) + 
    geom_boxplot() + 
    ggtitle('Gamma')
  p2 <- ggplot(to_plot, aes(x = eigengene_mode, y = diff)) + 
    geom_boxplot() + 
    ggtitle('Eigengene mode')
  p3 <- ggplot(to_plot, aes(x = ae_standardize, y = diff)) + 
    geom_boxplot() + 
    ggtitle('AE std')
  
  q <- (p2 + p3) / p1
  png(filename = paste0(to_analyze[index], '.png'), 
      width = 2100, height = 2100, res = 300)
  plot(q)
  dev.off()
  
}

# writing
write.csv(tmp, row.names = FALSE, file = 'all_results.csv')

# all cell and zscore, overall good solution
#eigengene_mode all_cells
#ae_standardize zscore
# Norman 0p9 --> 0.119
# Joung2023_random_subsample 1p0 --> 0.0885

# global and zscore_ref, too many negative values
#eigengene_mode global, too many negative values
#ae_standardize zscore_ref
# Norman 0p9 --> 0.0727 or 1p0 --> 0.0842
# Joung2023_random_subsample 1p0 --> 0.0612

# all cell and no scale, overall poor solution (positive small values)
#eigengene_mode all_cells
#ae_standardize zscore
# Norman 0p9 --> 0.063
# Joung2023_random_subsample 1p6 --> 0.1266

# Do the following make sense?
# all cell and zscore_ref 
# global zscore
# global none

