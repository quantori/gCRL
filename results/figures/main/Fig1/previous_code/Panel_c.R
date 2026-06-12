# Script for contrasting original and permuted repetitions

# set up
rm(list = ls())
library(tidyverse)
library(ggplot2)

# control panel
dataset_name <- 'Norman' 
#dataset_name <- 'Joun2013'
global <- FALSE

# loading files
if(dataset_name == 'Norman'){
  if(global){
    tmp_folder <- '../results/Norman_results/poly_ae_results/partial_mcc_8_communities'
    control_results_file <- file.path(tmp_folder, 'norman_partial_mcc_new_data_MCC_results_8_clusters_control.txt')
    permuted_results_file <- file.path(tmp_folder, 'norman_partial_mcc_new_data_MCC_results_8_clusters_permuted.txt')
    res_folder <- './Norman/Panel_c_global'
  }else{
    tmp_folder <- '../results/Norman_results/poly_ae_results/partial_mcc_7_communities'
    control_results_file <- file.path(tmp_folder, 'norman_partial_mcc_new_data_MCC_results_7_clusters_control_2.txt')
    permuted_results_file <- file.path(tmp_folder, 'norman_partial_mcc_new_data_MCC_results_7_clusters_permuted.txt')
    res_folder <- './Norman/Panel_c'
  }

}
if(dataset_name == 'Joun2013'){
  if(global){
    tmp_folder <- '../results/Joun2013_results/poly_ae_results/partial_mcc_6_communities'
    control_results_file <- file.path(tmp_folder, 'partial_mcc_new_data_MCC_results_6_clusters_control.txt')
    permuted_results_file <- file.path(tmp_folder, 'partial_mcc_new_data_MCC_results_6_clusters_permuted.txt')
    res_folder <- './Joun2013/Panel_c_global'
  }else{
    tmp_folder <- '../results/Joun2013_results/poly_ae_results/partial_MCC_5_communities'
    control_results_file <- file.path(tmp_folder, 'partial_mcc_new_data_MCC_results_5_clusters_control.txt')
    permuted_results_file <- file.path(tmp_folder, 'partial_mcc_new_data_MCC_results_5_clusters_permuted.txt')
    res_folder <- './Joun2013/Panel_c'
  }
}
dir.create(res_folder, showWarnings = FALSE, recursive = TRUE)

# loading data
control_results <- read.table(control_results_file, header = FALSE)[[1]]
permuted_results <- read.table(permuted_results_file, header = FALSE)[[1]]

# t test
res <- t.test(control_results, permuted_results)
# wilcox.test(control_results, permuted_results) # same conclusion
res.var <- var.test(control_results, permuted_results)

# plotting density with t-test results
to_plot <- data.frame(group = c(rep('control', length(control_results)),
                                rep('permuted', length(permuted_results))), 
                      value = c(control_results, permuted_results))
p <- ggplot(data = to_plot, mapping = aes(x = value, fill = group)) + 
  geom_density(alpha = 0.5, color = 'grey') + 
  geom_vline(xintercept = mean(control_results), color = 'red4', 
             linewidth = 0.5, linetype = 'dashed') + 
  geom_vline(xintercept = mean(permuted_results), color = 'blue4', # 'turquoise4', 
             linewidth = 0.5, linetype = 'dashed') + 
  scale_fill_manual(values = c('red4', 'blue4')) + 
  theme_bw() + 
  theme(legend.position = 'bottom') +
  ggtitle(paste0('Difference: ', 
                 round(mean(control_results) - mean(permuted_results), 3), '\n', 
                 't-test p-value: ', round(res$p.value, 3), '\n', 
                 'F-test p-value: ', round(res.var$p.value, 3)))

# saving
svg(filename = file.path(res_folder, 'density_plot.svg'), 
    height = 4, width = 4)
plot(p)
dev.off()
