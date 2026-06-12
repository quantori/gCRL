# Script for performing enrichment analysis on TF communities

# set up
rm(list = ls())
library(tidyverse)
library(ggplot2)
library(clusterProfiler)
library(org.Hs.eg.db)
# library(reticulate)
# virtualenv_install(
#   envname = 'r-reticulate',
#   packages = 'pandas')
# virtualenv_install(
#   envname = 'r-reticulate',
#   packages = 'pyarrow')
# pd <- import('pandas')

# control panel
dataset_name <- 'Norman' 
#dataset_name <- 'Joun2013'
if(dataset_name == 'Norman'){
  tmp_folder <- '../results/Norman_results/GRN'
  tfs_file <- file.path(tmp_folder, 'tf_communities_step_10pct.csv')
  res_folder <- './Norman/Panel_d'
}
if(dataset_name == 'Joun2013'){
  tmp_folder <- '../results/Joun2013_results/GRN'
  tfs_file <- file.path(tmp_folder, 'tf_communities_step_30pct.csv')
  res_folder <- './Joun2013/Panel_d'
}
# all_tfs_file <- '../../../../data/hg38_TFinfo_dataframe_gimmemotifsv5_fpr2_threshold_10_20210630.parquet'
dir.create(res_folder, showWarnings = FALSE, recursive = TRUE)

# loading data
tfs <- read.csv(tfs_file)
# all_tfs <- pd$read_parquet(all_tfs_file)
# all_tfs_names <- colnames(all_tfs)
# all_tfs_names <- setdiff(all_tfs_names, c('peak_id', 'gene_short_name'))
# head(all_tfs_names)
# all(tfs$tf %in% all_tfs_names)

# looping over communities
for(i in unique(tfs$community)){
  
  # transforming tf ids
  translated_ids <- bitr(tfs$tf[tfs$community == i], 
                         fromType = 'SYMBOL', toType = 'ENTREZID', 
                         OrgDb = org.Hs.eg.db)
  translated_ids <- unique(translated_ids$ENTREZID)
  
  # enrichment analysis
  resGO <- enrichGO(gene = tfs$tf[tfs$community == i], 
                  OrgDb = org.Hs.eg.db, minGSSize = 20, maxGSSize = 200,
                  keyType = 'SYMBOL', pvalueCutoff = 1) # universe = all_tfs_names,
  resKEGG <- enrichKEGG(gene = translated_ids, organism = 'hsa', 
                        minGSSize = 20, maxGSSize = 200,
                        keyType = 'ncbi-geneid', pvalueCutoff = 1) # universe = all_tfs_names,
  
  # saving results
  write.csv(resGO@result, file = file.path(res_folder, 
                                         paste0('enrichGO_', i, '.csv')))
  write.csv(resKEGG@result, file = file.path(res_folder, 
                                           paste0('enrichKEGG_', i, '.csv')))
  
  # dot plots
  p <- dotplot(resGO)
  svg(filename = file.path(res_folder, paste0('enrichGO_', i, '.svg')), 
      height = 7, width = 6)
  plot(p)
  dev.off()
  p <- dotplot(resKEGG)
  svg(filename = file.path(res_folder, paste0('enrichKEGG_', i, '.svg')), 
      height = 7, width = 6)
  plot(p)
  dev.off()
  
}
