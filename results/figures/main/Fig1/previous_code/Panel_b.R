# Script for producing heatmaps of TF activities, Z and Z_hat matrices, T-Mex matrices

# set up
rm(list = ls())
library(ComplexHeatmap)
library(latex2exp)
library(reticulate)
library(clue)
library(igraph)
source('partial_mcc_matrix.R')
np <- import("numpy")

# control panel
num_cells <- 1000
color_pool <- c('yellow', 'darkmagenta', 'cyan', 'red3', 'green3', 'blue3',
                'violet', 'orange')
#dataset_name <- 'Norman' 
dataset_name <- 'Joun2013'
global_factor <- TRUE
# loading data
if(dataset_name == 'Norman'){
  
  if(global_factor){
    tmp_folder <- '../results/Norman_results/poly_ae_results/norman_seed_42_partial_mcc_community_global(8)_control'
    res_folder <- './Norman/Panel_b_global'
  }else{
    tmp_folder <- '../results/Norman_results/poly_ae_results/norman_seed_43_partial_mcc_community_7_control'
    res_folder <- './Norman/Panel_b'
  }

  tf_activities_file <- file.path(tmp_folder, 'Partial_MCC_A.npy')
  z_matrix_file <- file.path(tmp_folder, 'Partial_MCC_BX.npy')
  z_hat_matrix_file <- file.path(tmp_folder, 'Partial_MCC_B.npy')
  a_matrix_file <- file.path(tmp_folder, 'Partial_MCC_X.npy')
  
}
if(dataset_name == 'Joun2013'){
  
  if(global_factor){
    tmp_folder <- '../results/Joun2013_results/poly_ae_results/joun_seed_42_partial_mcc_community_global(6)_control'
    res_folder <- './Joun2013/Panel_b_global'
  }else{
    tmp_folder <- '../results/Joun2013_results/poly_ae_results/joun_seed_43_partial_mcc_community_5_control'
    res_folder <- './Joun2013/Panel_b'
  }
  tf_activities_file <- file.path(tmp_folder, 'Partial_MCC_A.npy')
  z_matrix_file <- file.path(tmp_folder, 'Partial_MCC_BX.npy')
  z_hat_matrix_file <- file.path(tmp_folder, 'Partial_MCC_B.npy')
  a_matrix_file <- file.path(tmp_folder, 'Partial_MCC_X.npy')
  
}
dir.create(res_folder, showWarnings = FALSE, recursive = TRUE)

# loading data
loading_npy <- function(file_name){
  tmp <- np$load(file_name, allow_pickle = TRUE)
  tmp <- py_to_r(tmp)
  do.call(rbind, tmp$tolist())
}
tf_activities_matrix <- loading_npy(tf_activities_file)
z_matrix <- loading_npy(z_matrix_file)
z_hat_matrix <- loading_npy(z_hat_matrix_file)
a_matrix <- loading_npy(a_matrix_file)

#### plotting heatmaps ####

# number of factors
num_factors <- dim(tf_activities_matrix)[2]

# choosing the num_cells to display. We rank the cell according to the 
# rank product of their sd between tf_activities and z_matrix
cell_rank_product <- log2(rank(apply(tf_activities_matrix, 1, sd)) + 1) * 
                        log2(rank(apply(z_matrix, 1, sd)) + 1)  
to_select <- order(cell_rank_product, decreasing = TRUE)[1:num_cells]

# creating column annotations for tf_activties and z heatmaps
col_names <- paste0('CF ', 1:num_factors)
col_cols <- color_pool[1:num_factors] # column colors....
names(col_cols) <- col_names
col_annotation <- HeatmapAnnotation(CF = col_names, 
                                    col = list(CF = col_cols), 
                                    show_annotation_name = FALSE,
                                    show_legend = FALSE)

# creating column annotations for and z_hat heatmap
col_names <- paste0('CF ', 1:num_factors)
col_cols <- paste0('grey', (1:num_factors) * 10) # column colors....
names(col_cols) <- col_names
col_annotation_z_hat <- HeatmapAnnotation(CF = col_names, 
                                          col = list(CF = col_cols), 
                                          show_annotation_name = FALSE,
                                          show_legend = FALSE)

# creating the heatmaps
suppressMessages({
tf_activities_hm <- Heatmap(tf_activities_matrix[to_select, ], 
                            column_split = 1:num_factors,
                            column_title = TeX(paste0('$TFA_', 1:num_factors, '$')),
                            show_heatmap_legend = FALSE, 
                            top_annotation = col_annotation,
                            cluster_rows = FALSE, cluster_columns = FALSE)
z_hm <- Heatmap(z_matrix[to_select, ], 
                column_split = 1:num_factors,
                column_title = TeX(paste0('$Z_', 1:num_factors, '$')),
                show_heatmap_legend = FALSE, 
                top_annotation = col_annotation,
                cluster_rows = FALSE, cluster_columns = FALSE)
z_hat_hm <- Heatmap(z_hat_matrix[to_select, ], 
                    column_split = 1:num_factors,
                    column_title = TeX(paste0('$\\hat{Z}_', 1:num_factors, '$')),
                    show_heatmap_legend = FALSE, 
                    top_annotation = col_annotation_z_hat,
                    cluster_rows = FALSE, cluster_columns = FALSE)
})

# plottig the heatmaps
png(filename = file.path(res_folder, 'heatmap_z.png'), 
    height = 2100, width = 2100, res = 300)
plot(z_hm)
dev.off()
svg(filename = file.path(res_folder, 'heatmap_z.svg'), height = 7, width = 7)
plot(z_hm)
dev.off()
png(filename = file.path(res_folder, 'heatmap_z_hat.png'), 
    height = 2100, width = 2100, res = 300)
plot(z_hat_hm)
dev.off()
svg(filename = file.path(res_folder, 'heatmap_z_hat.svg'), height = 7, width = 7)
plot(z_hat_hm)
dev.off()
png(filename = file.path(res_folder, 'heatmap_tf_activities.png'), 
    height = 2100, width = 2100, res = 300)
plot(tf_activities_hm)
dev.off()
svg(filename = file.path(res_folder, 'heatmap_tf_activities.svg'), height = 7, width = 7)
plot(tf_activities_hm)
dev.off()

#### plotting the A matrix ####

# Labels and colors
row_labels <- paste0("hat_Z_", 1:num_factors)
col_labels <- paste0("Z_", 1:num_factors)
row_colors <- paste0('grey', (1:num_factors) * 10)
col_colors <- color_pool

# Edge list
edges <- which(a_matrix != 0, arr.ind = TRUE)
edge_list <- data.frame(
  from = row_labels[edges[, 1]],
  to   = col_labels[edges[, 2]],
  weight = a_matrix[edges]
)

# Graph
g <- graph_from_data_frame(edge_list, directed = FALSE, vertices = c(row_labels, col_labels))
V(g)$type <- c(rep(0, length(row_labels)), rep(1, length(col_labels)))
V(g)$color <-c(row_colors[1:length(row_labels)], col_colors[1:length(col_labels)])
V(g)$shape <- "square"
V(g)$size <- 30

# Manual layout
y_pos <- seq(1, 0, length.out = num_factors)
row_pos <- cbind(x = -1, y = y_pos)
col_pos <- cbind(x = 1,  y = y_pos)
layout_manual <- rbind(row_pos, col_pos)
rownames(layout_manual) <- c(row_labels, col_labels)

# Edge aesthetics
weights <- E(g)$weight
weights_norm <- (abs(weights) - min(abs(weights))) / (max(abs(weights)) - min(abs(weights)) + 1e-9)
E(g)$width <- 1 + 4 * weights_norm
E(g)$color <- ifelse(weights > 0, rgb(1, 0.1, 0.1, alpha = weights_norm),
                     rgb(0.1, 0.1, 1, alpha = weights_norm))

# plotting!
svg(filename = file.path(res_folder, 'a_matrix.svg'), 
    height = 4 + num_factors / 5, width = 4 + num_factors / 5)

# Use plot.igraph and capture layout coordinates
igraph::plot.igraph(
  g,
  layout = layout_manual[V(g)$name, ],
  vertex.label = NA,
  edge.width = E(g)$width,
  edge.color = E(g)$color,
  margin = 0.2
)
coords <- layout_manual[V(g)$name, ]
coords[ , 2] <- (2 * (coords[ , 2] - min(coords[ , 2])) / 
                   (max(coords[ , 2]) - min(coords[ , 2]))) - 1 

# Add labels manually, aligned with rescaled node positions
offset_x <- 0.4  # fine-tune as needed
for (i in seq_along(V(g))) {
  name <- V(g)$name[i]
  x <- coords[i, 1]
  y <- coords[i, 2]
  if (V(g)$type[i] == 0) {
    name <- gsub('hat_', 'hat{', name, fixed = TRUE)
    name <- gsub('_', '}_', name, fixed = TRUE)
    text(x - offset_x, y, labels = TeX(paste0('$\\', name, '$')))
  } else {
    text(x + offset_x, y, labels = TeX(paste0('$', name, '$')))
  }
}

# end plotting
dev.off()

#### T-Mex analysis ####

# computing the T-Mex statistics
p_mcc_res <- partial_mcc_matrix(tf_activities_matrix, z_matrix)
mean(diag(p_mcc_res)) 

# creating annotations for the T-Mex heatmap
col_names <- paste0('CF ', 1:num_factors)
col_cols <- color_pool[1:num_factors] # column colors....
names(col_cols) <- col_names
col_annotation <- HeatmapAnnotation(CF = col_names, 
                                    col = list(CF = col_cols), 
                                    show_annotation_name = FALSE,
                                    show_legend = FALSE)
row_annotation <- HeatmapAnnotation(CF = col_names, 
                                    col = list(CF = col_cols), 
                                    show_annotation_name = FALSE,
                                    show_legend = FALSE, which = 'row')

# Identify for each row the index of the max absolute value
row_max_idx <- apply(abs(p_mcc_res), 1, which.max)

# Plot with annotation only at max locations
ht <- Heatmap(p_mcc_res,
              row_split = 1:num_factors,
              row_title = TeX(paste0('$TFA_', 1:num_factors, '$')),
              column_split = 1:num_factors,
              column_title = TeX(paste0('$Z_', 1:num_factors, '$')),
              cluster_rows = FALSE,
              cluster_columns = FALSE,
              show_row_names = FALSE,
              show_column_names = FALSE,
              left_annotation = row_annotation,
              top_annotation = col_annotation,
              show_heatmap_legend = FALSE,
              cell_fun = function(j, i, x, y, w, h, fill) {
                if (j == row_max_idx[i]) {
                  grid.text(sprintf("%.3f", p_mcc_res[i, j]), x, y, 
                            gp = gpar(fontsize = 10, fontface = "bold"))
                }
              })

# saving
svg(filename = file.path(res_folder, 'tmex_z.svg'), 
    height = 4 + num_factors / 5, width = 4 + num_factors / 5)
draw(ht, column_title = paste0('T-Mex = ', round(mean(diag(p_mcc_res)), 3)), 
     column_title_side = 'bottom',
     column_title_gp = gpar(fontsize = 20))
dev.off()

# optimal in terms of permutation partial MCC
res <- optimal_partial_mcc_matrix(tf_activities_matrix, z_hat_matrix)

# creating annotations for the T-Mex heatmap
col_names <- paste0('CF ', 1:num_factors)
col_cols <- paste0('grey', 1:num_factors * 10)[res$permutation] # we permute here!
row_cols <- color_pool[1:num_factors] 
names(row_cols) <- names(col_cols) <- col_names
col_annotation <- HeatmapAnnotation(CF = col_names, 
                                    col = list(CF = col_cols), 
                                    show_annotation_name = FALSE,
                                    show_legend = FALSE)
row_annotation <- HeatmapAnnotation(CF = col_names, 
                                    col = list(CF = row_cols), 
                                    show_annotation_name = FALSE,
                                    show_legend = FALSE, which = 'row')

# Identify for each row the index of the max absolute value
row_max_idx <- apply(abs(res$matrix), 1, which.max)

# Plot with annotation only at max locations
column_titles <- TeX(paste0('$\\hat{Z}^*_', 1:num_factors, '$')[res$permutation]) # we permute here!,
ht <- Heatmap(res$matrix,
              row_split = 1:num_factors,
              row_title = TeX(paste0('$TFA_', 1:num_factors, '$')),
              column_split = 1:num_factors,
              column_title = column_titles,
              cluster_rows = FALSE,
              cluster_columns = FALSE,
              show_row_names = FALSE,
              show_column_names = FALSE,
              left_annotation = row_annotation,
              top_annotation = col_annotation,
              show_heatmap_legend = FALSE,
              cell_fun = function(j, i, x, y, w, h, fill) {
                if (j == row_max_idx[i]) {
                  grid.text(sprintf("%.3f", res$matrix[i, j]), x, y, 
                            gp = gpar(fontsize = 10, fontface = "bold"))
                }
              })

# saving
svg(filename = file.path(res_folder, 'tmex_z_hat.svg'), 
    height = 4 + num_factors / 5, width = 4 + num_factors / 5)
draw(ht, column_title = paste0('T-Mex = ', round(res$mean_partial_mcc, 3)), 
     column_title_side = 'bottom',
     column_title_gp = gpar(fontsize = 20))
dev.off()


