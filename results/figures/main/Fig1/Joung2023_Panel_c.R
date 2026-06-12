# Panel_c for Joung2023: density plot contrasting real vs permuted T-Mex scores

rm(list = ls())
library(tidyverse)
library(ggplot2)
library(reticulate)
np <- import("numpy")

# ── Control panel ──────────────────────────────────────────────
pct        <- 50
gamma      <- '1p0'

mcc_folder <- file.path('../../../../results/real/Joung2023',
                         paste0('partial_mcc_permutation_', pct, 'pct_gamma', gamma))
res_folder <- './Joung2023/Panel_c'
dir.create(res_folder, showWarnings = FALSE, recursive = TRUE)

# ── Load scores ────────────────────────────────────────────────
# scores_real: (n_real_seeds,)               — one T-Mex score per seed
# scores_perm: (n_permutations, n_perm_seeds) — averaged across seeds → (n_permutations,)
load_npy_vec <- function(path) {
  arr <- py_to_r(np$load(path, allow_pickle = TRUE))
  as.vector(arr)
}

load_npy_mat <- function(path) {
  arr <- py_to_r(np$load(path, allow_pickle = TRUE))
  if (is.list(arr)) arr <- do.call(rbind, arr)
  as.matrix(arr)
}

scores_real <- load_npy_vec(file.path(mcc_folder, 'scores_real.npy'))
scores_perm_mat <- load_npy_mat(file.path(mcc_folder, 'scores_perm.npy'))
# Average across n_perm_seeds → one value per permutation
scores_perm <- rowMeans(scores_perm_mat)

cat('Real scores  — n:', length(scores_real),
    ' mean:', round(mean(scores_real), 4),
    ' sd:', round(sd(scores_real), 4), '\n')
cat('Perm scores  — n:', length(scores_perm),
    ' mean:', round(mean(scores_perm), 4),
    ' sd:', round(sd(scores_perm), 4), '\n')

# ── Statistical tests ──────────────────────────────────────────
res     <- t.test(scores_real, scores_perm)
res_var <- var.test(scores_real, scores_perm)

cat('t-test p-value:    ', round(res$p.value,     6), '\n')
cat('F-test p-value:    ', round(res_var$p.value,  6), '\n')
cat('Mean difference:   ', round(mean(scores_real) - mean(scores_perm), 4), '\n')

# ── Density plot ───────────────────────────────────────────────
to_plot <- data.frame(
  group = c(rep('real',      length(scores_real)),
            rep('permuted',  length(scores_perm))),
  value = c(scores_real, scores_perm))

p <- ggplot(data = to_plot, mapping = aes(x = value, fill = group)) +
  geom_density(alpha = 0.5, color = 'grey') +
  geom_vline(xintercept = mean(scores_real),  color = 'red4',
             linewidth = 0.5, linetype = 'dashed') +
  geom_vline(xintercept = mean(scores_perm), color = 'blue4',
             linewidth = 0.5, linetype = 'dashed') +
  scale_fill_manual(values = c(real = 'red4', permuted = 'blue4')) +
  theme_bw() +
  theme(legend.position = 'bottom') +
  xlab('T-Mex score') +
  ggtitle(paste0(
    #'Difference: ',   round(mean(scores_real) - mean(scores_perm), 3), '\n',
    #'t-test p-value: ', round(res$p.value,      3), '\n',
    #'F-test p-value: ', round(res_var$p.value,   3))
    'Difference: ',   round(mean(scores_real) - mean(scores_perm), 3),
    ', t-test p-value: ', round(res$p.value,      3)
    ))

svg(file.path(res_folder, 'density_plot.svg'), height = 4, width = 4)
plot(p)
dev.off()

cat('Saved to', res_folder, '\n')
