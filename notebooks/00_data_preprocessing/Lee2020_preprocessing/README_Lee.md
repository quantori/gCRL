# gCRL - Lee experiment (CVAE)

This repo documents and reproduces our conditional VAE experiments on the Lee dataset, modeling differentiation outcomes under TF perturbations across MAIT, NKT and γδT lineages.


# 1) Data

AnnData
We had the initial data file received from the paper authors (https://doi.org/10.1038/s41467-020-18155-8, https://zenodo.org/records/16777767), added cluster annotation, and created .h5ad file MAIT_NKT_gdT_paperlabels_exact.h5ad  
https://drive.google.com/file/d/1ypPvYVF2ohfA11YELF3yf4Y-RoItxX_v/view?usp=drive_link

Initial data overview notebook: gCRL/notebooks/00_data_preprocessing/Lee/00_data_overview.ipynb 

Preparation for CVAE training:
File: gCRL/data/real/Lee/lee_panel.h5ad 
Preprocessing of the adata file for training: gCRL/notebooks/00_data_preprocessing/Lee/02_data_preprocessing_Lee.ipynb
adata.X — log1p(norm) expression, shape (n_cells, n_genes)


**obs**


cell_type ∈ {MAIT, NKT, γδT}

intervention ∈ {unperturbed, Tbx21, Rorc}

set ∈ {training, test}
 (training contains controls per cell type)

ct_int: "{cell_type}|{intervention}"


**var**


kind ∈ {TF, TG} (163 TFs; ~25.8k TGs)

community: TF community ID from multiplex GRN,  TGs set to -1

panels used in our figures:
in_panel_NKT_H100 
in_panel_MAIT_H100_Ccr7 
in_panel_gdT_boost2
in_panel_union (union of the above)


**uns**
 precomputed PCA/UMAP for quick inspection 


# 2) Gene panels
Notebook for panel set choosing: gCRL/notebooks/00_data_preprocessing/Lee/additional/02_feature_sets_selection.ipynb 
UNION
Folder: https://drive.google.com/drive/folders/136tyErFNaxFYx5PUyBmx4P7dojEVfjkN?usp=drive_link 


For each MAIT / NKT / gdT is var_names.txt - lineage-specific panels:
NKT: F_+H100  - base + top-100 HVGs 
MAIT: F_+H100 ∪ {Ccr7} 
γδT: boost2 -boosted HVGs + lineage markers


# 3)GRN
We compared:
Pooled GRN: one merged network,  simpler, but blurs lineage specificity. 

Notebook:  gCRL/notebooks/00_data_preprocessing/Lee/additional/01_GRN_pooled.ipynb

and 

Multiplex GRN (80% stability): aggregate across experiments/conditions, keep robust edges, chosen for TF set & TF community IDs. 

Notebook: gCRL/notebooks/00_data_preprocessing/Lee/01_GRN_multiplex_leiden.ipynb
TF communities are stored in ad.var["community"] for TF genes (TG = -1)

# 4) Training regimes
Train on MAIT controls + finals (MAIT1 / MAIT17).

Zero-shot NKT: including NKTp controls only (there is no NKT finals in training).

γδT: optionally held out (it is not used for training, for completeness in evaluation).
