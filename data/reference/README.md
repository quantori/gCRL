# Reference Data

This directory contains shared reference data used across multiple analyses.

## Contents

### `ontologies/go-basic.obo`

Gene Ontology (GO) basic OBO file. Used by `gcrl.grn.enrichment` functions
(`compute_go_levels`, `run_ora_for_clusters`, etc.) to assign GO term hierarchy
levels and filter enrichment results.

**Source:** [Gene Ontology Consortium](http://geneontology.org/)

To update to the latest release:

```bash
wget http://purl.obolibrary.org/obo/go/go-basic.obo -O ontologies/go-basic.obo
```
