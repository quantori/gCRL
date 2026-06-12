# Reference Data

This directory contains shared reference data used across multiple analyses in the gCRL package.

## Directory Structure

```
reference/
├── ontologies/           # Biological ontologies
│   └── go-basic.obo     # Gene Ontology (GO) basic format
└── README.md            # This file
```

## Ontologies

### Gene Ontology (GO)

**File**: `ontologies/go-basic.obo`
**Source**: [Gene Ontology Consortium](http://geneontology.org/)
**Format**: OBO (Open Biomedical Ontologies)
**Description**: Basic GO ontology file containing biological process, molecular function, and cellular component terms.

**Update Instructions**:
```bash
# Download latest version
wget http://purl.obolibrary.org/obo/go/go-basic.obo -O ontologies/go-basic.obo
```

## Usage

### In Python Code

The recommended way to access reference data is through the `gcrl.data` module:

```python
from gcrl.data import get_go_obo_path

# Get path to GO OBO file
go_path = get_go_obo_path()
```

### In Functions

Many enrichment functions in `gcrl.grn.enrichment` automatically use these reference files:

```python
from gcrl.grn import compute_go_levels, extract_go_ids_from_terms

# Uses default go-basic.obo from gCRL/data/reference/ontologies/
go_term_levels = compute_go_levels(
    go_ids=your_go_ids,
    namespace="biological_process"
)

# Or specify a custom path if needed
go_term_levels = compute_go_levels(
    go_obo_path="/path/to/custom/go.obo",
    go_ids=your_go_ids,
    namespace="biological_process"
)
```

## Adding New Reference Data

When adding new reference datasets:

1. Create an appropriate subdirectory (e.g., `gene_sets/`, `networks/`)
2. Add the data files
3. Update `src/gcrl/data/__init__.py` with helper functions
4. Document the new data here in this README

## Notes

- Reference data files are **not** included in version control if they are large (>10MB)
- Use `.gitignore` to exclude large files, but document their source and how to obtain them
- Keep reference data separate from experimental/analysis-specific data (use `gCRL/data/real/` or `gCRL/data/example/` for those)
