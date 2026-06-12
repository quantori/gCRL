
# Makefile
.PHONY: init clean

# Directory tree for the gCRL project
INIT_DIRS = \
  src/gcrl/{data,grn,models,training,alignment,evaluation,utils} \
  scripts \
  configs \
  simulation/code/SERGIO \
  simulation/notebooks \
  simulation/generated_data \
  data/{example,real,simulated} \
  notebooks/{00_data_preprocessing,10_modeling_gcrl_ae,20_modeling_gcrl_vae,30_alignment,40_generalization,90_figures_for_paper} \
  results/{generalization/{zero_shot_single,double_perturb},mcc_alignment,ablations,figures/{main,supplementary},tables} \
  tests/data \
  docs/api

init:
	@echo "Creating directory tree..."
	@mkdir -p $(INIT_DIRS)
	@touch src/gcrl/__init__.py
	@for d in $(INIT_DIRS); do touch $$d/.gitkeep; done
	@echo "Done. Next:"
	@echo "  python -m venv .venv && . .venv/bin/activate && pip install -e ."

clean:
	@find . -name ".gitkeep" -delete
