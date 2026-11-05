QUARTO ?= quarto
QUARTO_PYTHON ?= .venv/bin/python
QUARTO_CMD = QUARTO_PYTHON=$(QUARTO_PYTHON) $(QUARTO)
CURRENT_DIR := $(shell pwd)

.PHONY: .uv
.uv:
	@uv -V || echo 'Please install uv: https://docs.astral.sh/uv/getting-started/installation/'

.PHONY: jupyter
jupyter: .uv
	uv run jupyter lab

.PHONY: lint
lint:
	uvx ruff check
	uvx ty check

.PHONY: html
html: pdf
	$(QUARTO_CMD) render --to html

.PHONY: pdf
pdf:
	$(QUARTO_CMD) render mlp.ipynb --to pdf --output-dir $(CURRENT_DIR)
	@if grep -q '<title>Redirect to mlp.html</title>' $(CURRENT_DIR)/index.html 2>/dev/null; then rm $(CURRENT_DIR)/index.html; fi

.PHONY: preview
preview: pdf
	$(QUARTO_CMD) preview --to html

.PHONY: check
check:
	$(QUARTO_CMD) check

.PHONY: clean
clean:
	rm -rf httdocs .quarto _freeze