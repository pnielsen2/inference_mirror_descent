# Makefile for building paper, presentation, and poster
# Run from project root directory
#
# Usage:
#   make paper         - Build paper.pdf
#   make presentation  - Build presentation.pdf
#   make poster        - Build poster.pdf
#   make all           - Build all documents
#   make clean         - Remove build artifacts
#   make view-paper    - Build and open paper.pdf
#
# Figures are automatically picked up from figures/ directory
#
# Style files:
#   - paper/icml2026-official/  : Official ICML 2026 style (DO NOT EDIT)
#   - paper/style/              : Helper packages only (cleveref, forloop, etc.)
# The official ICML style takes precedence.

PAPER_DIR = paper
BUILD_DIR = $(PAPER_DIR)/build
ICML_STYLE = $(PAPER_DIR)/icml2026-official
HELPER_STYLE = $(PAPER_DIR)/style
FIGURES_DIR = figures

# Set TEXINPUTS: official ICML first, then helpers
export TEXINPUTS := $(ICML_STYLE):$(HELPER_STYLE):$(TEXINPUTS)

# LaTeX commands
PDFLATEX = pdflatex -interaction=nonstopmode -output-directory=$(BUILD_DIR)
BIBTEX = bibtex

.PHONY: all paper presentation poster clean view-paper view-presentation view-poster

all: paper presentation poster

# Paper build
paper: $(BUILD_DIR)/paper.pdf
	@cp $(BUILD_DIR)/paper.pdf $(PAPER_DIR)/paper.pdf
	@echo "Paper built: $(PAPER_DIR)/paper.pdf"

$(BUILD_DIR)/paper.pdf: $(PAPER_DIR)/paper.tex $(PAPER_DIR)/refs.bib $(wildcard $(FIGURES_DIR)/*.png) | $(BUILD_DIR)
	@echo "Building paper..."
	@cd $(PAPER_DIR) && TEXINPUTS=icml2026-official:style:$(TEXINPUTS) pdflatex -interaction=nonstopmode -output-directory=build paper.tex || true
	@cd $(PAPER_DIR)/build && BIBINPUTS=..:$(BIBINPUTS) BSTINPUTS=../icml2026-official:$(BSTINPUTS) bibtex paper || true
	@cd $(PAPER_DIR) && TEXINPUTS=icml2026-official:style:$(TEXINPUTS) pdflatex -interaction=nonstopmode -output-directory=build paper.tex || true
	@cd $(PAPER_DIR) && TEXINPUTS=icml2026-official:style:$(TEXINPUTS) pdflatex -interaction=nonstopmode -output-directory=build paper.tex || true
	@test -f $(BUILD_DIR)/paper.pdf || (echo "ERROR: paper.pdf not generated" && exit 1)

# Presentation build
presentation: $(BUILD_DIR)/presentation.pdf
	@cp $(BUILD_DIR)/presentation.pdf $(PAPER_DIR)/presentation.pdf
	@echo "Presentation built: $(PAPER_DIR)/presentation.pdf"

$(BUILD_DIR)/presentation.pdf: $(PAPER_DIR)/presentation.tex $(wildcard $(FIGURES_DIR)/*.png) | $(BUILD_DIR)
	@echo "Building presentation..."
	@cd $(PAPER_DIR) && TEXINPUTS=style:$(TEXINPUTS) pdflatex -interaction=nonstopmode -output-directory=build presentation.tex || true
	@cd $(PAPER_DIR) && TEXINPUTS=style:$(TEXINPUTS) pdflatex -interaction=nonstopmode -output-directory=build presentation.tex || true
	@test -f $(BUILD_DIR)/presentation.pdf || (echo "ERROR: presentation.pdf not generated" && exit 1)

# Poster build  
poster: $(BUILD_DIR)/poster.pdf
	@cp $(BUILD_DIR)/poster.pdf $(PAPER_DIR)/poster.pdf
	@echo "Poster built: $(PAPER_DIR)/poster.pdf"

$(BUILD_DIR)/poster.pdf: $(PAPER_DIR)/poster.tex $(wildcard $(FIGURES_DIR)/*.png) | $(BUILD_DIR)
	@echo "Building poster..."
	@cd $(PAPER_DIR) && TEXINPUTS=style:$(TEXINPUTS) pdflatex -interaction=nonstopmode -output-directory=build poster.tex || true
	@cd $(PAPER_DIR) && TEXINPUTS=style:$(TEXINPUTS) pdflatex -interaction=nonstopmode -output-directory=build poster.tex || true
	@test -f $(BUILD_DIR)/poster.pdf || (echo "ERROR: poster.pdf not generated" && exit 1)

# Create build directory
$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)

# Clean build artifacts
clean:
	@rm -rf $(BUILD_DIR)
	@rm -f $(PAPER_DIR)/*.pdf
	@echo "Cleaned build artifacts"

# View commands (platform-dependent, adjust as needed)
view-paper: paper
	@echo "PDF ready at: $(PAPER_DIR)/paper.pdf"

view-presentation: presentation
	@echo "PDF ready at: $(PAPER_DIR)/presentation.pdf"

view-poster: poster
	@echo "PDF ready at: $(PAPER_DIR)/poster.pdf"

# Convenience: rebuild figures then paper
figures:
	@python scripts/plot_training_curves.py
	@echo "Figures regenerated in $(FIGURES_DIR)/"

paper-fresh: figures paper
