# GAHIB Paper Build
# Usage: make        (build PDF)
#        make clean   (remove build artifacts)
#        make figures (regenerate all figures from source data)

MAIN = gahib_paper
LATEX = pdflatex
BIBTEX = bibtex
PYTHON ?= .venv/bin/python

.PHONY: all clean figures

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex references.bib figures/*.pdf
	$(LATEX) $(MAIN)
	$(BIBTEX) $(MAIN)
	$(LATEX) $(MAIN)
	$(LATEX) $(MAIN)

clean:
	rm -f $(MAIN).{aux,bbl,blg,log,out,pdf,toc,lof,lot,fls,fdb_latexmk}

# Regenerate all figures from source data
figures:
	cd .. && $(PYTHON) -m gahib.viz.fig_overview
	cd .. && $(PYTHON) -m gahib.viz.fig_architecture
	cd .. && $(PYTHON) -m gahib.viz.fig_dataset_taxonomy
	cd .. && $(PYTHON) -m gahib.viz.run_all_visualizations
	cd .. && $(PYTHON) -m gahib.viz.fig_downstream_analysis
	cd .. && $(PYTHON) experiments/run_interpretation.py --figures-only
