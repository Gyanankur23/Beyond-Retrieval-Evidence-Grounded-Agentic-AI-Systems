#!/bin/sh
cd "$(dirname "$0")" && pdflatex -interaction=nonstopmode manuscript.tex >/dev/null && bibtex manuscript >/dev/null && pdflatex -interaction=nonstopmode manuscript.tex >/dev/null && pdflatex -interaction=nonstopmode manuscript.tex >/dev/null && rm -f *.aux *.log *.out *.bbl *.blg && echo "built paper/manuscript.pdf"
