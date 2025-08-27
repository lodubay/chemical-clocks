# Paper Template

This is a template repository for code-heavy papers. The directory structure is:

`src/data` - Data files and model outputs that won't be tracked by git.

`src/extra` - Non-paper plots and output files that aren't tracked by git.

`src/scripts` - Scripts and other code for generating paper figures and output files.

`src/scripts/styles` - matplotlib style files for figure scripts.

`src/scripts/paths.py` - Links to the directory structure using Python's pathlib library.

`src/tex` - LaTeX source directory.

`src/tex/figures` - LaTeX figures, static or generated from scripts. Tracked by git.

`src/tex/output` - Other script output such as tables. Tracked by git.

`src/tex/ms.tex` - LaTeX manuscript file.

`src/tex/references.bib` - Bibliography file.
