# CD704 Machine Learning

Coursework workspace for the University of Brighton MSc Data Analytics
machine learning module.

## Project layout

- `week-02/`, `week-03/`, etc. contain that week's exercises.
  - `notebooks/` is for exploratory Jupyter notebooks.
  - `scripts/` is for reusable Python scripts.
- `resources/` contains shared datasets, references, and resource notes.
- `requirements.txt` lists the Python packages used by the notebooks and
  scripts.

## Getting started

Create and activate the local virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m ipykernel install --user --name cd704-ml --display-name "Python (CD704 ML)"
```

In VS Code, select the `Python (CD704 ML)` kernel when opening a notebook.

## Working conventions

- Keep week-specific work in its corresponding `week-XX/` directory.
- Keep large or private datasets out of Git; place them in a local `data/`
  directory and document their source in `resources/RESOURCES.md`.
- Prefer one notebook per exercise or analysis, with a descriptive filename.

