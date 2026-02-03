# Basics and requirements
Install `uv` and `llmart`, and download/navigate to this folder.

To understand and run the basic `llmart` developer workflow, run the [notebook](basic_dev_workflow.ipynb) by first launching a `jupyter` server using:
```bash
uv run --with-requirements requirements.txt jupyter server
```

Followed by opening the notebook. Alternatively, you can run the standalone [script](main.py) using:
```bash
uv run --with-requirements requirements.txt python main.py
```
