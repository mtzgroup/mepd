# Installation

Install from GitHub:

```bash
pip install "git+https://github.com/mtzgroup/mepd.git"
```

For development:

```bash
git clone https://github.com/mtzgroup/mepd.git
cd mepd
uv sync
uv run pytest
```

Electronic-structure backends may require separate credentials or local program
installations. For ChemCloud-backed runs, configure ChemCloud credentials before
submitting calculations.
