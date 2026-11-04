# LogP Visualiser

Calculate the lipophilicity (Wildman–Crippen LogP) and topological polar surface area
(Ertl TPSA) of a molecule with RDKit, and see each atom's contribution drawn on the
structure. Switch between:

- **LogP map**: red atoms increase LogP (lipophilic), blue atoms decrease it (hydrophilic)
- **TPSA map**: white → green, darker green = larger polar surface contribution (only N/O, optionally S/P)
- **BOILED-Egg** (Daina & Zoete, 2016): WLOGP vs TPSA plot predicting passive
  gastrointestinal absorption (white) and blood–brain barrier permeation (yolk)

Stronger colour = larger contribution.

Molecules can be **drawn** (Ketcher editor), typed as **SMILES / InChI**, or **uploaded**
(`.mol`, `.sdf`, `.mol2`, `.pdb`, `.smi`). Hydrogen contributions are added to the heavy
atom they are bonded to, so the per-atom values sum to the total LogP.

## Run it locally

Requires Python 3.10+.

```bash
python -m venv venv
venv\Scripts\pip install -r requirements.txt      # macOS/Linux: venv/bin/pip ...
venv\Scripts\python -m streamlit run app.py       # macOS/Linux: venv/bin/python ...
```

The app opens in your browser at http://localhost:8501.

## Command line

```bash
python logp_core.py "CC(=O)Nc1ccc(O)cc1" -o paracetamol.png
python logp_core.py molecule.sdf --style heatmap --vmax 1.0 -o out.png
python logp_core.py "CC(=O)Nc1ccc(O)cc1" --property tpsa -o tpsa.png --egg egg.png
```

## Files

| File | Purpose |
|------|---------|
| `app.py` | Streamlit web interface |
| `logp_core.py` | Parsing, LogP/TPSA contributions and image rendering (also a CLI) |
| `boiled_egg.py` | BOILED-Egg ellipses, prediction and plot |
| `packages.txt` | System libraries for Streamlit Community Cloud |
| `requirements.txt` | Python dependencies |
