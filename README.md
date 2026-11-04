# LogP Visualiser

Calculate the lipophilicity (Wildman–Crippen LogP, via RDKit) of a molecule and see
each atom's contribution drawn on the structure:

- **Red** atoms increase LogP (lipophilic)
- **Blue** atoms decrease LogP (hydrophilic)
- Stronger colour = larger contribution

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
```

## Files

| File | Purpose |
|------|---------|
| `app.py` | Streamlit web interface |
| `logp_core.py` | Parsing, LogP contributions and image rendering (also a CLI) |
| `requirements.txt` | Python dependencies |
