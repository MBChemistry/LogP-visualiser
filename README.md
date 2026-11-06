# LogP Visualiser

Calculate the lipophilicity (Wildman–Crippen LogP) and topological polar surface area
(Ertl TPSA) of a molecule with RDKit, and see each atom's contribution drawn on the
structure. Switch between:

- **LogP map**: red atoms increase LogP (lipophilic), blue atoms decrease it (hydrophilic)
- **TPSA map**: white → green, darker green = larger polar surface contribution (only N/O, optionally S/P)
- **BOILED-Egg** (Daina & Zoete, 2016): WLOGP vs TPSA plot predicting passive
  gastrointestinal absorption (white) and blood–brain barrier permeation (yolk)

Stronger colour = larger contribution.

**Fsp3**, the fraction of carbons that are sp3-hybridised
([Wei et al., 2020](https://pubmed.ncbi.nlm.nih.gov/32712310/)), is shown with the other
metrics and included in the exported results.

A **Lipinski rule-of-five** table checks MW ≤ 500, LogP ≤ 5, H-bond donors ≤ 5 and
H-bond acceptors ≤ 10, giving pass/fail per rule and overall (pass = at most one violation).

**Batch mode** (sidebar → Mode) analyses many molecules at once from a pasted SMILES
list or an `.sdf` / `.smi` / `.csv` file. Each molecule gets a compound number; pick a
compound from the dropdown to see its maps and Lipinski table, view every compound on
one numbered BOILED-Egg plot, and download a results table (CSV) or a ZIP containing
each compound's maps, per-atom CSVs, Lipinski table and egg plot.

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
| `boiled_egg.py` | BOILED-Egg ellipses, prediction and plots (single and multi-compound) |
| `batch.py` | Batch mode: compound numbering, results table and ZIP export |
| `packages.txt` | System libraries for Streamlit Community Cloud |
| `requirements.txt` | Python dependencies |
