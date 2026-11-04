"""
Batch mode: analyse many molecules at once and package the results.

Each molecule gets a compound number (1, 2, 3, ...) in input order. That number
is shown in the app, on the BOILED-Egg markers, in the results table and in the
file names of the downloadable ZIP.
"""

from __future__ import annotations

import csv
import io
import re
import zipfile
from dataclasses import dataclass

from rdkit import Chem

import boiled_egg
import logp_core as core


@dataclass
class Compound:
    number: int
    name: str
    smiles: str          # canonical SMILES (used to rebuild the molecule)

    @property
    def label(self) -> str:
        return f"Cpd {self.number}" + (f" – {self.name}" if self.name else "")

    @property
    def mol(self) -> Chem.Mol:
        return Chem.MolFromSmiles(self.smiles)


def make_compounds(named_mols: list[tuple[str, Chem.Mol]]) -> list[Compound]:
    """Number molecules 1..N in input order."""
    out = []
    for i, (name, mol) in enumerate(named_mols, start=1):
        smi = Chem.MolToSmiles(mol)
        out.append(Compound(number=i, name=name if name and name != smi else "", smiles=smi))
    return out


def analyse(cpd: Compound) -> dict:
    """All whole-molecule results for one compound (one row of the results table)."""
    s = core.molecule_summary(cpd.mol)
    _, ro5 = core.lipinski_rule_of_five(s)
    egg = boiled_egg.predict(s["LogP"], s["TPSA"])
    return {
        "Cpd": cpd.number,
        "Name": cpd.name,
        "SMILES": cpd.smiles,
        "MW": round(s["MW"], 2),
        "LogP (WLOGP)": round(s["LogP"], 2),
        "TPSA (Å²)": round(s["TPSA"], 2),
        "HBD": s["HBD"],
        "HBA": s["HBA"],
        "Rotatable bonds": s["RotB"],
        "Ro5 violations": ro5["violations"],
        "Lipinski Ro5": "Pass" if ro5["pass"] else "Fail",
        "GI absorption": "High" if egg["hia"] else "Low",
        "BBB permeant": "Yes" if egg["bbb"] else "No",
        "Egg region": egg["region"],
    }


def results_table(compounds: list[Compound]) -> list[dict]:
    return [analyse(c) for c in compounds]


def _csv(rows: list[dict]) -> str:
    if not rows:
        return ""
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue()


def _safe(name: str, maxlen: int = 40) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")
    return s[:maxlen]


def build_zip(compounds: list[Compound], rows: list[dict], render_kwargs: dict,
              tpsa_include_sp: bool = False, progress=None) -> bytes:
    """
    ZIP layout:
      summary_results.csv
      boiled_egg_all_compounds.png
      cpd_001_<name>/  logp_map.png, tpsa_map.png, logp_contributions.csv,
                       tpsa_contributions.csv, lipinski_ro5.csv, boiled_egg.png
    `render_kwargs` are passed to core.render_figure (style, size, ...). Use
    'vmax_logp' / 'vmax_tpsa' keys for the fixed colour scales (None = relative).
    `progress(i, n)` is called after each compound.
    """
    kw = dict(render_kwargs)
    vmax_logp = kw.pop("vmax_logp", None)
    vmax_tpsa = kw.pop("vmax_tpsa", None)
    width = max(3, len(str(len(compounds))))

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("summary_results.csv", _csv(rows))
        z.writestr("boiled_egg_all_compounds.png", boiled_egg.plot_many(
            [(r["Cpd"], r["LogP (WLOGP)"], r["TPSA (Å²)"]) for r in rows]))

        for i, (cpd, row) in enumerate(zip(compounds, rows), start=1):
            folder = f"cpd_{cpd.number:0{width}d}" + (f"_{_safe(cpd.name)}" if cpd.name else "")
            mol = cpd.mol

            for res, vmax, stem in (
                (core.compute_logp_contributions(mol), vmax_logp, "logp"),
                (core.compute_tpsa_contributions(mol, include_s_and_p=tpsa_include_sp), vmax_tpsa, "tpsa"),
            ):
                title = f"{cpd.label} – {core.default_title(res)}"
                z.writestr(f"{folder}/{stem}_map.png",
                           core.render_figure(res, title=title, vmax=vmax, **kw))
                z.writestr(f"{folder}/{stem}_contributions.csv", _csv(core.contribution_table(res)))

            s = core.molecule_summary(mol)
            ro5_rows, ro5 = core.lipinski_rule_of_five(s)
            ro5_rows = ro5_rows + [{"Property": "Overall (≤ 1 violation allowed)",
                                    "Value": str(ro5["violations"]), "Rule": "≤ 1 violation",
                                    "Result": "Pass" if ro5["pass"] else "Fail"}]
            z.writestr(f"{folder}/lipinski_ro5.csv", _csv(ro5_rows))
            z.writestr(f"{folder}/boiled_egg.png",
                       boiled_egg.plot_many([(r["Cpd"], r["LogP (WLOGP)"], r["TPSA (Å²)"]) for r in rows],
                                            highlight=cpd.number,
                                            title=f"BOILED-Egg — {cpd.label}: GI {row['GI absorption']}, "
                                                  f"BBB {row['BBB permeant']}"))
            if progress:
                progress(i, len(compounds))
    return buf.getvalue()
