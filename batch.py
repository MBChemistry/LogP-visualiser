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
import druglikeness
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
    mol = cpd.mol
    s = core.molecule_summary(mol)
    desc = druglikeness.filter_descriptors(mol, s)
    filters = druglikeness.summary_columns(desc)
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
        "Fsp³": None if s["Fsp3"] is None else round(s["Fsp3"], 3),
        "Molar refractivity": round(desc["MR"], 2),
        "Atoms (incl. H)": desc["Atoms"],
        "Rings": desc["Rings"],
        "Carbon atoms": desc["Carbons"],
        "Heteroatoms": desc["Heteroatoms"],
        **filters,
        "Filters passed": sum(v == "Pass" for v in filters.values()),
        "GI absorption": "High" if egg["hia"] else "Low",
        "BBB permeant": "Yes" if egg["bbb"] else "No",
        "Egg region": egg["region"],
    }


FILTER_COLUMNS = [f.column for f in druglikeness.FILTERS.values()]


def results_table(compounds: list[Compound]) -> list[dict]:
    return [analyse(c) for c in compounds]


def all_atom_rows(compounds: list[Compound], include_s_and_p: bool = False) -> list[dict]:
    """Per-atom LogP + TPSA for every compound in one long table (one row per atom)."""
    out = []
    for c in compounds:
        for r in core.combined_contribution_table(c.mol, include_s_and_p=include_s_and_p):
            out.append({"Cpd": c.number, "Name": c.name, **r})
    return out


EGG_COLUMNS = ["Cpd", "Name", "SMILES", "LogP (WLOGP)", "TPSA (Å²)", "Egg region",
               "GI absorption", "BBB permeant"]


def all_filter_rows(compounds: list[Compound]) -> list[dict]:
    """Every compound's full filter criteria in one long table."""
    out = []
    for c in compounds:
        desc = druglikeness.filter_descriptors(c.mol, core.molecule_summary(c.mol))
        for r in druglikeness.all_tables_rows(desc):
            out.append({"Cpd": c.number, "Name": c.name, **r})
    return out


def egg_rows(rows: list[dict]) -> list[dict]:
    """The BOILED-Egg table: coordinates, region and predictions for each compound."""
    return [{k: r[k] for k in EGG_COLUMNS} for r in rows]


def _csv(rows: list[dict]) -> str:
    if not rows:
        return ""
    buf = io.StringIO()
    buf.write("﻿")  # BOM, so Excel reads Fsp³ and Å² correctly
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
      all_compounds_atom_contributions.csv   (per-atom LogP + TPSA, every compound)
      boiled_egg_table.csv
      all_compounds_druglikeness_filters.csv
      boiled_egg_all_compounds.png
      cpd_001_<name>/  logp_map.png, tpsa_map.png, atom_contributions.csv (LogP + TPSA),
                       summary.csv (all whole-molecule results, incl. Fsp³),
                       druglikeness_filters.csv (all five filters), boiled_egg.png
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
        z.writestr("all_compounds_atom_contributions.csv",
                   _csv(all_atom_rows(compounds, include_s_and_p=tpsa_include_sp)))
        z.writestr("boiled_egg_table.csv", _csv(egg_rows(rows)))
        z.writestr("all_compounds_druglikeness_filters.csv", _csv(all_filter_rows(compounds)))
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
            z.writestr(f"{folder}/atom_contributions.csv",
                       _csv(core.combined_contribution_table(mol, include_s_and_p=tpsa_include_sp)))

            s = core.molecule_summary(mol)
            desc = druglikeness.filter_descriptors(mol, s)
            z.writestr(f"{folder}/druglikeness_filters.csv",
                       _csv(druglikeness.all_tables_rows(desc)))
            z.writestr(f"{folder}/summary.csv", _csv([row]))
            z.writestr(f"{folder}/boiled_egg.png",
                       boiled_egg.plot_many([(r["Cpd"], r["LogP (WLOGP)"], r["TPSA (Å²)"]) for r in rows],
                                            highlight=cpd.number,
                                            title=f"BOILED-Egg — {cpd.label}: GI {row['GI absorption']}, "
                                                  f"BBB {row['BBB permeant']}"))
            if progress:
                progress(i, len(compounds))
    return buf.getvalue()
