"""
Core logic: parse a molecule, compute Crippen (Wildman-Crippen) LogP atom
contributions, and render a red/blue heatmap of those contributions.

Red  = atom increases LogP (more lipophilic)
Blue = atom decreases LogP (more hydrophilic)
Colour intensity scales with |contribution|.

Can also be used from the command line:
    python logp_core.py "CC(C)c1ccc(cc1)C(C)=O" -o ibuprofen_fragment.png
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from rdkit import Chem
from rdkit.Chem import Crippen, rdDepictor, rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import colors as mcolors  # noqa: E402

# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

SUPPORTED_FILE_TYPES = ["mol", "sdf", "mol2", "pdb", "smi", "smiles", "txt", "csv"]


def mol_from_text(text: str) -> Chem.Mol:
    """Parse SMILES, InChI, or a MOL block given as text."""
    text = text.strip()
    if not text:
        raise ValueError("No molecule given.")
    if text.startswith("InChI="):
        mol = Chem.MolFromInchi(text)
    elif "M  END" in text or "V2000" in text or "V3000" in text:
        mol = Chem.MolFromMolBlock(text)
    else:
        # first whitespace-separated token = SMILES (rest may be a name)
        mol = Chem.MolFromSmiles(text.split()[0])
    if mol is None:
        raise ValueError("Could not parse the molecule - check the SMILES/structure.")
    return mol


def mols_from_lines(text: str, skipped: list | None = None) -> list[tuple[str, Chem.Mol]]:
    """One molecule per line: 'SMILES [name]' (blank lines and # comments ignored)."""
    mols = []
    for i, line in enumerate(text.splitlines()):
        parts = line.strip().split(maxsplit=1)
        if not parts or parts[0].startswith("#"):
            continue
        m = Chem.MolFromSmiles(parts[0])
        if m is None:
            if skipped is not None:
                skipped.append(f"line {i + 1}: {line.strip()[:60]}")
            continue
        mols.append((parts[1].strip() if len(parts) > 1 else "", m))
    return mols


def _mols_from_csv(text: str, skipped: list | None) -> list[tuple[str, Chem.Mol]]:
    """CSV with a 'smiles' column (any case) and optional name/id column."""
    import csv

    rows = list(csv.reader(io.StringIO(text)))
    if not rows:
        return []
    header = [h.strip().lower() for h in rows[0]]
    if "smiles" in header:
        s_col = header.index("smiles")
        n_col = next((header.index(k) for k in ("name", "id", "compound", "title", "cpd")
                      if k in header), None)
        body = rows[1:]
    else:  # no header: first column = SMILES, second = name
        s_col, n_col, body = 0, (1 if len(rows[0]) > 1 else None), rows
    mols = []
    for i, r in enumerate(body):
        if not r or not r[s_col].strip():
            continue
        m = Chem.MolFromSmiles(r[s_col].strip())
        if m is None:
            if skipped is not None:
                skipped.append(f"row {i + 1}: {r[s_col].strip()[:60]}")
            continue
        mols.append(((r[n_col].strip() if n_col is not None and n_col < len(r) else ""), m))
    return mols


def mols_from_file(data: bytes, filename: str, skipped: list | None = None) -> list[tuple[str, Chem.Mol]]:
    """Parse an uploaded chemical file. Returns a list of (name, mol).
    Unparseable entries are skipped (and described in `skipped` if given)."""
    ext = Path(filename).suffix.lower().lstrip(".")
    text = data.decode("utf-8", errors="replace")
    mols: list[tuple[str, Chem.Mol]] = []

    if ext == "sdf":
        supplier = Chem.ForwardSDMolSupplier(io.BytesIO(data))
        for i, m in enumerate(supplier):
            if m is not None:
                name = m.GetProp("_Name") if m.HasProp("_Name") and m.GetProp("_Name") else f"Molecule {i + 1}"
                mols.append((name, m))
            elif skipped is not None:
                skipped.append(f"record {i + 1}")
    elif ext == "csv":
        mols = _mols_from_csv(text, skipped)
    elif ext == "mol":
        m = Chem.MolFromMolBlock(text)
        if m is not None:
            mols.append((m.GetProp("_Name") or Path(filename).stem, m))
    elif ext == "mol2":
        m = Chem.MolFromMol2Block(text)
        if m is not None:
            mols.append((m.GetProp("_Name") if m.HasProp("_Name") else Path(filename).stem, m))
    elif ext == "pdb":
        m = Chem.MolFromPDBBlock(text)
        if m is not None:
            mols.append((Path(filename).stem, m))
    else:  # .smi / .smiles / .txt  -> one SMILES per line, optional name
        mols = [(n or Chem.MolToSmiles(m), m) for n, m in mols_from_lines(text, skipped)]

    if not mols:
        raise ValueError(f"No valid molecules found in {filename}.")
    return mols


# ---------------------------------------------------------------------------
# LogP contributions
# ---------------------------------------------------------------------------


@dataclass
class AtomContribResult:
    mol: Chem.Mol                 # heavy-atom molecule (used for drawing)
    contributions: list[float]    # per heavy atom
    total: float                  # whole-molecule value (sum of contributions)
    prop: str                     # key into PROPERTIES ("LogP" or "TPSA")
    smiles: str

    @property
    def info(self) -> dict:
        return PROPERTIES[self.prop]


# How each property is coloured and labelled.
#   diverging=True  -> blue (-) / white (0) / red (+), symmetric scale
#   diverging=False -> white (0) -> colour (max), values are never negative
PROPERTIES = {
    "LogP": {
        "title": "Crippen LogP",
        "unit": "",
        "diverging": True,
        "value_fmt": "{:+.2f}",
        "total_fmt": "{:.2f}",
        "colorbar": "Atomic LogP contribution  (blue = hydrophilic, red = lipophilic)",
        "default_fixed_vmax": 1.0,
    },
    "TPSA": {
        "title": "TPSA",
        "unit": " Å²",
        "diverging": False,
        "value_fmt": "{:.1f}",
        "total_fmt": "{:.1f}",
        "colorbar": "Atomic TPSA contribution (Å²)  (darker green = more polar surface)",
        "default_fixed_vmax": 25.0,
    },
}


def _heavy_with_h_map(mol: Chem.Mol):
    """Return (molH, heavy, heavy_idx) where heavy_idx maps molH idx -> heavy idx."""
    mol = Chem.Mol(mol)
    Chem.SanitizeMol(mol)
    molH = Chem.AddHs(mol)
    heavy = Chem.RemoveHs(molH)
    heavy_idx = {}
    k = 0
    for atom in molH.GetAtoms():  # RemoveHs keeps heavy atoms in their original order
        if atom.GetAtomicNum() != 1 or atom.GetDegree() == 0:
            heavy_idx[atom.GetIdx()] = k
            k += 1
    if k != heavy.GetNumAtoms():  # e.g. explicit isotopic H kept - fall back
        heavy = molH
        heavy_idx = {i: i for i in range(molH.GetNumAtoms())}
    return molH, heavy, heavy_idx


def compute_tpsa_contributions(mol: Chem.Mol, include_s_and_p: bool = False) -> AtomContribResult:
    """
    Ertl topological polar surface area, per atom (Å²).

    Only N and O (and optionally S and P) contribute; the contribution of each
    polar atom already accounts for its attached hydrogens, so carbons are 0.
    """
    _molH, heavy, _ = _heavy_with_h_map(mol)
    contribs = list(rdMolDescriptors._CalcTPSAContribs(heavy, includeSandP=include_s_and_p))
    return AtomContribResult(
        mol=heavy,
        contributions=contribs,
        total=rdMolDescriptors.CalcTPSA(heavy, includeSandP=include_s_and_p),
        prop="TPSA",
        smiles=Chem.MolToSmiles(heavy),
    )


def compute_logp_contributions(mol: Chem.Mol) -> AtomContribResult:
    """
    Wildman-Crippen atom contributions.

    Crippen assigns a value to every atom *including hydrogens*. If computed on
    a molecule with implicit Hs, the H terms are lost and the contributions no
    longer sum to the real LogP. So: add explicit Hs, compute, then add each
    H's contribution onto the heavy atom it is attached to.
    """
    molH, heavy, heavy_idx = _heavy_with_h_map(mol)
    contribs = rdMolDescriptors._CalcCrippenContribs(molH)

    per_atom = [0.0] * heavy.GetNumAtoms()
    for atom, (logp_c, _mr) in zip(molH.GetAtoms(), contribs):
        idx = atom.GetIdx()
        if idx in heavy_idx:
            per_atom[heavy_idx[idx]] += logp_c
        else:  # hydrogen -> give to its neighbour
            nbr = atom.GetNeighbors()[0].GetIdx()
            per_atom[heavy_idx[nbr]] += logp_c

    return AtomContribResult(
        mol=heavy,
        contributions=per_atom,
        total=Crippen.MolLogP(molH),
        prop="LogP",
        smiles=Chem.MolToSmiles(heavy),
    )


def molecule_summary(mol: Chem.Mol) -> dict:
    """Whole-molecule descriptors shown alongside the maps."""
    from rdkit.Chem import Descriptors

    m = Chem.Mol(mol)
    Chem.SanitizeMol(m)
    return {
        "LogP": Crippen.MolLogP(Chem.AddHs(m)),
        "TPSA": rdMolDescriptors.CalcTPSA(m),
        "MW": Descriptors.MolWt(m),
        "MR": Crippen.MolMR(Chem.AddHs(m)),
        # Lipinski's own definitions: donors = NH + OH count, acceptors = N + O count
        "HBD": rdMolDescriptors.CalcNumLipinskiHBD(m),
        "HBA": rdMolDescriptors.CalcNumLipinskiHBA(m),
        "RotB": rdMolDescriptors.CalcNumRotatableBonds(m),
    }


def lipinski_rule_of_five(summary: dict) -> tuple[list[dict], dict]:
    """
    Lipinski's rule of five (Lipinski et al., Adv. Drug Deliv. Rev. 1997).

    Poor oral absorption is more likely when MORE THAN ONE rule is broken,
    so the molecule passes overall with at most one violation.
    Returns (rows, overall).
    """
    checks = [
        ("Molecular weight", summary["MW"], 500, "≤ 500 Da", "{:.1f}"),
        ("LogP (Crippen)", summary["LogP"], 5, "≤ 5", "{:.2f}"),
        ("H-bond donors (OH + NH)", summary["HBD"], 5, "≤ 5", "{:d}"),
        ("H-bond acceptors (N + O)", summary["HBA"], 10, "≤ 10", "{:d}"),
    ]
    rows = []
    for name, value, limit, rule, fmt in checks:
        ok = value <= limit
        rows.append({"Property": name, "Value": fmt.format(value), "Rule": rule,
                     "Result": "Pass" if ok else "Fail"})
    violations = sum(r["Result"] == "Fail" for r in rows)
    overall = {"violations": violations, "pass": violations <= 1}
    return rows, overall


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

CMAP = plt.get_cmap("bwr")  # blue (negative) -> white (0) -> red (positive)
# TPSA: white (0) -> green (max); green kept light enough for black labels
CMAP_SEQ = mcolors.LinearSegmentedColormap.from_list("white_green", ["#ffffff", "#1fb84a"])


def colour_scale(result: AtomContribResult, vmax: float | None):
    """(cmap, norm, vmax) for this result; vmax None/<=0 -> scale to this molecule."""
    w = np.asarray(result.contributions, dtype=float)
    if vmax is None or vmax <= 0:
        vmax = float(np.max(np.abs(w))) if len(w) and np.max(np.abs(w)) > 0 else 1.0
    if result.info["diverging"]:
        return CMAP, mcolors.Normalize(vmin=-vmax, vmax=vmax, clip=True), vmax
    return CMAP_SEQ, mcolors.Normalize(vmin=0, vmax=vmax, clip=True), vmax


def render_heatmap(
    result: AtomContribResult,
    size: int = 800,
    vmax: float | None = None,
    sigma: float = 0.40,
    show_values: bool = True,
    show_indices: bool = False,
) -> Image.Image:
    """
    Draw the molecule over a Gaussian heatmap of the atom contributions.

    vmax  : contribution that maps to full colour intensity. None = use the
            largest |contribution| in this molecule (relative scale). Set a
            fixed value (e.g. 1.0) to compare molecules on the same scale.
    sigma : Gaussian width as a fraction of the average bond length.
    """
    mol = Chem.Mol(result.mol)
    w = np.asarray(result.contributions, dtype=float)
    cmap, norm, vmax = colour_scale(result, vmax)
    fmt = result.info["value_fmt"]

    if mol.GetNumConformers() == 0 or mol.GetConformer().Is3D():
        rdDepictor.Compute2DCoords(mol)
    rdDepictor.StraightenDepiction(mol)

    # 1) draw molecule on a transparent canvas, get atom pixel positions
    drawer = rdMolDraw2D.MolDraw2DCairo(size, size)
    opts = drawer.drawOptions()
    opts.clearBackground = False
    opts.padding = 0.12
    opts.bondLineWidth = 2
    opts.addAtomIndices = show_indices
    opts.annotationFontScale = 0.6
    if show_values:
        for atom, c in zip(mol.GetAtoms(), w):
            if result.info["diverging"] or c > 0:
                atom.SetProp("atomNote", fmt.format(c))
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    mol_layer = Image.open(io.BytesIO(drawer.GetDrawingText())).convert("RGBA")

    pts = np.array([[drawer.GetDrawCoords(i).x, drawer.GetDrawCoords(i).y]
                    for i in range(mol.GetNumAtoms())])

    # average bond length in pixels -> Gaussian width
    if mol.GetNumBonds():
        bl = np.mean([np.linalg.norm(pts[b.GetBeginAtomIdx()] - pts[b.GetEndAtomIdx()])
                      for b in mol.GetBonds()])
    else:
        bl = size / 6
    s = sigma * bl

    # 2) Gaussian field: sum_i w_i * exp(-d^2 / 2s^2)
    yy, xx = np.mgrid[0:size, 0:size].astype(float)
    field = np.zeros((size, size))
    for (px, py), wi in zip(pts, w):
        field += wi * np.exp(-((xx - px) ** 2 + (yy - py) ** 2) / (2 * s * s))

    rgba = (cmap(norm(field)) * 255).astype(np.uint8)
    heat = Image.fromarray(rgba)  # (H, W, 4) uint8 -> RGBA

    # 3) composite: white background <- heatmap <- molecule
    out = Image.new("RGBA", (size, size), (255, 255, 255, 255))
    out = Image.alpha_composite(out, heat)
    out = Image.alpha_composite(out, mol_layer)
    return out.convert("RGB")


from functools import lru_cache  # noqa: E402


@lru_cache(maxsize=256)
def _load_font(px: int):
    """Bold sans font: Arial on Windows, else the DejaVu font bundled with matplotlib."""
    from PIL import ImageFont
    from matplotlib import font_manager

    for name in ("arialbd.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, px)
        except OSError:
            pass
    path = font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans", weight="bold"))
    return ImageFont.truetype(path, px)


def render_circles(
    result: AtomContribResult,
    size: int = 800,
    vmax: float | None = None,
    radius: float = 0.38,
    show_values: bool = True,
    show_indices: bool = False,
    show_hydrogens: bool = False,
    **_ignored,
) -> Image.Image:
    """
    Draw each atom as a circle filled with its heatmap colour and the atom
    label in black inside it. Bonds are drawn by RDKit underneath.

    radius : circle radius as a fraction of the average bond length.
    """
    from PIL import ImageDraw

    mol = Chem.Mol(result.mol)
    w = np.asarray(result.contributions, dtype=float)
    cmap, norm, vmax = colour_scale(result, vmax)
    fmt = result.info["value_fmt"]

    if mol.GetNumConformers() == 0 or mol.GetConformer().Is3D():
        rdDepictor.Compute2DCoords(mol)
    rdDepictor.StraightenDepiction(mol)

    # draw at 3x and downsample at the end -> smooth (anti-aliased) circles
    final_size, ss = size, 3
    size = final_size * ss

    # 1) bonds only (no atom labels), black
    drawer = rdMolDraw2D.MolDraw2DCairo(size, size)
    opts = drawer.drawOptions()
    opts.noAtomLabels = True
    opts.useBWAtomPalette()
    opts.padding = 0.15
    opts.bondLineWidth = 3 * ss
    opts.scaleBondWidth = False
    opts.multipleBondOffset = 0.2
    opts.fixedBondLength = size * 0.11  # stop small molecules being drawn huge
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    img = Image.open(io.BytesIO(drawer.GetDrawingText())).convert("RGB")

    pts = np.array([[drawer.GetDrawCoords(i).x, drawer.GetDrawCoords(i).y]
                    for i in range(mol.GetNumAtoms())])
    if mol.GetNumBonds():
        bl = np.mean([np.linalg.norm(pts[b.GetBeginAtomIdx()] - pts[b.GetEndAtomIdx()])
                      for b in mol.GetBonds()])
    else:
        bl = size / 6
    r = max(radius * bl, 8)

    # 2) circles with everything inside them, stacked vertically:
    #       #index (optional)  /  atom label  /  value (optional)
    draw = ImageDraw.Draw(img)
    outline_w = max(int(r * 0.07), 1)
    black = (0, 0, 0)

    # (centre-y offset, font size) for each line, in units of r
    if show_indices and show_values:
        lay = {"index": (-0.56, 0.30), "label": (-0.06, 0.56), "value": (0.47, 0.36)}
    elif show_values:
        lay = {"label": (-0.24, 0.66), "value": (0.36, 0.44)}
    elif show_indices:
        lay = {"index": (-0.50, 0.34), "label": (0.10, 0.70)}
    else:
        lay = {"label": (0.0, 0.78)}

    def fit(text, size, max_w):
        """Largest font <= size whose rendering of text fits in max_w pixels."""
        f = _load_font(int(size))
        while draw.textlength(text, font=f) > max_w and f.size > 8:
            f = _load_font(f.size - 1)
        return f

    def half_chord(dy):
        """Half the circle's width at vertical offset dy (keeps text inside)."""
        return r * np.sqrt(max(1 - (abs(dy) / r + 0.15) ** 2, 0.2))

    value_font = index_font = None
    if "value" in lay:
        dy, s_ = lay["value"]
        widest = max((fmt.format(c) for c in w), key=len, default="-0.00")
        value_font = fit(max(widest, "-0.00", key=len), s_ * r, 1.8 * half_chord(dy * r))
    if "index" in lay:
        dy, s_ = lay["index"]
        index_font = fit(f"#{mol.GetNumAtoms() - 1}", s_ * r, 1.8 * half_chord(dy * r))

    for atom, (x, y), c in zip(mol.GetAtoms(), pts, w):
        fill = tuple(int(v * 255) for v in cmap(norm(c))[:3])
        draw.ellipse([x - r, y - r, x + r, y + r], fill=fill,
                     outline=(60, 60, 60), width=outline_w)

        # label as runs: symbol, H, subscript count, superscript charge  (e.g. NH3+)
        runs = [(atom.GetSymbol(), "n")]
        nh = atom.GetTotalNumHs() if show_hydrogens else 0
        if nh:
            runs.append(("H", "n"))
            if nh > 1:
                runs.append((str(nh), "sub"))
        q = atom.GetFormalCharge()
        if q:
            runs.append(((str(abs(q)) if abs(q) > 1 else "") + ("+" if q > 0 else "−"), "sup"))

        dy, s_ = lay["label"]
        _draw_rich_label(draw, x, y + dy * r, runs, s_ * r, 1.5 * half_chord(dy * r), black)

        if value_font is not None:
            draw.text((x, y + lay["value"][0] * r), fmt.format(c), fill=black, font=value_font, anchor="mm")
        if index_font is not None:
            draw.text((x, y + lay["index"][0] * r), f"#{atom.GetIdx()}", fill=black, font=index_font, anchor="mm")

    return img.resize((final_size, final_size), Image.LANCZOS)


def _draw_rich_label(draw, x, yc, runs, size, max_w, fill):
    """
    Draw text runs centred at (x, yc). kind: 'n' normal, 'sub' subscript,
    'sup' superscript. Shrinks the font until the whole label fits in max_w.
    """
    size = int(size)
    while True:
        big = _load_font(size)
        small = _load_font(max(int(size * 0.62), 6))
        widths = [draw.textlength(t, font=big if k == "n" else small) for t, k in runs]
        if sum(widths) <= max_w or size <= 8:
            break
        size -= 1
    cap = size * 0.72                       # approx. capital height
    baseline = yc + cap / 2                 # so capitals are vertically centred
    xx = x - sum(widths) / 2
    for (t, k), wdt in zip(runs, widths):
        if k == "n":
            draw.text((xx, baseline), t, fill=fill, font=big, anchor="ls")
        elif k == "sub":
            draw.text((xx, baseline + size * 0.22), t, fill=fill, font=small, anchor="ls")
        else:  # sup
            draw.text((xx, baseline - cap * 0.55), t, fill=fill, font=small, anchor="ls")
        xx += wdt


def render_figure(result: AtomContribResult, title: str | None = None,
                  style: str = "circles", **kwargs) -> bytes:
    """Circles/heatmap + colour bar + title, returned as PNG bytes."""
    if style == "circles":
        img = render_circles(result, **kwargs)
    else:
        kwargs.pop("radius", None)
        kwargs.pop("show_hydrogens", None)
        img = render_heatmap(result, **kwargs)

    # crop surrounding white space (keeps a small margin)
    from PIL import ImageChops
    bbox = ImageChops.difference(img, Image.new("RGB", img.size, (255, 255, 255))).getbbox()
    if bbox:
        m = int(0.04 * img.width)
        img = img.crop((max(bbox[0] - m, 0), max(bbox[1] - m, 0),
                        min(bbox[2] + m, img.width), min(bbox[3] + m, img.height)))

    cmap, norm, _ = colour_scale(result, kwargs.get("vmax"))
    info = result.info

    aspect = img.height / img.width
    fig, ax = plt.subplots(figsize=(7, 7 * aspect + 1.2), dpi=150)
    ax.imshow(img)
    ax.axis("off")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cb = fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.045, pad=0.02)
    cb.set_label(info["colorbar"])
    ax.set_title(title or default_title(result), fontsize=14)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return buf.getvalue()


def default_title(result: AtomContribResult) -> str:
    info = result.info
    return f"{info['title']} = {info['total_fmt'].format(result.total)}{info['unit']}"


def contribution_table(result: AtomContribResult) -> list[dict]:
    col = f"{result.prop} contribution{result.info['unit'].replace(' ', ' (') + ')' if result.info['unit'] else ''}"
    rows = []
    for atom, c in zip(result.mol.GetAtoms(), result.contributions):
        rows.append({
            "Atom index": atom.GetIdx(),
            "Element": atom.GetSymbol(),
            "Attached H": atom.GetTotalNumHs(),
            "Aromatic": atom.GetIsAromatic(),
            col: round(c, 3),
        })
    return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="LogP / TPSA atom-contribution maps")
    p.add_argument("molecule", help="SMILES/InChI string, or path to .mol/.sdf/.mol2/.pdb/.smi file")
    p.add_argument("-o", "--output", default="logp_heatmap.png")
    p.add_argument("--vmax", type=float, default=None, help="fixed colour scale (default: per molecule)")
    p.add_argument("--no-values", action="store_true", help="hide per-atom numbers")
    p.add_argument("--style", choices=["circles", "heatmap"], default="circles")
    p.add_argument("--property", choices=["logp", "tpsa"], default="logp")
    p.add_argument("--egg", metavar="PNG", help="also save a BOILED-Egg plot to this file")
    a = p.parse_args()

    path = Path(a.molecule)
    if path.is_file():
        name, mol = mols_from_file(path.read_bytes(), path.name)[0]
    else:
        name, mol = a.molecule, mol_from_text(a.molecule)

    res = compute_tpsa_contributions(mol) if a.property == "tpsa" else compute_logp_contributions(mol)
    Path(a.output).write_bytes(render_figure(res, style=a.style, vmax=a.vmax,
                                              show_values=not a.no_values))
    print(f"{res.smiles}  {default_title(res)}  -> {a.output}")

    if a.egg:
        import boiled_egg

        s = molecule_summary(mol)
        pred = boiled_egg.predict(s["LogP"], s["TPSA"])
        Path(a.egg).write_bytes(boiled_egg.plot(s["LogP"], s["TPSA"], label=name))
        print(f"BOILED-Egg: GI absorption {'High' if pred['hia'] else 'Low'}, "
              f"BBB permeant {'Yes' if pred['bbb'] else 'No'}  -> {a.egg}")
