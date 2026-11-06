"""
Druglikeness filters: Lipinski, Ghose, Veber, Egan and Muegge.

Each filter is a list of criteria on whole-molecule descriptors. A molecule
passes a criterion when its value falls inside the stated range.

Only Lipinski tolerates a violation (poor absorption is "more likely when more
than one" rule is broken), so it passes with at most one. The others are
all-or-nothing, as implemented in SwissADME (Daina, Michielin & Zoete,
Sci. Rep. 2017, 7, 42717).

LogP throughout is Wildman-Crippen (WLOGP), the value the rest of the app uses.
The original papers used other estimates (Lipinski MLOGP, Muegge XLOGP), so
borderline molecules can differ from other tools.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rdkit import Chem
from rdkit.Chem import Crippen, rdMolDescriptors


# ---------------------------------------------------------------------------
# Extra descriptors (beyond molecule_summary) needed by these filters
# ---------------------------------------------------------------------------

def filter_descriptors(mol: Chem.Mol, summary: dict) -> dict:
    """summary (from logp_core.molecule_summary) plus the counts these filters need."""
    m = Chem.Mol(mol)
    Chem.SanitizeMol(m)
    return {
        **summary,
        # Ghose counts every atom, hydrogens included
        "Atoms": Chem.AddHs(m).GetNumAtoms(),
        "Rings": rdMolDescriptors.CalcNumRings(m),
        "Carbons": sum(a.GetAtomicNum() == 6 for a in m.GetAtoms()),
        "Heteroatoms": rdMolDescriptors.CalcNumHeteroatoms(m),
        "MR": Crippen.MolMR(Chem.AddHs(m)),
    }


# ---------------------------------------------------------------------------
# Criteria
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Criterion:
    label: str                      # shown in the table
    key: str                        # key into the descriptor dict
    low: float | None = None        # inclusive lower bound
    high: float | None = None       # inclusive upper bound
    fmt: str = "{:.2f}"
    unit: str = ""

    @property
    def rule(self) -> str:
        lo = None if self.low is None else self.fmt.format(self.low)
        hi = None if self.high is None else self.fmt.format(self.high)
        if lo is not None and hi is not None:
            return f"{lo} – {hi}{self.unit}"
        if hi is not None:
            return f"≤ {hi}{self.unit}"
        return f"≥ {lo}{self.unit}"

    def check(self, descriptors: dict) -> dict:
        v = descriptors.get(self.key)
        if v is None:
            return {"Property": self.label, "Value": "n/a", "Rule": self.rule, "Result": "n/a"}
        ok = (self.low is None or v >= self.low) and (self.high is None or v <= self.high)
        return {"Property": self.label, "Value": self.fmt.format(v) + self.unit,
                "Rule": self.rule, "Result": "Pass" if ok else "Fail"}


@dataclass(frozen=True)
class Filter:
    name: str                       # e.g. "Lipinski"
    full_name: str                  # e.g. "Lipinski's rule of five"
    reference: str
    description: str                # what the filter is for, shown in the app
    criteria: list[Criterion]
    allowed_violations: int = 0
    notes: str = ""                 # caveats, e.g. a differing logP estimate
    column: str = field(default="")  # results-table column name

    def evaluate(self, descriptors: dict) -> tuple[list[dict], dict]:
        rows = [c.check(descriptors) for c in self.criteria]
        violations = sum(r["Result"] == "Fail" for r in rows)
        undefined = sum(r["Result"] == "n/a" for r in rows)
        return rows, {
            "violations": violations,
            "undefined": undefined,
            "pass": violations <= self.allowed_violations,
            "allowed": self.allowed_violations,
        }


MW = dict(key="MW", fmt="{:.1f}", unit=" Da")
LOGP = dict(key="LogP", fmt="{:.2f}")
TPSA = dict(key="TPSA", fmt="{:.1f}", unit=" Å²")
COUNT = dict(fmt="{:.0f}")

FILTERS: dict[str, Filter] = {
    "Lipinski": Filter(
        name="Lipinski",
        full_name="Lipinski's rule of five",
        reference="Lipinski et al., Adv. Drug Deliv. Rev. 1997, 23, 3–25",
        description=(
            "The original and most widely used oral-bioavailability filter, derived from "
            "compounds that reached phase II trials. It flags molecules unlikely to be well "
            "absorbed after oral dosing. Poor absorption is more likely when **more than one** "
            "rule is broken, so a molecule passes with at most one violation."
        ),
        criteria=[
            Criterion("Molecular weight", high=500, **MW),
            Criterion("LogP (WLOGP)", high=5, **LOGP),
            Criterion("H-bond donors (OH + NH)", key="HBD", high=5, **COUNT),
            Criterion("H-bond acceptors (N + O)", key="HBA", high=10, **COUNT),
        ],
        allowed_violations=1,
        notes="Lipinski originally used MLOGP; donors/acceptors use Lipinski's simple counting.",
        column="Lipinski",
    ),
    "Ghose": Filter(
        name="Ghose",
        full_name="Ghose filter",
        reference="Ghose, Viswanadhan & Wendoloski, J. Comb. Chem. 1999, 1, 55–68",
        description=(
            "A qualifying range built from drug databases, bounding molecules on both sides "
            "rather than only above. It adds molar refractivity (a size and polarisability "
            "measure) and a total atom count, so very small molecules fail as well as very "
            "large ones. All four criteria must be met."
        ),
        criteria=[
            Criterion("Molecular weight", low=160, high=480, **MW),
            Criterion("LogP (WLOGP)", low=-0.4, high=5.6, **LOGP),
            Criterion("Molar refractivity", key="MR", low=40, high=130, fmt="{:.1f}"),
            Criterion("Atoms (incl. H)", key="Atoms", low=20, high=70, **COUNT),
        ],
        notes="Molar refractivity is the Wildman–Crippen estimate.",
        column="Ghose",
    ),
    "Veber": Filter(
        name="Veber",
        full_name="Veber filter",
        reference="Veber et al., J. Med. Chem. 2002, 45, 2615–2623",
        description=(
            "Derived from oral bioavailability measured in rats. It ignores molecular weight "
            "entirely and instead limits flexibility (rotatable bonds) and polarity (TPSA), "
            "arguing these predict absorption better than weight does. Both criteria must be met."
        ),
        criteria=[
            Criterion("Rotatable bonds", key="RotB", high=10, **COUNT),
            Criterion("TPSA", high=140, **TPSA),
        ],
        notes="Veber also allows H-bond donors + acceptors ≤ 12 as an alternative to the TPSA "
              "limit; the TPSA form is used here, as in SwissADME.",
        column="Veber",
    ),
    "Egan": Filter(
        name="Egan",
        full_name="Egan filter (the 'egg')",
        reference="Egan, Merz & Baldwin, J. Med. Chem. 2000, 43, 3867–3877",
        description=(
            "A two-descriptor model of passive intestinal absorption, using lipophilicity and "
            "polar surface area. These same two axes underlie the BOILED-Egg plot elsewhere in "
            "this app, though the egg's ellipses differ from this rectangular cut-off. Both "
            "criteria must be met."
        ),
        criteria=[
            Criterion("LogP (WLOGP)", high=5.88, **LOGP),
            Criterion("TPSA", high=131.6, **TPSA),
        ],
        column="Egan",
    ),
    "Muegge": Filter(
        name="Muegge",
        full_name="Muegge filter (pharmacophore point filter)",
        reference="Muegge, Heald & Brittelli, J. Med. Chem. 2001, 44, 1841–1846",
        description=(
            "The strictest filter here, designed to separate drugs from non-drug chemicals. "
            "Alongside the usual properties it requires some chemical variety: enough carbons, "
            "at least one heteroatom, and a cap on rings. All eight criteria must be met."
        ),
        criteria=[
            Criterion("Molecular weight", low=200, high=600, **MW),
            Criterion("LogP (WLOGP)", low=-2, high=5, **LOGP),
            Criterion("TPSA", high=150, **TPSA),
            Criterion("Rings", key="Rings", high=7, **COUNT),
            Criterion("Carbon atoms", key="Carbons", low=5, **COUNT),
            Criterion("Heteroatoms", key="Heteroatoms", low=2, **COUNT),
            Criterion("Rotatable bonds", key="RotB", high=15, **COUNT),
            Criterion("H-bond acceptors (N + O)", key="HBA", high=10, **COUNT),
            Criterion("H-bond donors (OH + NH)", key="HBD", high=5, **COUNT),
        ],
        notes="Muegge used XLOGP. 'Carbon atoms ≥ 5' and 'heteroatoms ≥ 2' are the published "
              "> 4 and > 1 rules.",
        column="Muegge",
    ),
}

FILTER_NAMES = list(FILTERS)


def evaluate(name: str, descriptors: dict) -> tuple[list[dict], dict]:
    return FILTERS[name].evaluate(descriptors)


def evaluate_all(descriptors: dict) -> dict[str, dict]:
    """{filter name: overall} for every filter."""
    return {n: f.evaluate(descriptors)[1] for n, f in FILTERS.items()}


def summary_columns(descriptors: dict) -> dict:
    """Pass/Fail per filter, for the results table and exports."""
    out = {}
    for name, overall in evaluate_all(descriptors).items():
        out[FILTERS[name].column] = "Pass" if overall["pass"] else "Fail"
    return out


def all_tables_rows(descriptors: dict) -> list[dict]:
    """Every filter's criteria in one table, with a Filter column."""
    return [{"Filter": n, **r} for n in FILTERS for r in table_rows(n, descriptors)]


def table_rows(name: str, descriptors: dict) -> list[dict]:
    """Criteria rows plus a final overall row, as shown in the app and exported."""
    f = FILTERS[name]
    rows, overall = f.evaluate(descriptors)
    v = overall["violations"]
    allowed = f"≤ {f.allowed_violations} violation" if f.allowed_violations else "0 violations"
    return rows + [{
        "Property": f"Overall ({allowed} allowed)" if f.allowed_violations else "Overall",
        "Value": f"{v} violation" + ("" if v == 1 else "s"),
        "Rule": allowed,
        "Result": "Pass" if overall["pass"] else "Fail",
    }]
