"""
BOILED-Egg model (Daina & Zoete, ChemMedChem 2016, 11, 1117-1121).

Plots WLOGP (Wildman-Crippen LogP) against TPSA (Ertl, N and O only):
  - "white" ellipse -> high probability of passive gastrointestinal absorption (HIA)
  - "yolk"  ellipse -> high probability of passive blood-brain barrier permeation (BBB)

Ellipses are given in (TPSA, WLOGP) space as published:
  centre, full axis lengths along (TPSA, WLOGP), rotation angle in degrees.
"""

from __future__ import annotations

import io

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402

#            centre (TPSA, WLOGP)   axes (TPSA, WLOGP)   angle (deg)
WHITE = dict(cx=71.051, cy=2.292, a=142.081, b=8.740, angle=-1.031325)
YOLK = dict(cx=38.117, cy=3.177, a=82.061, b=5.557, angle=-0.171887)


def _ellipse_value(tpsa: float, wlogp: float, e: dict) -> float:
    """<= 1 inside the ellipse, > 1 outside."""
    t = np.radians(e["angle"])
    dx, dy = tpsa - e["cx"], wlogp - e["cy"]
    u = dx * np.cos(t) + dy * np.sin(t)
    v = -dx * np.sin(t) + dy * np.cos(t)
    return (u / (e["a"] / 2)) ** 2 + (v / (e["b"] / 2)) ** 2


def predict(wlogp: float, tpsa: float) -> dict:
    """GI absorption (in white or yolk) and BBB permeation (in yolk)."""
    in_yolk = _ellipse_value(tpsa, wlogp, YOLK) <= 1
    in_white = _ellipse_value(tpsa, wlogp, WHITE) <= 1
    return {"hia": bool(in_white or in_yolk), "bbb": bool(in_yolk),
            "region": "yolk" if in_yolk else ("white" if in_white else "grey")}


def _outline(e: dict, n: int = 400) -> np.ndarray:
    """Ellipse boundary as (WLOGP, TPSA) points for plotting (x = WLOGP)."""
    s = np.linspace(0, 2 * np.pi, n)
    t = np.radians(e["angle"])
    u, v = e["a"] / 2 * np.cos(s), e["b"] / 2 * np.sin(s)
    tpsa = e["cx"] + u * np.cos(t) - v * np.sin(t)
    wlogp = e["cy"] + u * np.sin(t) + v * np.cos(t)
    return np.column_stack([wlogp, tpsa])


REGION_COLOUR = {"yolk": "#c0392b", "white": "#1f4fbf", "grey": "#222222"}


def _egg_axes(xs, ys, figsize=(7, 5.6)):
    """Figure with the grey background, white and yolk drawn; limits fit all points."""
    fig, ax = plt.subplots(figsize=figsize, dpi=150)
    xmin, xmax = min([-4.0] + [x - 0.8 for x in xs]), max([8.0] + [x + 0.8 for x in xs])
    ymin, ymax = min([-10.0] + [y - 10 for y in ys]), max([200.0] + [y + 15 for y in ys])
    ax.set_facecolor("#c8c8c8")  # grey: outside both regions
    ax.add_patch(Polygon(_outline(WHITE), closed=True, facecolor="white",
                         edgecolor="#9a9a9a", lw=1.2, label="HIA (white)"))
    ax.add_patch(Polygon(_outline(YOLK), closed=True, facecolor="#f8d66d",
                         edgecolor="#d9a400", lw=1.2, label="BBB (yolk)"))
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("WLOGP  (Wildman–Crippen LogP)")
    ax.set_ylabel("TPSA (Å²)")
    ax.grid(alpha=0.25, color="white")
    return fig, ax


def _png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return buf.getvalue()


def plot(wlogp: float, tpsa: float, label: str | None = None) -> bytes:
    """BOILED-Egg plot with one molecule marked and labelled; returns PNG bytes."""
    pred = predict(wlogp, tpsa)
    fig, ax = _egg_axes([wlogp], [tpsa])

    ax.scatter([wlogp], [tpsa], s=120, c=REGION_COLOUR[pred["region"]],
               edgecolors="black", linewidths=1.2, zorder=5)
    txt = label if label else "Input molecule"
    ax.annotate(f"{txt}\n(WLOGP {wlogp:.2f}, TPSA {tpsa:.1f} Å²)", (wlogp, tpsa),
                xytext=(12, 12), textcoords="offset points", fontsize=9, zorder=6,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#888", alpha=0.9))
    ax.set_title(
        f"BOILED-Egg  —  GI absorption: {'High' if pred['hia'] else 'Low'}   |   "
        f"BBB permeant: {'Yes' if pred['bbb'] else 'No'}",
        fontsize=11,
    )
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    return _png(fig)


def plot_many(points: list[tuple[int, float, float]], highlight: int | None = None,
              title: str | None = None) -> bytes:
    """
    BOILED-Egg with several molecules. points = [(compound_number, wlogp, tpsa), ...].
    Each marker shows its compound number; colour = predicted region.
    `highlight` draws a thick ring around that compound number.
    """
    from matplotlib.lines import Line2D

    xs = [p[1] for p in points]
    ys = [p[2] for p in points]
    fig, ax = _egg_axes(xs, ys, figsize=(8, 6.4))

    n_digits = max((len(str(p[0])) for p in points), default=1)
    size = {1: 230, 2: 300, 3: 420}.get(n_digits, 520)
    font = {1: 8, 2: 7.5, 3: 7}.get(n_digits, 6.5)

    for num, x, y in points:
        region = predict(x, y)["region"]
        hl = highlight is not None and num == highlight
        ax.scatter([x], [y], s=size * (1.35 if hl else 1), c=REGION_COLOUR[region],
                   edgecolors="#ffd400" if hl else "black", linewidths=3 if hl else 1,
                   zorder=6 if hl else 5, alpha=0.95)
        ax.text(x, y, str(num), ha="center", va="center", fontsize=font, fontweight="bold",
                color="white", zorder=7)

    handles = [
        Polygon([[0, 0]], facecolor="white", edgecolor="#9a9a9a", label="HIA (white)"),
        Polygon([[0, 0]], facecolor="#f8d66d", edgecolor="#d9a400", label="BBB (yolk)"),
        Line2D([], [], ls="", marker="o", ms=9, mfc=REGION_COLOUR["yolk"], mec="black",
               label="In yolk: GI+ / BBB+"),
        Line2D([], [], ls="", marker="o", ms=9, mfc=REGION_COLOUR["white"], mec="black",
               label="In white: GI+ / BBB−"),
        Line2D([], [], ls="", marker="o", ms=9, mfc=REGION_COLOUR["grey"], mec="black",
               label="Outside: GI− / BBB−"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.9)
    n_gi = sum(predict(x, y)["hia"] for _, x, y in points)
    n_bbb = sum(predict(x, y)["bbb"] for _, x, y in points)
    ax.set_title(title or f"BOILED-Egg — {len(points)} compounds   |   "
                          f"GI absorption high: {n_gi}   |   BBB permeant: {n_bbb}", fontsize=11)
    return _png(fig)
