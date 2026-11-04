"""
LogP Heatmap - interactive web app.

Run with:
    streamlit run app.py
"""

import pandas as pd
import streamlit as st
from streamlit_ketcher import st_ketcher

import logp_core as core

st.set_page_config(page_title="LogP Heatmap", page_icon="🧪", layout="wide")
st.title("LogP atom-contribution heatmap")
st.caption(
    "Wildman–Crippen LogP (RDKit). Each heavy atom's colour shows its contribution, "
    "including its attached hydrogens: **red** raises LogP (lipophilic), **blue** lowers it "
    "(hydrophilic). Stronger colour means a larger contribution."
)

EXAMPLES = {
    "4-Isopropylacetophenone": "CC(C)c1ccc(cc1)C(C)=O",
    "Ibuprofen": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "Aspirin": "CC(=O)Oc1ccccc1C(=O)O",
    "Caffeine": "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
    "Paracetamol": "CC(=O)Nc1ccc(O)cc1",
    "Diclofenac": "OC(=O)Cc1ccccc1Nc1c(Cl)cccc1Cl",
}

# ---------------- Sidebar: display options ----------------
with st.sidebar:
    st.header("Display options")
    style_label = st.radio("Style", ["Atom circles", "Smooth heatmap"])
    style = "circles" if style_label == "Atom circles" else "heatmap"
    scale_mode = st.radio(
        "Colour scale",
        ["Relative (this molecule)", "Fixed (compare molecules)"],
        help="Relative: the largest |contribution| in this molecule gets full colour. "
             "Fixed: the same value always gets the same colour across molecules.",
    )
    vmax = None
    if scale_mode.startswith("Fixed"):
        vmax = st.slider("Full-intensity at |contribution| =", 0.2, 2.0, 1.0, 0.1)
    sigma, radius, show_h = 0.4, 0.34, False
    if style == "circles":
        radius = st.slider("Circle size (× bond length)", 0.25, 0.48, 0.38, 0.02)
        show_h = st.checkbox("Include hydrogens in labels (e.g. CH3, OH)", False)
    else:
        sigma = st.slider("Heatmap spread (× bond length)", 0.2, 0.8, 0.4, 0.05)
    show_values = st.checkbox("Show per-atom values", True)
    show_indices = st.checkbox("Show atom indices", False)
    size = st.select_slider("Image resolution", [500, 700, 900, 1200], 900)

# ---------------- Input ----------------
mol = None
name = None
tab_draw, tab_text, tab_file = st.tabs(["✏️ Draw", "⌨️ SMILES / InChI", "📁 Upload file"])

with tab_text:
    ex = st.selectbox("Example molecules", ["—"] + list(EXAMPLES))
    default = EXAMPLES.get(ex, "")
    text = st.text_area("SMILES, InChI or MOL block", value=default, height=100,
                        placeholder="e.g. CC(C)Cc1ccc(cc1)C(C)C(=O)O")
    use_text = st.button("Calculate from text", type="primary")

with tab_draw:
    st.write("Draw a structure and click **Apply** in the editor.")
    drawn_smiles = st_ketcher(st.session_state.get("drawn", ""), height=500, key="ketcher")
    if drawn_smiles:
        st.session_state["drawn"] = drawn_smiles

with tab_file:
    up = st.file_uploader("Chemical file", type=core.SUPPORTED_FILE_TYPES)
    file_mols = []
    if up is not None:
        try:
            file_mols = core.mols_from_file(up.getvalue(), up.name)
        except Exception as e:  # noqa: BLE001
            st.error(str(e))
        if len(file_mols) > 1:
            choice = st.selectbox(f"{len(file_mols)} molecules in file - choose one",
                                  range(len(file_mols)), format_func=lambda i: file_mols[i][0])
        elif file_mols:
            choice = 0

# Which input wins: the most recently used one
if use_text:
    st.session_state["source"] = "text"
    st.session_state["text"] = text
if drawn_smiles and drawn_smiles != st.session_state.get("last_drawn"):
    st.session_state["source"] = "draw"
    st.session_state["last_drawn"] = drawn_smiles
if up is not None and up.file_id != st.session_state.get("last_file"):
    st.session_state["source"] = "file"
    st.session_state["last_file"] = up.file_id

source = st.session_state.get("source")
try:
    if source == "text" and st.session_state.get("text", "").strip():
        mol = core.mol_from_text(st.session_state["text"])
        name = ex if ex != "—" and st.session_state["text"].strip() == default else None
    elif source == "draw" and drawn_smiles:
        mol = core.mol_from_text(drawn_smiles)
        name = "Drawn structure"
    elif source == "file" and file_mols:
        name, mol = file_mols[choice]
except ValueError as e:
    st.error(str(e))

# ---------------- Output ----------------
st.divider()
if mol is None:
    st.info("Enter a SMILES, draw a molecule, or upload a file to get started.")
    st.stop()

res = core.compute_logp_contributions(mol)

col_img, col_info = st.columns([3, 2])
with col_img:
    png = core.render_figure(
        res,
        title=f"{name + ' - ' if name else ''}Crippen LogP = {res.logp:.2f}",
        style=style, size=size, vmax=vmax, sigma=sigma,
        radius=radius, show_hydrogens=show_h,
        show_values=show_values, show_indices=show_indices,
    )
    st.image(png, width="stretch")
    st.download_button("Download PNG", png, file_name="logp_heatmap.png", mime="image/png")

with col_info:
    m1, m2, m3 = st.columns(3)
    m1.metric("LogP (Crippen)", f"{res.logp:.2f}")
    m2.metric("Molar refractivity", f"{res.mr:.1f}")
    m3.metric("Heavy atoms", res.mol.GetNumAtoms())
    st.code(res.smiles, language=None)

    df = pd.DataFrame(core.contribution_table(res))
    st.subheader("Per-atom contributions")
    st.dataframe(
        df.style.background_gradient(
            subset=["LogP contribution"], cmap="bwr",
            vmin=-(vmax or df["LogP contribution"].abs().max()),
            vmax=(vmax or df["LogP contribution"].abs().max()),
        ).format({"LogP contribution": "{:+.3f}"}),
        hide_index=True, width="stretch", height=400,
    )
    st.download_button("Download CSV", df.to_csv(index=False), file_name="logp_contributions.csv")
    st.caption(
        f"Sum of contributions = {sum(res.contributions):.3f}. Hydrogen contributions "
        "are added to the heavy atom they are bonded to."
    )
