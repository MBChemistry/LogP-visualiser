"""
LogP / TPSA visualiser - interactive web app.

Run with:
    streamlit run app.py
"""

import pandas as pd
import streamlit as st
from streamlit_ketcher import st_ketcher

import boiled_egg
import logp_core as core

st.set_page_config(page_title="LogP & TPSA Visualiser", page_icon="🧪", layout="wide")
st.title("LogP & TPSA atom-contribution visualiser")
st.caption(
    "Per-atom contributions to **Wildman–Crippen LogP** (lipophilicity) and **Ertl TPSA** "
    "(topological polar surface area), calculated with RDKit, plus a **BOILED-Egg** prediction "
    "of passive gastrointestinal absorption and blood–brain barrier permeation."
)

EXAMPLES = {
    "4-Isopropylacetophenone": "CC(C)c1ccc(cc1)C(C)=O",
    "Ibuprofen": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "Aspirin": "CC(=O)Oc1ccccc1C(=O)O",
    "Caffeine": "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
    "Paracetamol": "CC(=O)Nc1ccc(O)cc1",
    "Diclofenac": "OC(=O)Cc1ccccc1Nc1c(Cl)cccc1Cl",
    "Atenolol": "CC(C)NCC(O)COc1ccc(CC(N)=O)cc1",
}

# ---------------- Sidebar: display options ----------------
with st.sidebar:
    st.header("Display options")
    style_label = st.radio("Style", ["Atom circles", "Smooth heatmap"])
    style = "circles" if style_label == "Atom circles" else "heatmap"
    scale_mode = st.radio(
        "Colour scale",
        ["Relative (this molecule)", "Fixed (compare molecules)"],
        help="Relative: the largest contribution in this molecule gets full colour. "
             "Fixed: the same value always gets the same colour across molecules.",
    )
    fixed = scale_mode.startswith("Fixed")
    vmax_logp = st.slider("LogP full intensity at |contribution| =", 0.2, 2.0, 1.0, 0.1,
                          disabled=not fixed)
    vmax_tpsa = st.slider("TPSA full intensity at contribution (Å²) =", 5.0, 40.0, 25.0, 1.0,
                          disabled=not fixed)
    sigma, radius, show_h = 0.4, 0.38, False
    if style == "circles":
        radius = st.slider("Circle size (× bond length)", 0.25, 0.48, 0.38, 0.02)
        show_h = st.checkbox("Include hydrogens in labels (e.g. CH₃, OH)", False)
    else:
        sigma = st.slider("Heatmap spread (× bond length)", 0.2, 0.8, 0.4, 0.05)
    show_values = st.checkbox("Show per-atom values", True)
    show_indices = st.checkbox("Show atom indices", False)
    tpsa_sp = st.checkbox(
        "TPSA: include S and P", False,
        help="Ertl's original TPSA counts only N and O. This affects the TPSA map only; "
             "the BOILED-Egg always uses the N/O-only TPSA it was trained on.",
    )
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

summary = core.molecule_summary(mol)

m = st.columns(3) + st.columns(3)
m[0].metric("LogP (WLOGP)", f"{summary['LogP']:.2f}")
m[1].metric("TPSA", f"{summary['TPSA']:.1f} Å²")
m[2].metric("Mol. weight", f"{summary['MW']:.1f}")
m[3].metric("H-bond donors", summary["HBD"])
m[4].metric("H-bond acceptors", summary["HBA"])
m[5].metric("Rotatable bonds", summary["RotB"])

view = st.segmented_control(
    "Visualise", ["LogP", "TPSA", "BOILED-Egg"], default="LogP", key="view",
    help="Switch between the per-atom LogP map, the per-atom TPSA map and the BOILED-Egg plot.",
)
view = view or "LogP"

if view in ("LogP", "TPSA"):
    if view == "LogP":
        res = core.compute_logp_contributions(mol)
        vmax = vmax_logp if fixed else None
        blurb = ("Each heavy atom includes its attached hydrogens. **Red** raises LogP "
                 "(lipophilic), **blue** lowers it (hydrophilic).")
    else:
        res = core.compute_tpsa_contributions(mol, include_s_and_p=tpsa_sp)
        vmax = vmax_tpsa if fixed else None
        blurb = ("Only polar atoms (N, O" + (", S, P" if tpsa_sp else "") + ") contribute; "
                 "**darker blue** = more polar surface area. Carbons are 0.")
    st.caption(blurb)

    col_img, col_info = st.columns([3, 2])
    with col_img:
        title = f"{name + ' - ' if name else ''}{core.default_title(res)}"
        png = core.render_figure(
            res, title=title,
            style=style, size=size, vmax=vmax, sigma=sigma,
            radius=radius, show_hydrogens=show_h,
            show_values=show_values, show_indices=show_indices,
        )
        st.image(png, width="stretch")
        st.download_button("Download PNG", png, file_name=f"{view.lower()}_map.png", mime="image/png")

    with col_info:
        st.code(res.smiles, language=None)
        df = pd.DataFrame(core.contribution_table(res))
        col = df.columns[-1]
        cmap, norm, _ = core.colour_scale(res, vmax)
        st.subheader("Per-atom contributions")
        st.dataframe(
            df.style.background_gradient(subset=[col], cmap=cmap, vmin=norm.vmin, vmax=norm.vmax)
            .format({col: "{:+.3f}" if res.info["diverging"] else "{:.2f}"}),
            hide_index=True, width="stretch", height=400,
        )
        st.download_button("Download CSV", df.to_csv(index=False),
                           file_name=f"{view.lower()}_contributions.csv")
        st.caption(f"Sum of contributions = {sum(res.contributions):.3f}"
                   f"{res.info['unit']} (= total {res.prop}).")

else:  # BOILED-Egg
    pred = boiled_egg.predict(summary["LogP"], summary["TPSA"])
    col_egg, col_pred = st.columns([3, 2])
    with col_egg:
        egg_png = boiled_egg.plot(summary["LogP"], summary["TPSA"], label=name or core.Chem.MolToSmiles(mol))
        st.image(egg_png, width="stretch")
        st.download_button("Download PNG", egg_png, file_name="boiled_egg.png", mime="image/png")
    with col_pred:
        st.subheader("Prediction")
        p1, p2 = st.columns(2)
        p1.metric("GI absorption", "High" if pred["hia"] else "Low")
        p2.metric("BBB permeant", "Yes" if pred["bbb"] else "No")
        where = {"yolk": "the **yolk** (BBB region, which also implies good GI absorption)",
                 "white": "the **white** (GI absorption region) but outside the yolk",
                 "grey": "**outside the egg** (grey region)"}[pred["region"]]
        st.markdown(f"The molecule falls in {where}.")
        st.markdown(
            "- **White**: high probability of passive absorption from the gastrointestinal tract.\n"
            "- **Yolk**: high probability of passive permeation through the blood–brain barrier.\n"
            "- **Grey**: predicted poorly absorbed and not brain-penetrant."
        )
        st.caption(
            "Model: Daina & Zoete, *ChemMedChem* 2016, 11, 1117–1121. Uses WLOGP (Wildman–Crippen) "
            "and TPSA (N/O only). Predicts *passive* diffusion only — it does not account for "
            "active transport or P-glycoprotein efflux."
        )
