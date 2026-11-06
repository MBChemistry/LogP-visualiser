"""
LogP / TPSA visualiser - interactive web app.

Run with:
    streamlit run app.py
"""

import pandas as pd
import streamlit as st
from rdkit import Chem, RDLogger
from streamlit_ketcher import st_ketcher

import batch
import boiled_egg
import druglikeness
import logp_core as core

RDLogger.DisableLog("rdApp.*")  # bad SMILES are reported in the UI instead

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
BATCH_EXAMPLE = "\n".join(f"{smi} {name}" for name, smi in EXAMPLES.items())

# ---------------- Sidebar ----------------
with st.sidebar:
    mode = st.radio("Mode", ["Single molecule", "Batch (multiple molecules)"], key="mode")
    st.divider()
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
    sigma, radius, show_h = 0.4, 0.33, True
    if style == "circles":
        radius = st.slider("Circle size (× bond length)", 0.25, 0.48, 0.33, 0.02)
        show_h = st.checkbox("Include hydrogens in labels (e.g. CH₃, OH)", True)
    else:
        sigma = st.slider("Heatmap spread (× bond length)", 0.2, 0.8, 0.4, 0.05)
    show_values = st.checkbox("Show per-atom values", True)
    show_indices = st.checkbox("Show atom indices", False)
    show_rotatable = st.checkbox(
        "Highlight rotatable bonds (yellow)", False,
        help="Marks the bonds counted in the Rotatable bonds metric on the LogP and TPSA maps, "
             "including images in the batch ZIP.",
    )
    tpsa_sp = st.checkbox(
        "TPSA: include S and P", False,
        help="Ertl's original TPSA counts only N and O. This affects the TPSA map only; "
             "the BOILED-Egg always uses the N/O-only TPSA it was trained on.",
    )
    size = st.select_slider("Image resolution", [500, 700, 900, 1200], 900)

RENDER = dict(style=style, size=size, sigma=sigma, radius=radius, show_hydrogens=show_h,
              show_values=show_values, show_indices=show_indices,
              highlight_rotatable=show_rotatable)


# ---------------- Reusable display pieces ----------------
def plural(n, word):
    return f"{n} {word}{'s' if n != 1 else ''}"


HELP = {
    "LogP": "Lipophilicity: how strongly a molecule prefers an oily phase (octanol) over water. "
            "Higher means more lipophilic. Calculated by summing Wildman–Crippen atom "
            "contributions (WLOGP). Lipinski's guide is ≤ 5.",
    "TPSA": "Topological polar surface area: the surface area of polar atoms (N and O, with "
            "their hydrogens), estimated from Ertl's fragment values. Higher values generally "
            "mean poorer membrane permeation. Used by the BOILED-Egg model.",
    "MW": "Molecular weight: the sum of the average atomic masses, in g/mol. Lipinski's guide "
          "is ≤ 500.",
    "Fsp3": "Fraction of sp³ carbons: sp³-hybridised carbons ÷ total carbons. Runs from 0 (no "
            "sp³ carbons, e.g. benzene) to 1 (all sp³, e.g. cyclohexane); higher means a more "
            "saturated, three-dimensional molecule. A drug-likeness parameter (Wei et al., "
            "Drug Discov. Today 2020), not a pass/fail rule. Shown as n/a if there is no carbon.",
    "HBD": "Hydrogen-bond donors: the number of OH and NH hydrogens (Lipinski's counting). "
           "Lipinski's guide is ≤ 5.",
    "HBA": "Hydrogen-bond acceptors: the number of N and O atoms (Lipinski's counting). "
           "Lipinski's guide is ≤ 10.",
    "RotB": "Rotatable bonds: single bonds that can rotate freely, excluding ring and terminal "
            "bonds. A measure of molecular flexibility.",
    "GI": "Predicted passive absorption from the gastrointestinal tract. High if the molecule "
          "falls in the white or yolk of the BOILED-Egg plot.",
    "BBB": "Predicted passive permeation of the blood–brain barrier. Yes if the molecule falls "
           "in the yolk of the BOILED-Egg plot.",
}


def show_metrics(summary):
    fsp3 = "n/a" if summary["Fsp3"] is None else f"{summary['Fsp3']:.2f}"
    top, bottom = st.columns(4), st.columns(4)
    top[0].metric("LogP (WLOGP)", f"{summary['LogP']:.2f}", help=HELP["LogP"])
    top[1].metric("TPSA", f"{summary['TPSA']:.1f} Å²", help=HELP["TPSA"])
    top[2].metric("Mol. weight", f"{summary['MW']:.1f}", help=HELP["MW"])
    top[3].metric("Fsp³", fsp3, help=HELP["Fsp3"])
    bottom[0].metric("H-bond donors", summary["HBD"], help=HELP["HBD"])
    bottom[1].metric("H-bond acceptors", summary["HBA"], help=HELP["HBA"])
    bottom[2].metric("Rotatable bonds", summary["RotB"], help=HELP["RotB"])


def summary_download(row, file_stem, key):
    """All whole-molecule results for one molecule (incl. Fsp³) as a one-row CSV."""
    st.download_button("Download molecule results (CSV)",
                       pd.DataFrame([row]).to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"{file_stem}_results.csv", key=f"summary_{key}")


def _pass_fail_style(v):
    if v in ("Pass", "High", "Yes"):
        return "background-color: #c8f0d0; color: #0b5d1e; font-weight: 600"
    if v in ("Fail", "Low", "No"):
        return "background-color: #f8c9c4; color: #8a1c10; font-weight: 600"
    return ""


def show_filters(mol, summary, key):
    """Druglikeness filters: a dropdown of filters, each with its criteria table."""
    desc = druglikeness.filter_descriptors(mol, summary)
    overalls = druglikeness.evaluate_all(desc)
    n_pass = sum(o["pass"] for o in overalls.values())

    with st.expander(f"Druglikeness filters: **{n_pass}/{len(overalls)} passed**", expanded=True):
        chips = "   ·   ".join(
            f"{'✅' if o['pass'] else '❌'} {n}" for n, o in overalls.items())
        st.markdown(chips)
        name = st.selectbox(
            "Filter", druglikeness.FILTER_NAMES, key=f"filter_{key}",
            format_func=lambda n: f"{druglikeness.FILTERS[n].full_name} — "
                                  f"{'Pass' if overalls[n]['pass'] else 'Fail'}",
            help="Each filter is a published rule of thumb for oral druglikeness. "
                 "Choosing one shows its criteria and this molecule's values.",
        )
        f = druglikeness.FILTERS[name]
        st.markdown(f.description)

        rows = druglikeness.table_rows(name, desc)
        df = pd.DataFrame(rows)
        styled = (df.style.map(_pass_fail_style, subset=["Result"])
                  .apply(lambda r: ["font-weight: 700; border-top: 2px solid #888"] * len(r)
                         if r.name == len(df) - 1 else [""] * len(r), axis=1))
        st.dataframe(styled, hide_index=True, width="stretch", key=f"filt_{key}_{name}")
        st.caption(f"{f.reference}." + (f" {f.notes}" if f.notes else ""))

        d1, d2 = st.columns(2)
        d1.download_button(f"Download {f.name} table (CSV)",
                           df.to_csv(index=False).encode("utf-8-sig"),
                           file_name=f"{key}_{f.name.lower()}.csv", key=f"filtdl_{key}_{name}")
        d2.download_button("Download all filters (CSV)",
                           pd.DataFrame(druglikeness.all_tables_rows(desc))
                           .to_csv(index=False).encode("utf-8-sig"),
                           file_name=f"{key}_druglikeness_filters.csv", key=f"filtall_{key}")


def show_atom_map(mol, view, title_prefix, file_stem, key):
    """LogP or TPSA per-atom map + contributions table."""
    if view == "LogP":
        res = core.compute_logp_contributions(mol)
        vmax = vmax_logp if fixed else None
        blurb = ("Each heavy atom includes its attached hydrogens. **Red** raises LogP "
                 "(lipophilic), **blue** lowers it (hydrophilic).")
    else:
        res = core.compute_tpsa_contributions(mol, include_s_and_p=tpsa_sp)
        vmax = vmax_tpsa if fixed else None
        blurb = ("Only polar atoms (N, O" + (", S, P" if tpsa_sp else "") + ") contribute; "
                 "**darker green** = more polar surface area. Carbons are 0.")
    st.caption(blurb)

    col_img, col_info = st.columns([3, 2])
    with col_img:
        png = core.render_figure(res, title=f"{title_prefix}{core.default_title(res)}", vmax=vmax, **RENDER)
        st.image(png, width="stretch")
        st.download_button("Download PNG", png, file_name=f"{file_stem}_{view.lower()}_map.png",
                           mime="image/png", key=f"png_{key}_{view}")
    with col_info:
        st.code(res.smiles, language=None)
        show_combined_atom_table(mol, file_stem, key=f"{key}_{view}")


def show_combined_atom_table(mol, file_stem, key):
    """Per-atom LogP and TPSA side by side, each coloured on its own scale."""
    logp_res = core.compute_logp_contributions(mol)
    tpsa_res = core.compute_tpsa_contributions(mol, include_s_and_p=tpsa_sp)
    lcmap, lnorm, _ = core.colour_scale(logp_res, vmax_logp if fixed else None)
    tcmap, tnorm, _ = core.colour_scale(tpsa_res, vmax_tpsa if fixed else None)

    df = pd.DataFrame(core.combined_contribution_table(mol, include_s_and_p=tpsa_sp))
    st.subheader("Per-atom contributions")
    st.dataframe(
        df.style
        .background_gradient(subset=[core.LOGP_COL], cmap=lcmap, vmin=lnorm.vmin, vmax=lnorm.vmax)
        .background_gradient(subset=[core.TPSA_COL], cmap=tcmap, vmin=tnorm.vmin, vmax=tnorm.vmax)
        .format({core.LOGP_COL: "{:+.3f}", core.TPSA_COL: "{:.2f}"}),
        hide_index=True, width="stretch", height=400, key=f"atoms_{key}",
    )
    st.download_button("Download LogP + TPSA per-atom CSV", df.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"{file_stem}_atom_contributions.csv", key=f"csv_{key}")
    st.caption(f"Totals: LogP = {sum(logp_res.contributions):.2f}, "
               f"TPSA = {sum(tpsa_res.contributions):.1f} Å² (sums of the per-atom values).")


EGG_LEGEND = (
    "- **White**: high probability of passive absorption from the gastrointestinal tract.\n"
    "- **Yolk**: high probability of passive permeation through the blood–brain barrier.\n"
    "- **Grey**: predicted poorly absorbed and not brain-penetrant."
)
EGG_REF = (
    "Model: Daina & Zoete, *ChemMedChem* 2016, 11, 1117–1121. Uses WLOGP (Wildman–Crippen) "
    "and TPSA (N/O only). Predicts *passive* diffusion only — it does not account for "
    "active transport or P-glycoprotein efflux."
)


# =====================================================================
# SINGLE MOLECULE MODE
# =====================================================================
def single_mode():
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
        choice = 0
        if up is not None:
            try:
                file_mols = core.mols_from_file(up.getvalue(), up.name)
            except Exception as e:  # noqa: BLE001
                st.error(str(e))
            if len(file_mols) > 1:
                st.info(f"This file has {len(file_mols)} molecules. Switch to **Batch** mode in the "
                        "sidebar to analyse them all at once.")
                choice = st.selectbox(f"{len(file_mols)} molecules in file - choose one",
                                      range(len(file_mols)), format_func=lambda i: file_mols[i][0])

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

    st.divider()
    if mol is None:
        st.info("Enter a SMILES, draw a molecule, or upload a file to get started.")
        return

    summary = core.molecule_summary(mol)
    show_metrics(summary)
    single_row = batch.analyse(batch.Compound(1, name or "", Chem.MolToSmiles(mol)))
    single_row.pop("Cpd")
    summary_download(single_row, "molecule", key="single")
    show_filters(mol, summary, key="single")

    view = st.segmented_control(
        "Visualise", ["LogP", "TPSA", "BOILED-Egg"], default="LogP", key="view",
        help="Switch between the per-atom LogP map, the per-atom TPSA map and the BOILED-Egg plot.",
    ) or "LogP"

    if view in ("LogP", "TPSA"):
        show_atom_map(mol, view, f"{name + ' - ' if name else ''}", "molecule", key="single")
        return

    pred = boiled_egg.predict(summary["LogP"], summary["TPSA"])
    col_egg, col_pred = st.columns([3, 2])
    with col_egg:
        egg_png = boiled_egg.plot(summary["LogP"], summary["TPSA"], label=name or Chem.MolToSmiles(mol))
        st.image(egg_png, width="stretch")
        st.download_button("Download PNG", egg_png, file_name="boiled_egg.png", mime="image/png")
    with col_pred:
        st.subheader("Prediction")
        p1, p2 = st.columns(2)
        p1.metric("GI absorption", "High" if pred["hia"] else "Low", help=HELP["GI"])
        p2.metric("BBB permeant", "Yes" if pred["bbb"] else "No", help=HELP["BBB"])
        where = {"yolk": "the **yolk** (BBB region, which also implies good GI absorption)",
                 "white": "the **white** (GI absorption region) but outside the yolk",
                 "grey": "**outside the egg** (grey region)"}[pred["region"]]
        st.markdown(f"The molecule falls in {where}.")
        egg_row = pd.DataFrame([{
            "Name": name or "", "SMILES": Chem.MolToSmiles(mol),
            "LogP (WLOGP)": round(summary["LogP"], 2), "TPSA (Å²)": round(summary["TPSA"], 2),
            "Egg region": pred["region"],
            "GI absorption": "High" if pred["hia"] else "Low",
            "BBB permeant": "Yes" if pred["bbb"] else "No",
        }])
        st.download_button("Download BOILED-Egg values (CSV)", egg_row.to_csv(index=False).encode("utf-8-sig"),
                           file_name="boiled_egg_values.csv", key="single_egg_csv")
        st.markdown(EGG_LEGEND)
        st.caption(EGG_REF)


# =====================================================================
# BATCH MODE
# =====================================================================
@st.cache_data(show_spinner=False)
def _results(smiles_and_names: tuple) -> list[dict]:
    cpds = [batch.Compound(i + 1, n, s) for i, (s, n) in enumerate(smiles_and_names)]
    return batch.results_table(cpds)


@st.cache_data(show_spinner=False)
def _atom_rows(smiles_and_names: tuple, include_sp: bool) -> list[dict]:
    cpds = [batch.Compound(i + 1, n, s) for i, (s, n) in enumerate(smiles_and_names)]
    return batch.all_atom_rows(cpds, include_s_and_p=include_sp)


@st.cache_data(show_spinner=False)
def _filter_rows(smiles_and_names: tuple) -> list[dict]:
    cpds = [batch.Compound(i + 1, n, s) for i, (s, n) in enumerate(smiles_and_names)]
    return batch.all_filter_rows(cpds)


def batch_mode():
    st.subheader("Batch analysis")
    tab_paste, tab_file = st.tabs(["⌨️ Paste SMILES list", "📁 Upload file"])
    with tab_paste:
        pasted = st.text_area(
            "One molecule per line: `SMILES` optionally followed by a name",
            height=180, key="batch_text",
            placeholder="CC(C)Cc1ccc(cc1)C(C)C(=O)O ibuprofen\nCC(=O)Oc1ccccc1C(=O)O aspirin",
        )
        c1, c2 = st.columns([1, 4])
        run_paste = c1.button("Run batch", type="primary", key="run_paste")
        c2.button("Load example list", key="batch_example",
                  on_click=lambda: st.session_state.update(batch_text=BATCH_EXAMPLE))
    with tab_file:
        up = st.file_uploader(
            "Multi-molecule file: .sdf, .smi/.txt (SMILES [name] per line) or .csv (with a "
            "'smiles' column and optional 'name'/'id' column)",
            type=["sdf", "smi", "smiles", "txt", "csv"], key="batch_file",
        )
        run_file = st.button("Run batch on file", type="primary", key="run_file", disabled=up is None)

    if run_paste or run_file:
        skipped: list[str] = []
        try:
            if run_paste:
                named = core.mols_from_lines(pasted, skipped)
            else:
                named = core.mols_from_file(up.getvalue(), up.name, skipped)
        except ValueError as e:
            named = []
            st.error(str(e))
        cpds = batch.make_compounds(named)
        st.session_state["batch"] = [(c.smiles, c.name) for c in cpds]
        st.session_state["batch_skipped"] = skipped
        st.session_state.pop("batch_zip", None)

    data = st.session_state.get("batch")
    st.divider()
    if not data:
        st.info("Paste a list of SMILES or upload a multi-molecule file, then click **Run batch**.")
        return

    compounds = [batch.Compound(i + 1, n, s) for i, (s, n) in enumerate(data)]
    rows = _results(tuple(data))
    skipped = st.session_state.get("batch_skipped", [])
    msg = f"**{plural(len(compounds), 'compound')}** analysed."
    if skipped:
        msg += f" {plural(len(skipped), 'entry')} could not be parsed and {'was' if len(skipped) == 1 else 'were'} skipped."
    st.success(msg)
    if skipped:
        with st.expander("Skipped entries"):
            st.code("\n".join(skipped), language=None)

    # ---- Results table (all compounds) ----
    st.subheader("Results")
    df = pd.DataFrame(rows).drop(columns=["Egg region"])
    st.dataframe(
        df.style.map(_pass_fail_style,
                     subset=batch.FILTER_COLUMNS + ["GI absorption", "BBB permeant"])
        .format({"MW": "{:.1f}", "LogP (WLOGP)": "{:.2f}", "TPSA (Å²)": "{:.1f}", "Fsp³": "{:.2f}"},
                na_rep="n/a"),
        hide_index=True, width="stretch", key="batch_table",
        column_config={col: st.column_config.Column(help=HELP[k]) for col, k in {
            "MW": "MW", "LogP (WLOGP)": "LogP", "TPSA (Å²)": "TPSA", "HBD": "HBD", "HBA": "HBA",
            "Rotatable bonds": "RotB", "Fsp³": "Fsp3", "GI absorption": "GI", "BBB permeant": "BBB",
        }.items()},
    )
    n = len(rows)
    counts = "   ·   ".join(f"{c}: {sum(r[c] == 'Pass' for r in rows)}/{n}"
                            for c in batch.FILTER_COLUMNS)
    st.caption(f"{counts}   ·   High GI absorption: "
               f"{sum(r['GI absorption'] == 'High' for r in rows)}/{n}   ·   BBB permeant: "
               f"{sum(r['BBB permeant'] == 'Yes' for r in rows)}/{n}")

    d1, d2, d3, d4 = st.columns(4)
    d1.download_button("Download results table (CSV)", df.to_csv(index=False).encode("utf-8-sig"),
                       file_name="batch_results.csv", key="batch_csv")
    d2.download_button(
        "Download per-atom LogP + TPSA, all compounds (CSV)",
        pd.DataFrame(_atom_rows(tuple(data), tpsa_sp)).to_csv(index=False).encode("utf-8-sig"),
        file_name="all_compounds_atom_contributions.csv", key="batch_atoms_csv",
        help="One file, one row per atom: compound number, name, atom, LogP and TPSA contribution.",
    )
    d3.download_button(
        "Download druglikeness filters, all compounds (CSV)",
        pd.DataFrame(_filter_rows(tuple(data))).to_csv(index=False).encode("utf-8-sig"),
        file_name="all_compounds_druglikeness_filters.csv", key="batch_filters_csv",
        help="Every criterion of all five filters, for every compound.",
    )
    with d4:
        kw = dict(RENDER, vmax_logp=vmax_logp if fixed else None,
                  vmax_tpsa=vmax_tpsa if fixed else None)
        sig = repr((sorted(kw.items()), tpsa_sp, tuple(data)))
        cached = st.session_state.get("batch_zip")
        if cached is None or cached[0] != sig:  # none yet, or display options changed
            if st.button("Prepare ZIP of all results and images", key="make_zip",
                         help="Per compound: LogP and TPSA maps (current display settings), results "
                              "summary (incl. Fsp³), per-atom CSV, druglikeness filters and BOILED-Egg "
                              "plot. Plus the summary table and an all-compounds BOILED-Egg."):
                bar = st.progress(0.0, text="Rendering compounds…")
                zbytes = batch.build_zip(
                    compounds, rows, kw, tpsa_include_sp=tpsa_sp,
                    progress=lambda i, n: bar.progress(i / n, text=f"Rendering compound {i} of {n}…"))
                st.session_state["batch_zip"] = (sig, zbytes)
                bar.empty()
                st.rerun()
        else:
            st.download_button("Download ZIP of all results and images", cached[1],
                               file_name="batch_results.zip", mime="application/zip",
                               type="primary", key="batch_zipdl")
            st.caption(f"{len(cached[1]) / 1e6:.1f} MB · images use the current display options.")

    # ---- Per-compound viewer ----
    st.divider()
    st.subheader("Compound viewer")
    num = st.selectbox("Compound", [c.number for c in compounds],
                       format_func=lambda n: compounds[n - 1].label + "   (" + (
                           compounds[n - 1].smiles if len(compounds[n - 1].smiles) <= 60
                           else compounds[n - 1].smiles[:57] + "…") + ")",
                       key="batch_cpd")
    cpd = compounds[num - 1]
    mol = cpd.mol
    st.markdown(f"### {cpd.label}")
    summary = core.molecule_summary(mol)
    show_metrics(summary)
    summary_download(rows[cpd.number - 1], f"cpd{cpd.number:03d}", key=f"cpd{cpd.number}")
    show_filters(mol, summary, key=f"cpd{cpd.number:03d}")

    view = st.segmented_control(
        "Visualise", ["LogP", "TPSA", "BOILED-Egg"], default="LogP", key="batch_view",
        help="The BOILED-Egg shows every compound in the batch; the selected one is ringed in yellow.",
    ) or "LogP"

    if view in ("LogP", "TPSA"):
        show_atom_map(mol, view, f"{cpd.label} - ", f"cpd{cpd.number:03d}", key=f"cpd{cpd.number}")
        return

    col_egg, col_pred = st.columns([3, 2])
    with col_egg:
        egg_png = boiled_egg.plot_many([(r["Cpd"], r["LogP (WLOGP)"], r["TPSA (Å²)"]) for r in rows],
                                       highlight=cpd.number)
        st.image(egg_png, width="stretch")
        st.download_button("Download PNG", egg_png, file_name="boiled_egg_all_compounds.png",
                           mime="image/png", key="egg_all_dl")
    with col_pred:
        r = rows[cpd.number - 1]
        st.markdown(f"**{cpd.label}** (ringed in yellow)")
        p1, p2 = st.columns(2)
        p1.metric("GI absorption", r["GI absorption"], help=HELP["GI"])
        p2.metric("BBB permeant", r["BBB permeant"], help=HELP["BBB"])
        st.markdown("**All compounds**")
        egg_df = pd.DataFrame(batch.egg_rows(rows))
        st.dataframe(egg_df.drop(columns=["SMILES"]).style
                     .map(_pass_fail_style, subset=["GI absorption", "BBB permeant"])
                     .format({"LogP (WLOGP)": "{:.2f}", "TPSA (Å²)": "{:.1f}"}),
                     hide_index=True, width="stretch", key="egg_table")
        st.download_button("Download BOILED-Egg table (CSV)", egg_df.to_csv(index=False).encode("utf-8-sig"),
                           file_name="boiled_egg_table.csv", key="egg_table_dl")
        st.markdown(EGG_LEGEND)
        st.caption(EGG_REF)


if mode.startswith("Single"):
    single_mode()
else:
    batch_mode()
