"""
Builds manuscript tables from data/ and results/ (no hand-entered numbers except labels).
Outputs: tables/Research_Tables.xlsx and tables/tables_md.json (markdown versions for the manuscript build).
Usage: python 06_make_tables.py <data_dir> <results_dir> <tables_dir>
"""
import sys, os, json, numpy as np, pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Font, Alignment, Border, Side
D, R, T = sys.argv[1:4] if len(sys.argv) > 3 else ("data", "results", "tables")
os.makedirs(T, exist_ok=True)
df = pd.read_csv(f"{D}/GO_ZTA_master_dataset_N20.csv")
perf = pd.read_csv(f"{R}/cv_performance_all_models.csv")
NAMES = {"RD_pct": "Relative density (%)", "HV_GPa": "Vickers hardness H_V (GPa)", "E_GPa": "Young's modulus E (GPa)",
         "KIC_MPa_m05": "Fracture toughness K_IC (MPa m^1/2)", "d_Al2O3_um": "Al2O3 grain size (µm)",
         "d_YSZ_um": "YSZ grain size (µm)", "t_ZrO2_wt_pct": "t-ZrO2 (wt%)", "m_ZrO2_wt_pct": "m-ZrO2 (wt%)"}
SHORT = {"RD_pct": "RD", "HV_GPa": "H_V", "E_GPa": "E", "KIC_MPa_m05": "K_IC", "d_Al2O3_um": "d_Al2O3", "d_YSZ_um": "d_YSZ", "t_ZrO2_wt_pct": "t-ZrO2"}
TG = list(SHORT)
tabs, caps, notes = {}, {}, {}

# Table 1 dataset
src = {"RD_pct": "Digitised, Fig. 6 of [12] (mean marker)", "HV_GPa": "Digitised, Fig. 7 of [12] (box median)",
       "E_GPa": "Digitised, Fig. 8 of [12] (box median)", "KIC_MPa_m05": "Digitised, Fig. 9 of [12] (box median)",
       "d_Al2O3_um": "Table 2 of [12] (exact)", "d_YSZ_um": "Table 3 of [12] (exact)", "t_ZrO2_wt_pct": "Table 1 of [12] (exact)",
       "m_ZrO2_wt_pct": "Table 1 of [12] (exact)"}
meth = {"RD_pct": "Archimedes (ASTM B311)", "HV_GPa": "Vickers, 9.807 N, 15 s, 5 indents", "E_GPa": "As reported in [12]",
        "KIC_MPa_m05": "Indentation crack length (Eq. (2) of [12])", "d_Al2O3_um": "Line intercept, FESEM (ASTM E112)",
        "d_YSZ_um": "Line intercept, FESEM (ASTM E112)", "t_ZrO2_wt_pct": "XRD Rietveld", "m_ZrO2_wt_pct": "XRD Rietveld"}
rows = [dict(Variable="GO content", Role="Input", Unit="wt%", Values_or_range="0, 0.25, 0.50, 0.75, 1.00", Source="Design of [12]", Measurement="—"),
        dict(Variable="Sintering temperature", Role="Input", Unit="°C", Values_or_range="1200, 1300, 1400, 1500", Source="Design of [12]", Measurement="—")]
for t in list(SHORT) + ["m_ZrO2_wt_pct"]:
    n, u = NAMES[t].rsplit(" (", 1)
    rows.append(dict(Variable=n, Role="Output" if t != "m_ZrO2_wt_pct" else "Output (not modelled)", Unit=u.rstrip(")"),
                     Values_or_range=f"{df[t].min():.4g} – {df[t].max():.4g}", Source=src[t], Measurement=meth[t]))
tabs[1] = pd.DataFrame(rows)
caps[1] = "Variables of the GO-doped ZTA dataset (N = 20 processing conditions), their provenance and measurement methods."
notes[1] = ("Constant in all samples (not used as inputs): Al2O3:YSZ = 80:20 wt%, 300 rpm/6 h milling, 300 N mm-2 compaction, "
            "20 °C min-1 heating, 10 min dwell, 2.45 GHz. Digitised values carry a read-out uncertainty of about one pixel "
            "(0.0075 % RD, 0.009 GPa H_V, 0.08 GPa E, 0.0076 MPa m^1/2 K_IC) in addition to the experimental scatter.")

# Table 2 model settings
tabs[2] = pd.DataFrame([
    ("Mean (null)", "Training-fold mean", "—"),
    ("Linear", "y = β0 + β1·GO + β2·T*", "—"),
    ("Quadratic RSM", "Full 2nd-order polynomial in (GO, T*), ridge-penalised", "α ∈ [10^-4, 10^2], inner LOOCV (RidgeCV)"),
    ("Dopant-step linear", "y = β0 + β1·I(GO>0) + β2·GO + β3·T*", "—"),
    ("GPR (ARD-RBF)", "k = σf² exp(−½Σ(Δx_j/ℓ_j)²) + σn² δ", "σf, ℓ_GO, ℓ_T, σn by marginal likelihood (5 restarts) in each fold"),
    ("SVR (RBF)", "ε-insensitive kernel regression", "C ∈ {1,10,100}, γ ∈ {0.1,0.5,2}, ε ∈ {0.01,0.1}; inner 5-fold CV"),
    ("Random forest", "500 trees, min. leaf size 2", "fixed"),
    ("Gradient boosting", "300 trees, depth 2, learning rate 0.05, subsample 0.8", "fixed"),
    ("MLP (16-16)", "2 hidden layers × 16 ReLU units, L-BFGS, L2 = 0.01", "fixed")],
    columns=["Model", "Form / architecture", "Hyper-parameters (selected inside training folds)"])
caps[2] = "Surrogate models compared. Inputs are GO content (wt%) and T* = (T − 1200 °C)/300 °C; targets are standardised within each training fold."
notes[2] = "All models were implemented in scikit-learn 1.9.1 with random seed 42."

# Table 3 validation RMSE
order = ["Mean (null)", "Linear", "Quadratic RSM", "Dopant-step linear", "GPR (ARD-RBF)", "SVR (RBF)", "Random forest", "Gradient boosting", "MLP (16-16)"]
blocks = []
for sch, lab in [("LOOCV", "LOOCV"), ("LOGO-GO", "Leave-one-GO-level-out")]:
    p = perf[perf.scheme == sch].pivot_table(index="model", columns="target", values="RMSE").loc[order, TG]
    p.columns = [SHORT[c] for c in p.columns]; p.insert(0, "Scheme", lab); p = p.reset_index().rename(columns={"model": "Model"})
    blocks.append(p)
t3 = pd.concat(blocks, ignore_index=True)
for c in ["RD", "H_V", "K_IC", "d_Al2O3", "d_YSZ", "t-ZrO2"]: t3[c] = t3[c].map(lambda v: f"{v:.3f}")
t3["E"] = t3["E"].map(lambda v: f"{v:.2f}")
tabs[3] = t3
caps[3] = "Out-of-sample root-mean-square error (RMSE) of the surrogate models for interpolation (LOOCV) and for prediction at an unseen GO level (leave-one-GO-level-out)."
notes[3] = ("Units as in Table 1. Leave-one-temperature-out results and R², MAE and maximum errors for every model, target and scheme are given in "
            "Research_Tables.xlsx (sheet 'Full_CV'). Values below the 'Mean (null)' row indicate a skilful model.")

# Table 4 noise / GPR
nc = pd.read_csv(f"{R}/noise_ceiling.csv").set_index("target")
cov = pd.read_csv(f"{R}/gpr_loocv_interval_coverage.csv").set_index("target")
rows = []
for t in TG:
    g = perf[(perf.scheme == "LOOCV") & (perf.model == "GPR (ARD-RBF)") & (perf.target == t)].iloc[0]
    rows.append(dict(Target=SHORT[t], SD_between_conditions=f"{df[t].std():.3g}",
                     Within_condition_scatter=(f"{nc.loc[t,'scatter_SD']:.3g}" if t in nc.index else "n.a."),
                     GPR_LOOCV_RMSE=f"{g.RMSE:.3g}", GPR_LOOCV_R2=f"{g.R2:.3f}",
                     Coverage_95=f"{cov.loc[t,'coverage_95']:.2f}", Mean_predictive_SD=f"{cov.loc[t,'mean_pred_sd']:.3g}"))
tabs[4] = pd.DataFrame(rows)
caps[4] = "Between-condition variability, within-condition experimental scatter and Gaussian-process (GPR) LOOCV accuracy and calibration."
notes[4] = ("Within-condition scatter: IQR/1.349 of the published box plots (H_V, E, K_IC) or half-length of the published error bar (RD), median over the 20 conditions. "
            "n.a.: the published grain-size SDs describe the grain population, not the uncertainty of the mean, and Rietveld uncertainties were not reported. "
            "Coverage: fraction of conditions whose reported value lies inside the 95% predictive interval.")

# Table 5 ANOVA
an = pd.read_csv(f"{R}/two_way_anova.csv")
tabs[5] = pd.DataFrame(dict(Target=[SHORT[t] for t in an.target], eta2_GO=an.eta2_GO.map(lambda v: f"{v:.3f}"),
                            p_GO=an.p_GO.map(lambda v: f"{v:.1e}"), eta2_T=an.eta2_T.map(lambda v: f"{v:.3f}"),
                            p_T=an.p_T.map(lambda v: f"{v:.3f}" if v >= 0.001 else f"{v:.1e}"),
                            eta2_residual=an.eta2_residual.map(lambda v: f"{v:.3f}")))
caps[5] = "Two-way additive analysis of variance of the 5 × 4 full-factorial design: fraction of the total variance (η²) attributable to GO content, sintering temperature and the residual (interaction + error)."
notes[5] = "One value per cell; F tests use 4, 3 and 12 degrees of freedom for GO, T and residual, respectively."

# Table 6 physics
ph = json.load(open(f"{R}/physics_descriptor_results.json"))
ch = pd.read_csv(f"{R}/chain_model_performance.csv")
e, h = ph["porosity_law_E_GPa"], ph["porosity_law_HV_GPa"]; j = ph["HV_joint_porosity_dinv"]; hp = ph["hall_petch_alone"]
rows = [
 ("E = E0 exp(−b p)", f"E0 = {e['P0']:.0f} GPa (95% CI {e['P0_ci95'][0]:.0f}–{e['P0_ci95'][1]:.0f}); b = {e['b']:.2f} ± {e['b_ci95']:.2f}", f"{e['r2']:.2f}", f"{e['p_value']:.1e}"),
 ("H_V = H0 exp(−b p)", f"H0 = {h['P0']:.2f} GPa (95% CI {h['P0_ci95'][0]:.2f}–{h['P0_ci95'][1]:.2f}); b = {h['b']:.2f} ± {h['b_ci95']:.2f}", f"{h['r2']:.2f}", f"{h['p_value']:.1e}"),
 ("H_V = a + k d_Al2O3^−1/2", f"k = {hp['slope']:.2f} ± {hp['slope_ci95']:.2f} GPa µm^1/2", f"{hp['r2']:.2f}", f"{hp['p_value']:.3f}"),
 ("H_V = a + c p + k d_Al2O3^−1/2", f"c = {j['coef_porosity']:.1f} GPa (p = {j['p_values'][1]:.1e}); k = {j['coef_dinv']:.3f} (p = {j['p_values'][2]:.2f}); partial r(H_V, d^−1/2 at fixed p) = {j['partial_r_HV_dinv_given_porosity']:.3f}", f"{j['r2']:.2f}", "—"),
]
for _, r in ch.iterrows():
    rows.append((f"Chain: GPR(RD) → porosity law → {SHORT[r.target]} [{r.scheme}]", f"RMSE = {r.RMSE:.3g}", f"{r.R2:.2f}", "—"))
tabs[6] = pd.DataFrame(rows, columns=["Relation", "Fitted parameters", "R²", "p"])
caps[6] = "Physically motivated descriptor relations fitted to the 20 conditions (p = porosity fraction = 1 − RD/100)."
notes[6] = "± values are 95% confidence half-widths from ordinary least squares. Chain-model R² values are out-of-sample."

# Table 7 augmentation
au = pd.read_csv(f"{R}/augmentation_leakage_experiment.csv")
tabs[7] = pd.DataFrame(dict(Target=[SHORT[t] for t in au.target], Model=au.model,
     R2_leaky=au.R2_leaky_augmented_random5fold.map(lambda v: f"{v:.3f}"),
     R2_no_aug=au.R2_LOOCV_no_augmentation.map(lambda v: f"{v:.3f}"),
     R2_fold_aug=au.R2_LOOCV_train_fold_augmentation.map(lambda v: f"{v:.3f}")))
caps[7] = "Effect of SMOTE-type synthetic augmentation (980 synthetic + 20 real samples) on apparent and true predictive accuracy."
notes[7] = ("Leaky: synthesis on the full data set followed by random 5-fold CV on the pooled 1000 samples. No aug.: LOOCV on the 20 real conditions. "
            "Fold aug.: LOOCV in which synthetic samples are generated only from the 19 training conditions and the model is scored on the held-out real condition.")

# Table S1 digitisation validation, S2 CSV comparison
dv = pd.read_csv(f"{R}/digitisation_validation.csv")
tabs["S1"] = dv.assign(target=dv.target.map(SHORT)).round(4)
caps["S1"] = "Validation of the digitised values against values stated in the text of [12]."
notes["S1"] = "The 0 wt% GO/1200 °C K_IC difference (stated 4.06, mean; digitised 4.11, median) reflects mean-vs-median and is within the reported ±0.07."
cmpc = pd.read_csv(f"{D}/unverified_CSV_vs_published_comparison.csv")
tabs["S2"] = cmpc
caps["S2"] = "Comparison of the unverified file GO_ZTA_experimental_data.csv with the published record."
notes["S2"] = "Agreement tolerance = approximately one digitisation pixel (mechanical properties) or rounding (tabulated values)."

# write xlsx
xl = f"{T}/Research_Tables.xlsx"
with pd.ExcelWriter(xl, engine="openpyxl") as w:
    readme = pd.DataFrame({"Item": [f"Table {k}" for k in tabs] + ["Full_CV", "Dataset_N20"],
                           "Caption": [caps[k] for k in tabs] + ["All models x targets x CV schemes (RMSE, MAE, R2, MaxAE, RMSE/scatter)",
                                                                 "Analysis dataset used for every model (see Dataset_Description.docx)"]})
    readme.to_excel(w, sheet_name="README", index=False)
    for k, t in tabs.items():
        t.to_excel(w, sheet_name=f"Table_{k}", index=False, startrow=2)
    perf.round(4).to_excel(w, sheet_name="Full_CV", index=False)
    df.to_excel(w, sheet_name="Dataset_N20", index=False)
wb = load_workbook(xl)
for k in tabs:
    ws = wb[f"Table_{k}"]; ws["A1"] = f"Table {k}. {caps[k]}"; ws["A1"].font = Font(bold=True)
    last = ws.max_row + 2; ws.cell(last, 1, "Note: " + notes[k]).alignment = Alignment(wrap_text=False)
    for c in ws[3]: c.font = Font(bold=True); c.border = Border(bottom=Side(style="thin"))
for ws in wb.worksheets:
    for col in ws.columns:
        L = max((len(str(c.value)) if c.value is not None and c.row > 1 else 8) for c in col)
        ws.column_dimensions[col[0].column_letter].width = min(max(10, L + 2), 70)
wb.save(xl)
json.dump({str(k): dict(caption=caps[k], note=notes[k], md=tabs[k].to_markdown(index=False)) for k in tabs},
          open(f"{T}/tables_md.json", "w"), indent=1)
print("ok", list(tabs))
