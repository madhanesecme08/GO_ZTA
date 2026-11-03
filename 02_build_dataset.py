"""
Build the analysis dataset (N = 20 processing conditions) for GO-doped ZTA.
Sources (all from Mohankumar et al., Phys. Scr. 101 (2026) 025007, doi:10.1088/1402-4896/ae32bb):
  * Table 1  - Rietveld phase fractions (exact, tabulated)            -> transcribed below
  * Table 2  - Al2O3 grain size mean +/- SD (exact, tabulated)          -> transcribed below
  * Table 3  - YSZ grain size mean +/- SD (exact, tabulated)            -> transcribed below
  * Figs 6-9 - relative density, HV, E, K_IC (digitised, see script 01)-> data/digitized_long.csv
Also compares the unverified CSV supplied in the project folder against the published record.
Usage: python 02_build_dataset.py <data_dir> [<unverified_csv>]
"""
import sys, numpy as np, pandas as pd
D = sys.argv[1] if len(sys.argv) > 1 else "data"
GO = [0.0, 0.25, 0.5, 0.75, 1.0]; TT = [1200, 1300, 1400, 1500]
# Table 1: Rp, Rwp, Rexp, chi2, alpha-Al2O3, t-ZrO2, m-ZrO2 (rows: GO level, then T)
T1 = [
 [9.2,15.4,11.3,1.88,76.2,18.6,5.2],[11.3,14.0,10.9,1.66,75.9,18.4,5.7],[14.3,12.6,11.5,1.78,76.6,18.8,4.6],[9.8,13.4,9.5,2.89,76.3,19.0,4.7],
 [10.5,14.8,8.6,1.43,76.5,19.1,4.4],[12.7,13.6,12.0,1.67,76.6,18.7,4.7],[14.4,12.8,8.8,1.44,76.3,18.9,4.8],[8.9,16.2,11.8,1.62,76.3,19.4,4.3],
 [11.6,14.3,12.4,1.29,75.9,18.2,5.9],[16.7,10.9,7.9,1.14,76.3,18.5,5.2],[13.6,14.3,8.9,1.23,76.2,17.6,6.2],[13.2,10.6,12.5,1.34,76.5,18.1,5.4],
 [11.4,12.5,13.4,1.24,76.4,17.4,6.2],[10.6,16.8,9.7,1.31,76.5,17.8,5.7],[14.8,12.4,10.8,1.59,76.2,17.2,6.6],[15.7,13.9,10.1,1.64,76.5,17.5,6.0],
 [14.7,15.2,9.2,1.33,76.2,16.8,7.0],[13.4,13.5,11.1,1.42,75.9,16.5,7.6],[13.8,14.9,8.6,1.13,76.3,17.2,6.5],[10.7,13.8,7.9,1.45,76.5,16.9,6.6]]
# Table 2: Al2O3 grain size mean, SD  (rows GO; cols T)
T2m = [[5.51,5.23,5.48,5.60],[3.71,2.49,2.76,3.39],[3.44,3.23,3.29,3.83],[5.45,5.04,5.16,5.62],[5.62,5.38,5.46,5.95]]
T2s = [[1.24,0.92,1.62,1.73],[0.72,0.47,0.38,0.70],[0.51,0.66,0.48,0.60],[1.20,0.88,1.62,1.64],[1.67,1.17,1.26,1.52]]
# Table 3: YSZ grain size mean, SD
T3m = [[2.36,2.36,2.99,2.99],[2.00,1.59,1.94,2.06],[2.29,1.85,2.35,2.43],[3.11,2.75,2.95,3.04],[3.18,2.85,3.47,3.49]]
T3s = [[0.74,0.60,0.87,0.74],[0.31,0.43,0.45,0.64],[0.34,0.41,0.63,0.62],[0.94,0.40,0.96,0.85],[0.65,0.47,1.17,0.80]]
rows = []
k = 0
for i, g in enumerate(GO):
    for j, t in enumerate(TT):
        Rp, Rwp, Rexp, chi2, a, tz, mz = T1[k]; k += 1
        rows.append(dict(GO_wt_pct=g, Sintering_T_C=t, alpha_Al2O3_wt_pct=a, t_ZrO2_wt_pct=tz, m_ZrO2_wt_pct=mz,
                         Rietveld_Rp=Rp, Rietveld_Rwp=Rwp, Rietveld_Rexp=Rexp, Rietveld_chi2=chi2,
                         d_Al2O3_um=T2m[i][j], d_Al2O3_SD_um=T2s[i][j], d_YSZ_um=T3m[i][j], d_YSZ_SD_um=T3s[i][j]))
tab = pd.DataFrame(rows)
tab["t_fraction_of_ZrO2"] = tab.t_ZrO2_wt_pct / (tab.t_ZrO2_wt_pct + tab.m_ZrO2_wt_pct)
tab["phase_sum_wt_pct"] = tab.alpha_Al2O3_wt_pct + tab.t_ZrO2_wt_pct + tab.m_ZrO2_wt_pct
tab.to_csv(f"{D}/published_tabulated_Tables1-3.csv", index=False)

dig = pd.read_csv(f"{D}/digitized_long.csv")
def pick(prop, col, name):
    d = dig[dig.property == prop][["GO_wt_pct", "Sintering_T_C", col]].rename(columns={col: name})
    return d
m = tab.copy()
rd = dig[dig.property == "Relative_Density_pct"]
m = m.merge(rd[["GO_wt_pct","Sintering_T_C","mean","whisker_upper","whisker_lower"]].rename(columns={
    "mean":"RD_pct","whisker_upper":"RD_err_upper","whisker_lower":"RD_err_lower"}), on=["GO_wt_pct","Sintering_T_C"])
for prop, short in [("Vickers_Hardness_GPa","HV_GPa"),("Youngs_Modulus_GPa","E_GPa"),("Fracture_Toughness_MPa_m0p5","KIC_MPa_m05")]:
    d = dig[dig.property == prop][["GO_wt_pct","Sintering_T_C","median","Q1","Q3","whisker_lower","whisker_upper"]]
    d = d.rename(columns={"median":short,"Q1":short+"_Q1","Q3":short+"_Q3","whisker_lower":short+"_min","whisker_upper":short+"_max"})
    m = m.merge(d, on=["GO_wt_pct","Sintering_T_C"])
m["porosity_pct"] = 100 - m.RD_pct
m["RD_halfbar"] = (m.RD_err_upper - m.RD_err_lower) / 2
for s in ["HV_GPa","E_GPa","KIC_MPa_m05"]:
    m[s+"_IQR"] = m[s+"_Q3"] - m[s+"_Q1"]
m["data_provenance"] = "Tables1-3 exact; RD/HV/E/KIC digitised from published Figs 6-9"
m = m.round(4)
m.to_csv(f"{D}/GO_ZTA_master_dataset_N20.csv", index=False)
print(m[["GO_wt_pct","Sintering_T_C","RD_pct","HV_GPa","E_GPa","KIC_MPa_m05","d_Al2O3_um","d_YSZ_um","t_ZrO2_wt_pct"]].to_string())
print("phase sum range:", tab.phase_sum_wt_pct.min(), tab.phase_sum_wt_pct.max())

# ---- comparison with the unverified CSV ----
if len(sys.argv) > 2:
    u = pd.read_csv(sys.argv[2])
    c = u.merge(m, on=["GO_wt_pct","Sintering_T_C"])
    pairs = [("Relative_Density_pct","RD_pct"),("Vickers_Hardness_GPa","HV_GPa"),("Youngs_Modulus_GPa","E_GPa"),
             ("Fracture_Toughness_MPa_m0p5","KIC_MPa_m05"),("Al2O3_Grain_Size_um","d_Al2O3_um"),
             ("YSZ_Grain_Size_um","d_YSZ_um"),("t_ZrO2_wt_pct_x","t_ZrO2_wt_pct_y"),("m_ZrO2_wt_pct_x","m_ZrO2_wt_pct_y")]
    out = []
    for a, b in pairs:
        diff = c[a] - c[b]
        tol = {"Relative_Density_pct":0.15,"Vickers_Hardness_GPa":0.1,"Youngs_Modulus_GPa":1.5,"Fracture_Toughness_MPa_m0p5":0.06}.get(a, 0.005)
        out.append(dict(variable=a.replace("_x",""), csv_min=c[a].min(), csv_max=c[a].max(), published_min=c[b].min(), published_max=c[b].max(),
                        mean_abs_diff=diff.abs().mean(), max_abs_diff=diff.abs().max(),
                        n_conditions_agreeing=int((diff.abs() <= tol).sum()), agreement_tolerance=tol))
    cmp_ = pd.DataFrame(out).round(3)
    cmp_.to_csv(f"{D}/unverified_CSV_vs_published_comparison.csv", index=False)
    long = c[["GO_wt_pct","Sintering_T_C"] + [p for pr in pairs for p in pr]]
    long.to_csv(f"{D}/unverified_CSV_vs_published_by_condition.csv", index=False)
    print(cmp_.to_string())
