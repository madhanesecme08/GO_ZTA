"""
Physically motivated descriptor analysis for GO-doped ZTA (N = 20).
  (1) Porosity dependence of E and HV: exponential (Spriggs-type) model  P_x = P0 * exp(-b * p), p = porosity fraction
  (2) Hardness vs Al2O3 grain size (Hall-Petch-type d^-1/2 term) with and without porosity
  (3) Two-stage 'chain' surrogate: GPR(GO,T) -> relative density -> exponential porosity law -> E, HV
      (validated by LOOCV and LOGO-GO with the GPR refitted inside every fold)
  (4) Spearman rank-correlation matrix between measured/derived quantities
  (5) Rule-of-mixtures (Voigt / Reuss) bounds for the dense 80/20 wt% Al2O3/YSZ composite
Usage: python 04_physics_descriptor_analysis.py <data_dir> <results_dir>
"""
import sys, json, numpy as np, pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
D, R = (sys.argv[1], sys.argv[2]) if len(sys.argv) > 2 else ("data", "results")
SEED = 42
df = pd.read_csv(f"{D}/GO_ZTA_master_dataset_N20.csv")
p = (100 - df.RD_pct.values) / 100.0
out = {}

def ols(x, y):
    r = stats.linregress(x, y)
    n = len(x); tcrit = stats.t.ppf(0.975, n - 2)
    return dict(slope=r.slope, slope_ci95=tcrit * r.stderr, intercept=r.intercept,
                intercept_ci95=tcrit * r.intercept_stderr, r2=r.rvalue ** 2, p_value=r.pvalue)

# (1) exponential porosity law: ln P = ln P0 - b p
for prop in ["E_GPa", "HV_GPa"]:
    f = ols(p, np.log(df[prop].values))
    out[f"porosity_law_{prop}"] = dict(P0=float(np.exp(f["intercept"])),
        P0_ci95=[float(np.exp(f["intercept"] - f["intercept_ci95"])), float(np.exp(f["intercept"] + f["intercept_ci95"]))],
        b=float(-f["slope"]), b_ci95=float(f["slope_ci95"]), r2=float(f["r2"]), p_value=float(f["p_value"]))

# (2) hardness: Hall-Petch-type term alone and jointly with porosity
dinv = df.d_Al2O3_um.values ** -0.5
hp = ols(dinv, df.HV_GPa.values)
A = np.c_[np.ones(20), p, dinv]
coef, *_ = np.linalg.lstsq(A, df.HV_GPa.values, rcond=None)
res = df.HV_GPa.values - A @ coef
r2_joint = 1 - (res ** 2).sum() / ((df.HV_GPa - df.HV_GPa.mean()) ** 2).sum()
s2 = (res ** 2).sum() / (20 - 3); cov = s2 * np.linalg.inv(A.T @ A); se = np.sqrt(np.diag(cov))
tvals = coef / se; pvals = 2 * (1 - stats.t.cdf(np.abs(tvals), 17))
# partial correlation of HV with d^-1/2 controlling for porosity
def resid(y, x): b = np.polyfit(x, y, 1); return y - np.polyval(b, x)
pc = stats.pearsonr(resid(df.HV_GPa.values, p), resid(dinv, p))
out["hall_petch_alone"] = {k: float(v) for k, v in hp.items()}
out["HV_joint_porosity_dinv"] = dict(coef_intercept=float(coef[0]), coef_porosity=float(coef[1]), coef_dinv=float(coef[2]),
                                     se=[float(x) for x in se], p_values=[float(x) for x in pvals], r2=float(r2_joint),
                                     partial_r_HV_dinv_given_porosity=float(pc[0]), partial_p=float(pc[1]))
# porosity alone, linear, for HV (for comparison)
out["HV_vs_porosity_linear"] = {k: float(v) for k, v in ols(p, df.HV_GPa.values).items()}

# (3) chain surrogate
X = np.c_[df.GO_wt_pct.values, (df.Sintering_T_C.values - 1200) / 300.0]
k = C(1.0, (1e-3, 1e3)) * RBF([0.3, 0.5], (1e-2, 1e2)) + WhiteKernel(0.05, (1e-6, 1.0))
gpr = make_pipeline(StandardScaler(), GaussianProcessRegressor(kernel=k, normalize_y=True, n_restarts_optimizer=5, random_state=SEED))
def splits(kind):
    if kind == "LOOCV": return [(np.setdiff1d(np.arange(20), [i]), np.array([i])) for i in range(20)]
    g = df.GO_wt_pct.values
    return [(np.where(g != v)[0], np.where(g == v)[0]) for v in np.unique(g)]
chain = []
for kind in ["LOOCV", "LOGO-GO"]:
    for prop in ["E_GPa", "HV_GPa"]:
        pred = np.zeros(20)
        for tr, te in splits(kind):
            m = clone(gpr).fit(X[tr], df.RD_pct.values[tr])
            p_te = (100 - m.predict(X[te])) / 100
            f = ols(p[tr], np.log(df[prop].values[tr]))          # porosity law refitted on training fold only
            pred[te] = np.exp(f["intercept"] + f["slope"] * p_te)
        y = df[prop].values; e = y - pred
        chain.append(dict(scheme=kind, target=prop, model="Chain: GPR(RD) + porosity law",
                          RMSE=float(np.sqrt(np.mean(e ** 2))), MAE=float(np.mean(np.abs(e))),
                          R2=float(1 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum()), MaxAE=float(np.abs(e).max())))
        df[f"chain_{kind}_{prop}"] = pred
pd.DataFrame(chain).to_csv(f"{R}/chain_model_performance.csv", index=False)
df.to_csv(f"{R}/chain_model_predictions.csv", index=False)

# (4) Spearman correlation matrix
cols = ["GO_wt_pct", "Sintering_T_C", "RD_pct", "HV_GPa", "E_GPa", "KIC_MPa_m05", "d_Al2O3_um", "d_YSZ_um", "t_ZrO2_wt_pct", "m_ZrO2_wt_pct"]
rho, pv = stats.spearmanr(df[cols].values)
pd.DataFrame(rho, index=cols, columns=cols).to_csv(f"{R}/spearman_rho.csv")
pd.DataFrame(pv, index=cols, columns=cols).to_csv(f"{R}/spearman_p.csv")
# doped-only subset (GO > 0): correlations of K_IC with descriptors
sub = df[df.GO_wt_pct > 0]
kc = {}
for c in ["RD_pct", "HV_GPa", "E_GPa", "d_Al2O3_um", "d_YSZ_um", "t_ZrO2_wt_pct"]:
    r_all = stats.spearmanr(df.KIC_MPa_m05, df[c]); r_sub = stats.spearmanr(sub.KIC_MPa_m05, sub[c])
    kc[c] = dict(rho_all=float(r_all[0]), p_all=float(r_all[1]), rho_doped=float(r_sub[0]), p_doped=float(r_sub[1]))
out["KIC_spearman"] = kc

# (5) rule-of-mixtures bounds (constituent values from literature; see manuscript)
rho_A, rho_Z = 3.98, 6.05          # g cm-3  (alpha-Al2O3; 3Y-TZP)
E_A, E_Z = 416.0, 210.0            # GPa
wA, wZ = 0.80, 0.20
vZ = (wZ / rho_Z) / (wA / rho_A + wZ / rho_Z); vA = 1 - vZ
out["rule_of_mixtures"] = dict(v_YSZ=vZ, rho_theoretical=1 / (wA / rho_A + wZ / rho_Z),
                               E_Voigt=vA * E_A + vZ * E_Z, E_Reuss=1 / (vA / E_A + vZ / E_Z),
                               max_reported_E=float(df.E_GPa.max()))
json.dump(out, open(f"{R}/physics_descriptor_results.json", "w"), indent=2)
print(json.dumps(out, indent=1))
print(pd.DataFrame(chain).round(3))
