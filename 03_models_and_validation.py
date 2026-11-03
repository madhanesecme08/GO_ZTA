"""
Small-data surrogate modelling of processing-property relations in microwave-sintered GO-doped ZTA.

Inputs : GO content (wt%), sintering temperature (deg C)       -- the only two variables varied (N = 20, 5 x 4 full factorial)
Targets: relative density, Vickers hardness, Young's modulus, fracture toughness (digitised medians/means),
         Al2O3 and YSZ grain sizes, t-ZrO2 phase fraction (tabulated)

Validation schemes (all predictions are out-of-sample; all hyper-parameters are tuned inside the training folds):
  LOOCV       : leave-one-condition-out (20 folds)          -> interpolation inside the design grid
  LOGO-GO     : leave-one-GO-level-out (5 folds)            -> prediction at an unseen composition
  LOGO-T      : leave-one-temperature-out (4 folds)         -> prediction at an unseen temperature

Usage: python 03_models_and_validation.py <data_dir> <results_dir>
"""
import sys, json, warnings
import numpy as np, pandas as pd
from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, FunctionTransformer
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV, LeaveOneOut, KFold
from sklearn.dummy import DummyRegressor
warnings.filterwarnings("ignore")
SEED = 42
D = sys.argv[1] if len(sys.argv) > 1 else "data"
R = sys.argv[2] if len(sys.argv) > 2 else "results"
import os; os.makedirs(R, exist_ok=True)

df = pd.read_csv(f"{D}/GO_ZTA_master_dataset_N20.csv")
TARGETS = {"RD_pct": "Relative density (%)", "HV_GPa": "Vickers hardness (GPa)", "E_GPa": "Young's modulus (GPa)",
           "KIC_MPa_m05": "Fracture toughness (MPa m^0.5)", "d_Al2O3_um": "Al2O3 grain size (um)",
           "d_YSZ_um": "YSZ grain size (um)", "t_ZrO2_wt_pct": "t-ZrO2 fraction (wt%)"}
X = np.c_[df.GO_wt_pct.values, (df.Sintering_T_C.values - 1200.0) / 300.0]   # GO in wt%, T rescaled to [0,1]
groups_go = df.GO_wt_pct.values; groups_t = df.Sintering_T_C.values

# experimental scatter estimates (per-condition repeatability) ------------------------------------
# HV, E, KIC: IQR/1.349 of the published box plots (normal-equivalent SD); RD: half-length of published error bar
noise = {"HV_GPa": float(np.median(df.HV_GPa_IQR / 1.349)), "E_GPa": float(np.median(df.E_GPa_IQR / 1.349)),
         "KIC_MPa_m05": float(np.median(df.KIC_MPa_m05_IQR / 1.349)), "RD_pct": float(np.median(df.RD_halfbar))}

def doped(Xa):  # descriptor: dopant present (1) / absent (0)
    return np.c_[(Xa[:, 0] > 0).astype(float), Xa]

def models():
    inner = KFold(5, shuffle=True, random_state=SEED)
    k = C(1.0, (1e-3, 1e3)) * RBF(length_scale=[0.3, 0.5], length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(0.05, (1e-6, 1.0))
    return {
        "Mean (null)": DummyRegressor(),
        "Linear": LinearRegression(),
        "Quadratic RSM": make_pipeline(PolynomialFeatures(2, include_bias=False), StandardScaler(),
                                       RidgeCV(alphas=np.logspace(-4, 2, 25))),
        "Dopant-step linear": make_pipeline(FunctionTransformer(doped), LinearRegression()),
        "GPR (ARD-RBF)": make_pipeline(StandardScaler(), GaussianProcessRegressor(kernel=k, normalize_y=True,
                                       n_restarts_optimizer=5, random_state=SEED)),
        "SVR (RBF)": GridSearchCV(make_pipeline(StandardScaler(), SVR(kernel="rbf")),
                                  {"svr__C": [1, 10, 100], "svr__gamma": [0.1, 0.5, 2.0], "svr__epsilon": [0.01, 0.1]},
                                  cv=inner, scoring="neg_mean_squared_error"),
        "Random forest": RandomForestRegressor(n_estimators=500, min_samples_leaf=2, random_state=SEED),
        "Gradient boosting": GradientBoostingRegressor(n_estimators=300, max_depth=2, learning_rate=0.05,
                                                       subsample=0.8, random_state=SEED),
        "MLP (16-16)": make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(16, 16), solver="lbfgs", alpha=1e-2,
                                     max_iter=5000, random_state=SEED)),
    }

def splits(kind):
    n = len(df)
    if kind == "LOOCV":   return [(np.setdiff1d(np.arange(n), [i]), np.array([i])) for i in range(n)]
    g = groups_go if kind == "LOGO-GO" else groups_t
    return [(np.where(g != v)[0], np.where(g == v)[0]) for v in np.unique(g)]

def ystd_scale(y):  # targets scaled for SVR/MLP stability; inverse after prediction
    return y.mean(), y.std()

def oof(model, y, kind, return_std=False):
    pred = np.full(len(y), np.nan); sd = np.full(len(y), np.nan)
    for tr, te in splits(kind):
        mu, s = ystd_scale(y[tr])
        m = clone(model).fit(X[tr], (y[tr] - mu) / s)
        if return_std:
            p, ps = m.predict(X[te], return_std=True); sd[te] = ps * s
        else:
            p = m.predict(X[te])
        pred[te] = p * s + mu
    return (pred, sd) if return_std else pred

def metrics(y, p):
    res = y - p; ss = ((y - y.mean()) ** 2).sum()
    return dict(RMSE=float(np.sqrt(np.mean(res ** 2))), MAE=float(np.mean(np.abs(res))),
                R2=float(1 - (res ** 2).sum() / ss), MaxAE=float(np.abs(res).max()))

rows, preds = [], {}
for kind in ["LOOCV", "LOGO-GO", "LOGO-T"]:
    for name, mdl in models().items():
        for t in TARGETS:
            y = df[t].values.astype(float)
            if name.startswith("GPR"):
                p, s = oof(mdl, y, kind, return_std=True); preds[(kind, name, t, "sd")] = s
            else:
                p = oof(mdl, y, kind)
            preds[(kind, name, t)] = p
            r = metrics(y, p); r.update(scheme=kind, model=name, target=t)
            r["RMSE_over_noise"] = r["RMSE"] / noise[t] if t in noise else np.nan
            rows.append(r)
    print("done", kind, flush=True)
perf = pd.DataFrame(rows)[["scheme", "model", "target", "RMSE", "MAE", "R2", "MaxAE", "RMSE_over_noise"]]
perf.to_csv(f"{R}/cv_performance_all_models.csv", index=False)

# out-of-fold predictions table
op = df[["GO_wt_pct", "Sintering_T_C"] + list(TARGETS)].copy()
for (k, *rest), v in preds.items():
    if len(rest) == 2:
        op[f"{k}|{rest[0]}|{rest[1]}"] = v
    else:
        op[f"{k}|{rest[0]}|{rest[1]}|sd"] = v
op.to_csv(f"{R}/out_of_fold_predictions.csv", index=False)

# GP interval coverage (LOOCV) --------------------------------------------------------------------
cov = []
for t in TARGETS:
    y = df[t].values; p = preds[("LOOCV", "GPR (ARD-RBF)", t)]; s = preds[("LOOCV", "GPR (ARD-RBF)", t, "sd")]
    z = (y - p) / s
    cov.append(dict(target=t, coverage_95=float(np.mean(np.abs(z) <= 1.96)), mean_pred_sd=float(np.mean(s)),
                    mean_abs_z=float(np.mean(np.abs(z)))))
pd.DataFrame(cov).to_csv(f"{R}/gpr_loocv_interval_coverage.csv", index=False)

# noise ceiling ---------------------------------------------------------------------------------------
nc = []
for t, s in noise.items():
    v = df[t].var(ddof=1)
    nc.append(dict(target=t, scatter_SD=s, between_condition_SD=float(np.sqrt(v)), R2_ceiling=float(1 - s ** 2 / v)))
pd.DataFrame(nc).to_csv(f"{R}/noise_ceiling.csv", index=False)
json.dump(noise, open(f"{R}/noise_estimates.json", "w"), indent=2)

# two-way additive ANOVA (GO x T full factorial, one value per cell) ----------------------------------
an = []
for t in TARGETS:
    y = df.pivot(index="GO_wt_pct", columns="Sintering_T_C", values=t).values
    g = y.mean(); a, b = y.shape
    ss_go = b * ((y.mean(1) - g) ** 2).sum(); ss_t = a * ((y.mean(0) - g) ** 2).sum()
    ss_tot = ((y - g) ** 2).sum(); ss_res = ss_tot - ss_go - ss_t
    df_go, df_t, df_r = a - 1, b - 1, (a - 1) * (b - 1)
    from scipy.stats import f as fdist
    F_go = (ss_go / df_go) / (ss_res / df_r); F_t = (ss_t / df_t) / (ss_res / df_r)
    an.append(dict(target=t, eta2_GO=ss_go / ss_tot, eta2_T=ss_t / ss_tot, eta2_residual=ss_res / ss_tot,
                   F_GO=F_go, p_GO=1 - fdist.cdf(F_go, df_go, df_r), F_T=F_t, p_T=1 - fdist.cdf(F_t, df_t, df_r)))
pd.DataFrame(an).to_csv(f"{R}/two_way_anova.csv", index=False)

# augmentation experiment ------------------------------------------------------------------------------
rng = np.random.default_rng(SEED)
def smote_like(Xa, Ya, n_new, k=5, jitter=0.0):
    """SMOTE-style interpolation in the joint (X, y) space, as used for regression augmentation."""
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=min(k + 1, len(Xa))).fit(Xa)
    idx = nn.kneighbors(Xa, return_distance=False)
    Xn, Yn = [], []
    for _ in range(n_new):
        i = rng.integers(len(Xa)); j = idx[i][rng.integers(1, idx.shape[1])]; lam = rng.random()
        Xn.append(Xa[i] + lam * (Xa[j] - Xa[i])); Yn.append(Ya[i] + lam * (Ya[j] - Ya[i]))
    Yn = np.array(Yn); Yn = Yn + jitter * Ya.std() * rng.standard_normal(Yn.shape)
    return np.array(Xn), Yn

aug_rows = []
# GPR is excluded here only because exact GP training on ~1000 points x 20 folds x 4 targets is prohibitively slow.
aug_models = {"Random forest": models()["Random forest"], "Gradient boosting": models()["Gradient boosting"],
              "MLP (16-16)": models()["MLP (16-16)"]}
for t in ["HV_GPa", "KIC_MPa_m05", "E_GPa", "RD_pct"]:
    y = df[t].values
    for name, mdl in aug_models.items():
        # (a) leaky protocol: augment the full data set first (N=1000), then random 5-fold CV on the pooled set
        Xa, Ya = smote_like(X, y, 980, jitter=0.02)
        Xall, Yall = np.r_[X, Xa], np.r_[y, Ya]
        p_leak = np.zeros_like(Yall)
        for tr, te in KFold(5, shuffle=True, random_state=SEED).split(Xall):
            mu, s = Yall[tr].mean(), Yall[tr].std()
            mm = clone(mdl)
            p_leak[te] = mm.fit(Xall[tr], (Yall[tr] - mu) / s).predict(Xall[te]) * s + mu
        r_leak = metrics(Yall, p_leak)
        # (b) correct protocol: LOOCV on real samples; augmentation generated from the training fold only
        p_ok = np.zeros_like(y); p_noaug = preds[("LOOCV", name, t)]
        for tr, te in splits("LOOCV"):
            Xa, Ya = smote_like(X[tr], y[tr], 980, jitter=0.02)
            Xt, Yt = np.r_[X[tr], Xa], np.r_[y[tr], Ya]
            mu, s = Yt.mean(), Yt.std()
            p_ok[te] = clone(mdl).fit(Xt, (Yt - mu) / s).predict(X[te]) * s + mu
        r_ok = metrics(y, p_ok); r_no = metrics(y, p_noaug)
        aug_rows.append(dict(target=t, model=name,
                             R2_leaky_augmented_random5fold=r_leak["R2"], RMSE_leaky=r_leak["RMSE"],
                             R2_LOOCV_no_augmentation=r_no["R2"], RMSE_no_aug=r_no["RMSE"],
                             R2_LOOCV_train_fold_augmentation=r_ok["R2"], RMSE_train_fold_aug=r_ok["RMSE"]))
    print("aug", t, flush=True)
pd.DataFrame(aug_rows).to_csv(f"{R}/augmentation_leakage_experiment.csv", index=False)
print(perf.pivot_table(index=["scheme", "model"], columns="target", values="R2").round(2).to_string())
