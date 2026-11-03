"""
Publication figures for the GO-ZTA small-data modelling study.
Every plotted value is read from data/ or results/ (no hand-entered numbers).
Output per figure: PDF (vector), EPS (vector), TIFF (1000 dpi, LZW) and PNG preview.
Usage: python 05_make_figures.py <data_dir> <results_dir> <fig_dir>
"""
import sys, os, json, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
D, R, F = sys.argv[1:4] if len(sys.argv) > 3 else ("data", "results", "figures")
os.makedirs(F, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "lines.linewidth": 1.0,
                     "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42, "ps.fonttype": 42})
MM = 1 / 25.4
TCOL = {1200: "#1b6ca8", 1300: "#2ca25f", 1400: "#e6a100", 1500: "#c0392b"}   # colour-blind-safe-ish, plus markers
TMARK = {1200: "o", 1300: "s", 1400: "^", 1500: "D"}
df = pd.read_csv(f"{D}/GO_ZTA_master_dataset_N20.csv")
perf = pd.read_csv(f"{R}/cv_performance_all_models.csv")
oof = pd.read_csv(f"{R}/out_of_fold_predictions.csv")
LAB = {"RD_pct": "Relative density (%)", "HV_GPa": "$H_V$ (GPa)", "E_GPa": "$E$ (GPa)", "KIC_MPa_m05": "$K_{IC}$ (MPa m$^{1/2}$)",
       "d_Al2O3_um": "$d_{Al_2O_3}$ (µm)", "d_YSZ_um": "$d_{YSZ}$ (µm)", "t_ZrO2_wt_pct": "t-ZrO$_2$ (wt%)", "m_ZrO2_wt_pct": "m-ZrO$_2$ (wt%)"}
TARGETS = ["RD_pct", "HV_GPa", "E_GPa", "KIC_MPa_m05", "d_Al2O3_um", "d_YSZ_um", "t_ZrO2_wt_pct"]

def save(fig, name):
    for ext in ["pdf", "eps"]:
        fig.savefig(f"{F}/{name}.{ext}")
    fig.savefig(f"{F}/{name}.tiff", dpi=1000, pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(f"{F}/{name}.png", dpi=200)
    plt.close(fig)

def panel(ax, s):
    ax.text(-0.02, 1.02, s, transform=ax.transAxes, fontweight="bold", fontsize=9, ha="right", va="bottom")

# ---------------------------------------------------------------- Figure 1: workflow
fig, ax = plt.subplots(figsize=(190 * MM, 70 * MM)); ax.set_xlim(0, 100); ax.set_ylim(0, 32); ax.axis("off")
boxes = [
    (1, 17, 17, 13, "Published record\nPhys. Scr. 101,\n025007 (2026)\n5 GO × 4 T levels\n(N = 20)"),
    (21, 17, 17, 13, "Data curation\nTables 1–3\ntranscribed;\nFigs 6–9 digitised\nand validated"),
    (41, 17, 17, 13, "Surrogate models\nlinear, RSM,\ndopant-step, GPR,\nSVR, RF, GBR, MLP\n(GO wt%, T)"),
    (61, 17, 17, 13, "Validation\nLOOCV,\nleave-one-GO-out,\nleave-one-T-out,\nnoise ceiling"),
    (81, 17, 17, 13, "Outputs\naccuracy vs.\nscatter,\nuncertainty maps,\nnext experiments"),
    (21, 1, 17, 12, "Integrity checks\nCSV vs. record;\nphase sums;\nRoM bounds"),
    (41, 1, 17, 12, "Physical\ndescriptors\nporosity law,\nHall–Petch test,\nt-ZrO$_2$ vs $K_{IC}$"),
    (61, 1, 17, 12, "Augmentation\ntest\nleaky vs.\nfold-internal\nsynthesis"),
]
for x, y, w, h, t in boxes:
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.2", fc="#f3f6fa", ec="#34495e", lw=0.7))
    ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=6.4, linespacing=1.25)
arr = dict(arrowstyle="-|>", color="#34495e", lw=0.8)
for x0, x1 in [(18.4, 20.6), (38.4, 40.6), (58.4, 60.6), (78.4, 80.6)]:
    ax.annotate("", (x1, 23.5), (x0, 23.5), arrowprops=arr)
for xc in [29.5, 49.5, 69.5]:
    ax.annotate("", (xc, 16.7), (xc, 13.3), arrowprops=arr)
save(fig, "Figure_01")

# ---------------------------------------------------------------- Figure 2: data overview
fig, axs = plt.subplots(2, 4, figsize=(190 * MM, 95 * MM))
spec = [("RD_pct", "RD_err_lower", "RD_err_upper", "bar"), ("HV_GPa", "HV_GPa_Q1", "HV_GPa_Q3", "iqr"),
        ("E_GPa", "E_GPa_Q1", "E_GPa_Q3", "iqr"), ("KIC_MPa_m05", "KIC_MPa_m05_Q1", "KIC_MPa_m05_Q3", "iqr"),
        ("d_Al2O3_um", "d_Al2O3_SD_um", None, "sd"), ("d_YSZ_um", "d_YSZ_SD_um", None, "sd"),
        ("t_ZrO2_wt_pct", None, None, None), ("m_ZrO2_wt_pct", None, None, None)]
for k, (ax, (v, lo, hi, kind)) in enumerate(zip(axs.flat, spec)):
    for j, T in enumerate([1200, 1300, 1400, 1500]):
        s = df[df.Sintering_T_C == T].sort_values("GO_wt_pct"); x = s.GO_wt_pct + (j - 1.5) * 0.018
        if kind in ("bar", "iqr"):
            yerr = [s[v] - s[lo], s[hi] - s[v]]
        elif kind == "sd":
            yerr = s[lo]
        else:
            yerr = None
        ax.errorbar(x, s[v], yerr=yerr, fmt=TMARK[T] + "-", ms=3, color=TCOL[T], lw=0.7, elinewidth=0.6, capsize=1.5,
                    label=f"{T} °C")
    ax.set_xlabel("GO content (wt%)"); ax.set_ylabel(LAB[v]); ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    panel(ax, f"({'abcdefgh'[k]})")
    if k < 4:
        ax.text(0.98, 0.04, "digitised", transform=ax.transAxes, ha="right", fontsize=6, style="italic", color="0.35")
hh, ll = axs[0, 0].get_legend_handles_labels()
fig.tight_layout(w_pad=0.8, h_pad=0.8, rect=(0, 0, 1, 0.95))
fig.legend(hh, ll, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.0))
save(fig, "Figure_02")

# ---------------------------------------------------------------- Figure 3: benchmark heatmaps (RMSE / between-condition SD)
order = ["Mean (null)", "Linear", "Quadratic RSM", "Dopant-step linear", "GPR (ARD-RBF)", "SVR (RBF)",
         "Random forest", "Gradient boosting", "MLP (16-16)"]
sdy = {t: df[t].std(ddof=0) for t in TARGETS}
fig, axs = plt.subplots(1, 3, figsize=(190 * MM, 70 * MM), sharey=True)
for ax, sch, lab in zip(axs, ["LOOCV", "LOGO-T", "LOGO-GO"], ["Leave-one-condition-out", "Leave-one-temperature-out", "Leave-one-GO-level-out"]):
    M = np.array([[perf[(perf.scheme == sch) & (perf.model == m) & (perf.target == t)].RMSE.values[0] / sdy[t]
                   for t in TARGETS] for m in order])
    im = ax.imshow(M, cmap="RdYlBu_r", vmin=0, vmax=2, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=5.6,
                    color="white" if (M[i, j] > 1.6 or M[i, j] < 0.25) else "black")
    ax.set_xticks(range(len(TARGETS))); ax.set_xticklabels([LAB[t].split(" (")[0] for t in TARGETS], rotation=45, ha="right")
    ax.set_yticks(range(len(order))); ax.set_yticklabels(order); ax.set_title(f"({'abc'[['LOOCV','LOGO-T','LOGO-GO'].index(sch)]}) " + lab, fontsize=8)
cb = fig.colorbar(im, ax=axs, shrink=0.8, pad=0.01); cb.set_label("RMSE / SD$_y$  (1 = mean predictor)")
save(fig, "Figure_03")

# ---------------------------------------------------------------- Figure 4: GPR LOOCV parity with 95% intervals
cov = pd.read_csv(f"{R}/gpr_loocv_interval_coverage.csv").set_index("target")
noise = json.load(open(f"{R}/noise_estimates.json"))
fig, axs = plt.subplots(2, 4, figsize=(190 * MM, 92 * MM))
for k, (ax, t) in enumerate(zip(axs.flat, TARGETS)):
    y = df[t].values; p = oof[f"LOOCV|GPR (ARD-RBF)|{t}"].values; s = oof[f"LOOCV|GPR (ARD-RBF)|{t}|sd"].values
    for T in TCOL:
        m = df.Sintering_T_C.values == T
        ax.errorbar(y[m], p[m], yerr=1.96 * s[m], fmt=TMARK[T], ms=3, color=TCOL[T], elinewidth=0.5, capsize=1, lw=0)
    lo, hi = min(y.min(), (p - 1.96 * s).min()), max(y.max(), (p + 1.96 * s).max()); pad = 0.05 * (hi - lo)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=0.6)
    if t in noise:
        ax.fill_between([lo - pad, hi + pad], [lo - pad - noise[t], hi + pad - noise[t]], [lo - pad + noise[t], hi + pad + noise[t]],
                        color="0.85", lw=0, zorder=0)
    ax.set_xlim(lo - pad, hi + pad); ax.set_ylim(lo - pad, hi + pad); ax.set_aspect("equal")
    r = perf[(perf.scheme == "LOOCV") & (perf.model == "GPR (ARD-RBF)") & (perf.target == t)].iloc[0]
    ax.text(0.04, 0.96, f"RMSE = {r.RMSE:.3g}\n95% PI cover. = {cov.loc[t, 'coverage_95']:.2f}", transform=ax.transAxes,
            va="top", fontsize=6)
    ax.set_xlabel("Reported " + LAB[t], fontsize=7); ax.set_ylabel("LOOCV predicted", fontsize=7); panel(ax, f"({'abcdefg'[k]})")
axs.flat[-1].axis("off")
h = [plt.Line2D([], [], marker=TMARK[T], color=TCOL[T], ls="", ms=4, label=f"{T} °C") for T in TCOL]
h.append(plt.Rectangle((0, 0), 1, 1, fc="0.85", label="± experimental scatter"))
axs.flat[-1].legend(handles=h, loc="center", frameon=False)
fig.tight_layout(w_pad=0.6, h_pad=0.8)
save(fig, "Figure_04")

# ---------------------------------------------------------------- Figure 5: LOGO-GO failure at unseen compositions
fig, axs = plt.subplots(1, 3, figsize=(190 * MM, 58 * MM))
for k, (ax, t) in enumerate(zip(axs, ["HV_GPa", "KIC_MPa_m05", "E_GPa"])):
    g = df.groupby("GO_wt_pct")[t].mean()
    ax.plot(g.index, g.values, "ko-", ms=4, label="reported (mean over T)")
    for mname, c, mk in [("GPR (ARD-RBF)", "#c0392b", "s"), ("Dopant-step linear", "#1b6ca8", "^"), ("Random forest", "#2ca25f", "v")]:
        pg = pd.Series(oof[f"LOGO-GO|{mname}|{t}"].values, index=df.GO_wt_pct).groupby(level=0).mean()
        ax.plot(pg.index, pg.values, mk + "--", color=c, ms=3.5, lw=0.8, label=f"{mname} (GO level held out)")
    ax.set_xlabel("GO content (wt%)"); ax.set_ylabel(LAB[t]); ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0]); panel(ax, f"({'abc'[k]})")
hh, ll = axs[0].get_legend_handles_labels()
fig.tight_layout(w_pad=1.0, rect=(0, 0.1, 1, 1))
fig.legend(hh, ll, loc="lower center", ncol=4, frameon=False, fontsize=6.5)
save(fig, "Figure_05")

# ---------------------------------------------------------------- Figure 6: physical descriptors
phys = json.load(open(f"{R}/physics_descriptor_results.json"))
p = 100 - df.RD_pct.values
fig, axs = plt.subplots(1, 4, figsize=(190 * MM, 50 * MM))
GOC = plt.cm.viridis(np.linspace(0, 0.9, 5)); GOL = [0, 0.25, 0.5, 0.75, 1.0]
def goscatter(ax, x, y):
    for c, gl in zip(GOC, GOL):
        m = df.GO_wt_pct.values == gl
        ax.scatter(x[m], y[m], s=12, color=c, edgecolor="k", lw=0.3, label=f"{gl:g} wt% GO", zorder=3)
pp = np.linspace(0.8, 3.9, 50)
for ax, prop, key, lt in [(axs[0], "HV_GPa", "porosity_law_HV_GPa", "a"), (axs[1], "E_GPa", "porosity_law_E_GPa", "b")]:
    goscatter(ax, p, df[prop].values)
    q = phys[key]; ax.plot(pp, q["P0"] * np.exp(-q["b"] * pp / 100), "k-", lw=0.8)
    ax.text(0.97, 0.96, f"$P_0$ = {q['P0']:.3g}\n$b$ = {q['b']:.2f} ± {q['b_ci95']:.2f}\n$R^2$ = {q['r2']:.2f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=6)
    ax.set_xlabel("Porosity (%)"); ax.set_ylabel(LAB[prop]); panel(ax, f"({lt})")
# partial residual plot: HV residual (after porosity) vs d^-1/2
lin = phys["HV_vs_porosity_linear"]
resid = df.HV_GPa.values - (lin["intercept"] + lin["slope"] * p / 100)
goscatter(axs[2], df.d_Al2O3_um.values ** -0.5, resid)
axs[2].axhline(0, color="k", lw=0.5)
axs[2].text(0.97, 0.96, f"partial r = {phys['HV_joint_porosity_dinv']['partial_r_HV_dinv_given_porosity']:.2f}\n"
            f"p = {phys['HV_joint_porosity_dinv']['partial_p']:.2f}", transform=axs[2].transAxes, ha="right", va="top", fontsize=6)
axs[2].set_xlabel("$d_{Al_2O_3}^{-1/2}$ (µm$^{-1/2}$)"); axs[2].set_ylabel("$H_V$ residual after porosity (GPa)"); panel(axs[2], "(c)")
goscatter(axs[3], df.t_ZrO2_wt_pct.values, df.KIC_MPa_m05.values)
kc = phys["KIC_spearman"]["t_ZrO2_wt_pct"]
axs[3].text(0.03, 0.96, f"doped: ρ = {kc['rho_doped']:.2f}\nall: ρ = {kc['rho_all']:.2f}", transform=axs[3].transAxes, va="top", fontsize=6)
axs[3].set_xlabel("t-ZrO$_2$ (wt%)"); axs[3].set_ylabel(LAB["KIC_MPa_m05"]); panel(axs[3], "(d)")
hh, ll = axs[3].get_legend_handles_labels()
fig.tight_layout(w_pad=0.8, rect=(0, 0.1, 1, 1))
fig.legend(hh, ll, loc="lower center", ncol=5, frameon=False, fontsize=6.5)
save(fig, "Figure_06")

# ---------------------------------------------------------------- Figure 7: augmentation / leakage
aug = pd.read_csv(f"{R}/augmentation_leakage_experiment.csv")
fig, axs = plt.subplots(1, 4, figsize=(190 * MM, 52 * MM), sharey=True)
for k, (ax, t) in enumerate(zip(axs, ["RD_pct", "HV_GPa", "E_GPa", "KIC_MPa_m05"])):
    s = aug[aug.target == t]; x = np.arange(len(s)); w = 0.26
    ax.bar(x - w, s.R2_leaky_augmented_random5fold, w, color="#c0392b", label="augment-then-split (leaky)")
    ax.bar(x, s.R2_LOOCV_no_augmentation, w, color="#7f8c8d", label="LOOCV, no augmentation")
    ax.bar(x + w, s.R2_LOOCV_train_fold_augmentation, w, color="#1b6ca8", label="LOOCV, fold-internal augmentation")
    ax.set_xticks(x); ax.set_xticklabels([{"Random forest": "RF", "Gradient boosting": "GBR", "MLP (16-16)": "MLP"}[m] for m in s.model], fontsize=7)
    ax.set_title(LAB[t]); ax.axhline(0, color="k", lw=0.5); ax.set_ylim(-0.15, 1.05); panel(ax, f"({'abcd'[k]})")
axs[0].set_ylabel("$R^2$ on evaluation samples")
hh, ll = axs[0].get_legend_handles_labels()
fig.tight_layout(w_pad=0.5, rect=(0, 0.08, 1, 1))
fig.legend(hh, ll, loc="lower center", ncol=3, frameon=False, fontsize=6.5)
save(fig, "Figure_07")

# ---------------------------------------------------------------- Figure 8: GPR mean and uncertainty maps
X = np.c_[df.GO_wt_pct.values, (df.Sintering_T_C.values - 1200) / 300]
go = np.linspace(0, 1, 101); tt = np.linspace(1200, 1500, 61); G, TT = np.meshgrid(go, tt)
Xg = np.c_[G.ravel(), (TT.ravel() - 1200) / 300]
fig, axs = plt.subplots(2, 3, figsize=(190 * MM, 100 * MM))
maps = {}
for r, t in enumerate(["HV_GPa", "KIC_MPa_m05"]):
    k = C(1.0, (1e-3, 1e3)) * RBF([0.3, 0.5], (1e-2, 1e2)) + WhiteKernel(0.05, (1e-6, 1.0))
    gp = make_pipeline(StandardScaler(), GaussianProcessRegressor(kernel=k, normalize_y=True, n_restarts_optimizer=10, random_state=42))
    gp.fit(X, df[t].values)
    mu, sd = gp.predict(Xg, return_std=True); mu = mu.reshape(G.shape); sd = sd.reshape(G.shape)
    maps[t] = dict(kernel=str(gp[-1].kernel_), argmax_mu=[float(G.ravel()[mu.argmax()]), float(TT.ravel()[mu.argmax()])],
                   max_mu=float(mu.max()), sd_at_max=float(sd.ravel()[mu.argmax()]),
                   argmax_sd=[float(G.ravel()[sd.argmax()]), float(TT.ravel()[sd.argmax()])], max_sd=float(sd.max()),
                   sd_at_design_points_mean=float(gp.predict(X, return_std=True)[1].mean()))
    # expected-improvement-type acquisition (upper confidence bound, kappa = 2) for the next experiment
    ucb = mu + 2 * sd
    maps[t]["argmax_ucb"] = [float(G.ravel()[ucb.argmax()]), float(TT.ravel()[ucb.argmax()])]
    for c, (Z, lab, cm) in enumerate([(mu, "GPR mean " + LAB[t], "viridis"), (sd, "GPR s.d. " + LAB[t].split(" (")[0] + " (" + LAB[t].split(" (")[1], "magma"),
                                     (ucb, "UCB = mean + 2 s.d.", "cividis")]):
        ax = axs[r, c]; cs = ax.contourf(G, TT, Z, 20, cmap=cm); cb = fig.colorbar(cs, ax=ax, pad=0.02); cb.ax.tick_params(labelsize=6)
        cb.set_label(lab, fontsize=6.5)
        ax.scatter(df.GO_wt_pct, df.Sintering_T_C, s=8, c="white", edgecolor="k", lw=0.4)
        if c == 2:
            ax.plot(*maps[t]["argmax_ucb"], "r*", ms=8, mec="k", mew=0.4)
        ax.set_xlabel("GO content (wt%)"); ax.set_ylabel("Sintering temperature (°C)"); panel(ax, f"({'abcdef'[r * 3 + c]})")
fig.tight_layout(w_pad=0.6, h_pad=0.8)
save(fig, "Figure_08")
json.dump(maps, open(f"{R}/gpr_maps_summary.json", "w"), indent=2)

# ---------------------------------------------------------------- Figure S1: digitisation validation
val = pd.DataFrame([  # values stated in the text of the published article (doi:10.1088/1402-4896/ae32bb)
    ("RD_pct", 0.0, 1200, 96.5), ("RD_pct", 0.0, 1400, 96.8), ("RD_pct", 0.25, 1500, 99.0),
    ("HV_GPa", 0.25, 1500, 18.75), ("E_GPa", 0.25, 1500, 398.0), ("E_GPa", 1.0, 1200, 369.0), ("E_GPa", 1.0, 1500, 374.0),
    ("KIC_MPa_m05", 0.25, 1500, 5.73), ("KIC_MPa_m05", 0.0, 1200, 4.06)], columns=["target", "GO", "T", "stated"])
val["digitised"] = [df[(df.GO_wt_pct == r.GO) & (df.Sintering_T_C == r.T)][r.target].values[0] for r in val.itertuples()]
val["abs_diff"] = (val.digitised - val.stated).abs(); val["rel_diff_pct"] = 100 * val.abs_diff / val.stated
val.to_csv(f"{R}/digitisation_validation.csv", index=False)
fig, ax = plt.subplots(figsize=(90 * MM, 70 * MM))
ax.scatter(val.stated / val.stated, val.rel_diff_pct, s=0)  # placeholder for limits
ax.cla()
ax.bar(range(len(val)), val.rel_diff_pct, color="#1b6ca8")
ax.set_xticks(range(len(val))); ax.set_xticklabels([f"{LAB[r.target].split(' (')[0]}\n{r.GO:g}/{r.T}" for r in val.itertuples()], fontsize=5.5, rotation=60, ha="right")
ax.set_ylabel("|digitised − stated| / stated (%)")
save(fig, "Figure_S1")
print(val.round(3).to_string()); print(json.dumps(maps, indent=1))
