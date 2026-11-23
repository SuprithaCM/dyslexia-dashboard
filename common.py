"""Shared feature aggregation + research indices (same definitions as the notebook)."""
import numpy as np, pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut

SEED = 42
SUM_COLS = ["n_fix", "n_sacc", "n_progress", "n_regress", "n_revisits", "rereading_time",
            "n_fix_aoi", "n_between_line_regress", "n_within_line_regress"]
PPEI_COLS = [("progressive_rate", 1), ("regressive_rate", -1), ("rereading_rate", -1), ("AFD", -1)]
RFI_COLS = ["AFD", "P_R_ratio", "SA"]
LPI_COLS = ["word_fix_dur", "n_revisits", "rereading_time"]
ADDED = ["PPEI", "RFI_fold_score", "LPI", "OPE"]


def ratio(n, d):
    d = float(d) if pd.notna(d) else 0.0
    return float(n) / d if d else 0.0


def _sd(a):
    s = np.nanstd(a)
    return float(s) if np.isfinite(s) and s > 0 else 1.0


def aggregate(metric_rows, fix_rows, sacc_rows):
    """Per-task rows -> one row per participant (identical to notebook cell 4B)."""
    mr = pd.DataFrame(metric_rows)
    m = mr.groupby("subject_id").mean(numeric_only=True).reset_index()
    for c in SUM_COLS:
        m[c] = m["subject_id"].map(mr.groupby("subject_id")[c].sum())
    m["progressive_rate"] = [ratio(a, b) for a, b in zip(m.n_progress, m.n_fix)]
    m["regressive_rate"] = [ratio(a, b) for a, b in zip(m.n_regress, m.n_fix)]
    m["rereading_rate"] = [ratio(a, b) for a, b in zip(m.n_revisits, m.n_fix_aoi)]
    m["P_R_ratio"] = [ratio(a, b) for a, b in zip(m.n_progress, m.n_regress)]
    fx = pd.DataFrame(fix_rows).groupby("subject_id").sum(numeric_only=True).reset_index()
    sc = pd.DataFrame(sacc_rows).groupby("subject_id").mean(numeric_only=True).reset_index()
    return m.merge(fx, on="subject_id", how="left").merge(sc, on="subject_id", how="left")


def fit_indices(df, y):
    """Fit index parameters on the full training cohort. Returns (params, out-of-fold RFI scores)."""
    y = np.asarray(y).astype(int)
    p = {"ppei": {c: (float(np.nanmean(df[c])), _sd(df[c])) for c, _ in PPEI_COLS}}
    Xr = df[RFI_COLS].to_numpy(float)

    def fit_lr(X, yy):
        imp = SimpleImputer(strategy="median").fit(X)
        sc = StandardScaler().fit(imp.transform(X))
        lr = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED)
        lr.fit(sc.transform(imp.transform(X)), yy)
        return imp, sc, lr

    oof = np.zeros(len(df))
    for tr, va in LeaveOneOut().split(Xr):          # cross-fitted RFI for training rows
        i, s, l = fit_lr(Xr[tr], y[tr])
        oof[va[0]] = l.decision_function(s.transform(i.transform(Xr[va])))[0]
    p["rfi"] = fit_lr(Xr, y)
    p["lpi"] = {}
    for c in LPI_COLS:
        v = df[c].fillna(df[c].median())
        p["lpi"][c] = (float(df[c].median()), float(v.min()), float(v.max()))
    return p, oof


def apply_indices(df, p, rfi_override=None):
    out = df.copy()
    out["PPEI"] = sum(s * (out[c] - p["ppei"][c][0]) / p["ppei"][c][1] for c, s in PPEI_COLS)
    imp, sc, lr = p["rfi"]
    out["RFI_fold_score"] = rfi_override if rfi_override is not None else \
        lr.decision_function(sc.transform(imp.transform(out[RFI_COLS].to_numpy(float))))
    parts = []
    for c in LPI_COLS:
        med, mn, mx = p["lpi"][c]
        parts.append((out[c].fillna(med) - mn) / ((mx - mn) or 1.0))
    out["LPI"] = np.mean(np.vstack(parts), axis=0)
    out["OPE"] = out["mean_sacc_ampl_trial"] / (out["mean_fix_dur_trial"] * (1 + out["n_regress"])).replace(0, np.nan)
    return out
