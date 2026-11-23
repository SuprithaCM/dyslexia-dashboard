"""Train the final Model 2 (XGBoost + PPEI/RFI/LPI/OPE) on all participants and save model.joblib.
Usage: python train_model.py --zip data.zip --labels dyslexia_class_label.csv"""
import argparse, re, zipfile, joblib
from pathlib import Path
import numpy as np, pandas as pd
from xgboost import XGBClassifier
from sklearn.impute import SimpleImputer
from common import *

ap = argparse.ArgumentParser()
ap.add_argument("--zip", required=True); ap.add_argument("--labels", required=True)
ap.add_argument("--out", default="model.joblib")
a = ap.parse_args()

num = lambda s: pd.to_numeric(s, errors="coerce")
labels = pd.read_csv(a.labels)[["subject_id", "class_id", "label"]].drop_duplicates("subject_id")
labels[["subject_id", "class_id"]] = labels[["subject_id", "class_id"]].astype(int)

M, F, S = [], [], []
with zipfile.ZipFile(a.zip) as zf:
    for n in zf.namelist():
        m = re.search(r"Subject_(\d+)_(.+)_(metrics|fixations|saccades|raw)\.csv$", Path(n).name)
        if not m or not n.startswith("data/"):
            continue
        sid, kind = int(m.group(1)), m.group(3)
        d = pd.read_csv(zf.open(n))
        if kind == "metrics":
            trial = d[d["aoi_kind"] == "line"] if "aoi_kind" in d and (d["aoi_kind"] == "line").any() else d
            if "trialid" in trial: trial = trial.drop_duplicates("trialid")
            aoi = d[d["aoi_kind"] == "subline"] if "aoi_kind" in d and (d["aoi_kind"] == "subline").any() else d
            cs = lambda f, c: num(f[c]).fillna(0).sum() if c in f else 0.0
            cm = lambda f, c: num(f[c]).mean() if c in f else np.nan
            nf, npg, nr = cs(trial, "n_fix_trial"), cs(trial, "n_progress_trial"), cs(trial, "n_regress_trial")
            nfa, rev = cs(aoi, "n_fix_aoi"), cs(aoi, "n_revisits_aoi")
            M.append(dict(subject_id=sid, AFD=cm(trial, "mean_fix_dur_trial"), mean_sacc_dur=cm(trial, "mean_sacc_dur_trial"),
                          SA=cm(trial, "mean_sacc_ampl_trial"), n_fix=nf, n_sacc=cs(trial, "n_sacc_trial"),
                          n_progress=npg, n_regress=nr, progressive_rate=ratio(npg, nf), regressive_rate=ratio(nr, nf),
                          rereading_rate=ratio(rev, nfa), P_R_ratio=ratio(npg, nr), n_revisits=rev,
                          rereading_time=cs(aoi, "dwell_time_rereading_aoi"), word_fix_dur=cm(aoi, "mean_fix_dur_aoi"),
                          n_fix_aoi=nfa, n_between_line_regress=cs(trial, "n_between_line_regress_trial"),
                          n_within_line_regress=cs(trial, "n_within_line_regress_trial"),
                          mean_sacc_ampl_trial=cm(trial, "mean_sacc_ampl_trial"), mean_fix_dur_trial=cm(trial, "mean_fix_dur_trial")))
        elif kind == "fixations":
            du = num(d["duration_ms"])
            F.append(dict(subject_id=sid, fix_event_count=len(d), fix_event_mean_ms=du.mean(), fix_event_median_ms=du.median(),
                          fix_event_std_ms=du.std(), fix_event_total_ms=du.sum()))
        elif kind == "saccades":
            du, am = num(d["duration_ms"]), num(d["ampl"])
            S.append(dict(subject_id=sid, sacc_event_count=len(d), sacc_event_mean_duration_ms=du.mean(),
                          sacc_event_mean_amplitude=am.mean(), sacc_event_median_amplitude=am.median(),
                          sacc_event_std_amplitude=am.std()))

data = labels.merge(aggregate(M, F, S), on="subject_id", how="left")
assert not data["AFD"].isna().any(), "Some participants have no metrics"
y = data["class_id"].astype(int)
base_cols = [c for c in data.columns if c not in {"subject_id", "class_id", "label"}]
Xb = data[base_cols].replace([np.inf, -np.inf], np.nan)

params, oof = fit_indices(Xb, y)
Xtr = apply_indices(Xb, params, rfi_override=oof).replace([np.inf, -np.inf], np.nan)
all_cols = base_cols + ADDED
Xtr = Xtr[all_cols]
imp = SimpleImputer(strategy="median").fit(Xtr)
model = XGBClassifier(n_estimators=250, max_depth=3, learning_rate=0.03, subsample=0.9, colsample_bytree=0.9,
                      reg_lambda=2.0, objective="binary:logistic", eval_metric="logloss", random_state=SEED, n_jobs=-1)
model.fit(imp.transform(Xtr), y)

ref = Xtr.assign(y=y.values).groupby("y").median().T.rename(columns={0: "control", 1: "dyslexic"})
joblib.dump(dict(model=model, imputer=imp, params=params, base_cols=base_cols, all_cols=all_cols,
                 ref=ref, n_train=len(data)), a.out)
print(f"Saved {a.out}: {len(data)} participants, {len(all_cols)} features")
