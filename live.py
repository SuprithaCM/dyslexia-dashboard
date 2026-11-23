"""Turn raw webcam gaze + word boxes into the same participant-level features used in training."""
import math
import numpy as np
from common import aggregate

DISP_PX, MIN_FIX_MS, TOL_PX = 100, 100, 45     # webcam-tuned I-DT thresholds


def px_to_deg(px, px_per_cm, dist_cm):
    return math.degrees(2 * math.atan((px / px_per_cm) / (2 * dist_cm)))


def detect_fixations(g):
    """Dispersion-threshold (I-DT) on an (n,3) array of t,x,y."""
    out, i, n = [], 0, len(g)
    disp = lambda s: np.ptp(s[:, 1]) + np.ptp(s[:, 2])
    while i < n:
        j = i
        while j < n and g[j, 0] - g[i, 0] < MIN_FIX_MS: j += 1
        if j >= n: break
        if disp(g[i:j + 1]) <= DISP_PX:
            while j + 1 < n and disp(g[i:j + 2]) <= DISP_PX: j += 1
            out.append(dict(t0=float(g[i, 0]), t1=float(g[j, 0]), x=float(g[i:j + 1, 1].mean()), y=float(g[i:j + 1, 2].mean())))
            i = j + 1
        else:
            i += 1
    return out


def nearest_word(x, y, words):
    best, bd = None, 1e9
    for w in words:
        d = math.hypot(max(w["x"] - x, 0, x - w["x"] - w["w"]), max(w["y"] - y, 0, y - w["y"] - w["h"]))
        if d < bd: best, bd = w, d
    return best if bd <= TOL_PX else None


def task_features(sid, task, ppcm, dist):
    g = np.array([[s["t"], s["x"], s["y"]] for s in task["gaze"]], float)
    g = g[np.argsort(g[:, 0])] if len(g) else g.reshape(0, 3)
    fx = detect_fixations(g) if len(g) > 3 else []
    for f in fx:
        f["dur"] = f["t1"] - f["t0"]
        w = nearest_word(f["x"], f["y"], task["words"])
        f["word"], f["line"] = (w["i"], w["line"]) if w else (None, None)
    sacc = []
    for a, b in zip(fx, fx[1:]):
        sacc.append((max(b["t0"] - a["t1"], 1.0), px_to_deg(math.hypot(b["x"] - a["x"], b["y"] - a["y"]), ppcm, dist)))
    prog = reg = btw = wth = rev = 0
    rev_time, seen, prev = 0.0, set(), None
    for f in fx:
        if f["word"] is None: continue
        if f["word"] in seen and prev is not None and f["word"] != prev["word"]:
            rev += 1; rev_time += f["dur"]
        if prev is not None:
            if f["word"] > prev["word"]: prog += 1
            elif f["word"] < prev["word"]:
                reg += 1
                if f["line"] != prev["line"]: btw += 1
                else: wth += 1
        seen.add(f["word"]); prev = f
    durs = np.array([f["dur"] for f in fx]) if fx else np.array([np.nan])
    wd = np.array([f["dur"] for f in fx if f["word"] is not None]) if fx else np.array([])
    sd = np.array([s[0] for s in sacc]) if sacc else np.array([np.nan])
    sa = np.array([s[1] for s in sacc]) if sacc else np.array([np.nan])
    n_aoi = int(len(wd))
    M = dict(subject_id=sid, AFD=np.nanmean(durs), mean_sacc_dur=np.nanmean(sd), SA=np.nanmean(sa), n_fix=len(fx), n_sacc=len(sacc),
             n_progress=prog, n_regress=reg, progressive_rate=0, regressive_rate=0, rereading_rate=0, P_R_ratio=0,
             n_revisits=rev, rereading_time=rev_time, word_fix_dur=wd.mean() if n_aoi else np.nan, n_fix_aoi=n_aoi,
             n_between_line_regress=btw, n_within_line_regress=wth, mean_sacc_ampl_trial=np.nanmean(sa), mean_fix_dur_trial=np.nanmean(durs))
    F = dict(subject_id=sid, fix_event_count=len(fx), fix_event_mean_ms=np.nanmean(durs), fix_event_median_ms=np.nanmedian(durs),
             fix_event_std_ms=np.nanstd(durs, ddof=1) if len(fx) > 1 else np.nan, fix_event_total_ms=float(np.nansum(durs)) if fx else 0.0)
    S = dict(subject_id=sid, sacc_event_count=len(sacc), sacc_event_mean_duration_ms=np.nanmean(sd), sacc_event_mean_amplitude=np.nanmean(sa),
             sacc_event_median_amplitude=np.nanmedian(sa), sacc_event_std_amplitude=np.nanstd(sa, ddof=1) if len(sacc) > 1 else np.nan)
    return M, F, S, fx


def extract(tasks, ppcm, dist, sid=0):
    Ms, Fs, Ss, fixs = [], [], [], []
    for t in tasks:
        M, F, S, fx = task_features(sid, t, ppcm, dist)
        Ms.append(M); Fs.append(F); Ss.append(S); fixs.append(fx)
    return aggregate(Ms, Fs, Ss), fixs
