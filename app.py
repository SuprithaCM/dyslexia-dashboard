"""Flask backend: receives gaze + word layout, computes features/indices, predicts, explains (TreeSHAP), builds PDF."""
import io, uuid, math
import joblib, numpy as np, pandas as pd, xgboost as xgb
from flask import Flask, request, jsonify, render_template, send_file, redirect
from common import apply_indices, ADDED
from live import extract

app = Flask(__name__)
try:
    B = joblib.load("model.joblib")
except FileNotFoundError:
    B = None
REPORTS = {}

LABELS = {"AFD": "Average fixation duration", "mean_sacc_dur": "Saccade duration", "SA": "Saccade amplitude",
          "n_fix": "Number of fixations", "n_sacc": "Number of saccades", "n_progress": "Forward (progressive) movements",
          "n_regress": "Backward (regressive) movements", "progressive_rate": "Progressive rate", "regressive_rate": "Regression rate",
          "rereading_rate": "Re-reading rate", "P_R_ratio": "Progressive/regressive ratio", "n_revisits": "Word revisits",
          "rereading_time": "Re-reading time (ms)", "word_fix_dur": "Fixation duration on words (lexical)",
          "n_fix_aoi": "Fixations on words", "n_between_line_regress": "Between-line regressions",
          "n_within_line_regress": "Within-line regressions", "mean_sacc_ampl_trial": "Mean saccade amplitude (trial)",
          "mean_fix_dur_trial": "Mean fixation duration (trial)", "fix_event_count": "Fixation event count",
          "fix_event_mean_ms": "Fixation mean (ms)", "fix_event_median_ms": "Fixation median (ms)", "fix_event_std_ms": "Fixation variability (ms)",
          "fix_event_total_ms": "Total fixation time (ms)", "sacc_event_count": "Saccade event count",
          "sacc_event_mean_duration_ms": "Saccade mean duration (ms)", "sacc_event_mean_amplitude": "Saccade mean amplitude",
          "sacc_event_median_amplitude": "Saccade median amplitude", "sacc_event_std_amplitude": "Saccade amplitude variability",
          "PPEI": "PPEI (progressive-pattern efficiency)", "RFI_fold_score": "RFI (reading fluency index)",
          "LPI": "LPI (lexical processing index)", "OPE": "OPE (oculomotor efficiency)"}


def category(c):
    if c in ADDED: return "Composite indices"
    if "regress" in c or "progress" in c or c == "P_R_ratio": return "Regressions / reading flow"
    if "rerea" in c or "revisit" in c or c in ("word_fix_dur", "n_fix_aoi"): return "Lexical / re-reading"
    if "sacc" in c or c == "SA": return "Saccades"
    return "Fixations"


def fnum(x, nd=3):
    return None if x is None or (isinstance(x, float) and not math.isfinite(x)) else round(float(x), nd)


def analyze_payload(d):
    tasks = d["tasks"]
    ppcm = float(d["px_per_cm"]); dist = float(d.get("distance_cm", 60))
    base, fixs = extract(tasks, ppcm, dist)
    row = base.reindex(columns=B["base_cols"]).replace([np.inf, -np.inf], np.nan)
    full = apply_indices(row, B["params"]).replace([np.inf, -np.inf], np.nan)[B["all_cols"]]
    A = B["imputer"].transform(full)
    model = B["model"]
    prob = float(model.predict_proba(A)[0, 1])
    contrib = model.get_booster().predict(xgb.DMatrix(A), pred_contribs=True)[0][:-1]   # TreeSHAP, log-odds
    ref = B["ref"]
    items = []
    for c, sv in zip(B["all_cols"], contrib):
        items.append(dict(feature=c, label=LABELS.get(c, c), category=category(c), value=fnum(full[c].iloc[0]),
                          control=fnum(ref.loc[c, "control"]), dyslexic=fnum(ref.loc[c, "dyslexic"]), shap=fnum(sv, 4)))
    items.sort(key=lambda r: -abs(r["shap"]))
    cats = {}
    for it in items: cats[it["category"]] = cats.get(it["category"], 0) + it["shap"]
    # data-quality gate
    dur_s = sum(t["duration_ms"] for t in tasks) / 1000
    n_samp = sum(len(t["gaze"]) for t in tasks)
    n_fix = int(base["n_fix"].iloc[0]); cov = float(base["n_fix_aoi"].iloc[0]) / n_fix if n_fix else 0
    hz = n_samp / dur_s if dur_s else 0
    q = dict(sampling_hz=fnum(hz, 1), fixations=n_fix, word_coverage=fnum(cov, 2), duration_s=fnum(dur_s, 1))
    warn = []
    if hz < 8: warn.append("Gaze sampling rate is very low (<8 Hz).")
    if n_fix < 40: warn.append("Too few fixations detected (<40).")
    if cov < 0.3: warn.append("Most fixations did not land on the text (poor calibration).")
    status = "inconclusive" if warn else ("dyslexic" if prob >= 0.5 else "non_dyslexic")
    conf = "borderline" if 0.35 <= prob <= 0.65 else "clear"
    top_to = [i for i in items if i["shap"] > 0][:3]
    top_away = [i for i in items if i["shap"] < 0][:2]
    idx = {k: next(i for i in items if i["feature"] == f) for k, f in
           [("PPEI", "PPEI"), ("RFI", "RFI_fold_score"), ("LPI", "LPI"), ("OPE", "OPE")]}
    rid = uuid.uuid4().hex[:10]
    res = dict(id=rid, status=status, confidence=conf, probability=fnum(prob, 4), quality=q, warnings=warn,
               indices=idx, contributions=items, categories={k: fnum(v, 4) for k, v in cats.items()},
               top_toward=top_to, top_away=top_away, parameters={k: fnum(row[k].iloc[0]) for k in B["base_cols"]},
               fixations=[[dict(x=f["x"], y=f["y"], dur=f["dur"], word=f["word"]) for f in fx] for fx in fixs])
    REPORTS[rid] = dict(res, name=d.get("name", "Participant"))
    return res

@app.get("/mediapipe/face_mesh/<path:f>")
def mediapipe_files(f):
    """WebGazer requests its face-mesh model from this origin; forward to the CDN."""
    return redirect("https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh@0.4.1633559619/" + f, code=302)
@app.get("/")
def home(): return render_template("index.html")


@app.post("/api/analyze")
def analyze():
    if B is None: return jsonify(error="model.joblib not found. Run train_model.py first."), 503
    try: return jsonify(analyze_payload(request.get_json(force=True)))
    except Exception as e: return jsonify(error=str(e)), 400


@app.get("/api/report/<rid>.pdf")
def pdf(rid):
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors
    r = REPORTS.get(rid)
    if not r: return "Report not found", 404
    st = getSampleStyleSheet(); mm = 2.835
    verdict = {"dyslexic": "Dyslexia-like reading pattern detected", "non_dyslexic": "No dyslexia-like pattern detected",
               "inconclusive": "Inconclusive (poor data quality)"}[r["status"]]
    S = [Paragraph(f"Eye-Tracking Dyslexia Screening Report — {r['name']}", st["Title"]),
         Paragraph("<b>Research screening output — not a clinical diagnosis.</b>", st["Heading3"]),
         Paragraph(f"<b>Result:</b> {verdict} ({r['confidence']}). Model probability: {r['probability']:.1%}", st["BodyText"]), Spacer(1, 8)]
    def tbl(data, w):
        t = Table(data, colWidths=[x * mm for x in w], repeatRows=1)
        t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), .4, colors.grey), ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                               ("FONTSIZE", (0, 0), (-1, -1), 8)])); return t
    S += [Paragraph("Research indices", st["Heading2"]),
          tbl([["Index", "Value", "Non-dys. median", "Dys. median", "Model push (log-odds)"]] +
              [[k, v["value"], v["control"], v["dyslexic"], f"{v['shap']:+.3f}"] for k, v in r["indices"].items()], [45, 30, 35, 30, 40]), Spacer(1, 8),
          Paragraph("Explainable AI — parameters driving the prediction (TreeSHAP)", st["Heading2"]),
          tbl([["Parameter", "Your value", "Non-dys. median", "Direction", "SHAP"]] +
              [[i["label"], i["value"], i["control"], "→ dyslexia" if i["shap"] > 0 else "→ typical", f"{i['shap']:+.3f}"]
               for i in r["contributions"][:12]], [65, 25, 35, 25, 20]), Spacer(1, 8)]
    if r["top_toward"]:
        S.append(Paragraph("<b>Main driver:</b> " + r["top_toward"][0]["label"] + " pushed the model most strongly toward the dyslexia class. "
                           "Attribution describes model influence, not a cause.", st["BodyText"]))
    S += [Paragraph(f"Data quality: {r['quality']['sampling_hz']} Hz, {r['quality']['fixations']} fixations, word coverage {r['quality']['word_coverage']}.", st["BodyText"]),
          Spacer(1, 8), Paragraph("Webcam gaze is less accurate than research eye trackers and the model was trained on tracker data. "
                                  "Clinical assessment by a qualified professional is required.", st["Italic"])]
    buf = io.BytesIO(); SimpleDocTemplate(buf, pagesize=A4).build(S); buf.seek(0)
    return send_file(buf, mimetype="application/pdf", download_name=f"report_{rid}.pdf")


if __name__ == "__main__":
    app.run(debug=False, port=5000)
