# Eye-Tracking Dyslexia Dashboard
1. `pip install -r requirements.txt`
2. `python train_model.py --zip data.zip --labels dyslexia_class_label.csv`  -> creates model.joblib
3. `python app.py` -> open http://localhost:5000 in Chrome (camera works on localhost; use HTTPS elsewhere)

Flow: calibration (9 pts) -> accuracy check -> 3 passages -> report (verdict, PPEI/RFI/LPI/OPE, SHAP drivers, gaze paths, PDF).
Files: common.py (features+indices), live.py (gaze -> fixations/saccades/regressions), app.py (API, SHAP, PDF), templates/index.html.
Research screening only, not a clinical diagnosis.
