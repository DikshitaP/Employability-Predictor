# CareerLens — Student Employability Prediction System v2

## Bug Fixes in this version
- **emp_model now uses RAW unscaled features** — matches how the GBT was trained in the notebook (Cell 9 uses raw values, no scaler). Previous version incorrectly applied scaler_student before calling predict_proba, causing near-zero employability scores.
- **Guardrails replaced with data-calibrated penalties** from the notebook's apply_guardrails() function. backlog=1 now applies a 0.52× multiplier (23.4% actual placement rate) instead of a flat 10% cap.

## New Features
- `/api/shortlist` — Company Shortlister with Dream vs Safe Bet comparison
- `/api/learning_path` — 30/60/90-day personalised improvement plan
- `/api/resume_score` — Recruiter-style résumé strength score (A+ to C)
- `/api/interview_tips` — Tier-specific rounds, resources, and tips

## How to run
1. Place your model artifacts in the same folder as app.py:
   - placement_model_best.pt
   - emp_model.pkl
   - scaler_hybrid.pkl  (used only for placement model)
   - scaler_student.pkl (kept for compatibility but NOT used for emp_model)
2. pip install -r requirements.txt
3. python app.py
4. Open http://localhost:5000

## Viva answer — "Zero skill match but high probability?"
The model uses two separate predictors:
1. **Overall Employability (GBT)** — student-only features. Skill match with a specific company is irrelevant here; a student with great CGPA, internships, and aptitude is genuinely employable regardless of any one company's skill list.
2. **Company Placement (BERT+LSTM)** — this DOES use skill match ratio as a tabular feature. If skill match is 0, it reduces the placement probability for that specific company. But the model also sees 20 other features — CGPA, projects, communication, etc. — so a very strong student profile can still yield a moderate probability. This mirrors reality: a FAANG-calibre student who doesn't know a niche tool can still get through based on fundamentals.
