# Hybrid AI-Based Network Intrusion Detection System

Masters Dissertation — Aiswariya Akhil (E4318387), CIS4055, Supervisor: Nauman Issar

## What's in this zip

This is the **complete, working codebase** — every file has been run
end-to-end and confirmed working before packaging. `data/raw/`,
`models/saved/`, and `results/confusion_matrices/` are intentionally
empty (just `.gitkeep` placeholders): those hold large generated files
that should come from YOUR machine, not be overwritten by mine. Your own
real, already-trained models and dataset should go in here — see
"Restoring your existing work" below if you already have them.

`results/metrics/comparison.csv` IS included with real numbers — these
are your own genuine evaluation results, captured from your terminal
output on 26 June:

```
Random Forest:            99.03% accuracy
Hybrid (RF+Autoencoder):  97.44% accuracy
```

## Project structure

```
Dissertation-/
├── requirements.txt
├── data/raw/                      <- put UNSW-NB15_clean.csv here (you have this already)
├── models/saved/                  <- put rf_final.pkl etc. here (you have this already)
├── results/
│   ├── metrics/comparison.csv     <- included, real
│   └── confusion_matrices/        <- regenerate via evaluation_simple.py if needed
├── src/
│   ├── data/build_dataset.py      <- rebuilds the dataset from public UNSW-NB15 (only if needed again)
│   ├── models/train_simple.py     <- trains RF + Autoencoder
│   ├── evaluation/evaluation_simple.py
│   └── features/flow_extractor.py <- for future live-capture phase
└── nids_dashboard/                <- the Django SOC dashboard, chart-growth bug already fixed
```

## Restoring your existing work

You already have a real, working `data/raw/UNSW-NB15_clean.csv` and
trained models on your Kali machine (from your successful training run).
**Don't retrain from scratch** — just copy those into this new structure:

```bash
cp ~/project/data/raw/UNSW-NB15_clean.csv  Dissertation-/data/raw/
cp ~/project/models/saved/*                Dissertation-/models/saved/
```

(Adjust the source path if your existing files live somewhere other than
`~/project` — run `find ~ -iname "rf_final.pkl"` if unsure where they are.)

## First-time setup (if starting completely fresh)

```bash
cd Dissertation-
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python3 src/data/build_dataset.py        # ~1-2 min, downloads public UNSW-NB15
python3 src/models/train_simple.py       # ~2-5 min, trains RF + Autoencoder
python3 src/evaluation/evaluation_simple.py   # writes results/metrics/comparison.csv
```

## Running the dashboard

```bash
source venv/bin/activate
cd nids_dashboard
python manage.py migrate
python manage.py createsuperuser
python manage.py seed_metrics
python manage.py runserver
```
Visit `http://127.0.0.1:8000/` and log in.

## What's genuinely working right now

- Real CSV upload → real `RandomForestClassifier.predict()` +
  Autoencoder reconstruction-error predictions (no randomness, no
  placeholders)
- Alerts auto-raised per non-benign detection, with acknowledge/resolve
- IP blocking (admin-only)
- CSV + PDF export
- Dashboard charts read live from the database via JSON APIs
  (`api/traffic-overview/`, `api/attack-distribution/`,
  `api/algorithm-comparison/`) — chart container sizing bug fixed
- `seed_metrics` management command loads real evaluation numbers

## Known, documented limitations (for your methodology/discussion chapter)

1. **Source/destination IPs shown on the dashboard are synthetic** —
   generated at ingest time for display purposes only. UNSW-NB15's
   features are flow-level statistics (duration, byte counts, TTLs),
   not packet headers, so there's no real IP to show. State this
   plainly; it's an honest, explainable design choice.
2. **No live packet capture yet.** `flow_extractor.py` produces a
   different feature set than the trained model expects. Live detection
   needs either retraining on flow-extractor features, or extending
   flow_extractor to reproduce the UNSW-NB15 feature engineering exactly.
3. **Hybrid model has a higher false-positive rate than RF alone**
   (5.7% vs 1.2% in your last run) — a genuine, reproducible finding
   worth investigating and discussing (e.g. try a stricter anomaly
   threshold percentile in train_simple.py), not a bug to silently fix.
4. **Algorithm comparison chart only shows models actually trained**
   (RF, Hybrid) — no fabricated SVM/Neural Network numbers. Train real
   baselines if you want that comparison.
