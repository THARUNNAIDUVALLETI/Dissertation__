# Hybrid NIDS Dashboard — Setup

This is a clean rebuild of `nids_dashboard/` with **real** model predictions
(no `np.random.random()` placeholders). Drop it in to replace your existing
`nids_dashboard/` folder.

## 1. Where this expects your files

```
Dissertation-/
├── models/saved/
│   ├── rf_final.pkl
│   ├── autoencoder_final.keras
│   ├── scaler_final.pkl
│   └── threshold_final.npy
├── results/metrics/comparison.csv
└── nids_dashboard/        <-- this folder replaces your existing one
```

If your layout differs, set the `NIDS_MODELS_DIR` environment variable to
the absolute path of `models/saved/`, e.g.:

```bash
export NIDS_MODELS_DIR=/home/kali/Dissertation-/models/saved
```

## 2. Install dependencies

```bash
cd ~/Dissertation-
source venv/bin/activate
pip install django pandas numpy scikit-learn tensorflow joblib reportlab
```

(`reportlab` is new — it's what generates the PDF report. Everything else
you already had.)

## 3. First-time setup

```bash
cd nids_dashboard
python manage.py migrate
python manage.py createsuperuser      # this is your login
python manage.py seed_metrics         # loads comparison.csv into the Accuracy KPI + bar chart
python manage.py runserver
```

Visit `http://127.0.0.1:8000/` — it redirects to `/dashboard/login/`.

## 4. Using it

- **Upload CSV**: upload any CSV with `feature_00 .. feature_48` columns
  (optionally a `label` column) — e.g. carve a slice off
  `data/raw/UNSW-NB15_clean.csv` for a demo run. The model genuinely runs
  on every row; nothing is faked.
- **Dashboard**: KPI cards, traffic-over-24h line chart, attack-distribution
  doughnut, and an accuracy bar chart all read from the database, which is
  populated by your uploads.
- **Alerts**: every non-benign detection raises an Alert you can
  acknowledge or resolve.
- **Access Control**: blocking/unblocking IPs requires `is_staff` (your
  superuser has this; create extra non-staff users via `createsuperuser`'s
  sibling, `python manage.py shell` + `User.objects.create_user(...)`, to
  demo the permission boundary).
- **Reports**: CSV export of all detections, PDF summary report.

## 5. Known, deliberate limitation (read before your viva)

Source/destination IPs and ports shown in the dashboard are **synthetic**,
generated at ingest time — `UNSW-NB15_clean.csv` has no IP columns (it's
49 anonymized numeric features). The verdict and confidence for each row
are genuine model output; the network identity wrapped around it is a
display placeholder. State this plainly in your write-up — it's an honest
limitation, not a bug, and it's exactly the gap that real PCAP/live capture
would need to close (see point 6).

The "Algorithm Accuracy Comparison" chart only plots Random Forest and the
Hybrid model, because those are the only two you actually trained. It does
not show Decision Tree/SVM bars with invented numbers — if you want that
comparison, train those baselines with `train_simple.py`-style scripts and
add rows to `comparison.csv` before re-running `seed_metrics`.

## 6. Next phase: live/PCAP detection

`flow_extractor.py` produces ~11 named flow features from real packets.
The trained model expects 49 anonymized columns from `UNSW-NB15_clean.csv`.
These are not the same feature space — wiring live capture into this model
as-is would produce predictions that look real but aren't. To do live
detection properly, either:
- retrain RF + Autoencoder on `flow_extractor.py`'s feature set, or
- extend `flow_extractor.py` to reproduce the original 49-column UNSW-NB15
  feature engineering exactly.

Happy to build either once you're ready for that phase.
