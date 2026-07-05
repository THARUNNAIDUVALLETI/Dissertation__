#!/usr/bin/env python3
"""
main.py — Zero Day Hunter NIDS
================================
Aiswariya Akhil (E4318387) | CIS4055 | Supervisor: Nauman Issar

Commands:
  python main.py                  → show menu + project status
  python main.py dashboard        → start the SOC dashboard
  python main.py train            → train RF + Autoencoder
  python main.py train-baselines  → train DT, LR, KNN, NB
  python main.py evaluate         → run evaluation_simple.py
  python main.py demo             → create demo CSV files for upload
  python main.py build-dataset    → re-download + rebuild UNSW-NB15
  python main.py status           → check all files exist
"""
import os, sys, subprocess
from pathlib import Path

ROOT      = Path(__file__).resolve().parent
DASH      = ROOT / "nids_dashboard"
SRC_M     = ROOT / "src" / "models"
SRC_E     = ROOT / "src" / "evaluation"
SRC_D     = ROOT / "src" / "data"
MODELS    = ROOT / "models" / "saved"
DATA      = ROOT / "data" / "raw"

BANNER = """
╔══════════════════════════════════════════════════════════════╗
║            ZERO DAY HUNTER — AI-Based Hybrid NIDS           ║
║         Random Forest + Autoencoder Intrusion Detection      ║
╚══════════════════════════════════════════════════════════════╝
"""

def status():
    print("=== PROJECT STATUS ===\n")
    dataset = DATA / "UNSW-NB15_clean.csv"
    if dataset.exists():
        size = dataset.stat().st_size/1024/1024
        print(f"  [OK]  Dataset:  UNSW-NB15_clean.csv ({size:.1f} MB)")
    else:
        print("  [!!]  Dataset:  MISSING → run: python main.py build-dataset")
    print()
    for fname, label in [
        ("rf_final.pkl","Random Forest"), ("autoencoder_final.keras","Autoencoder"),
        ("scaler_final.pkl","Scaler"), ("threshold_final.npy","Threshold"),
        ("categorical_encoders.pkl","Encoders")
    ]:
        ok = (MODELS/fname).exists()
        print(f"  {'[OK]' if ok else '[!!]'}  {label}")
    baselines = list(MODELS.glob("*_baseline.pkl"))
    print(f"\n  {'[OK]' if baselines else '[!!]'}  Baseline models: {[b.stem for b in baselines] or 'none — run train-baselines'}")
    comp = ROOT/"results"/"metrics"/"comparison.csv"
    if comp.exists():
        import pandas as pd
        df = pd.read_csv(comp)
        print(f"\n  [OK]  comparison.csv ({len(df)} models):")
        for _, r in df.iterrows():
            print(f"        {r['model']:<40} acc={float(r['accuracy']):.4f}")
    else:
        print("\n  [!!]  comparison.csv MISSING → run: python main.py evaluate")
    print()

def dashboard():
    print("Starting Zero Day Hunter at http://127.0.0.1:8000/")
    print("Press Ctrl+C to stop.\n")
    os.environ.setdefault("NIDS_MODELS_DIR", str(MODELS))
    subprocess.run([sys.executable, "manage.py", "runserver"], cwd=DASH)

def train():
    print("Training RF + Autoencoder...\n")
    subprocess.run([sys.executable, "train_simple.py"], cwd=SRC_M)

def train_baselines():
    print("Training baseline models...\n")
    subprocess.run([sys.executable, "train_baselines.py"], cwd=SRC_M)

def evaluate():
    print("Running evaluation...\n")
    subprocess.run([sys.executable, "evaluation_simple.py"], cwd=SRC_E)

def build_dataset():
    print("Building dataset...\n")
    subprocess.run([sys.executable, "build_dataset.py"], cwd=SRC_D)

def demo():
    dataset = DATA / "UNSW-NB15_clean.csv"
    if not dataset.exists():
        print("ERROR: UNSW-NB15_clean.csv not found. Run: python main.py build-dataset"); return
    import pandas as pd
    df = pd.read_csv(dataset)
    print(f"Loaded {len(df)} rows. Creating demo files...\n")
    df.sample(500,random_state=42).to_csv(DATA/"demo_500.csv",index=False)
    print("  Created: demo_500.csv          (500 rows, natural mix)")
    pd.concat([df[df["label"]==0].sample(100,random_state=1),
               df[df["label"]==1].sample(400,random_state=1)]).sample(frac=1,random_state=99).to_csv(
        DATA/"demo_attack_heavy.csv",index=False)
    print("  Created: demo_attack_heavy.csv  (80% attacks)")
    pd.concat([df[df["label"]==0].sample(400,random_state=2),
               df[df["label"]==1].sample(100,random_state=2)]).sample(frac=1,random_state=88).to_csv(
        DATA/"demo_benign_heavy.csv",index=False)
    print("  Created: demo_benign_heavy.csv  (80% benign)")
    df.sample(2000,random_state=7).to_csv(DATA/"demo_2000.csv",index=False)
    print("  Created: demo_2000.csv          (2000 rows)")
    print(f"\nAll files saved to {DATA}/")
    print("Upload any of these from the dashboard Upload & Analyse page.")

CMDS = {"dashboard":dashboard,"train":train,"train-baselines":train_baselines,
        "evaluate":evaluate,"build-dataset":build_dataset,"demo":demo,"status":status}

if __name__ == "__main__":
    print(BANNER)
    cmd = sys.argv[1] if len(sys.argv)>1 else None
    if cmd is None:
        print("Commands:")
        for c in CMDS: print(f"  python main.py {c}")
        print()
        status()
    elif cmd in CMDS:
        CMDS[cmd]()
    else:
        print(f"Unknown: {cmd}")
        sys.exit(1)
