"""
FINAL – BULLETPROOF DATASET PREPARATION
Aligned with base paper:
Explainable Misinformation Detection Across Multiple Social Media Platforms
"""

import os
import glob
import pandas as pd
from tqdm import tqdm

# ===================== PATHS =====================
DATASET_ROOT = r"C:\Users\Sasi Kiruthik\Downloads\EXPLAINABLE MISINFO\dataset"
CIP_ROOT = r"C:\Users\Sasi Kiruthik\OneDrive\Desktop\CIP"
OUTPUT_DIR = os.path.join(CIP_ROOT, "data", "raw")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===================== HELPERS =====================
def normalize_label(x):
    if pd.isna(x):
        return None
    x = str(x).strip().lower()
    if x in ["fake", "false", "1", "misinformation"]:
        return 1
    if x in ["real", "true", "0"]:
        return 0
    return None


def finalize_df(df, name):
    if df.empty:
        raise RuntimeError(f"{name} DATASET EMPTY — CHECK SOURCE FILES")

    if "text" not in df.columns or "label" not in df.columns:
        raise RuntimeError(f"{name} missing required columns")

    df = df.dropna(subset=["text", "label"])
    df["label"] = df["label"].astype(int)

    dist = df["label"].value_counts().to_dict()
    print(f"{name} label distribution: {dist}")

    if len(dist) != 2:
        raise RuntimeError(f"{name} INVALID — SINGLE CLASS DATASET")

    return df.reset_index(drop=True)

# ===================== CoAID =====================
def prepare_coaid():
    print("\nPreparing CoAID (Source Domain)")

    # 🔎 Auto-detect CoAID directory
    possible_dirs = [
        d for d in os.listdir(DATASET_ROOT)
        if "coaid" in d.lower()
    ]

    if not possible_dirs:
        raise RuntimeError("CoAID directory NOT FOUND inside dataset/")

    coaid_dir = os.path.join(DATASET_ROOT, possible_dirs[0])
    print(f"Using CoAID directory: {coaid_dir}")

    csv_files = glob.glob(os.path.join(coaid_dir, "*.csv"))
    if not csv_files:
        raise RuntimeError("No CSV files found in CoAID directory")

    rows = []
    for file in tqdm(csv_files, desc="Reading CoAID files"):
        fname = os.path.basename(file).lower()
        label = 1 if ("fake" in fname or "false" in fname) else 0

        df = pd.read_csv(file)
        for _, r in df.iterrows():
            parts = []
            for col in ["content", "abstract", "title", "newstitle"]:
                if col in df.columns and pd.notna(r.get(col)):
                    parts.append(str(r[col]))

            text = " ".join(parts).strip()
            if len(text) > 10:
                rows.append({"text": text, "label": label})

    df = finalize_df(pd.DataFrame(rows), "CoAID")
    out = os.path.join(OUTPUT_DIR, "coaid_source.csv")
    df.to_csv(out, index=False)
    print(f"Saved → {out}")
    return df

# ===================== MiSoVac =====================
def prepare_misovac():
    print("\nPreparing MiSoVac (Target Domain)")
    file = os.path.join(DATASET_ROOT, "MiSoVac-Target domain in base paper.csv")
    if not os.path.exists(file):
        raise RuntimeError("MiSoVac file not found")

    df = pd.read_csv(file)
    df["label"] = df["label"].apply(normalize_label)
    df = df[["text", "label"]]

    df = finalize_df(df, "MiSoVac")
    out = os.path.join(OUTPUT_DIR, "misovac_target.csv")
    df.to_csv(out, index=False)
    print(f"Saved → {out}")
    return df

# ===================== FakeNewsNet =====================
def prepare_fakenewsnet():
    print("\nPreparing FakeNewsNet (Target Domain)")
    file = os.path.join(DATASET_ROOT, "fakenewsnet_dataset-Target domain created.xlsx")
    if not os.path.exists(file):
        raise RuntimeError("FakeNewsNet file not found")

    df = pd.read_excel(file)

    text_col = next(c for c in df.columns if "text" in c.lower() or "content" in c.lower())
    label_col = next(c for c in df.columns if "label" in c.lower() or "class" in c.lower())

    df = df[[text_col, label_col]]
    df.columns = ["text", "label"]
    df["label"] = df["label"].apply(normalize_label)

    df = finalize_df(df, "FakeNewsNet")
    out = os.path.join(OUTPUT_DIR, "fakenewsnet_target.csv")
    df.to_csv(out, index=False)
    print(f"Saved → {out}")
    return df

# ===================== WELFake =====================
def prepare_welfake():
    print("\nPreparing WELFake (Target Domain)")
    file = os.path.join(DATASET_ROOT, "WELFake-Target domain created.xlsx")
    if not os.path.exists(file):
        raise RuntimeError("WELFake file not found")

    df = pd.read_excel(file)

    text_col = next(c for c in df.columns if "text" in c.lower() or "content" in c.lower())
    label_col = next(c for c in df.columns if "label" in c.lower() or "class" in c.lower())

    df = df[[text_col, label_col]]
    df.columns = ["text", "label"]
    df["label"] = df["label"].apply(normalize_label)

    df = finalize_df(df, "WELFake")
    out = os.path.join(OUTPUT_DIR, "welfake_target.csv")
    df.to_csv(out, index=False)
    print(f"Saved → {out}")
    return df

# ===================== MAIN =====================
if __name__ == "__main__":
    print("=" * 70)
    print("FINAL DATASET PREPARATION (BASE PAPER ALIGNED)")
    print("=" * 70)

    prepare_coaid()
    prepare_misovac()
    prepare_fakenewsnet()
    prepare_welfake()

    print("\nALL DATASETS PREPARED SUCCESSFULLY")
