"""
FINAL DATA PREPROCESSING
Zero warnings. Zero guessing. Dataset-trusting logic.
"""

import os
import re
import pickle
import numpy as np
import pandas as pd
import nltk
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
from config import Config

# ---------------- NLTK ----------------
for pkg in ["punkt", "stopwords", "wordnet"]:
    try:
        nltk.data.find(pkg)
    except LookupError:
        nltk.download(pkg, quiet=True)

# ---------------- SAFE SPLIT ----------------
def split(indices, labels, test_size, seed):
    return train_test_split(
        indices,
        test_size=test_size,
        stratify=labels,
        random_state=seed
    )

# ---------------- PREPROCESSOR ----------------
class DataPreprocessor:

    def __init__(self, config: Config):
        self.config = config
        self.tokenizer = BertTokenizer.from_pretrained(config.BERT_MODEL_NAME)

    # -------- TEXT CLEAN --------
    def clean_text(self, text):
        if pd.isna(text):
            return ""
        text = str(text).lower()
        text = re.sub(r"http\S+|www\S+", "", text)
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    # -------- LABEL (NO GUESSING) --------
    def normalize_label(self, x):
        if pd.isna(x):
            return np.nan

        # numeric → trust dataset
        if isinstance(x, (int, np.integer)):
            return int(x)

        x = str(x).strip().lower()

        if x in ["fake", "false"]:
            return 1
        if x in ["real", "true"]:
            return 0

        return np.nan

    # -------- LOAD DATASET --------
    def load_dataset(self, path, platform):
        df = pd.read_csv(path)

        df["cleaned_text"] = df["text"].apply(self.clean_text)
        df["label"] = df["label"].apply(self.normalize_label)

        df = df.dropna(subset=["label", "cleaned_text"])
        df["label"] = df["label"].astype(int)
        df["platform"] = platform

        dist = df["label"].value_counts().to_dict()
        print(f"{platform} label distribution: {dist}")

        # HARD ASSERT — NO WARNINGS
        assert len(dist) == 2, f"{platform} is single-class — DATASET IS WRONG"

        return df

    # -------- PREPARE --------
    def prepare_datasets(self, paths, platforms):
        dfs = [self.load_dataset(p, plat) for p, plat in zip(paths, platforms)]
        data = pd.concat(dfs, ignore_index=True)

        encoded = self.tokenizer(
            data["cleaned_text"].tolist(),
            padding="max_length",
            truncation=True,
            max_length=self.config.MAX_SEQUENCE_LENGTH,
            return_tensors="pt"
        )

        os.makedirs(self.config.PROCESSED_DATA_DIR, exist_ok=True)
        pickle.dump(encoded, open(f"{self.config.PROCESSED_DATA_DIR}/encoded.pkl", "wb"))
        pickle.dump(data, open(f"{self.config.PROCESSED_DATA_DIR}/data.pkl", "wb"))

        print("✓ Preprocessing complete. No warnings. No collapse.")
        return data, encoded
