"""
SOURCE → TARGET DOMAIN ADAPTATION TRAINING
Base paper aligned
No legacy dataset assumptions
"""

import os
import pickle
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from config import Config

# ================= DEVICE =================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= DATASET =================
class SourceTargetDataset(Dataset):
    def __init__(self, indices, labels, encoded_data, domain_labels):
        self.indices = indices
        self.labels = labels
        self.encoded_data = encoded_data
        self.domain_labels = domain_labels

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        return {
            "input_ids": self.encoded_data["input_ids"][i],
            "attention_mask": self.encoded_data["attention_mask"][i],
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
            "domain_labels": self.domain_labels[idx],
        }

# ================= SPLIT =================
def stratified_split(df, test_size=0.2, seed=42):
    return train_test_split(
        df.index.tolist(),
        test_size=test_size,
        stratify=df["label"].values,
        random_state=seed
    )

# ================= MAIN =================
def main():
    config = Config()

    print("Loading processed data...")
    with open(os.path.join(config.PROCESSED_DATA_DIR, "encoded.pkl"), "rb") as f:
        encoded_data = pickle.load(f)

    with open(os.path.join(config.PROCESSED_DATA_DIR, "data.pkl"), "rb") as f:
        data = pickle.load(f)

    encoded_data["input_ids"] = encoded_data["input_ids"].to(device)
    encoded_data["attention_mask"] = encoded_data["attention_mask"].to(device)

    # ================= DOMAIN SPLIT =================
    source_df = data[data["platform"] == "CoAID"]
    target_df = data[data["platform"] != "CoAID"]

    src_idx, _ = stratified_split(source_df)
    tgt_idx, _ = stratified_split(target_df)

    # ================= DATASETS =================
    source_dataset = SourceTargetDataset(
        src_idx,
        source_df.loc[src_idx, "label"].values,
        encoded_data,
        domain_labels=torch.zeros(len(src_idx), dtype=torch.long).to(device)
    )

    target_dataset = SourceTargetDataset(
        tgt_idx,
        target_df.loc[tgt_idx, "label"].values,
        encoded_data,
        domain_labels=torch.ones(len(tgt_idx), dtype=torch.long).to(device)
    )

    source_loader = DataLoader(source_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    target_loader = DataLoader(target_dataset, batch_size=config.BATCH_SIZE, shuffle=True)

    # ================= MODEL =================
    from models.dann_model import create_model
    model = create_model(config).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    criterion_class = torch.nn.CrossEntropyLoss()
    criterion_domain = torch.nn.CrossEntropyLoss()

    print("Starting Source–Target Domain Adaptation Training")

    # ================= TRAIN =================
    for epoch in range(config.NUM_EPOCHS):
        model.train()
        total_loss = 0.0

        for (src, tgt) in zip(source_loader, target_loader):
            optimizer.zero_grad()

            # ---- SOURCE ----
            src_label_logits, src_domain_logits = model(
                src["input_ids"],
                src["attention_mask"],
                alpha=1.0
            )

            loss_cls = criterion_class(src_label_logits, src["labels"])
            loss_dom_src = criterion_domain(src_domain_logits, src["domain_labels"])

            # ---- TARGET (DOMAIN ONLY) ----
            _, tgt_domain_logits = model(
                tgt["input_ids"],
                tgt["attention_mask"],
                alpha=1.0
            )

            loss_dom_tgt = criterion_domain(tgt_domain_logits, tgt["domain_labels"])

            loss = loss_cls + loss_dom_src + loss_dom_tgt
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{config.NUM_EPOCHS}] Loss: {total_loss:.4f}")

    print("Training completed successfully.")

# ================= RUN =================
if __name__ == "__main__":
    main()
