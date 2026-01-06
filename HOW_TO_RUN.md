# 🚀 HOW TO RUN - Simple Instructions

## Quick Start (Easiest Way)

### Step 1: Open Terminal/Command Prompt
Navigate to your project folder:
```bash
cd "C:\Users\Sasi Kiruthik\OneDrive\Desktop\CIP"
```

### Step 2: Install Dependencies (One Time Only)
```bash
pip install -r requirements.txt
pip install openpyxl
```

### Step 3: Run Everything Automatically
```bash
python complete_workflow.py
```

This will automatically:
1. ✅ Prepare all datasets
2. ✅ Preprocess the data
3. ✅ Train the model
4. ✅ Evaluate the model

---

## Manual Step-by-Step (If You Prefer Control)

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
pip install openpyxl
```

### Step 2: Prepare Datasets
```bash
python prepare_datasets.py
```
**Wait for**: "✓ All datasets prepared successfully!"

### Step 3: Preprocess Data
```bash
python main.py --mode preprocess --data_files data/raw/coaid_source.csv data/raw/misovac_target.csv data/raw/fakenewsnet_target.csv data/raw/welfake_target.csv --platforms CoAID MiSoVac FakeNewsNet WELFake
```
**Wait for**: Processing to complete (shows progress bars)

### Step 4: Train Model
```bash
python train_source_target.py
```
**Wait for**: Training to complete (this takes the longest, 30 min - 2 hours)

### Step 5: Evaluate Model
```bash
python evaluate.py
```
**Wait for**: Evaluation to complete and results to be saved

---

## What You'll See

### During Dataset Preparation:
```
==================================================
Preparing CoAID Source Domain
==================================================
Found 33 CSV files
Processing CoAID files: 100%|████████| 33/33
CoAID dataset: 5000 samples
Label distribution: {0: 2500, 1: 2500}
```

### During Training:
```
Epoch 1/50
Train - Loss: 0.6234, Label Loss: 0.5123, Domain Loss: 0.1111, Acc: 0.7234
Validation Metrics:
  MiSoVac - Loss: 0.5432, Acc: 0.7890, F1: 0.7654, AUC: 0.8234
  FakeNewsNet - Loss: 0.5678, Acc: 0.7654, F1: 0.7432, AUC: 0.8012
  WELFake - Loss: 0.5890, Acc: 0.7523, F1: 0.7321, AUC: 0.7890
```

### After Completion:
- Check `models/checkpoints/best_model.pt` - Your trained model
- Check `results/evaluation/` - All evaluation results
- Check `results/evaluation/evaluation_results.json` - Metrics

---

## Test Your Model (After Training)

```bash
# Test with a text
python inference.py --text "COVID-19 is a hoax created by the government"

# Get explanation
python main.py --mode explain --text "COVID-19 is a hoax created by the government"
```

---

## Troubleshooting

### If you get "File not found" error:
- Make sure you're in the correct directory: `CIP` folder
- Check that the dataset folder path is correct

### If training is too slow:
- It's normal! Training takes 30 minutes to 2 hours
- You can reduce `NUM_EPOCHS` in `config.py` for faster testing

### If you get "CUDA out of memory":
- Edit `config.py` and change `BATCH_SIZE = 32` to `BATCH_SIZE = 16` or `8`

### If you get "openpyxl not found":
- Run: `pip install openpyxl`

---

## Expected Timeline

- **Step 1 (Install)**: 2-5 minutes
- **Step 2 (Prepare Data)**: 5-10 minutes
- **Step 3 (Preprocess)**: 10-30 minutes
- **Step 4 (Train)**: 30 minutes - 2 hours ⏰ (longest step)
- **Step 5 (Evaluate)**: 5-10 minutes

**Total**: 1-3 hours (mostly training time)

---

## That's It! 🎉

Just run these commands in order and wait for completion. The model will be saved automatically and you can check the results in the `results/` folder.



