# 🚀 START HERE - Complete Training Guide

## Quick Start (3 Commands)

If you want to run everything automatically:

```bash
# 1. Install dependencies
pip install -r requirements.txt
pip install openpyxl

# 2. Run automated workflow
python complete_workflow.py

# 3. Check results
# Results will be in: results/evaluation/
```

---

## Manual Step-by-Step (Recommended for First Time)

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
pip install openpyxl
```

### Step 2: Prepare Datasets
```bash
python prepare_datasets.py
```
**Expected**: Creates 4 CSV files in `data/raw/` directory

### Step 3: Preprocess Data
```bash
python main.py --mode preprocess --data_files data/raw/coaid_source.csv data/raw/misovac_target.csv data/raw/fakenewsnet_target.csv data/raw/welfake_target.csv --platforms CoAID MiSoVac FakeNewsNet WELFake
```
**Expected**: Processes and encodes all data, saves to `data/processed/`

### Step 4: Train Model
```bash
python train_source_target.py
```
**Expected**: Trains for multiple epochs, saves best model to `models/checkpoints/best_model.pt`

### Step 5: Evaluate Model
```bash
python evaluate.py
```
**Expected**: Generates evaluation metrics and visualizations in `results/evaluation/`

---

## What You'll See

### During Training:
```
Epoch 1/50
Train - Loss: 0.6234, Label Loss: 0.5123, Domain Loss: 0.1111, Acc: 0.7234
Validation Metrics:
  MiSoVac - Loss: 0.5432, Acc: 0.7890, F1: 0.7654, AUC: 0.8234
  FakeNewsNet - Loss: 0.5678, Acc: 0.7654, F1: 0.7432, AUC: 0.8012
  WELFake - Loss: 0.5890, Acc: 0.7523, F1: 0.7321, AUC: 0.7890
```

### After Evaluation:
- **Metrics**: Accuracy, Precision, Recall, F1, AUC for each domain
- **Visualizations**: Confusion matrices and ROC curves
- **Results File**: `results/evaluation/evaluation_results.json`

---

## Output Files Location

After completion, check these directories:

1. **`models/checkpoints/`** - Trained models
   - `best_model.pt` - Best model (use this for inference)
   - `training_history.json` - Training metrics over time

2. **`results/evaluation/`** - Evaluation results
   - `evaluation_results.json` - All metrics
   - `confusion_matrix_*.png` - Confusion matrices
   - `roc_curve_*.png` - ROC curves

3. **`data/processed/`** - Preprocessed data (for reuse)

---

## Test Your Model

After training, test with:

```bash
# Simple prediction
python inference.py --text "COVID-19 is a hoax created by the government"

# With explanation
python main.py --mode explain --text "COVID-19 is a hoax created by the government"
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "File not found" | Check you're in the `CIP` directory |
| "CUDA out of memory" | Reduce `BATCH_SIZE` in `config.py` to 16 or 8 |
| "openpyxl not found" | Run `pip install openpyxl` |
| Training too slow | Use GPU or reduce `NUM_EPOCHS` in `config.py` |

---

## Full Documentation

- **Complete Steps**: See `COMPLETE_STEPS.md` for detailed instructions
- **Quick Start**: See `QUICKSTART.md` for basic usage
- **Implementation Details**: See `IMPLEMENTATION_SUMMARY.md`

---

## Expected Timeline

- **Data Preparation**: 5-10 minutes
- **Preprocessing**: 10-30 minutes (depends on data size)
- **Training**: 30 minutes - 2 hours (depends on GPU/CPU)
- **Evaluation**: 5-10 minutes

**Total**: Approximately 1-3 hours

---

## Success Indicators

✅ All 4 datasets prepared successfully  
✅ Preprocessing completes without errors  
✅ Training loss decreases over epochs  
✅ Validation F1 scores improve  
✅ Evaluation generates metrics and visualizations  

---

**Ready to start? Run the commands above!** 🎯

