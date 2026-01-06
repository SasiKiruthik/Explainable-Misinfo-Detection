# Complete Steps to Train and See Output

## Overview
This guide will help you train the model on:
- **Source Domain**: CoAID
- **Target Domains**: MiSoVac, FakeNewsNet, WELFake

## Prerequisites
1. Python 3.8+ installed
2. All dependencies installed (see Step 1)

---

## Step-by-Step Instructions

### STEP 1: Install Dependencies

```bash
pip install -r requirements.txt
pip install openpyxl
```

**What this does**: Installs all required packages including openpyxl for reading Excel files.

---

### STEP 2: Prepare Datasets

```bash
python prepare_datasets.py
```

**What this does**:
- Combines all CoAID CSV files into one source dataset
- Processes MiSoVac, FakeNewsNet, and WELFake target domains
- Saves prepared datasets to `data/raw/` directory

**Expected output**:
- `data/raw/coaid_source.csv` - Combined CoAID source data
- `data/raw/misovac_target.csv` - MiSoVac target data
- `data/raw/fakenewsnet_target.csv` - FakeNewsNet target data
- `data/raw/welfake_target.csv` - WELFake target data

**You should see**: Summary of samples and label distributions for each dataset.

---

### STEP 3: Preprocess Data

```bash
python main.py --mode preprocess --data_files data/raw/coaid_source.csv data/raw/misovac_target.csv data/raw/fakenewsnet_target.csv data/raw/welfake_target.csv --platforms CoAID MiSoVac FakeNewsNet WELFake
```

**What this does**:
- Cleans and tokenizes all texts
- Encodes using BERT tokenizer
- Splits into train/validation/test sets
- Saves processed data for training

**Expected output**:
- Processed data saved to `data/processed/`
- Encoded texts and dataset splits

**You should see**: 
- Processing progress
- Total samples count
- Platform and label distributions

---

### STEP 4: Train the Model

```bash
python train_source_target.py
```

**What this does**:
- Creates DANN model
- Trains on CoAID (source) with domain adaptation to target domains
- Saves checkpoints during training
- Implements early stopping

**Expected output**:
- Training progress with loss values
- Validation metrics for each target domain
- Best model saved to `models/checkpoints/best_model.pt`

**You should see**:
```
Epoch 1/50
Train - Loss: 0.xxxx, Label Loss: 0.xxxx, Domain Loss: 0.xxxx, Acc: 0.xxxx
Validation Metrics:
  MiSoVac - Loss: 0.xxxx, Acc: 0.xxxx, F1: 0.xxxx, AUC: 0.xxxx
  FakeNewsNet - Loss: 0.xxxx, Acc: 0.xxxx, F1: 0.xxxx, AUC: 0.xxxx
  WELFake - Loss: 0.xxxx, Acc: 0.xxxx, F1: 0.xxxx, AUC: 0.xxxx
```

**Training time**: Depends on data size and GPU, typically 30 minutes to 2 hours.

---

### STEP 5: Evaluate the Model

```bash
python evaluate.py
```

**What this does**:
- Loads the best trained model
- Evaluates on test sets from all domains
- Generates confusion matrices and ROC curves
- Saves comprehensive evaluation results

**Expected output**:
- Evaluation metrics for each domain
- Confusion matrices saved as images
- ROC curves saved as images
- Results saved to `results/evaluation/evaluation_results.json`

**You should see**:
```
Evaluating platform: CoAID
Results for CoAID:
Accuracy: 0.xxxx
Precision: 0.xxxx
Recall: 0.xxxx
F1 Score: 0.xxxx
AUC: 0.xxxx
...
```

---

### STEP 6: Test Inference (Optional)

```bash
python inference.py --text "Your text to classify here"
```

**What this does**: Predicts if a text contains misinformation.

**Expected output**:
```
Prediction: Misinformation (or Real)
Confidence: 0.xxxx
Probabilities - Real: 0.xxxx, Misinformation: 0.xxxx
```

---

### STEP 7: Generate Explanations (Optional)

```bash
python main.py --mode explain --text "Your text to explain here"
```

**What this does**: Generates LIME explanations showing which features influenced the prediction.

**Expected output**:
```
PREDICTION RESULT WITH EXPLANATION
Text: Your text...
Prediction: Misinformation
Confidence: 0.xxxx
Top Features:
  feature1: 0.xxxx
  feature2: 0.xxxx
  ...
```

---

## Quick Run (Automated)

If you want to run all steps automatically:

```bash
python complete_workflow.py
```

This will execute all steps sequentially and show progress.

---

## Expected Output Locations

After completing all steps, you'll find:

1. **Prepared Data**: `data/raw/*.csv`
2. **Processed Data**: `data/processed/*.pkl`
3. **Model Checkpoints**: `models/checkpoints/`
   - `best_model.pt` - Best model
   - `checkpoint_epoch_*.pt` - Periodic checkpoints
   - `training_history.json` - Training metrics
4. **Evaluation Results**: `results/evaluation/`
   - `evaluation_results.json` - All metrics
   - `confusion_matrix_*.png` - Confusion matrices
   - `roc_curve_*.png` - ROC curves
5. **Explanations** (if generated): `results/explanations/`

---

## Troubleshooting

### Issue: "File not found" errors
- **Solution**: Make sure you're in the correct directory (`CIP` folder)
- Check that dataset path is correct in `prepare_datasets.py`

### Issue: "CUDA out of memory"
- **Solution**: Reduce `BATCH_SIZE` in `config.py` (try 16 or 8)

### Issue: "openpyxl not found"
- **Solution**: Run `pip install openpyxl`

### Issue: "No target domains found"
- **Solution**: Check that preprocessing step completed successfully
- Verify dataset files exist in `data/raw/`

### Issue: Training is slow
- **Solution**: 
  - Use GPU if available (automatically detected)
  - Reduce `NUM_EPOCHS` in `config.py` for testing
  - Reduce `MAX_SEQUENCE_LENGTH` in `config.py`

---

## Understanding the Output

### Training Metrics
- **Loss**: Lower is better (should decrease over epochs)
- **Accuracy**: Percentage of correct predictions
- **F1 Score**: Harmonic mean of precision and recall (0-1, higher is better)
- **AUC**: Area under ROC curve (0-1, higher is better)

### Domain Adaptation Success
- Compare F1 scores across domains
- Higher scores on target domains indicate successful domain adaptation
- Source domain (CoAID) should have good performance
- Target domains should also perform well due to domain adaptation

---

## Next Steps After Training

1. **Analyze Results**: Check `results/evaluation/evaluation_results.json`
2. **Visualize**: Look at confusion matrices and ROC curves
3. **Test**: Try inference on new texts
4. **Explain**: Generate explanations for interesting cases
5. **Fine-tune**: Adjust hyperparameters in `config.py` if needed

---

## Summary

The complete workflow:
1. ✅ Install dependencies
2. ✅ Prepare datasets
3. ✅ Preprocess data
4. ✅ Train model
5. ✅ Evaluate model
6. ✅ Test inference (optional)
7. ✅ Generate explanations (optional)

**Total time**: Approximately 1-3 hours depending on data size and hardware.

---

## Need Help?

If you encounter any issues:
1. Check error messages carefully
2. Verify all files are in correct locations
3. Ensure dependencies are installed
4. Check that dataset files are not corrupted
5. Review the troubleshooting section above

Good luck with your training! 🚀

