# 🚀 RUN THE WORKFLOW NOW

It looks like the workflow hasn't run yet. Here's how to run it:

## Step 1: Open Terminal in CIP Folder

Make sure you're in this directory:
```
C:\Users\Sasi Kiruthik\OneDrive\Desktop\CIP
```

## Step 2: Run These Commands One by One

### Command 1: Install Dependencies
```bash
pip install -r requirements.txt
pip install openpyxl
```

### Command 2: Prepare Datasets
```bash
python prepare_datasets.py
```
**Wait for**: "✓ All datasets prepared successfully!"

### Command 3: Preprocess Data
```bash
python main.py --mode preprocess --data_files data/raw/coaid_source.csv data/raw/misovac_target.csv data/raw/fakenewsnet_target.csv data/raw/welfake_target.csv --platforms CoAID MiSoVac FakeNewsNet WELFake
```
**Wait for**: Processing to complete (shows progress)

### Command 4: Train Model
```bash
python train_source_target.py
```
**Wait for**: Training to complete (30 min - 2 hours)

### Command 5: Evaluate Model
```bash
python evaluate.py
```
**Wait for**: Evaluation to complete

---

## OR Run Everything Automatically

If you want to run all steps at once:

```bash
python complete_workflow.py
```

This will ask for confirmation and run all steps automatically.

---

## After Running

Once complete, check results with:
```bash
python check_results.py
```

This will show you what was generated and the results.

---

## Need Help?

If you get any errors:
1. Check the error message
2. Make sure all dependencies are installed
3. Verify the dataset folder path is correct in `prepare_datasets.py`



