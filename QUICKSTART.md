# Quick Start Guide

## Installation

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Setup directories:**
```bash
python setup_directories.py
```

3. **Download NLTK data (if needed):**
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

## Quick Start with Sample Data

1. **Generate sample data:**
```bash
python main.py --mode sample_data
```
This creates sample datasets in `data/raw/` directory.

2. **Preprocess the data:**
```bash
python main.py --mode preprocess --data_files data/raw/twitter_sample_data.csv data/raw/facebook_sample_data.csv --platforms Twitter Facebook
```

3. **Train the model:**
```bash
python main.py --mode train
```
Training will take some time. The best model will be saved in `models/checkpoints/best_model.pt`.

4. **Evaluate the model:**
```bash
python main.py --mode evaluate
```

5. **Test inference:**
```bash
python main.py --mode inference --text "This is a test message to check if it's misinformation"
```

6. **Get explanation:**
```bash
python main.py --mode explain --text "This is a test message to check if it's misinformation"
```

## Using Your Own Data

1. **Prepare your data files:**
   - Format: CSV or JSON
   - Required columns: `text`, `label`
   - Labels: 0 for real, 1 for misinformation
   - Place files in `data/raw/` directory

2. **Preprocess:**
```bash
python main.py --mode preprocess --data_files data/raw/your_file1.csv data/raw/your_file2.csv --platforms Platform1 Platform2
```

3. **Train and evaluate as above**

## Direct Script Usage

You can also use individual scripts directly:

- **Training:** `python train.py`
- **Evaluation:** `python evaluate.py`
- **Inference:** `python inference.py --text "Your text"`
- **Explanations:** Use `explainability.py` module

## Configuration

Edit `config.py` to adjust:
- Batch size
- Learning rate
- Number of epochs
- Model architecture parameters
- LIME parameters

## Troubleshooting

1. **CUDA out of memory:** Reduce `BATCH_SIZE` in `config.py`
2. **Model not found:** Make sure you've trained the model first
3. **Data format error:** Check that your CSV has `text` and `label` columns
4. **Import errors:** Make sure all dependencies are installed

## Next Steps

- Read the full `README.md` for detailed documentation
- Experiment with different hyperparameters
- Try with real-world datasets
- Customize the model architecture for your needs

