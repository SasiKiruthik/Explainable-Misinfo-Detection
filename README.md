<<<<<<< HEAD
About the Project

This project implements the IEEE research paper “Explainable Misinformation Detection Across Multiple Social Media Platforms”.
The objective is to build a generalized and explainable misinformation detection system that can effectively identify fake or misleading information across different social media platforms.

Traditional misinformation detection models are often trained on a single platform and fail to generalize due to domain shift. Additionally, most deep learning models lack interpretability, reducing user trust. This project addresses both challenges using domain adaptation and explainable AI.

🧠 Core Idea

The system is based on a Domain Adversarial Neural Network (DANN) that learns domain-invariant textual features, enabling cross-platform generalization.
To improve transparency and trust, LIME (Local Interpretable Model-Agnostic Explanations) is integrated to provide word-level explanations for each prediction.

🏗️ Architecture Overview

Input Data:

Source Domain: CoAID dataset

Target Domain: MiSoVac dataset

Text Preprocessing:
Tokenization, stop-word removal, lemmatization, and noise removal

Text Representation:
GloVe word embeddings

Model:

Feature Extractor

Label Predictor (Misinformation / Genuine)

Domain Classifier with Gradient Reversal Layer

Explainability:
LIME generates human-understandable explanations for predictions

🎯 Key Objectives

Detect misinformation across multiple social media platforms

Reduce domain shift using adversarial domain adaptation

Build a generalized classification model

Provide explainable and interpretable predictions

Improve trust in misinformation detection systems

🛠️ Technologies Used

Python

TensorFlow

Scikit-learn

NLTK, spaCy

GloVe word embeddings

LIME (Explainable AI)

Jupyter Notebook

Git & GitHub

📊 Evaluation Metrics

Accuracy

Precision

Recall

F1-Score

AUC-ROC

📚 Reference

Gargi Joshi et al.,
“Explainable Misinformation Detection Across Multiple Social Media Platforms”,
IEEE Access, 2023.
=======
# Explainable Misinformation Detection Across Multiple Social Media Platforms

A complete implementation of the IEEE Access 2023 paper "Explainable Misinformation Detection Across Multiple Social Media Platforms" using Domain Adversarial Neural Networks (DANN) and LIME for explainability.

## Overview

This system provides:
- **Cross-platform misinformation detection** using DANN for domain adaptation
- **Explainable AI** using LIME to understand model decisions
- **Multi-platform support** for detecting misinformation across different social media platforms
- **Complete pipeline** from data preprocessing to model training, evaluation, and inference

## Features

1. **Domain Adversarial Neural Network (DANN)**
   - Feature extractor using BERT
   - Label predictor for misinformation classification
   - Domain classifier for adversarial domain adaptation
   - Gradient reversal layer for learning domain-invariant features

2. **Explainability with LIME**
   - Local interpretable model-agnostic explanations
   - Feature importance visualization
   - Explanation analysis and reporting

3. **Comprehensive Evaluation**
   - Multiple metrics (Accuracy, Precision, Recall, F1, AUC)
   - Per-class performance analysis
   - Confusion matrices and ROC curves
   - Cross-platform evaluation

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd CIP
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download NLTK data (if not already downloaded):
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

## Project Structure

```
CIP/
├── config.py                 # Configuration settings
├── data_preprocessing.py      # Data loading and preprocessing
├── models/
│   └── dann_model.py         # DANN model architecture
├── train.py                  # Training script
├── evaluate.py               # Evaluation script
├── inference.py              # Inference script
├── explainability.py         # LIME explainability module
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── data/                     # Data directory
│   ├── raw/                 # Raw data files
│   └── processed/           # Processed data
├── models/                   # Model directory
│   └── checkpoints/         # Model checkpoints
└── results/                 # Results directory
    ├── evaluation/          # Evaluation results
    └── explanations/        # LIME explanations
```

## Usage

### 1. Data Preparation

Prepare your data files in CSV or JSON format with the following columns:
- `text`: The text content to classify
- `label`: Binary label (0: Real, 1: Misinformation)

Example data format:
```csv
text,label
"This is a real news article",0
"This is fake information",1
```

Place your data files in the `data/raw/` directory.

### 2. Data Preprocessing

Preprocess your data from multiple platforms:

```python
from config import Config
from data_preprocessing import DataPreprocessor

config = Config()
preprocessor = DataPreprocessor(config)

# Define your data files and platform names
data_files = [
    "data/raw/platform1_data.csv",
    "data/raw/platform2_data.csv"
]
platform_names = ["Twitter", "Facebook"]

# Preprocess data
datasets, encoded_data = preprocessor.prepare_datasets(data_files, platform_names)
```

### 3. Training

Train the DANN model:

```bash
python train.py
```

The training script will:
- Load processed data
- Create DANN model
- Train with adversarial domain adaptation
- Save checkpoints and training history
- Implement early stopping

Training parameters can be adjusted in `config.py`.

### 4. Evaluation

Evaluate the trained model:

```bash
python evaluate.py
```

This will:
- Load the best model checkpoint
- Evaluate on test sets from all platforms
- Generate confusion matrices and ROC curves
- Save comprehensive evaluation results

### 5. Inference

Use the trained model for predictions:

**Single text prediction:**
```bash
python inference.py --text "Your text here"
```

**With explanation:**
```bash
python inference.py --text "Your text here" --explain
```

**Batch prediction from file:**
```bash
python inference.py --file input_texts.txt
```

**Interactive mode:**
```bash
python inference.py
```

### 6. Generate Explanations

Generate LIME explanations for test data:

```python
from config import Config
from explainability import MisinformationExplainer

config = Config()
explainer = MisinformationExplainer(config)

# Explain single instance
explanation = explainer.explain_instance("Your text here")
print(explanation.as_list())

# Explain test set
explanations, analysis = explainer.explain_test_set(
    test_texts, test_labels, save_dir="results/explanations"
)
```

## Configuration

Key parameters in `config.py`:

- `BATCH_SIZE`: Batch size for training (default: 32)
- `LEARNING_RATE`: Learning rate (default: 0.001)
- `NUM_EPOCHS`: Number of training epochs (default: 50)
- `LAMBDA_DOMAIN`: Weight for domain adversarial loss (default: 1.0)
- `GAMMA`: Gradient reversal parameter (default: 10.0)
- `HIDDEN_DIM`: Hidden dimension size (default: 256)
- `MAX_SEQUENCE_LENGTH`: Maximum sequence length (default: 128)
- `LIME_NUM_FEATURES`: Number of features in LIME explanations (default: 10)
- `LIME_NUM_SAMPLES`: Number of samples for LIME (default: 5000)

## Model Architecture

The DANN model consists of:

1. **Feature Extractor**: BERT-based feature extraction with additional transformations
2. **Label Predictor**: Multi-layer classifier for misinformation detection
3. **Domain Classifier**: Adversarial classifier for domain adaptation
4. **Gradient Reversal Layer**: Enables adversarial training

## Results

The model outputs:
- Training history with loss curves
- Evaluation metrics per platform
- Confusion matrices and ROC curves
- LIME explanations with feature importance
- Explanation analysis reports

## Performance Metrics

The system evaluates using:
- Accuracy
- Precision (macro and per-class)
- Recall (macro and per-class)
- F1 Score (macro and per-class)
- AUC-ROC

## Explainability

LIME explanations provide:
- Top features supporting misinformation classification
- Top features supporting real information classification
- Feature weights indicating importance
- Visualization capabilities

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- CUDA-capable GPU (recommended for training)

## Citation

If you use this implementation, please cite the original paper:

```
@article{joshi2023explainable,
  title={Explainable Misinformation Detection Across Multiple Social Media Platforms},
  author={Joshi, Gargi and Srivastava, Ananya and Yagnik, Bhargav and others},
  journal={IEEE Access},
  year={2023},
  volume={11},
  pages={23634--23646}
}
```

## License

This implementation is provided for research purposes.

## Contact

For questions or issues, please open an issue on the repository.

## Acknowledgments

- Original paper authors: Gargi Joshi, Ananya Srivastava, Bhargav Yagnik, et al.
- BERT model: Hugging Face Transformers
- LIME: Marco Tulio Ribeiro et al.

>>>>>>> 4f21f3f (Initial commit)
