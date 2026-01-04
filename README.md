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
