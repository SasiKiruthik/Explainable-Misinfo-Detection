"""
Configuration file for Explainable Misinformation Detection System
"""
import os

class Config:
    # Data paths
    DATA_DIR = "data"
    RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
    PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

    # Model paths
    MODEL_DIR = "models"
    CHECKPOINT_DIR = os.path.join(MODEL_DIR, "checkpoints")

    # Results paths
    RESULTS_DIR = "results"
    EXPLANATIONS_DIR = os.path.join(RESULTS_DIR, "explanations")

    # Training parameters
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 50
    EARLY_STOPPING_PATIENCE = 10

    # DANN parameters
    LAMBDA_DOMAIN = 1.0
    GAMMA = 10.0
    FEATURE_DIM = 768
    HIDDEN_DIM = 256
    NUM_DOMAINS = 4  # CoAID + 3 targets

    # Model architecture
    DROPOUT_RATE = 0.3
    BERT_MODEL_NAME = "bert-base-uncased"

    # Data preprocessing
    MAX_SEQUENCE_LENGTH = 128
    RANDOM_SEED = 42

    # Device
    DEVICE = "cuda" if os.path.exists("/proc/driver/nvidia") else "cpu"
