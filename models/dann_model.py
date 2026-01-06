"""
Domain Adversarial Neural Network (DANN) for Cross-Platform Misinformation Detection
Implements the DANN architecture with feature extractor, label predictor, and domain classifier
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
from config import Config


# =========================
# Gradient Reversal Layer
# =========================
class GradientReversalLayer(torch.autograd.Function):
    """Gradient Reversal Layer for adversarial domain adaptation"""

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


# =========================
# Feature Extractor
# =========================
class FeatureExtractor(nn.Module):
    """Feature extractor using BERT"""

    def __init__(self, config):
        super().__init__()
        self.bert = BertModel.from_pretrained(config.BERT_MODEL_NAME)

        self.feature_transform = nn.Sequential(
            nn.Linear(config.FEATURE_DIM, config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT_RATE),
            nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT_RATE),
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output  # [CLS]
        features = self.feature_transform(pooled_output)
        return features


# =========================
# Label Predictor
# =========================
class LabelPredictor(nn.Module):
    """Label predictor for misinformation classification"""

    def __init__(self, config):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM // 2),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT_RATE),
            nn.Linear(config.HIDDEN_DIM // 2, config.HIDDEN_DIM // 4),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT_RATE),
            nn.Linear(config.HIDDEN_DIM // 4, 2)
        )

    def forward(self, features):
        return self.classifier(features)


# =========================
# Domain Classifier
# =========================
class DomainClassifier(nn.Module):
    """Domain classifier for adversarial training"""

    def __init__(self, config):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM // 2),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT_RATE),
            nn.Linear(config.HIDDEN_DIM // 2, config.HIDDEN_DIM // 4),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT_RATE),
            nn.Linear(config.HIDDEN_DIM // 4, config.NUM_DOMAINS)
        )

    def forward(self, features):
        return self.classifier(features)


# =========================
# DANN Model
# =========================
class DANNModel(nn.Module):
    """Complete DANN model"""

    def __init__(self, config):
        super().__init__()
        self.feature_extractor = FeatureExtractor(config)
        self.label_predictor = LabelPredictor(config)
        self.domain_classifier = DomainClassifier(config)

    def forward(self, input_ids, attention_mask, alpha=1.0, return_features=False):
        features = self.feature_extractor(input_ids, attention_mask)

        # Label prediction
        label_logits = self.label_predictor(features)

        # Domain prediction with Gradient Reversal
        reversed_features = GradientReversalLayer.apply(features, alpha)
        domain_logits = self.domain_classifier(reversed_features)

        if return_features:
            return label_logits, domain_logits, features

        return label_logits, domain_logits

    def predict(self, input_ids, attention_mask):
        self.eval()
        with torch.no_grad():
            features = self.feature_extractor(input_ids, attention_mask)
            logits = self.label_predictor(features)
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
        return preds, probs

    def get_features(self, input_ids, attention_mask):
        self.eval()
        with torch.no_grad():
            return self.feature_extractor(input_ids, attention_mask)


# =========================
# Factory
# =========================
def create_model(config):
    return DANNModel(config)


# =========================
# Sanity Test
# =========================
if __name__ == "__main__":
    config = Config()
    model = create_model(config)

    x = torch.randint(0, 1000, (2, 128))
    mask = torch.ones(2, 128)

    y_label, y_domain = model(x, mask, alpha=0.5)
    print("Label logits:", y_label.shape)
    print("Domain logits:", y_domain.shape)
