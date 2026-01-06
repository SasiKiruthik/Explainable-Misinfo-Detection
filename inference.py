"""
Inference script for Misinformation Detection
Provides easy-to-use interface for predicting misinformation in new texts
"""
import os
import torch
import numpy as np
from transformers import BertTokenizer
import argparse

from config import Config
from models.dann_model import create_model
from explainability import MisinformationExplainer


class MisinformationDetector:
    """Simple interface for misinformation detection"""
    
    def __init__(self, config, model_path=None):
        self.config = config
        self.device = torch.device(config.DEVICE)
        self.tokenizer = BertTokenizer.from_pretrained(config.BERT_MODEL_NAME)
        
        # Load model
        if model_path is None:
            model_path = os.path.join(config.CHECKPOINT_DIR, 'best_model.pt')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}. Please train the model first.")
        
        print(f"Loading model from {model_path}...")
        self.model = create_model(config)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print("Model loaded successfully!")
    
    def predict(self, text, return_probabilities=False):
        """
        Predict if a text contains misinformation
        
        Args:
            text: Input text to classify
            return_probabilities: Whether to return probability scores
        
        Returns:
            Prediction (0: Real, 1: Misinformation) and optionally probabilities
        """
        # Clean and tokenize
        encoded = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.config.MAX_SEQUENCE_LENGTH,
            return_tensors='pt'
        )
        
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)
        
        # Predict
        with torch.no_grad():
            label_logits, _ = self.model(input_ids, attention_mask)
            probabilities = torch.softmax(label_logits, dim=1)
            prediction = torch.argmax(probabilities, dim=1).item()
        
        if return_probabilities:
            return prediction, probabilities[0].cpu().numpy()
        return prediction
    
    def predict_batch(self, texts, return_probabilities=False):
        """
        Predict for a batch of texts
        
        Args:
            texts: List of texts to classify
            return_probabilities: Whether to return probability scores
        
        Returns:
            List of predictions and optionally probabilities
        """
        # Tokenize
        encoded = self.tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=self.config.MAX_SEQUENCE_LENGTH,
            return_tensors='pt'
        )
        
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)
        
        # Predict
        with torch.no_grad():
            label_logits, _ = self.model(input_ids, attention_mask)
            probabilities = torch.softmax(label_logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1).cpu().numpy()
        
        if return_probabilities:
            return predictions, probabilities.cpu().numpy()
        return predictions
    
    def predict_with_explanation(self, text, num_features=10):
        """
        Predict with LIME explanation
        
        Args:
            text: Input text to classify
            num_features: Number of features to show in explanation
        
        Returns:
            Dictionary with prediction and explanation
        """
        explainer = MisinformationExplainer(self.config)
        explanation = explainer.explain_instance(text, num_features=num_features)
        
        # Get prediction
        prediction, probabilities = self.predict(text, return_probabilities=True)
        
        result = {
            'text': text,
            'prediction': 'Misinformation' if prediction == 1 else 'Real',
            'confidence': float(probabilities[prediction]),
            'probabilities': {
                'real': float(probabilities[0]),
                'misinformation': float(probabilities[1])
            },
            'explanation': explanation.as_list()
        }
        
        return result


def main():
    """Main inference function"""
    parser = argparse.ArgumentParser(description='Misinformation Detection Inference')
    parser.add_argument('--text', type=str, help='Text to classify')
    parser.add_argument('--file', type=str, help='File containing texts (one per line)')
    parser.add_argument('--explain', action='store_true', help='Generate LIME explanation')
    parser.add_argument('--model', type=str, default=None, help='Path to model checkpoint')
    
    args = parser.parse_args()
    
    config = Config()
    detector = MisinformationDetector(config, args.model)
    
    if args.text:
        # Single text prediction
        if args.explain:
            result = detector.predict_with_explanation(args.text)
            print("\n" + "="*50)
            print("PREDICTION RESULT")
            print("="*50)
            print(f"Text: {result['text']}")
            print(f"Prediction: {result['prediction']}")
            print(f"Confidence: {result['confidence']:.4f}")
            print(f"Probabilities - Real: {result['probabilities']['real']:.4f}, "
                  f"Misinformation: {result['probabilities']['misinformation']:.4f}")
            print("\nTop Features:")
            for feature, weight in result['explanation'][:10]:
                print(f"  {feature}: {weight:.4f}")
        else:
            prediction, probabilities = detector.predict(args.text, return_probabilities=True)
            label = "Misinformation" if prediction == 1 else "Real"
            print(f"\nPrediction: {label}")
            print(f"Confidence: {probabilities[prediction]:.4f}")
            print(f"Probabilities - Real: {probabilities[0]:.4f}, "
                  f"Misinformation: {probabilities[1]:.4f}")
    
    elif args.file:
        # Batch prediction from file
        if not os.path.exists(args.file):
            print(f"Error: File not found: {args.file}")
            return
        
        with open(args.file, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
        
        print(f"Processing {len(texts)} texts...")
        predictions, probabilities = detector.predict_batch(texts, return_probabilities=True)
        
        # Save results
        output_file = args.file.replace('.txt', '_predictions.txt')
        with open(output_file, 'w', encoding='utf-8') as f:
            for text, pred, prob in zip(texts, predictions, probabilities):
                label = "Misinformation" if pred == 1 else "Real"
                f.write(f"{label}\t{prob[pred]:.4f}\t{text}\n")
        
        print(f"Results saved to {output_file}")
    
    else:
        # Interactive mode
        print("Interactive Misinformation Detection")
        print("Type 'quit' to exit\n")
        
        while True:
            text = input("Enter text to classify: ").strip()
            if text.lower() == 'quit':
                break
            
            if not text:
                continue
            
            prediction, probabilities = detector.predict(text, return_probabilities=True)
            label = "Misinformation" if prediction == 1 else "Real"
            print(f"Prediction: {label}")
            print(f"Confidence: {probabilities[prediction]:.4f}")
            print(f"Probabilities - Real: {probabilities[0]:.4f}, "
                  f"Misinformation: {probabilities[1]:.4f}\n")


if __name__ == "__main__":
    main()

