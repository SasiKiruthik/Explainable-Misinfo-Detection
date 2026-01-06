"""
LIME-based Explainability Module for Misinformation Detection
Provides local interpretable explanations for model predictions
"""
import os
import torch
import numpy as np
from lime.lime_text import LimeTextExplainer
import pickle
import json
from tqdm import tqdm
from transformers import BertTokenizer

from config import Config
from models.dann_model import create_model


class MisinformationExplainer:
    """LIME-based explainer for misinformation detection"""
    
    def __init__(self, config, model_path=None):
        self.config = config
        self.device = torch.device(config.DEVICE)
        self.tokenizer = BertTokenizer.from_pretrained(config.BERT_MODEL_NAME)
        
        # Load model
        if model_path is None:
            model_path = os.path.join(config.CHECKPOINT_DIR, 'best_model.pt')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        print(f"Loading model from {model_path}...")
        self.model = create_model(config)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize LIME explainer
        self.explainer = LimeTextExplainer(
            class_names=['Real', 'Misinformation'],
            split_expression=r'\s+'
        )
    
    def predict_proba(self, texts):
        """Predict probabilities for a batch of texts"""
        if isinstance(texts, str):
            texts = [texts]
        
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
        
        return probabilities.cpu().numpy()
    
    def explain_instance(self, text, num_features=10, num_samples=5000):
        """
        Explain a single instance
        
        Args:
            text: Input text to explain
            num_features: Number of features to show in explanation
            num_samples: Number of samples for LIME
        
        Returns:
            LIME explanation object
        """
        def predict_fn(texts):
            probs = self.predict_proba(texts)
            return probs
        
        explanation = self.explainer.explain_instance(
            text,
            predict_fn,
            num_features=num_features,
            num_samples=num_samples
        )
        
        return explanation
    
    def explain_batch(self, texts, num_features=10, num_samples=5000, save_dir=None):
        """
        Explain a batch of texts
        
        Args:
            texts: List of texts to explain
            num_features: Number of features to show
            num_samples: Number of samples for LIME
            save_dir: Directory to save explanations
        
        Returns:
            List of explanation dictionaries
        """
        explanations = []
        
        for i, text in enumerate(tqdm(texts, desc="Generating explanations")):
            try:
                explanation = self.explain_instance(text, num_features, num_samples)
                
                # Get prediction
                probs = self.predict_proba(text)[0]
                prediction = int(np.argmax(probs))
                confidence = float(probs[prediction])
                
                # Extract explanation features
                exp_list = explanation.as_list()
                
                exp_dict = {
                    'text': text,
                    'prediction': 'Misinformation' if prediction == 1 else 'Real',
                    'confidence': confidence,
                    'probabilities': {
                        'real': float(probs[0]),
                        'misinformation': float(probs[1])
                    },
                    'explanation': [
                        {
                            'feature': feature,
                            'weight': float(weight)
                        }
                        for feature, weight in exp_list
                    ],
                    'top_features': {
                        'supporting_misinformation': [
                            {'feature': f, 'weight': float(w)}
                            for f, w in exp_list if w > 0
                        ][:num_features],
                        'supporting_real': [
                            {'feature': f, 'weight': float(w)}
                            for f, w in exp_list if w < 0
                        ][:num_features]
                    }
                }
                
                explanations.append(exp_dict)
                
                # Save individual explanation
                if save_dir:
                    os.makedirs(save_dir, exist_ok=True)
                    exp_path = os.path.join(save_dir, f'explanation_{i}.json')
                    with open(exp_path, 'w', encoding='utf-8') as f:
                        json.dump(exp_dict, f, indent=2, ensure_ascii=False)
                
            except Exception as e:
                print(f"Error explaining text {i}: {str(e)}")
                explanations.append({
                    'text': text,
                    'error': str(e)
                })
        
        return explanations
    
    def visualize_explanation(self, explanation, save_path=None):
        """
        Visualize LIME explanation
        
        Args:
            explanation: LIME explanation object
            save_path: Path to save visualization
        """
        try:
            import matplotlib.pyplot as plt
            
            fig = explanation.as_pyplot_figure()
            
            if save_path:
                plt.savefig(save_path, bbox_inches='tight', dpi=300)
                print(f"Explanation visualization saved to {save_path}")
            else:
                plt.show()
            
            plt.close()
        except Exception as e:
            print(f"Error visualizing explanation: {str(e)}")
    
    def explain_test_set(self, test_texts, test_labels, save_dir=None):
        """
        Explain test set and analyze explanations
        
        Args:
            test_texts: List of test texts
            test_labels: List of test labels
            save_dir: Directory to save explanations
        """
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        explanations = self.explain_batch(
            test_texts,
            num_features=self.config.LIME_NUM_FEATURES,
            num_samples=self.config.LIME_NUM_SAMPLES,
            save_dir=save_dir
        )
        
        # Analyze explanations
        analysis = {
            'total_explanations': len(explanations),
            'correct_predictions': 0,
            'incorrect_predictions': 0,
            'average_confidence': 0.0,
            'common_misinformation_features': {},
            'common_real_features': {}
        }
        
        misinformation_features = []
        real_features = []
        confidences = []
        
        for i, exp in enumerate(explanations):
            if 'error' in exp:
                continue
            
            pred = 1 if exp['prediction'] == 'Misinformation' else 0
            true_label = test_labels[i]
            
            if pred == true_label:
                analysis['correct_predictions'] += 1
            else:
                analysis['incorrect_predictions'] += 1
            
            confidences.append(exp['confidence'])
            
            # Collect features
            if pred == 1:
                for feat in exp['top_features']['supporting_misinformation']:
                    feature = feat['feature']
                    weight = feat['weight']
                    misinformation_features.append((feature, weight))
            else:
                for feat in exp['top_features']['supporting_real']:
                    feature = feat['feature']
                    weight = abs(feat['weight'])
                    real_features.append((feature, weight))
        
        analysis['average_confidence'] = float(np.mean(confidences)) if confidences else 0.0
        
        # Count common features
        from collections import Counter
        misinfo_counter = Counter([f[0] for f in misinformation_features])
        real_counter = Counter([f[0] for f in real_features])
        
        analysis['common_misinformation_features'] = dict(misinfo_counter.most_common(20))
        analysis['common_real_features'] = dict(real_counter.most_common(20))
        
        # Save analysis
        if save_dir:
            analysis_path = os.path.join(save_dir, 'explanation_analysis.json')
            with open(analysis_path, 'w', encoding='utf-8') as f:
                json.dump(analysis, f, indent=2, ensure_ascii=False)
            
            # Save all explanations
            all_exp_path = os.path.join(save_dir, 'all_explanations.json')
            with open(all_exp_path, 'w', encoding='utf-8') as f:
                json.dump(explanations, f, indent=2, ensure_ascii=False)
        
        return explanations, analysis


if __name__ == "__main__":
    # Example usage
    config = Config()
    explainer = MisinformationExplainer(config)
    
    # Example text
    test_text = "COVID-19 is a hoax created by the government to control the population."
    
    print(f"Explaining: {test_text}")
    explanation = explainer.explain_instance(test_text)
    
    print("\nExplanation:")
    print(explanation.as_list())
    
    # Visualize
    explainer.visualize_explanation(explanation, save_path="explanation_example.png")

