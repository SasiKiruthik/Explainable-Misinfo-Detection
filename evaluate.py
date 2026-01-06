"""
Evaluation script for Misinformation Detection Model
Tests the model on test sets and generates comprehensive evaluation metrics
"""
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import pickle
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from config import Config
from models.dann_model import create_model
from train import MisinformationDataset


class Evaluator:
    """Evaluator for misinformation detection model"""
    
    def __init__(self, config, model_path=None):
        self.config = config
        self.device = torch.device(config.DEVICE)
        
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
    
    def evaluate_dataset(self, dataset, dataset_name="Test"):
        """
        Evaluate model on a dataset
        
        Args:
            dataset: MisinformationDataset object
            dataset_name: Name of the dataset
        
        Returns:
            Dictionary of evaluation metrics
        """
        data_loader = DataLoader(
            dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            num_workers=0
        )
        
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        print(f"Evaluating on {dataset_name} set...")
        with torch.no_grad():
            for batch in tqdm(data_loader):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Predict
                label_logits, _ = self.model(input_ids, attention_mask)
                probabilities = torch.softmax(label_logits, dim=1)
                predictions = torch.argmax(probabilities, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities[:, 1].cpu().numpy())
        
        # Compute metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_predictions, average='weighted', zero_division=0)
        f1 = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)
        
        # Per-class metrics
        precision_per_class = precision_score(all_labels, all_predictions, average=None, zero_division=0)
        recall_per_class = recall_score(all_labels, all_predictions, average=None, zero_division=0)
        f1_per_class = f1_score(all_labels, all_predictions, average=None, zero_division=0)
        
        try:
            auc = roc_auc_score(all_labels, all_probabilities)
        except:
            auc = 0.0
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)
        
        metrics = {
            'dataset_name': dataset_name,
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'auc': float(auc),
            'precision_per_class': {
                'real': float(precision_per_class[0]),
                'misinformation': float(precision_per_class[1])
            },
            'recall_per_class': {
                'real': float(recall_per_class[0]),
                'misinformation': float(recall_per_class[1])
            },
            'f1_per_class': {
                'real': float(f1_per_class[0]),
                'misinformation': float(f1_per_class[1])
            },
            'confusion_matrix': cm.tolist(),
            'num_samples': len(all_labels)
        }
        
        return metrics, all_predictions, all_labels, all_probabilities
    
    def plot_confusion_matrix(self, cm, labels, save_path=None):
        """Plot confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=labels,
            yticklabels=labels
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            print(f"Confusion matrix saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_roc_curve(self, y_true, y_scores, save_path=None):
        """Plot ROC curve"""
        from sklearn.metrics import roc_curve
        
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        auc = roc_auc_score(y_true, y_scores)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            print(f"ROC curve saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def evaluate_all_platforms(self, save_dir=None):
        """Evaluate on all platforms"""
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        # Load processed data
        print("Loading processed data...")
        with open(os.path.join(self.config.PROCESSED_DATA_DIR, 'encoded_data.pkl'), 'rb') as f:
            encoded_data = pickle.load(f)
        
        with open(os.path.join(self.config.PROCESSED_DATA_DIR, 'datasets.pkl'), 'rb') as f:
            datasets = pickle.load(f)
        
        # Convert to tensors
        encoded_data['input_ids'] = encoded_data['input_ids'].to(self.device)
        encoded_data['attention_mask'] = encoded_data['attention_mask'].to(self.device)
        
        all_results = {}
        
        # Evaluate each platform
        for platform_name, platform_data in datasets.items():
            print(f"\n{'='*50}")
            print(f"Evaluating platform: {platform_name}")
            print(f"{'='*50}")
            
            # Test set
            test_data = platform_data['test']
            test_dataset = MisinformationDataset(
                test_data['indices'],
                test_data['labels'],
                encoded_data,
                test_data['texts']
            )
            
            metrics, predictions, labels, probabilities = self.evaluate_dataset(
                test_dataset,
                f"{platform_name}_Test"
            )
            
            all_results[platform_name] = metrics
            
            # Plot confusion matrix
            cm = np.array(metrics['confusion_matrix'])
            if save_dir:
                cm_path = os.path.join(save_dir, f'confusion_matrix_{platform_name}.png')
                self.plot_confusion_matrix(cm, ['Real', 'Misinformation'], cm_path)
                
                roc_path = os.path.join(save_dir, f'roc_curve_{platform_name}.png')
                self.plot_roc_curve(labels, probabilities, roc_path)
            
            # Print results
            print(f"\nResults for {platform_name}:")
            print(f"Accuracy: {metrics['accuracy']:.4f}")
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"Recall: {metrics['recall']:.4f}")
            print(f"F1 Score: {metrics['f1_score']:.4f}")
            print(f"AUC: {metrics['auc']:.4f}")
            print(f"\nPer-class metrics:")
            print(f"Real - Precision: {metrics['precision_per_class']['real']:.4f}, "
                  f"Recall: {metrics['recall_per_class']['real']:.4f}, "
                  f"F1: {metrics['f1_per_class']['real']:.4f}")
            print(f"Misinformation - Precision: {metrics['precision_per_class']['misinformation']:.4f}, "
                  f"Recall: {metrics['recall_per_class']['misinformation']:.4f}, "
                  f"F1: {metrics['f1_per_class']['misinformation']:.4f}")
        
        # Save all results
        if save_dir:
            results_path = os.path.join(save_dir, 'evaluation_results.json')
            with open(results_path, 'w') as f:
                json.dump(all_results, f, indent=2)
            print(f"\nAll results saved to {results_path}")
        
        return all_results


def main():
    """Main evaluation function"""
    config = Config()
    evaluator = Evaluator(config)
    
    # Create results directory
    results_dir = os.path.join(config.RESULTS_DIR, 'evaluation')
    os.makedirs(results_dir, exist_ok=True)
    
    # Evaluate all platforms
    results = evaluator.evaluate_all_platforms(save_dir=results_dir)
    
    print("\n" + "="*50)
    print("Evaluation completed!")
    print("="*50)


if __name__ == "__main__":
    main()

