"""
Training script for DANN-based Misinformation Detection Model
Implements adversarial training with domain adaptation
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tqdm import tqdm
import pickle
import json
from datetime import datetime

from config import Config
from models.dann_model import create_model


class MisinformationDataset(Dataset):
    """Dataset class for misinformation detection"""
    
    def __init__(self, indices, labels, encoded_data, texts=None, platform=None, platform_to_domain=None):
        self.indices = indices
        self.labels = labels
        self.input_ids = encoded_data['input_ids'][indices]
        self.attention_mask = encoded_data['attention_mask'][indices]
        self.texts = texts if texts else [None] * len(indices)
        self.platform = platform
        self.platform_to_domain = platform_to_domain or {}
    
    def get_domain_labels(self):
        """Get domain labels for this dataset"""
        if self.platform and self.platform in self.platform_to_domain:
            domain_id = self.platform_to_domain[self.platform]
            return torch.full((len(self.labels),), domain_id, dtype=torch.long)
        return torch.zeros(len(self.labels), dtype=torch.long)
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        item = {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'label': torch.tensor(self.labels[idx], dtype=torch.long),
            'text': self.texts[idx] if idx < len(self.texts) else None
        }
        # Add domain label if available
        if hasattr(self, 'domain_labels'):
            item['domain_label'] = self.domain_labels[idx]
        return item


class Trainer:
    """Trainer class for DANN model"""
    
    def __init__(self, config, model, train_loader, val_loader, device):
        self.config = config
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Loss functions
        self.label_criterion = nn.CrossEntropyLoss()
        self.domain_criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=1e-5
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_label_loss': [],
            'train_domain_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_label_loss': [],
            'val_domain_loss': [],
            'val_acc': [],
            'val_f1': [],
            'val_auc': []
        }
        
        self.best_val_f1 = 0.0
        self.patience_counter = 0
        
        # Create checkpoint directory
        os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    
    def compute_alpha(self, epoch, num_epochs):
        """Compute gradient reversal parameter (progressive training)"""
        p = epoch / num_epochs
        alpha = 2.0 / (1.0 + np.exp(-10 * p)) - 1.0
        return alpha
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        total_label_loss = 0.0
        total_domain_loss = 0.0
        all_predictions = []
        all_labels = []
        
        alpha = self.compute_alpha(epoch, self.config.NUM_EPOCHS)
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.NUM_EPOCHS}")
        
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Get domain labels from batch
            domain_labels = batch.get('domain_label', torch.zeros(len(labels), dtype=torch.long)).to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            label_logits, domain_logits = self.model(input_ids, attention_mask, alpha=alpha)
            
            # Compute losses
            label_loss = self.label_criterion(label_logits, labels)
            domain_loss = self.domain_criterion(domain_logits, domain_labels)
            
            # Combined loss
            loss = label_loss + self.config.LAMBDA_DOMAIN * domain_loss
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            total_label_loss += label_loss.item()
            total_domain_loss += domain_loss.item()
            
            predictions = torch.argmax(label_logits, dim=1).cpu().numpy()
            all_predictions.extend(predictions)
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'label_loss': f'{label_loss.item():.4f}',
                'domain_loss': f'{domain_loss.item():.4f}'
            })
        
        # Compute metrics
        avg_loss = total_loss / len(self.train_loader)
        avg_label_loss = total_label_loss / len(self.train_loader)
        avg_domain_loss = total_domain_loss / len(self.train_loader)
        accuracy = accuracy_score(all_labels, all_predictions)
        
        return avg_loss, avg_label_loss, avg_domain_loss, accuracy
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        total_label_loss = 0.0
        total_domain_loss = 0.0
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Get domain labels from batch
                domain_labels = batch.get('domain_label', torch.zeros(len(labels), dtype=torch.long)).to(self.device)
                
                # Forward pass
                label_logits, domain_logits = self.model(input_ids, attention_mask)
                
                # Compute losses
                label_loss = self.label_criterion(label_logits, labels)
                domain_loss = self.domain_criterion(domain_logits, domain_labels)
                loss = label_loss + self.config.LAMBDA_DOMAIN * domain_loss
                
                # Statistics
                total_loss += loss.item()
                total_label_loss += label_loss.item()
                total_domain_loss += domain_loss.item()
                
                probabilities = torch.softmax(label_logits, dim=1)
                predictions = torch.argmax(probabilities, dim=1).cpu().numpy()
                
                all_predictions.extend(predictions)
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities[:, 1].cpu().numpy())
        
        # Compute metrics
        avg_loss = total_loss / len(self.val_loader)
        avg_label_loss = total_label_loss / len(self.val_loader)
        avg_domain_loss = total_domain_loss / len(self.val_loader)
        accuracy = accuracy_score(all_labels, all_predictions)
        precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_predictions, average='weighted', zero_division=0)
        f1 = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)
        
        try:
            auc = roc_auc_score(all_labels, all_probabilities)
        except:
            auc = 0.0
        
        return avg_loss, avg_label_loss, avg_domain_loss, accuracy, f1, auc
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_f1': self.best_val_f1,
            'history': self.history
        }
        
        checkpoint_path = os.path.join(self.config.CHECKPOINT_DIR, f'checkpoint_epoch_{epoch+1}.pt')
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_path = os.path.join(self.config.CHECKPOINT_DIR, 'best_model.pt')
            torch.save(checkpoint, best_path)
            print(f"Saved best model with F1: {self.best_val_f1:.4f}")
    
    def train(self):
        """Main training loop"""
        print("Starting training...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(self.config.NUM_EPOCHS):
            # Train
            train_loss, train_label_loss, train_domain_loss, train_acc = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_label_loss, val_domain_loss, val_acc, val_f1, val_auc = self.validate()
            
            # Update learning rate
            self.scheduler.step(val_f1)
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_label_loss'].append(train_label_loss)
            self.history['train_domain_loss'].append(train_domain_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_label_loss'].append(val_label_loss)
            self.history['val_domain_loss'].append(val_domain_loss)
            self.history['val_acc'].append(val_acc)
            self.history['val_f1'].append(val_f1)
            self.history['val_auc'].append(val_auc)
            
            # Print metrics
            print(f"\nEpoch {epoch+1}/{self.config.NUM_EPOCHS}")
            print(f"Train - Loss: {train_loss:.4f}, Label Loss: {train_label_loss:.4f}, "
                  f"Domain Loss: {train_domain_loss:.4f}, Acc: {train_acc:.4f}")
            print(f"Val - Loss: {val_loss:.4f}, Label Loss: {val_label_loss:.4f}, "
                  f"Domain Loss: {val_domain_loss:.4f}, Acc: {val_acc:.4f}, "
                  f"F1: {val_f1:.4f}, AUC: {val_auc:.4f}")
            
            # Early stopping and checkpointing
            is_best = val_f1 > self.best_val_f1
            if is_best:
                self.best_val_f1 = val_f1
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            # Save checkpoint
            if (epoch + 1) % 5 == 0 or is_best:
                self.save_checkpoint(epoch, is_best)
            
            # Early stopping
            if self.patience_counter >= self.config.EARLY_STOPPING_PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        print("\nTraining completed!")
        print(f"Best validation F1: {self.best_val_f1:.4f}")
        
        # Save training history
        history_path = os.path.join(self.config.CHECKPOINT_DIR, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        return self.history


def main():
    """Main training function"""
    config = Config()
    device = torch.device(config.DEVICE)
    
    # Load processed data
    print("Loading processed data...")
    with open(os.path.join(config.PROCESSED_DATA_DIR, 'encoded_data.pkl'), 'rb') as f:
        encoded_data = pickle.load(f)
    
    with open(os.path.join(config.PROCESSED_DATA_DIR, 'datasets.pkl'), 'rb') as f:
        datasets = pickle.load(f)
    
    # Convert encoded data to tensors
    encoded_data['input_ids'] = encoded_data['input_ids'].to(device)
    encoded_data['attention_mask'] = encoded_data['attention_mask'].to(device)
    
    # Combine datasets from all platforms for training
    # Assuming we have at least one platform
    platform_names = list(datasets.keys())
    if len(platform_names) == 0:
        raise ValueError("No datasets found!")
    
    # Create platform to domain mapping
    platform_to_domain = {platform: idx for idx, platform in enumerate(platform_names)}
    config.NUM_DOMAINS = len(platform_names)
    
    # Combine training data from all platforms with domain labels
    train_indices = []
    train_labels = []
    train_texts = []
    train_domain_labels = []
    
    for platform in platform_names:
        train_data = datasets[platform]['train']
        domain_id = platform_to_domain[platform]
        num_samples = len(train_data['indices'])
        
        train_indices.extend(train_data['indices'])
        train_labels.extend(train_data['labels'])
        train_texts.extend(train_data['texts'])
        train_domain_labels.extend([domain_id] * num_samples)
    
    # Create datasets with domain information
    train_dataset = MisinformationDataset(
        train_indices, train_labels, encoded_data, train_texts
    )
    # Add domain labels to dataset items
    train_dataset.domain_labels = torch.tensor(train_domain_labels, dtype=torch.long)
    
    # Use first platform's validation set
    val_data = datasets[platform_names[0]]['val']
    val_domain_id = platform_to_domain[platform_names[0]]
    val_dataset = MisinformationDataset(
        val_data['indices'], val_data['labels'], encoded_data, val_data['texts']
    )
    val_dataset.domain_labels = torch.full((len(val_data['labels']),), val_domain_id, dtype=torch.long)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )
    
    # Create model
    print("Creating model...")
    model = create_model(config)
    
    # Create trainer
    trainer = Trainer(config, model, train_loader, val_loader, device)
    
    # Train
    history = trainer.train()
    
    print("Training completed successfully!")


if __name__ == "__main__":
    main()

