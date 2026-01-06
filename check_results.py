"""
Script to check what was generated and show results
"""
import os
import json

def check_directory(path, name):
    """Check if directory exists and list files"""
    print(f"\n{'='*60}")
    print(f"{name}")
    print('='*60)
    
    if os.path.exists(path):
        files = os.listdir(path)
        if files:
            print(f"[OK] Found {len(files)} files:")
            for file in files[:10]:  # Show first 10
                file_path = os.path.join(path, file)
                size = os.path.getsize(file_path) / (1024 * 1024)  # MB
                print(f"  - {file} ({size:.2f} MB)")
            if len(files) > 10:
                print(f"  ... and {len(files) - 10} more files")
        else:
            print("  Directory exists but is empty")
    else:
        print(f"  X Directory does not exist: {path}")

def main():
    print("="*60)
    print("CHECKING WORKFLOW RESULTS")
    print("="*60)
    
    # Check data directories
    check_directory("data/raw", "Prepared Datasets (data/raw)")
    check_directory("data/processed", "Processed Data (data/processed)")
    
    # Check model directories
    check_directory("models/checkpoints", "Model Checkpoints (models/checkpoints)")
    
    # Check results directories
    check_directory("results/evaluation", "Evaluation Results (results/evaluation)")
    check_directory("results/explanations", "Explanations (results/explanations)")
    
    # Check for training history
    history_path = "models/checkpoints/training_history.json"
    if os.path.exists(history_path):
        print("\n" + "="*60)
        print("TRAINING HISTORY FOUND")
        print("="*60)
        try:
            with open(history_path, 'r') as f:
                history = json.load(f)
            
            if 'train_loss' in history:
                print(f"Training epochs: {len(history['train_loss'])}")
                if history['train_loss']:
                    print(f"Final train loss: {history['train_loss'][-1]:.4f}")
                    print(f"Final train accuracy: {history['train_acc'][-1]:.4f}")
            
            if 'val_f1' in history:
                print("\nValidation F1 Scores (final):")
                for platform, f1_scores in history['val_f1'].items():
                    if f1_scores:
                        print(f"  {platform}: {f1_scores[-1]:.4f}")
        except Exception as e:
            print(f"Error reading history: {e}")
    
    # Check for evaluation results
    eval_path = "results/evaluation/evaluation_results.json"
    if os.path.exists(eval_path):
        print("\n" + "="*60)
        print("EVALUATION RESULTS FOUND")
        print("="*60)
        try:
            with open(eval_path, 'r') as f:
                results = json.load(f)
            
            for platform, metrics in results.items():
                print(f"\n{platform}:")
                print(f"  Accuracy: {metrics.get('accuracy', 0):.4f}")
                print(f"  Precision: {metrics.get('precision', 0):.4f}")
                print(f"  Recall: {metrics.get('recall', 0):.4f}")
                print(f"  F1 Score: {metrics.get('f1_score', 0):.4f}")
                print(f"  AUC: {metrics.get('auc', 0):.4f}")
        except Exception as e:
            print(f"Error reading results: {e}")
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    # Check what's complete
    steps_complete = []
    if os.path.exists("data/raw") and os.listdir("data/raw"):
        steps_complete.append("[OK] Dataset Preparation")
    if os.path.exists("data/processed") and os.listdir("data/processed"):
        steps_complete.append("[OK] Data Preprocessing")
    if os.path.exists("models/checkpoints/best_model.pt"):
        steps_complete.append("[OK] Model Training")
    if os.path.exists("results/evaluation/evaluation_results.json"):
        steps_complete.append("[OK] Model Evaluation")
    
    if steps_complete:
        print("Completed steps:")
        for step in steps_complete:
            print(f"  {step}")
    else:
        print("No completed steps found. The workflow may still be running or encountered an error.")
    
    print("\nNext steps:")
    if os.path.exists("models/checkpoints/best_model.pt"):
        print("  1. Test inference: python inference.py --text 'Your text here'")
        print("  2. Generate explanation: python main.py --mode explain --text 'Your text here'")
    else:
        print("  1. Check if training completed successfully")
        print("  2. Run: python train_source_target.py")

if __name__ == "__main__":
    main()

