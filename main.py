"""
Main entry point for the Explainable Misinformation Detection System
Provides a unified interface for all operations
"""
import argparse
import os
import sys

from config import Config


def main():
    parser = argparse.ArgumentParser(
        description='Explainable Misinformation Detection System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create sample data
  python main.py --mode sample_data
  
  # Preprocess data
  python main.py --mode preprocess --data_files data/raw/twitter.csv data/raw/facebook.csv --platforms Twitter Facebook
  
  # Train model
  python main.py --mode train
  
  # Evaluate model
  python main.py --mode evaluate
  
  # Run inference
  python main.py --mode inference --text "Your text here"
  
  # Generate explanations
  python main.py --mode explain --text "Your text here"
        """
    )
    
    parser.add_argument('--mode', type=str, required=True,
                       choices=['sample_data', 'preprocess', 'train', 'evaluate', 'inference', 'explain'],
                       help='Operation mode')
    
    # Data preprocessing arguments
    parser.add_argument('--data_files', type=str, nargs='+',
                       help='Data files for preprocessing')
    parser.add_argument('--platforms', type=str, nargs='+',
                       help='Platform names corresponding to data files')
    
    # Inference arguments
    parser.add_argument('--text', type=str,
                       help='Text to classify or explain')
    parser.add_argument('--file', type=str,
                       help='File containing texts for batch processing')
    parser.add_argument('--model', type=str, default=None,
                       help='Path to model checkpoint')
    
    args = parser.parse_args()
    
    config = Config()
    
    # Create necessary directories
    os.makedirs(config.DATA_DIR, exist_ok=True)
    os.makedirs(config.RAW_DATA_DIR, exist_ok=True)
    os.makedirs(config.PROCESSED_DATA_DIR, exist_ok=True)
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    os.makedirs(config.EXPLANATIONS_DIR, exist_ok=True)
    
    if args.mode == 'sample_data':
        print("Creating sample data...")
        from create_sample_data import create_sample_data
        create_sample_data()
        print("Sample data created successfully!")
    
    elif args.mode == 'preprocess':
        if not args.data_files or not args.platforms:
            print("Error: --data_files and --platforms are required for preprocessing")
            sys.exit(1)
        
        if len(args.data_files) != len(args.platforms):
            print("Error: Number of data files must match number of platforms")
            sys.exit(1)
        
        print("Preprocessing data...")
        from data_preprocessing import DataPreprocessor
        
        preprocessor = DataPreprocessor(config)
        datasets, encoded_data = preprocessor.prepare_datasets(args.data_files, args.platforms)
        print("Data preprocessing completed successfully!")
    
    elif args.mode == 'train':
        print("Starting training...")
        from train import main as train_main
        train_main()
        print("Training completed!")
    
    elif args.mode == 'evaluate':
        print("Evaluating model...")
        from evaluate import main as eval_main
        eval_main()
        print("Evaluation completed!")
    
    elif args.mode == 'inference':
        if not args.text and not args.file:
            print("Error: --text or --file is required for inference")
            sys.exit(1)
        
        print("Running inference...")
        from inference import MisinformationDetector
        
        detector = MisinformationDetector(config, args.model)
        
        if args.text:
            prediction, probabilities = detector.predict(args.text, return_probabilities=True)
            label = "Misinformation" if prediction == 1 else "Real"
            print(f"\nPrediction: {label}")
            print(f"Confidence: {probabilities[prediction]:.4f}")
            print(f"Probabilities - Real: {probabilities[0]:.4f}, "
                  f"Misinformation: {probabilities[1]:.4f}")
        
        elif args.file:
            if not os.path.exists(args.file):
                print(f"Error: File not found: {args.file}")
                sys.exit(1)
            
            with open(args.file, 'r', encoding='utf-8') as f:
                texts = [line.strip() for line in f if line.strip()]
            
            predictions, probabilities = detector.predict_batch(texts, return_probabilities=True)
            
            output_file = args.file.replace('.txt', '_predictions.txt')
            with open(output_file, 'w', encoding='utf-8') as f:
                for text, pred, prob in zip(texts, predictions, probabilities):
                    label = "Misinformation" if pred == 1 else "Real"
                    f.write(f"{label}\t{prob[pred]:.4f}\t{text}\n")
            
            print(f"Results saved to {output_file}")
    
    elif args.mode == 'explain':
        if not args.text:
            print("Error: --text is required for explanation")
            sys.exit(1)
        
        print("Generating explanation...")
        from inference import MisinformationDetector
        
        detector = MisinformationDetector(config, args.model)
        result = detector.predict_with_explanation(args.text)
        
        print("\n" + "="*50)
        print("PREDICTION RESULT WITH EXPLANATION")
        print("="*50)
        print(f"Text: {result['text']}")
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print(f"Probabilities - Real: {result['probabilities']['real']:.4f}, "
              f"Misinformation: {result['probabilities']['misinformation']:.4f}")
        print("\nTop Features:")
        for feature, weight in result['explanation'][:10]:
            print(f"  {feature}: {weight:.4f}")


if __name__ == "__main__":
    main()

