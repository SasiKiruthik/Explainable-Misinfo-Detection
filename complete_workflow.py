"""
Complete workflow script for training on CoAID (source) and target domains
This script automates the entire process from data preparation to training
"""
import os
import sys
import subprocess

def run_step(step_num, description, command):
    """Run a step and handle errors"""
    print("\n" + "="*70)
    print(f"STEP {step_num}: {description}")
    print("="*70)
    print(f"Running: {command}")
    print("-"*70)
    
    result = subprocess.run(command, shell=True, capture_output=False)
    
    if result.returncode != 0:
        print(f"\nERROR in Step {step_num}!")
        print("Please check the error messages above.")
        return False
    
    print(f"\n✓ Step {step_num} completed successfully!")
    return True


def main():
    """Complete workflow"""
    print("="*70)
    print("COMPLETE WORKFLOW: Source-Target Domain Adaptation Training")
    print("="*70)
    print("Source Domain: CoAID")
    print("Target Domains: MiSoVac, FakeNewsNet, WELFake")
    print("="*70)
    
    steps = [
        (1, "Install openpyxl for Excel file support", 
         "pip install openpyxl"),
        
        (2, "Prepare datasets (combine CoAID files, process target domains)",
         "python prepare_datasets.py"),
        
        (3, "Preprocess all datasets",
         'python main.py --mode preprocess --data_files data/raw/coaid_source.csv data/raw/misovac_target.csv data/raw/fakenewsnet_target.csv data/raw/welfake_target.csv --platforms CoAID MiSoVac FakeNewsNet WELFake'),
        
        (4, "Train model with source-target domain adaptation",
         "python train_source_target.py"),
        
        (5, "Evaluate model on all domains",
         "python evaluate.py"),
    ]
    
    print("\nThis workflow will execute the following steps:")
    for step_num, desc, _ in steps:
        print(f"  {step_num}. {desc}")
    
    response = input("\nDo you want to proceed? (yes/no): ").strip().lower()
    if response not in ['yes', 'y']:
        print("Workflow cancelled.")
        return
    
    # Execute steps
    for step_num, description, command in steps:
        success = run_step(step_num, description, command)
        if not success:
            print(f"\nWorkflow stopped at Step {step_num}.")
            print("Please fix the issue and run the remaining steps manually.")
            return
    
    print("\n" + "="*70)
    print("WORKFLOW COMPLETED SUCCESSFULLY!")
    print("="*70)
    print("\nNext steps:")
    print("1. Check results in: results/evaluation/")
    print("2. Check model checkpoints in: models/checkpoints/")
    print("3. Run inference: python inference.py --text 'Your text here'")
    print("4. Generate explanations: python main.py --mode explain --text 'Your text here'")


if __name__ == "__main__":
    main()

