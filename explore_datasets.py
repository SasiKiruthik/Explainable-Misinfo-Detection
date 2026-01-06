"""
Script to explore dataset structure
"""
import pandas as pd
import os

dataset_path = r"C:\Users\Sasi Kiruthik\Downloads\EXPLAINABLE MISINFO\dataset"

# Check CoAID files
print("="*50)
print("CoAID Source Domain Files")
print("="*50)
coaid_path = os.path.join(dataset_path, "CoAID-Source domain")
if os.path.exists(coaid_path):
    files = [f for f in os.listdir(coaid_path) if f.endswith('.csv')]
    print(f"Found {len(files)} CSV files")
    
    # Check one file
    sample_file = os.path.join(coaid_path, "NewsRealCOVID-19.csv")
    if os.path.exists(sample_file):
        try:
            df = pd.read_csv(sample_file, nrows=5)
            print(f"\nFile: NewsRealCOVID-19.csv")
            print(f"Columns: {list(df.columns)}")
            print(f"Shape: {df.shape}")
            print(f"Sample data:")
            print(df.head(2))
        except Exception as e:
            print(f"Error reading {sample_file}: {e}")

# Check MiSoVac
print("\n" + "="*50)
print("MiSoVac Target Domain")
print("="*50)
misovac_file = os.path.join(dataset_path, "MiSoVac-Target domain in base paper.csv")
if os.path.exists(misovac_file):
    try:
        df = pd.read_csv(misovac_file, nrows=5)
        print(f"Columns: {list(df.columns)}")
        print(f"Shape: {df.shape}")
        print(f"Sample data:")
        print(df.head(2))
    except Exception as e:
        print(f"Error reading {misovac_file}: {e}")

# Check FakeNewsNet
print("\n" + "="*50)
print("FakeNewsNet Target Domain")
print("="*50)
fakenewsnet_file = os.path.join(dataset_path, "fakenewsnet_dataset-Target domain created.xlsx")
if os.path.exists(fakenewsnet_file):
    try:
        df = pd.read_excel(fakenewsnet_file, nrows=5)
        print(f"Columns: {list(df.columns)}")
        print(f"Shape: {df.shape}")
        print(f"Sample data:")
        print(df.head(2))
    except Exception as e:
        print(f"Error reading {fakenewsnet_file}: {e}")

# Check WELFake
print("\n" + "="*50)
print("WELFake Target Domain")
print("="*50)
welfake_file = os.path.join(dataset_path, "WELFake-Target domain created.xlsx")
if os.path.exists(welfake_file):
    try:
        df = pd.read_excel(welfake_file, nrows=5)
        print(f"Columns: {list(df.columns)}")
        print(f"Shape: {df.shape}")
        print(f"Sample data:")
        print(df.head(2))
    except Exception as e:
        print(f"Error reading {welfake_file}: {e}")

