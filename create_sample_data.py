"""
Script to create sample data for testing the misinformation detection system
This is a utility script to generate sample datasets if you don't have real data
"""
import os
import pandas as pd
import numpy as np

from config import Config


def create_sample_data(num_samples=1000, output_dir=None):
    """Create sample misinformation detection datasets"""
    if output_dir is None:
        output_dir = Config.RAW_DATA_DIR
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Sample real news texts
    real_texts = [
        "Scientists have discovered a new method to improve renewable energy efficiency.",
        "The government announced new healthcare policies to benefit citizens.",
        "Research shows that regular exercise improves mental health.",
        "New technology helps reduce carbon emissions in manufacturing.",
        "Education programs show positive results in student performance.",
        "Medical breakthrough offers hope for treating rare diseases.",
        "Economic indicators suggest steady growth in the market.",
        "Environmental initiatives lead to cleaner air in urban areas.",
        "Social programs help reduce poverty rates significantly.",
        "Innovation in agriculture increases crop yields sustainably."
    ]
    
    # Sample misinformation texts
    misinformation_texts = [
        "Secret government experiment causes all diseases - they don't want you to know!",
        "One simple trick that doctors hate - cure any illness instantly!",
        "Shocking truth: vaccines are actually mind control devices!",
        "This one food will make you lose 50 pounds in a week - guaranteed!",
        "They're hiding the cure for cancer - big pharma doesn't want you healthy!",
        "Alien technology discovered but kept secret by governments worldwide!",
        "Miracle cure that works instantly - but doctors won't tell you about it!",
        "Secret ingredient in your food that's slowly poisoning everyone!",
        "One weird trick that makes you rich overnight - banks don't want you to know!",
        "Hidden truth about COVID-19 that the media is covering up!"
    ]
    
    # Generate datasets for two platforms
    platforms = ["Twitter", "Facebook"]
    
    for platform in platforms:
        data = []
        
        # Generate real samples
        for i in range(num_samples // 2):
            base_text = np.random.choice(real_texts)
            # Add some variation
            text = base_text + " " + str(np.random.randint(1, 100))
            data.append({
                'text': text,
                'label': 0,
                'platform': platform
            })
        
        # Generate misinformation samples
        for i in range(num_samples // 2):
            base_text = np.random.choice(misinformation_texts)
            # Add some variation
            text = base_text + " " + str(np.random.randint(1, 100))
            data.append({
                'text': text,
                'label': 1,
                'platform': platform
            })
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Shuffle
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Save
        output_file = os.path.join(output_dir, f"{platform.lower()}_sample_data.csv")
        df[['text', 'label']].to_csv(output_file, index=False)
        print(f"Created sample data: {output_file} ({len(df)} samples)")
    
    print(f"\nSample data created in {output_dir}")
    print("You can now use these files for training the model.")


if __name__ == "__main__":
    create_sample_data(num_samples=1000)

