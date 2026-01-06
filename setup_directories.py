"""
Utility script to create necessary directories
"""
import os
from config import Config

def setup_directories():
    """Create all necessary directories"""
    config = Config()
    
    directories = [
        config.DATA_DIR,
        config.RAW_DATA_DIR,
        config.PROCESSED_DATA_DIR,
        config.MODEL_DIR,
        config.CHECKPOINT_DIR,
        config.RESULTS_DIR,
        config.EXPLANATIONS_DIR,
        config.LOG_DIR
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        # Create .gitkeep file
        gitkeep = os.path.join(directory, '.gitkeep')
        if not os.path.exists(gitkeep):
            with open(gitkeep, 'w') as f:
                pass
    
    print("All directories created successfully!")

if __name__ == "__main__":
    setup_directories()

