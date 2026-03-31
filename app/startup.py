import os
import sys

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR    = os.path.join(BASE_DIR, 'src')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'best_model.pkl')

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

def ensure_model_exists():
    os.makedirs(os.path.join(BASE_DIR, 'models'), exist_ok=True)
    if not os.path.exists(MODEL_PATH):
        print("No model found — training now...")
        from data_loader   import DataLoader
        from data_cleaner  import DataCleaner
        from model_trainer import ModelTrainer
        loader    = DataLoader()
        master_df = loader.load_all()
        cleaner   = DataCleaner(master_df)
        clean_df  = cleaner.run_all()
        trainer   = ModelTrainer(clean_df)
        trainer.prepare_data()
        trainer.train_all()
        trainer.save_best_model()
        print("Model trained and saved!")
    else:
        print("Model already exists.")
