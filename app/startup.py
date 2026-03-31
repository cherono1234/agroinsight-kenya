import os, sys

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR    = os.path.join(BASE_DIR, 'src')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'best_model.pkl')

sys.path.insert(0, SRC_DIR)

def ensure_model_exists():
    os.makedirs(os.path.join(BASE_DIR, 'models'), exist_ok=True)
    if not os.path.exists(MODEL_PATH):
        print(f"SRC_DIR is: {SRC_DIR}")
        print(f"sys.path is: {sys.path}")
        print(f"src contents: {os.listdir(SRC_DIR)}")
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "model_trainer",
            os.path.join(SRC_DIR, "model_trainer.py")
        )
        mt = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mt)
        ModelTrainer = mt.ModelTrainer

        spec2 = importlib.util.spec_from_file_location(
            "data_loader",
            os.path.join(SRC_DIR, "data_loader.py")
        )
        dl = importlib.util.module_from_spec(spec2)
        spec2.loader.exec_module(dl)
        DataLoader = dl.DataLoader

        spec3 = importlib.util.spec_from_file_location(
            "data_cleaner",
            os.path.join(SRC_DIR, "data_cleaner.py")
        )
        dc = importlib.util.module_from_spec(spec3)
        spec3.loader.exec_module(dc)
        DataCleaner = dc.DataCleaner

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
