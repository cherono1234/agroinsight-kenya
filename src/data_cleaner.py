import pandas as pd, numpy as np, os
CLEAN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "clean")
class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        os.makedirs(CLEAN_DIR, exist_ok=True)
    def run_all(self):
        self.df = self.df.drop_duplicates()
        if "county" in self.df.columns:
            self.df["county"] = self.df["county"].str.strip().str.title()
        for col in self.df.select_dtypes(include=[np.number]).columns:
            self.df[col] = self.df[col].fillna(self.df[col].median())
        for col in self.df.select_dtypes(include=["object"]).columns:
            self.df[col] = self.df[col].fillna("Unknown")
        if "season" in self.df.columns:
            self.df["season_encoded"] = self.df["season"].map({"Long Rains":1,"Short Rains":0}).fillna(0).astype(int)
        if "crop" in self.df.columns:
            crops = sorted(self.df["crop"].unique())
            self.df["crop_encoded"] = self.df["crop"].map({c:i for i,c in enumerate(crops)})
        if "county" in self.df.columns:
            counties = sorted(self.df["county"].unique())
            self.df["county_encoded"] = self.df["county"].map({c:i for i,c in enumerate(counties)})
        out = os.path.join(CLEAN_DIR, "master_clean.csv")
        self.df.to_csv(out, index=False)
        return self.df