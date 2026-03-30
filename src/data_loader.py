"""
AgroInsight Kenya - Data Loader
--------------------------------
Loads agricultural, weather, and soil datasets
from the data/raw/ folder into clean DataFrames.
"""

import pandas as pd
import numpy as np
import os

# ── Paths ──────────────────────────────────────────────────────────────
RAW_DIR   = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
CLEAN_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'clean')


class DataLoader:
    """
    Loads all AgroInsight Kenya datasets.

    Usage:
        loader = DataLoader()
        yield_df   = loader.load_yield_data()
        weather_df = loader.load_weather_data()
        soil_df    = loader.load_soil_data()
    """

    def __init__(self):
        self.raw_dir   = RAW_DIR
        self.clean_dir = CLEAN_DIR
        os.makedirs(self.clean_dir, exist_ok=True)
        print("✅ DataLoader initialized")
        print(f"   Raw data folder  : {os.path.abspath(self.raw_dir)}")
        print(f"   Clean data folder: {os.path.abspath(self.clean_dir)}")

    # ── 1. Yield Data ──────────────────────────────────────────────────
    def load_yield_data(self, filename="yield_data.csv"):
        filepath = os.path.join(self.raw_dir, filename)
        if not os.path.exists(filepath):
            print(f"⚠️  File not found: {filepath}")
            print("   Generating sample yield data for testing...")
            return self._generate_sample_yield_data()
        df = pd.read_csv(filepath)
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
        print(f"✅ Yield data loaded     : {df.shape[0]} rows, {df.shape[1]} columns")
        return df

    # ── 2. Weather Data ────────────────────────────────────────────────
    def load_weather_data(self, filename="weather_data.csv"):
        filepath = os.path.join(self.raw_dir, filename)
        if not os.path.exists(filepath):
            print(f"⚠️  File not found: {filepath}")
            print("   Generating sample weather data for testing...")
            return self._generate_sample_weather_data()
        df = pd.read_csv(filepath)
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
        print(f"✅ Weather data loaded   : {df.shape[0]} rows, {df.shape[1]} columns")
        return df

    # ── 3. Soil Data ───────────────────────────────────────────────────
    def load_soil_data(self, filename="soil_data.csv"):
        filepath = os.path.join(self.raw_dir, filename)
        if not os.path.exists(filepath):
            print(f"⚠️  File not found: {filepath}")
            print("   Generating sample soil data for testing...")
            return self._generate_sample_soil_data()
        df = pd.read_csv(filepath)
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
        print(f"✅ Soil data loaded      : {df.shape[0]} rows, {df.shape[1]} columns")
        return df

    # ── 4. Load & Merge All ────────────────────────────────────────────
    def load_all(self):
        print("\n📦 Loading all datasets...")
        yield_df   = self.load_yield_data()
        weather_df = self.load_weather_data()
        soil_df    = self.load_soil_data()

        merged = pd.merge(yield_df, weather_df, on=['county', 'year', 'season'], how='left')
        merged = pd.merge(merged, soil_df, on='county', how='left')

        print(f"\n✅ Master dataset ready  : {merged.shape[0]} rows, {merged.shape[1]} columns")
        print(f"   Columns: {list(merged.columns)}\n")
        return merged

    # ── 5. Validate Schema ─────────────────────────────────────────────
    def validate_schema(self, df, required_columns):
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            print(f"❌ Missing columns: {missing}")
            return False
        print(f"✅ Schema valid — all required columns present")
        return True

    # ── Sample Data Generators ─────────────────────────────────────────
    def _generate_sample_yield_data(self):
        np.random.seed(42)
        counties = ['Nakuru','Kisumu','Machakos','Meru','Kakamega','Uasin Gishu','Trans Nzoia','Nyandarua','Kirinyaga',"Murang'a"]
        crops    = ['Maize','Beans','Wheat','Sorghum','Potatoes']
        seasons  = ['Long Rains','Short Rains']
        years    = list(range(2015, 2025))
        records  = []
        for county in counties:
            for crop in crops:
                for year in years:
                    for season in seasons:
                        records.append({'county': county,'crop': crop,'season': season,'year': year,
                            'yield_kg_per_ha': round(np.random.uniform(800, 4500), 2),
                            'area_planted_ha': round(np.random.uniform(100, 5000), 2)})
        df = pd.DataFrame(records)
        os.makedirs(self.raw_dir, exist_ok=True)
        df.to_csv(os.path.join(self.raw_dir, 'yield_data.csv'), index=False)
        print("   Sample yield data saved to data/raw/yield_data.csv")
        return df

    def _generate_sample_weather_data(self):
        np.random.seed(24)
        counties = ['Nakuru','Kisumu','Machakos','Meru','Kakamega','Uasin Gishu','Trans Nzoia','Nyandarua','Kirinyaga',"Murang'a"]
        seasons  = ['Long Rains','Short Rains']
        years    = list(range(2015, 2025))
        records  = []
        for county in counties:
            for year in years:
                for season in seasons:
                    records.append({'county': county,'year': year,'season': season,
                        'avg_rainfall_mm'   : round(np.random.uniform(300, 1200), 2),
                        'avg_temp_celsius'  : round(np.random.uniform(14, 32), 2),
                        'rainfall_deviation': round(np.random.uniform(-150, 150), 2)})
        df = pd.DataFrame(records)
        os.makedirs(self.raw_dir, exist_ok=True)
        df.to_csv(os.path.join(self.raw_dir, 'weather_data.csv'), index=False)
        print("   Sample weather data saved to data/raw/weather_data.csv")
        return df

    def _generate_sample_soil_data(self):
        np.random.seed(7)
        counties   = ['Nakuru','Kisumu','Machakos','Meru','Kakamega','Uasin Gishu','Trans Nzoia','Nyandarua','Kirinyaga',"Murang'a"]
        soil_types = ['Clay','Loam','Sandy Loam','Clay Loam','Silty Clay']
        records    = []
        for county in counties:
            records.append({'county': county,
                'ph_level'       : round(np.random.uniform(4.5, 7.5), 2),
                'soil_type'      : np.random.choice(soil_types),
                'fertility_index': round(np.random.uniform(0.3, 0.95), 2)})
        df = pd.DataFrame(records)
        os.makedirs(self.raw_dir, exist_ok=True)
        df.to_csv(os.path.join(self.raw_dir, 'soil_data.csv'), index=False)
        print("   Sample soil data saved to data/raw/soil_data.csv")
        return df


# ── Quick test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    loader    = DataLoader()
    master_df = loader.load_all()

    print("── First 5 rows ──")
    print(master_df.head())
    print(f"\nShape     : {master_df.shape}")
    print(f"Counties  : {master_df['county'].nunique()}")
    print(f"Crops     : {master_df['crop'].nunique()}")
    print(f"Year range: {master_df['year'].min()} – {master_df['year'].max()}")
    print(f"Missing   : {master_df.isnull().sum().sum()} total null values")