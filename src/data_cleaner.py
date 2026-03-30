"""
AgroInsight Kenya - Data Cleaner
----------------------------------
Cleans and prepares the master dataset for
machine learning model training.
"""

import pandas as pd
import numpy as np
import os

CLEAN_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'clean')


class DataCleaner:
    """
    Cleans the master DataFrame produced by DataLoader.

    Usage:
        cleaner  = DataCleaner(master_df)
        clean_df = cleaner.run_all()
    """

    def __init__(self, df):
        self.df = df.copy()
        os.makedirs(CLEAN_DIR, exist_ok=True)
        print("✅ DataCleaner initialized")
        print(f"   Input shape: {self.df.shape}")


    # ── 1. Remove Duplicates ───────────────────────────────────────────
    def remove_duplicates(self):
        before = len(self.df)
        self.df = self.df.drop_duplicates()
        removed = before - len(self.df)
        print(f"✅ Duplicates removed    : {removed} rows dropped")
        return self


    # ── 2. Standardise County Names ────────────────────────────────────
    def standardise_county_names(self):
        """
        Fixes inconsistent county name formatting.
        e.g. 'nakuru', 'NAKURU', ' Nakuru ' all become 'Nakuru'
        """
        self.df['county'] = (
            self.df['county']
            .str.strip()
            .str.title()
        )
        print(f"✅ County names standardised")
        return self


    # ── 3. Remove Outliers ─────────────────────────────────────────────
    def remove_outliers(self, column='yield_kg_per_ha', factor=3.0):
        """
        Removes rows where yield is unrealistically high or low
        using the IQR (Interquartile Range) method.
        """
        before = len(self.df)
        Q1  = self.df[column].quantile(0.25)
        Q3  = self.df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - factor * IQR
        upper = Q3 + factor * IQR
        self.df = self.df[
            (self.df[column] >= lower) &
            (self.df[column] <= upper)
        ]
        removed = before - len(self.df)
        print(f"✅ Outliers removed      : {removed} rows (yield outside {lower:.1f} – {upper:.1f} kg/ha)")
        return self


    # ── 4. Impute Missing Values ───────────────────────────────────────
    def impute_missing(self):
        """
        Fills missing values:
        - Numeric columns  → median of that column
        - Text columns     → 'Unknown'
        """
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        text_cols    = self.df.select_dtypes(include=['object']).columns

        for col in numeric_cols:
            missing = self.df[col].isnull().sum()
            if missing > 0:
                median_val = self.df[col].median()
                self.df[col] = self.df[col].fillna(median_val)
                print(f"   Imputed {missing} missing values in '{col}' with median ({median_val:.2f})")

        for col in text_cols:
            missing = self.df[col].isnull().sum()
            if missing > 0:
                self.df[col] = self.df[col].fillna('Unknown')
                print(f"   Imputed {missing} missing values in '{col}' with 'Unknown'")

        total_missing = self.df.isnull().sum().sum()
        print(f"✅ Missing values fixed  : {total_missing} remaining null values")
        return self


    # ── 5. Fix Data Types ──────────────────────────────────────────────
    def fix_dtypes(self):
        """
        Ensures columns have the correct data types.
        """
        if 'year' in self.df.columns:
            self.df['year'] = self.df['year'].astype(int)

        if 'yield_kg_per_ha' in self.df.columns:
            self.df['yield_kg_per_ha'] = self.df['yield_kg_per_ha'].astype(float)

        if 'area_planted_ha' in self.df.columns:
            self.df['area_planted_ha'] = self.df['area_planted_ha'].astype(float)

        if 'avg_rainfall_mm' in self.df.columns:
            self.df['avg_rainfall_mm'] = self.df['avg_rainfall_mm'].astype(float)

        if 'avg_temp_celsius' in self.df.columns:
            self.df['avg_temp_celsius'] = self.df['avg_temp_celsius'].astype(float)

        print(f"✅ Data types fixed")
        return self


    # ── 6. Engineer Features ───────────────────────────────────────────
    def engineer_features(self):
        """
        Creates new useful columns from existing data:
        - rainfall_category : Low / Medium / High
        - temp_category     : Cool / Warm / Hot
        - season_encoded    : Long Rains=1, Short Rains=0
        - crop_encoded      : numeric label for each crop
        - county_encoded    : numeric label for each county
        """
        # Rainfall category
        if 'avg_rainfall_mm' in self.df.columns:
            self.df['rainfall_category'] = pd.cut(
                self.df['avg_rainfall_mm'],
                bins=[0, 400, 800, 9999],
                labels=['Low', 'Medium', 'High']
            )

        # Temperature category
        if 'avg_temp_celsius' in self.df.columns:
            self.df['temp_category'] = pd.cut(
                self.df['avg_temp_celsius'],
                bins=[0, 18, 25, 99],
                labels=['Cool', 'Warm', 'Hot']
            )

        # Season encoding
        if 'season' in self.df.columns:
            self.df['season_encoded'] = self.df['season'].map({
                'Long Rains' : 1,
                'Short Rains': 0
            }).fillna(0).astype(int)

        # Crop encoding
        if 'crop' in self.df.columns:
            crops = sorted(self.df['crop'].unique())
            crop_map = {crop: idx for idx, crop in enumerate(crops)}
            self.df['crop_encoded'] = self.df['crop'].map(crop_map)
            print(f"   Crop encoding: {crop_map}")

        # County encoding
        if 'county' in self.df.columns:
            counties = sorted(self.df['county'].unique())
            county_map = {county: idx for idx, county in enumerate(counties)}
            self.df['county_encoded'] = self.df['county'].map(county_map)

        print(f"✅ Features engineered   : new columns added")
        return self


    # ── 7. Run All Steps ───────────────────────────────────────────────
    def run_all(self):
        """
        Runs every cleaning step in the correct order.
        Returns the fully cleaned DataFrame.
        """
        print("\n🧹 Starting data cleaning pipeline...\n")
        (self
            .remove_duplicates()
            .standardise_county_names()
            .fix_dtypes()
            .remove_outliers()
            .impute_missing()
            .engineer_features()
        )

        print(f"\n✅ Cleaning complete!")
        print(f"   Final shape : {self.df.shape}")
        print(f"   Columns     : {list(self.df.columns)}")

        # Save cleaned data
        out_path = os.path.join(CLEAN_DIR, 'master_clean.csv')
        self.df.to_csv(out_path, index=False)
        print(f"   Saved to    : {out_path}\n")

        return self.df


    # ── 8. Summary Report ─────────────────────────────────────────────
    def summary(self):
        """Prints a quick summary of the cleaned dataset."""
        print("\n── Dataset Summary ──────────────────────────────")
        print(f"  Shape         : {self.df.shape}")
        print(f"  Counties      : {sorted(self.df['county'].unique())}")
        print(f"  Crops         : {sorted(self.df['crop'].unique())}")
        print(f"  Years         : {self.df['year'].min()} – {self.df['year'].max()}")
        print(f"  Avg Yield     : {self.df['yield_kg_per_ha'].mean():.2f} kg/ha")
        print(f"  Min Yield     : {self.df['yield_kg_per_ha'].min():.2f} kg/ha")
        print(f"  Max Yield     : {self.df['yield_kg_per_ha'].max():.2f} kg/ha")
        print(f"  Missing values: {self.df.isnull().sum().sum()}")
        print("─────────────────────────────────────────────────\n")


# ── Quick test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.append(os.path.dirname(__file__))
    from data_loader import DataLoader

    # Load data
    loader    = DataLoader()
    master_df = loader.load_all()

    # Clean data
    cleaner  = DataCleaner(master_df)
    clean_df = cleaner.run_all()
    cleaner.summary()

    print("── First 5 rows of clean data ──")
    print(clean_df.head())