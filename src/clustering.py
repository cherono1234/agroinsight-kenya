import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pickle, os

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")

class CountyClusterer:
    """
    Groups Kenyan counties into yield-potential clusters using K-Means
    based on rainfall, temperature, soil pH, and fertility index.
    """
    FEATURES = ["avg_rainfall_mm", "avg_temp_celsius", "ph_level", "fertility_index"]

    def __init__(self, n_clusters=3):
        self.n_clusters = n_clusters
        self.scaler = StandardScaler()
        self.model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.cluster_labels = {}
        self.fitted_df = None

    def prepare_county_profile(self, df):
        """Aggregate to one row per county (mean of all features)."""
        profile = df.groupby("county")[self.FEATURES + ["yield_kg_per_ha"]].mean().reset_index()
        return profile.dropna()

    def fit(self, df):
        profile = self.prepare_county_profile(df)
        X = self.scaler.fit_transform(profile[self.FEATURES])
        profile["cluster"] = self.model.fit_predict(X)

        # Rank clusters by average yield to assign meaningful labels
        cluster_yield = profile.groupby("cluster")["yield_kg_per_ha"].mean().sort_values(ascending=False)
        rank_labels = ["High Yield Potential", "Moderate Yield Potential", "Low Yield Potential / Arid"]
        self.cluster_labels = {cl: rank_labels[i] if i < len(rank_labels) else f"Cluster {cl}"
                                for i, cl in enumerate(cluster_yield.index)}
        profile["cluster_label"] = profile["cluster"].map(self.cluster_labels)
        self.fitted_df = profile
        return profile

    def save(self):
        os.makedirs(MODELS_DIR, exist_ok=True)
        with open(os.path.join(MODELS_DIR, "cluster_model.pkl"), "wb") as f:
            pickle.dump({
                "model": self.model,
                "scaler": self.scaler,
                "cluster_labels": self.cluster_labels,
                "fitted_df": self.fitted_df,
                "features": self.FEATURES,
            }, f)
        print("Cluster model saved.")

    @staticmethod
    def load():
        path = os.path.join(MODELS_DIR, "cluster_model.pkl")
        if not os.path.exists(path):
            return None
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = CountyClusterer()
        obj.model = data["model"]
        obj.scaler = data["scaler"]
        obj.cluster_labels = data["cluster_labels"]
        obj.fitted_df = data["fitted_df"]
        return obj

    def predict_county(self, county_name):
        if self.fitted_df is None:
            return None
        row = self.fitted_df[self.fitted_df["county"] == county_name]
        if row.empty:
            return None
        return {
            "county": county_name,
            "cluster_label": row["cluster_label"].values[0],
            "avg_yield": round(row["yield_kg_per_ha"].values[0], 1),
            "avg_rainfall": round(row["avg_rainfall_mm"].values[0], 1),
            "avg_fertility": round(row["fertility_index"].values[0], 2),
        }