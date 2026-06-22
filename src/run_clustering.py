"""
Standalone script: K-Means Clustering of Kenyan Counties
Run this to generate cluster analysis results for your project report.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from clustering import CountyClusterer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CLEAN_PATH = os.path.join(BASE_DIR, "..", "data", "clean", "master_clean.csv")

def main():
    print("="*60)
    print("K-MEANS CLUSTERING — AgroInsight Kenya")
    print("="*60)

    df = pd.read_csv(CLEAN_PATH)
    print(f"Loaded {len(df)} records covering {df['county'].nunique()} counties\n")

    clusterer = CountyClusterer(n_clusters=3)
    profile = clusterer.fit(df)
    clusterer.save()

    print("CLUSTER SUMMARY (sorted by avg yield):\n")
    summary = profile.groupby("cluster_label").agg(
        counties=("county", "count"),
        avg_yield=("yield_kg_per_ha", "mean"),
        avg_rainfall=("avg_rainfall_mm", "mean"),
        avg_ph=("ph_level", "mean"),
        avg_fertility=("fertility_index", "mean"),
    ).round(2).sort_values("avg_yield", ascending=False)
    print(summary.to_string())

    print("\n\nCOUNTY -> CLUSTER ASSIGNMENT:\n")
    for label in profile["cluster_label"].unique():
        counties = profile[profile["cluster_label"] == label]["county"].tolist()
        print(f"\n{label} ({len(counties)} counties):")
        print(", ".join(sorted(counties)))

    out_path = os.path.join(BASE_DIR, "..", "data", "clean", "county_clusters.csv")
    profile.to_csv(out_path, index=False)
    print(f"\n\nFull cluster results saved to: {out_path}")

if __name__ == "__main__":
    main()