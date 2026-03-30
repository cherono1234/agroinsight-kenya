import pandas as pd
import numpy as np
import os
import pickle

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
KENYAN_COUNTIES = sorted(["Kakamega","Kirinyaga","Kisumu","Machakos","Meru","Murang\x27a","Nakuru","Nyandarua","Trans Nzoia","Uasin Gishu"])
CROPS   = sorted(["Maize","Beans","Wheat","Sorghum","Potatoes"])
SEASONS = ["Long Rains","Short Rains"]
CROP_MAP   = {crop: idx for idx, crop in enumerate(CROPS)}
COUNTY_MAP = {county: idx for idx, county in enumerate(KENYAN_COUNTIES)}
SEASON_MAP = {"Long Rains": 1, "Short Rains": 0}

class PredictionEngine:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.model_name = None
        self.features = None
        self.model_r2 = None
        self.model_rmse = None
        self._load_model()

    def _load_model(self):
        model_path = os.path.join(MODELS_DIR, "best_model.pkl")
        if not os.path.exists(model_path):
            print("No saved model found. Run model_trainer.py first.")
            return
        with open(model_path, "rb") as f:
            saved = pickle.load(f)
        self.model      = saved["model"]
        self.scaler     = saved["scaler"]
        self.model_name = saved["model_name"]
        self.features   = saved["features"]
        self.model_r2   = saved["r2"]
        self.model_rmse = saved["rmse"]
        print(f"Model loaded: {self.model_name} R2={self.model_r2}")

    def predict(self, county, crop, season, year,
                rainfall=700.0, temperature=22.0,
                rainfall_deviation=0.0, ph_level=6.0,
                fertility=0.7, area=1000.0):
        if self.model is None:
            return {"error": "No model loaded. Run model_trainer.py first."}
        feat = {
            "crop_encoded"      : CROP_MAP.get(crop, 0),
            "county_encoded"    : COUNTY_MAP.get(county, 0),
            "season_encoded"    : SEASON_MAP.get(season, 0),
            "year"              : int(year),
            "avg_rainfall_mm"   : float(rainfall),
            "avg_temp_celsius"  : float(temperature),
            "rainfall_deviation": float(rainfall_deviation),
            "ph_level"          : float(ph_level),
            "fertility_index"   : float(fertility),
            "area_planted_ha"   : float(area),
        }
        input_df  = pd.DataFrame([feat])
        available = [f for f in self.features if f in input_df.columns]
        X_scaled  = self.scaler.transform(input_df[available])
        pred      = round(float(self.model.predict(X_scaled)[0]), 2)
        pred      = max(0, pred)
        if pred >= 3500: rating = "Excellent"
        elif pred >= 2500: rating = "Good"
        elif pred >= 1500: rating = "Fair"
        else: rating = "Poor"
        tips = []
        if rating == "Excellent": tips.append("Conditions highly favourable.")
        elif rating == "Good": tips.append("Good yield expected.")
        elif rating == "Fair": tips.append("Moderate yield. Consider improving inputs.")
        else: tips.append("Low yield risk. Review farming inputs.")
        if rainfall < 400: tips.append("Low rainfall - consider irrigation.")
        if ph_level < 5.5: tips.append("Acidic soil - apply lime.")
        if fertility < 0.5: tips.append("Low fertility - add fertiliser.")
        return {
            "county": county, "crop": crop, "season": season, "year": year,
            "predicted_yield": pred, "rating": rating,
            "confidence": "High", "recommendation": " | ".join(tips),
            "model_used": self.model_name, "model_r2": self.model_r2,
        }

    def batch_predict(self, inputs_list):
        return pd.DataFrame([self.predict(**i) for i in inputs_list])

    @staticmethod
    def get_options():
        return {
            "counties": KENYAN_COUNTIES,
            "crops"   : CROPS,
            "seasons" : SEASONS,
            "years"   : list(range(2020, 2031)),
        }
