import pandas as pd, numpy as np, os, pickle
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
COUNTIES = sorted(["Kakamega","Kirinyaga","Kisumu","Machakos","Meru","Muranga","Nakuru","Nyandarua","Trans Nzoia","Uasin Gishu"])
CROPS    = sorted(["Maize","Beans","Wheat","Sorghum","Potatoes"])
SEASONS  = ["Long Rains","Short Rains"]
CROP_MAP   = {c:i for i,c in enumerate(CROPS)}
COUNTY_MAP = {c:i for i,c in enumerate(COUNTIES)}
SEASON_MAP = {"Long Rains":1,"Short Rains":0}
class PredictionEngine:
    def __init__(self):
        self.model=self.scaler=self.model_name=self.features=self.model_r2=self.model_rmse=None
        self._load()
    def _load(self):
        p = os.path.join(MODELS_DIR, "best_model.pkl")
        if not os.path.exists(p):
            print("No model found."); return
        with open(p,"rb") as f: s=pickle.load(f)
        self.model=s["model"]; self.scaler=s["scaler"]; self.model_name=s["model_name"]
        self.features=s["features"]; self.model_r2=s["r2"]; self.model_rmse=s["rmse"]
    def predict(self, county, crop, season, year, rainfall=700.0, temperature=22.0, rainfall_deviation=0.0, ph_level=6.0, fertility=0.7, area=1000.0):
        if self.model is None: return {"error":"No model loaded."}
        feat={"crop_encoded":CROP_MAP.get(crop,0),"county_encoded":COUNTY_MAP.get(county,0),"season_encoded":SEASON_MAP.get(season,0),"year":int(year),"avg_rainfall_mm":float(rainfall),"avg_temp_celsius":float(temperature),"rainfall_deviation":float(rainfall_deviation),"ph_level":float(ph_level),"fertility_index":float(fertility),"area_planted_ha":float(area)}
        df=pd.DataFrame([feat])
        avail=[f for f in self.features if f in df.columns]
        X=self.scaler.transform(df[avail])
        pred=round(float(self.model.predict(X)[0]),2)
        pred=max(0,pred)
        rating="Excellent" if pred>=3500 else "Good" if pred>=2500 else "Fair" if pred>=1500 else "Poor"
        tips=[]
        if rating=="Excellent": tips.append("Conditions highly favourable.")
        elif rating=="Good": tips.append("Good yield expected.")
        elif rating=="Fair": tips.append("Moderate yield. Consider improving inputs.")
        else: tips.append("Low yield risk. Review farming inputs.")
        if rainfall<400: tips.append("Low rainfall - consider irrigation.")
        if ph_level<5.5: tips.append("Acidic soil - apply lime.")
        if fertility<0.5: tips.append("Low fertility - add fertiliser.")
        return {"county":county,"crop":crop,"season":season,"year":year,"predicted_yield":pred,"rating":rating,"confidence":"High","recommendation":" | ".join(tips),"model_used":self.model_name,"model_r2":self.model_r2}
    def batch_predict(self,inputs):
        return pd.DataFrame([self.predict(**i) for i in inputs])
    @staticmethod
    def get_options():
        return {"counties":COUNTIES,"crops":CROPS,"seasons":SEASONS,"years":list(range(2020,2031))}