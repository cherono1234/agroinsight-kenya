import pandas as pd, numpy as np, os, pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
class ModelTrainer:
    FEATURES = ["crop_encoded","county_encoded","season_encoded","year","avg_rainfall_mm","avg_temp_celsius","rainfall_deviation","ph_level","fertility_index","area_planted_ha"]
    TARGET = "yield_kg_per_ha"
    def __init__(self, df):
        self.df = df.copy()
        self.results = {}
        self.best_model = None
        self.best_name = None
        self.feature_cols = []
        self.X_train = self.X_test = self.y_train = self.y_test = None
        os.makedirs(MODELS_DIR, exist_ok=True)
    def prepare_data(self, test_size=0.2, random_state=42):
        self.feature_cols = [f for f in self.FEATURES if f in self.df.columns]
        data = self.df[self.feature_cols + [self.TARGET]].dropna()
        X = data[self.feature_cols]
        y = data[self.TARGET]
        self.X_train,self.X_test,self.y_train,self.y_test = train_test_split(X,y,test_size=test_size,random_state=random_state)
        return self
    def _train_model(self, name, model):
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(self.X_train)
        Xte = scaler.transform(self.X_test)
        model.fit(Xtr, self.y_train)
        pred = model.predict(Xte)
        self.results[name] = {"model":model,"scaler":scaler,"r2":round(r2_score(self.y_test,pred),4),"rmse":round(np.sqrt(mean_squared_error(self.y_test,pred)),2),"mae":round(mean_absolute_error(self.y_test,pred),2)}
    def train_all(self):
        self._train_model("Linear Regression", LinearRegression())
        self._train_model("Random Forest", RandomForestRegressor(n_estimators=100,max_depth=10,random_state=42,n_jobs=-1))
        self._train_model("XGBoost", XGBRegressor(n_estimators=200,max_depth=6,learning_rate=0.1,random_state=42,verbosity=0))
        self.best_name = max(self.results, key=lambda k: self.results[k]["r2"])
        self.best_model = self.results[self.best_name]
        return self
    def print_results(self):
        for n,r in self.results.items():
            print(f"{n}: R2={r['r2']} RMSE={r['rmse']}")
    def save_best_model(self):
        out = os.path.join(MODELS_DIR, "best_model.pkl")
        with open(out,"wb") as f:
            pickle.dump({"model":self.best_model["model"],"scaler":self.best_model["scaler"],"model_name":self.best_name,"features":self.feature_cols,"r2":self.best_model["r2"],"rmse":self.best_model["rmse"],"mae":self.best_model["mae"]}, f)
        print(f"Model saved: {self.best_name}")