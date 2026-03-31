import pandas as pd, numpy as np, os
RAW_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "raw")
CLEAN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "clean")
class DataLoader:
    def __init__(self):
        self.raw_dir = RAW_DIR
        self.clean_dir = CLEAN_DIR
        os.makedirs(self.clean_dir, exist_ok=True)
        os.makedirs(self.raw_dir, exist_ok=True)
    def load_yield_data(self):
        fp = os.path.join(self.raw_dir, "yield_data.csv")
        if not os.path.exists(fp): return self._gen_yield()
        return pd.read_csv(fp)
    def load_weather_data(self):
        fp = os.path.join(self.raw_dir, "weather_data.csv")
        if not os.path.exists(fp): return self._gen_weather()
        return pd.read_csv(fp)
    def load_soil_data(self):
        fp = os.path.join(self.raw_dir, "soil_data.csv")
        if not os.path.exists(fp): return self._gen_soil()
        return pd.read_csv(fp)
    def load_all(self):
        y = self.load_yield_data()
        w = self.load_weather_data()
        s = self.load_soil_data()
        m = pd.merge(y, w, on=["county","year","season"], how="left")
        m = pd.merge(m, s, on="county", how="left")
        return m
    def _gen_yield(self):
        np.random.seed(42)
        counties=["Baringo","Bomet","Bungoma","Busia","Elgeyo Marakwet","Embu","Garissa","Homa Bay","Isiolo","Kajiado","Kakamega","Kericho","Kiambu","Kilifi","Kirinyaga","Kisii","Kisumu","Kitui","Kwale","Laikipia","Lamu","Machakos","Makueni","Mandera","Marsabit","Meru","Migori","Mombasa","Muranga","Nairobi","Nakuru","Nandi","Narok","Nyamira","Nyandarua","Nyeri","Samburu","Siaya","Taita Taveta","Tana River","Tharaka Nithi","Trans Nzoia","Turkana","Uasin Gishu","Vihiga","Wajir","West Pokot"]
        crops=["Maize","Beans","Wheat","Sorghum","Potatoes"]
        seasons=["Long Rains","Short Rains"]
        years=list(range(2015,2025))
        rows=[]
        for c in counties:
            for cr in crops:
                for y in years:
                    for s in seasons:
                        rows.append({"county":c,"crop":cr,"season":s,"year":y,"yield_kg_per_ha":round(np.random.uniform(800,4500),2),"area_planted_ha":round(np.random.uniform(100,5000),2)})
        df=pd.DataFrame(rows)
        df.to_csv(os.path.join(self.raw_dir,"yield_data.csv"),index=False)
        return df
    def _gen_weather(self):
        np.random.seed(24)
        counties=["Baringo","Bomet","Bungoma","Busia","Elgeyo Marakwet","Embu","Garissa","Homa Bay","Isiolo","Kajiado","Kakamega","Kericho","Kiambu","Kilifi","Kirinyaga","Kisii","Kisumu","Kitui","Kwale","Laikipia","Lamu","Machakos","Makueni","Mandera","Marsabit","Meru","Migori","Mombasa","Muranga","Nairobi","Nakuru","Nandi","Narok","Nyamira","Nyandarua","Nyeri","Samburu","Siaya","Taita Taveta","Tana River","Tharaka Nithi","Trans Nzoia","Turkana","Uasin Gishu","Vihiga","Wajir","West Pokot"]
        rows=[]
        for c in counties:
            for y in range(2015,2025):
                for s in ["Long Rains","Short Rains"]:
                    rows.append({"county":c,"year":y,"season":s,"avg_rainfall_mm":round(np.random.uniform(300,1200),2),"avg_temp_celsius":round(np.random.uniform(14,32),2),"rainfall_deviation":round(np.random.uniform(-150,150),2)})
        df=pd.DataFrame(rows)
        df.to_csv(os.path.join(self.raw_dir,"weather_data.csv"),index=False)
        return df
    def _gen_soil(self):
        np.random.seed(7)
        counties=["Baringo","Bomet","Bungoma","Busia","Elgeyo Marakwet","Embu","Garissa","Homa Bay","Isiolo","Kajiado","Kakamega","Kericho","Kiambu","Kilifi","Kirinyaga","Kisii","Kisumu","Kitui","Kwale","Laikipia","Lamu","Machakos","Makueni","Mandera","Marsabit","Meru","Migori","Mombasa","Muranga","Nairobi","Nakuru","Nandi","Narok","Nyamira","Nyandarua","Nyeri","Samburu","Siaya","Taita Taveta","Tana River","Tharaka Nithi","Trans Nzoia","Turkana","Uasin Gishu","Vihiga","Wajir","West Pokot"]
        rows=[{"county":c,"ph_level":round(np.random.uniform(4.5,7.5),2),"soil_type":np.random.choice(["Clay","Loam","Sandy Loam"]),"fertility_index":round(np.random.uniform(0.3,0.95),2)} for c in counties]
        df=pd.DataFrame(rows)
        df.to_csv(os.path.join(self.raw_dir,"soil_data.csv"),index=False)
        return df