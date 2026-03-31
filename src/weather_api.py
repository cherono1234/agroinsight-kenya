import requests

COUNTY_COORDINATES = {
    "Baringo": (-0.4667, 35.9667),"Bomet": (-0.7833, 35.3500),
    "Bungoma": (0.5667, 34.5667),"Busia": (0.4667, 34.1167),
    "Elgeyo Marakwet": (0.7833, 35.5167),"Embu": (-0.5333, 37.4500),
    "Garissa": (-0.4532, 39.6461),"Homa Bay": (-0.5167, 34.4500),
    "Isiolo": (0.3500, 37.5833),"Kajiado": (-1.8500, 36.7833),
    "Kakamega": (0.2833, 34.7500),"Kericho": (-0.3667, 35.2833),
    "Kiambu": (-1.0314, 36.8314),"Kilifi": (-3.6305, 39.8499),
    "Kirinyaga": (-0.5590, 37.2785),"Kisii": (-0.6817, 34.7667),
    "Kisumu": (-0.1022, 34.7617),"Kitui": (-1.3667, 38.0167),
    "Kwale": (-4.1833, 39.4500),"Laikipia": (0.2000, 36.7833),
    "Lamu": (-2.2686, 40.9020),"Machakos": (-1.5167, 37.2667),
    "Makueni": (-1.8035, 37.6202),"Mandera": (3.9366, 41.8670),
    "Marsabit": (2.3284, 37.9899),"Meru": (0.0467, 37.6496),
    "Migori": (-1.0634, 34.4731),"Mombasa": (-4.0435, 39.6682),
    "Muranga": (-0.7167, 37.1500),"Nairobi": (-1.2921, 36.8219),
    "Nakuru": (-0.3031, 36.0800),"Nandi": (0.1833, 35.1167),
    "Narok": (-1.0833, 35.8667),"Nyamira": (-0.5667, 34.9333),
    "Nyandarua": (-0.1833, 36.5167),"Nyeri": (-0.4167, 36.9500),
    "Samburu": (1.2167, 36.9667),"Siaya": (-0.0617, 34.2881),
    "Taita Taveta": (-3.4000, 38.3500),"Tana River": (-1.5000, 40.0000),
    "Tharaka Nithi": (-0.2833, 37.9167),"Trans Nzoia": (1.0167, 34.9500),
    "Turkana": (3.1167, 35.5960),"Uasin Gishu": (0.5167, 35.2833),
    "Vihiga": (0.0833, 34.7167),"Wajir": (1.7471, 40.0573),
    "West Pokot": (1.7500, 35.1167),
}

def get_weather(county):
    if county not in COUNTY_COORDINATES:
        return None
    lat, lon = COUNTY_COORDINATES[county]
    try:
        url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={lat}&longitude={lon}"
            f"&daily=precipitation_sum,temperature_2m_max,temperature_2m_min"
            f"&past_days=90&forecast_days=7"
            f"&timezone=Africa%2FNairobi"
        )
        r = requests.get(url, timeout=8)
        data = r.json().get("daily", {})
        rainfall = sum(data.get("precipitation_sum", [0]) or [0])
        temps_max = data.get("temperature_2m_max", [])
        temps_min = data.get("temperature_2m_min", [])
        avg_temp = round((sum(temps_max) + sum(temps_min)) / (len(temps_max) + len(temps_min) + 0.001), 1)
        return {
            "avg_rainfall_mm"   : round(min(rainfall, 1500), 1),
            "avg_temp_celsius"  : min(max(avg_temp, 10), 35),
            "county"            : county,
            "lat"               : lat,
            "lon"               : lon,
        }
    except Exception as e:
        print(f"Weather API error: {e}")
        return None

def get_counties():
    return sorted(COUNTY_COORDINATES.keys())