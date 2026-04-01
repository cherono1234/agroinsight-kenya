import requests

COUNTY_COORDINATES = {
    "Baringo": (-0.4667, 35.9667), "Bomet": (-0.7833, 35.3500),
    "Bungoma": (0.5667, 34.5667), "Busia": (0.4667, 34.1167),
    "Elgeyo Marakwet": (0.7833, 35.5167), "Embu": (-0.5333, 37.4500),
    "Garissa": (-0.4532, 39.6461), "Homa Bay": (-0.5167, 34.4500),
    "Isiolo": (0.3500, 37.5833), "Kajiado": (-1.8500, 36.7833),
    "Kakamega": (0.2833, 34.7500), "Kericho": (-0.3667, 35.2833),
    "Kiambu": (-1.0314, 36.8314), "Kilifi": (-3.6305, 39.8499),
    "Kirinyaga": (-0.5590, 37.2785), "Kisii": (-0.6817, 34.7667),
    "Kisumu": (-0.1022, 34.7617), "Kitui": (-1.3667, 38.0167),
    "Kwale": (-4.1833, 39.4500), "Laikipia": (0.2000, 36.7833),
    "Lamu": (-2.2686, 40.9020), "Machakos": (-1.5167, 37.2667),
    "Makueni": (-1.8035, 37.6202), "Mandera": (3.9366, 41.8670),
    "Marsabit": (2.3284, 37.9899), "Meru": (0.0467, 37.6496),
    "Migori": (-1.0634, 34.4731), "Mombasa": (-4.0435, 39.6682),
    "Muranga": (-0.7167, 37.1500), "Nairobi": (-1.2921, 36.8219),
    "Nakuru": (-0.3031, 36.0800), "Nandi": (0.1833, 35.1167),
    "Narok": (-1.0833, 35.8667), "Nyamira": (-0.5667, 34.9333),
    "Nyandarua": (-0.1833, 36.5167), "Nyeri": (-0.4167, 36.9500),
    "Samburu": (1.2167, 36.9667), "Siaya": (-0.0617, 34.2881),
    "Taita Taveta": (-3.4000, 38.3500), "Tana River": (-1.5000, 40.0000),
    "Tharaka Nithi": (-0.2833, 37.9167), "Trans Nzoia": (1.0167, 34.9500),
    "Turkana": (3.1167, 35.5960), "Uasin Gishu": (0.5167, 35.2833),
    "Vihiga": (0.0833, 34.7167), "Wajir": (1.7471, 40.0573),
    "West Pokot": (1.7500, 35.1167),
}

def get_weather(county):
    coords = COUNTY_COORDINATES.get(county)
    if not coords:
        return None
    lat, lon = coords
    try:
        url = (
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            f"&daily=precipitation_sum,temperature_2m_max,temperature_2m_min"
            f"&past_days=30&forecast_days=1"
            f"&timezone=Africa%2FNairobi"
        )
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        d = resp.json().get("daily", {})
        rain_list = [x for x in d.get("precipitation_sum", []) if x is not None]
        tmax_list = [x for x in d.get("temperature_2m_max", []) if x is not None]
        tmin_list = [x for x in d.get("temperature_2m_min", []) if x is not None]
        rainfall   = round(min(sum(rain_list), 1500), 1) if rain_list else 700
        avg_temp   = round((sum(tmax_list)/len(tmax_list) + sum(tmin_list)/len(tmin_list)) / 2, 1) if tmax_list else 22
        avg_temp   = min(max(avg_temp, 10), 35)
        return {
            "avg_rainfall_mm" : rainfall,
            "avg_temp_celsius": avg_temp,
            "county"          : county,
        }
    except Exception as e:
        print(f"Weather fetch error: {e}")
        return None

def get_counties():
    return sorted(COUNTY_COORDINATES.keys())