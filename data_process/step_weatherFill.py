import requests
import pandas as pd

# ---- 1. read PEMS 5min data ----
df_pems = pd.read_csv(
    "data/pems_5min.csv",
    parse_dates=["timestamp"]
).set_index("timestamp").asfreq("5T")

# ---- 2. note ----
station = "KLAX"  # Los Angeles Intl Airport
start   = "2025-01-01T00:00:00Z"
end     = "2025-04-26T23:55:00Z"

url = f"https://api.weather.gov/stations/{station}/observations"
params = {"start": start, "end": end, "limit": 1000, "sort": "asc"}
headers = {"User-Agent": "gonglz39@yahoo.com"}

records = []
while url:
    r = requests.get(url, params=params, headers=headers)
    r.raise_for_status()
    js = r.json()
    for feat in js["features"]:
        p = feat["properties"]
        records.append({
            "timestamp":   p["timestamp"],
            "temp_C":      p["temperature"]["value"],
            "precip_mm":   p["precipitationLastHour"]["value"]
        })
    url = js.get("pagination", {}).get("next")
    params = None  # note params

# note DataFrame, note 5 min
df_weather = pd.DataFrame(records)
df_weather["timestamp"] = pd.to_datetime(df_weather["timestamp"])
df_weather.set_index("timestamp", inplace=True)
df_weather = df_weather.resample("5T").ffill()

# ---- 3. note PEMS + note ----
df_all = df_pems.join(df_weather, how="left")
# note ffill note NaN
df_all[["temp_C","precip_mm"]] = df_all[["temp_C","precip_mm"]].ffill()

# ---- 4. note ----
df_all.to_parquet("output/pems_with_weather.parquet")
print("Merged shape:", df_all.shape)
print(df_all[['flow','temp_C','precip_mm']].head(10))
