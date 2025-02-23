import pandas as pd
import geopandas as gpd
import folium
from folium.plugins import HeatMap
from shapely.geometry import Point
import os

print(f"Current Working Directory: {os.getcwd()}")

# Load data
try:
    parking_df = pd.read_csv("parking.csv")
    print(f"✅ Successfully loaded parking.csv with {len(parking_df)} rows")
except FileNotFoundError:
    print("❌ parking.csv not found.")
    exit()

# Check columns
print("Columns in parking.csv:", parking_df.columns.tolist())

# Rename columns
if "LONG" in parking_df.columns and "LAT" in parking_df.columns:
    parking_df.rename(columns={"LONG": "longitude", "LAT": "latitude"}, inplace=True)
else:
    print("❌ 'LONG' or 'LAT' columns missing.")
    exit()

# Convert lat/lon to numeric
parking_df['latitude'] = pd.to_numeric(parking_df['latitude'], errors='coerce')
parking_df['longitude'] = pd.to_numeric(parking_df['longitude'], errors='coerce')

# Drop invalid data
parking_df.dropna(subset=['latitude', 'longitude'], inplace=True)

# GeoDataFrame conversion
parking_gdf = gpd.GeoDataFrame(
    parking_df,
    geometry=gpd.points_from_xy(parking_df.longitude, parking_df.latitude),
    crs="EPSG:4326"
)

# Check if data exists
if parking_gdf.empty:
    print("❌ No valid data.")
    exit()

# Create folium map
city_map = folium.Map(
    location=[parking_gdf.geometry.y.mean(), parking_gdf.geometry.x.mean()],
    zoom_start=12
)

# Prepare heatmap data
parking_heat_data = [
    [float(point.geometry.y), float(point.geometry.x)]
    for _, point in parking_gdf.iterrows()
]

print(f"✅ Prepared {len(parking_heat_data)} points for heatmap.")

# FIXED: Use strings as keys for gradient
gradient = {
    '0.2': 'green',
    '0.5': 'orange',
    '0.8': 'red'
}

# Add heatmap
HeatMap(
    parking_heat_data,
    radius=12,
    gradient=gradient,
    name="Illegal Parking Violations"
).add_to(city_map)

print("✅ Added heatmap to the map.")

# Verify folium layers before saving
print("DEBUG: Folium Layers Before Saving:")
for key, value in city_map._children.items():
    print(f"- {key}: {type(key)} -> {type(value)}")

# Save map
try:
    city_map.save("illegal_parking_heatmap.html")
    print("✅ Heatmap saved successfully! Open 'illegal_parking_heatmap.html' in your browser.")
except Exception as e:
    print(f"❌ ERROR saving heatmap: {e}")
    exit()
