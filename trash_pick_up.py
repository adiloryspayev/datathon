import pandas as pd
import geopandas as gpd
import folium
from folium.plugins import HeatMap
from shapely.geometry import Point
import os

# Load datasets with error handling
try:
    trash_df = pd.read_csv("../../datathon/cityline.csv")  # Trash complaints dataset
    print(f"Successfully loaded cityline.csv with {len(trash_df)} rows")
except FileNotFoundError:
    print("Error: cityline.csv not found. Make sure the file is in the correct directory.")
    print("Current working directory:", os.getcwd())  # Add this debug line
    exit()
except Exception as e:
    print(f"Error loading cityline.csv: {str(e)}")
    exit()

try:
    parking_df = pd.read_csv("../../datathon/parking.csv")  # Illegal parking dataset
    print(f"Successfully loaded parking.csv with {len(parking_df)} rows")
except FileNotFoundError:
    print("Error: parking.csv not found. Make sure the file is in the correct directory.")
    exit()
except Exception as e:
    print(f"Error loading parking.csv: {str(e)}")
    exit()

# After loading parking_df, add these debug prints
print("\nDEBUG: Parking Data Info:")
print("Columns in parking_df:", parking_df.columns.tolist())
print("\nSample of parking data:")
print(parking_df.head())

if 'VIOLATION_TYPE' in parking_df.columns or 'violation_type' in parking_df.columns:
    violation_column = 'VIOLATION_TYPE' if 'VIOLATION_TYPE' in parking_df.columns else 'violation_type'
    print("\nViolation types found:")
    print(parking_df[violation_column].value_counts())
else:
    print("\nWARNING: No violation type column found!")

# Rename columns
trash_df = trash_df.rename(columns={'X': 'longitude', 'Y': 'latitude'})
parking_df = parking_df.rename(columns={'LONG': 'longitude', 'LAT': 'latitude'})

# Convert lat/lon to numeric
trash_df[['latitude', 'longitude']] = trash_df[['latitude', 'longitude']].apply(pd.to_numeric, errors='coerce')
parking_df[['latitude', 'longitude']] = parking_df[['latitude', 'longitude']].apply(pd.to_numeric, errors='coerce')

# Drop NaN values
trash_df.dropna(subset=['latitude', 'longitude'], inplace=True)
parking_df.dropna(subset=['latitude', 'longitude'], inplace=True)

# Convert to GeoDataFrames
trash_gdf = gpd.GeoDataFrame(trash_df, geometry=gpd.points_from_xy(trash_df.longitude, trash_df.latitude), crs="EPSG:4326")
parking_gdf = gpd.GeoDataFrame(parking_df, geometry=gpd.points_from_xy(parking_df.longitude, parking_df.latitude), crs="EPSG:4326")

# Remove invalid geometries
trash_gdf = trash_gdf[trash_gdf.geometry.notna() & ~trash_gdf.geometry.is_empty]
parking_gdf = parking_gdf[parking_gdf.geometry.notna() & ~parking_gdf.geometry.is_empty]

# Ensure datasets are not empty
if trash_gdf.empty or parking_gdf.empty:
    print("Error: One or both datasets have no valid location data!")
    exit()

# Create the map
city_map = folium.Map(location=[trash_gdf.geometry.y.mean(), trash_gdf.geometry.x.mean()], zoom_start=12)

# Generate heatmap data
trash_heat_data = [
    [float(row.geometry.y), float(row.geometry.x)]
    for index, row in trash_gdf.iterrows()
    if pd.notna(row.geometry.y) and pd.notna(row.geometry.x)
]
print(f"DEBUG: Generated {len(trash_heat_data)} valid points for trash heatmap")

if trash_heat_data:
    HeatMap(trash_heat_data, radius=10, gradient={0.2: 'blue', 0.5: 'yellow', 0.8: 'red'}).add_to(city_map)

# Handle parking violations heatmap
valid_parking_heat_data = [
    [float(row.geometry.y), float(row.geometry.x)]
    for index, row in parking_gdf.iterrows()
    if pd.notna(row.geometry.y) and pd.notna(row.geometry.x)
]

# Ensure all entries are lists
valid_parking_heat_data = [x for x in valid_parking_heat_data if isinstance(x, list) and len(x) == 2]
print(f"DEBUG: Final processed parking heatmap data: {valid_parking_heat_data[:5]}")

gradient = {0.2: 'green', 0.5: 'orange', 0.8: 'red'}

# Ensure gradient is a dictionary
if not isinstance(gradient, dict):
    print(f"ERROR: Gradient is not a dictionary! Current value: {gradient}")
    exit()

if valid_parking_heat_data:
    try:
        HeatMap(
            valid_parking_heat_data,
            radius=15,
            gradient=gradient,
            name="All Parking Violations"
        ).add_to(city_map)
        print("✅ Successfully added parking heatmap!")
    except Exception as e:
        print(f"ERROR adding heatmap: {e}")
        exit()

# Add layer control
folium.LayerControl().add_to(city_map)

# Save the map
city_map.save("classified_parking_heatmap.html")
print("Heatmap saved! Open `classified_parking_heatmap.html` in a browser.")
