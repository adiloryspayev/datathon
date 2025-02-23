import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
from geopandas.tools import sjoin_nearest
import numpy as np
from geopy import distance
import seaborn as sns

# Load datasets
parking_data = pd.read_csv("/Users/adiloryspayev/datathon/Parking_Violations_-_2023_-_Present.csv")
requests_data = pd.read_csv("/Users/adiloryspayev/datathon/SYRCityline_Requests_(2021-Present).csv")

# Ensure that the relevant columns are treated as strings
requests_data['Request_type'] = requests_data['Request_type'].astype(str)
parking_data['description'] = parking_data['description'].astype(str)

# Filter for skipped trash/recycling requests
skipped_requests = requests_data[
    requests_data["Summary"].str.contains("skip", na=False)
].dropna(subset=["Lat", "Lng"]).copy()  # Drop NaN values for lat/lng and copy to avoid warnings

# Filter for illegally parked cars based on description
illegal_parking = parking_data[
    parking_data["description"].str.contains("ODD/EVEN PARKING",
                                             na=False)
].dropna(subset=["LAT", "LONG"]).copy()
# illegal_parking = parking_data.dropna(subset=["LAT", "LONG"]).copy()




# Output counts and samples to check the filters
print(f"Total skipped trash/recycling requests: {len(skipped_requests)}")
print(f"Total illegally parked cars: {len(illegal_parking)}")

# Print a sample of the filtered data to visually check
print("\nSample of skipped requests:")
print(skipped_requests[['Request_type', 'Lat', 'Lng']].head())

print("\nSample of illegally parked cars:")
print(illegal_parking[['description', 'LAT', 'LONG']].head())

# Create GeoDataFrames for both datasets
skipped_requests["geometry"] = skipped_requests.apply(
    lambda row: Point(row["Lng"], row["Lat"]), axis=1
)
illegal_parking["geometry"] = illegal_parking.apply(
    lambda row: Point(row["LONG"], row["LAT"]), axis=1
)

# Convert to GeoDataFrames
skipped_gdf = gpd.GeoDataFrame(skipped_requests, geometry="geometry")
parking_gdf = gpd.GeoDataFrame(illegal_parking, geometry="geometry")

# Set coordinate reference system (CRS) to WGS84 (EPSG:4326)
skipped_gdf.set_crs(epsg=4326, inplace=True)
parking_gdf.set_crs(epsg=4326, inplace=True)

# Reproject to a local CRS for accurate distance calculations
skipped_gdf = skipped_gdf.to_crs(epsg=32618)
parking_gdf = parking_gdf.to_crs(epsg=32618)

# Spatial join to find skipped pickups near illegally parked cars
nearby_events = sjoin_nearest(skipped_gdf, parking_gdf, max_distance=500, distance_col="distance")

# Check if the spatial join has results
if nearby_events.empty:
    print("No nearby events found.")
else:
    # # Analyze results
    # print("Skipped pickups near illegally parked cars:")
    # print(nearby_events[["Id", "ticket_number", "distance"]].head())

    # # Visualize the results on a map
    # fig, ax = plt.subplots(figsize=(10, 10))

    # # Plot skipped pickups
    # skipped_gdf.plot(ax=ax, color="blue", marker="o", label="Skipped Pickups", markersize=3)

    # # Plot illegally parked cars
    # parking_gdf.plot(ax=ax, color="red", marker="o", label="Illegally Parked Cars", markersize=3)

    # # Plot nearby events
    # nearby_events.plot(ax=ax, color="green", marker="s", label="Nearby Events")

    # # Add legend and title
    # plt.legend()
    # plt.title("Skipped Pickups and Illegally Parked Cars")
    # plt.show()

    # # Frequency chart of distances between skipped pickups and nearest illegally parked car
    # plt.figure(figsize=(10, 6))
    # bins = np.arange(0, nearby_events["distance"].max() + 20, 20)  # Create bins with step size of 20
    # plt.hist(nearby_events["distance"], bins=bins, color="green", edgecolor="black")
    # plt.title("Frequency of Distances Between Skipped Pickups and Nearest Illegally Parked Car")
    # plt.xlabel("Distance (meters)")
    # plt.ylabel("Frequency")
    # plt.grid(True)
    # plt.show()

    # # Count ticket types within specific distance ranges (0-10 meters, 10-20 meters, etc.)
    # def count_ticket_types_by_distance_range(events, max_distance, step=10):
    #     ticket_counts = []

    #     # Loop through each distance range (0-10 meters, 10-20 meters, ...)
    #     for distance in range(0, max_distance + 1, step):
    #         # Filter data for the current distance range
    #         filtered_data = events[(events['distance'] > distance - step) & (events['distance'] <= distance)]
            
    #         # Count the ticket types within this range
    #         ticket_type_counts = filtered_data['description'].value_counts()
    #         ticket_counts.append((distance, ticket_type_counts))
        
    #     return ticket_counts

    # # Get ticket counts by distance range
    # ticket_counts_by_radius = count_ticket_types_by_distance_range(nearby_events, max_distance=500, step=10)

    # # Print the results for each distance range
    # for distance, counts in ticket_counts_by_radius:
    #     print(f"\nTicket counts for distance range {distance}-{distance + 10} meters:")
    #     print(counts)
    # Filter the nearby_events DataFrame for the two ODD/EVEN PARKING violations
    odd_even_nov_mar = nearby_events[nearby_events['description'].str.contains("ODD/EVEN PARKING NOV-MAR")]
    odd_even_apr_oct = nearby_events[nearby_events['description'].str.contains("ODD/EVEN PARKING APR-OCT")]

    # Create bins for the histogram
    bins = np.arange(0, nearby_events["distance"].max() + 20, 20)

    # Plot the histograms for each violation type
    plt.figure(figsize=(10, 6))
    plt.hist(odd_even_nov_mar['distance'], bins=bins, color='blue', edgecolor='black', alpha=0.7, label='ODD/EVEN PARKING NOV-MAR')
    plt.hist(odd_even_apr_oct['distance'], bins=bins, color='orange', edgecolor='black', alpha=0.7, label='ODD/EVEN PARKING APR-OCT')

    # Add title, labels, and legend
    plt.title("Frequency of Distances Between Skipped Pickups and Nearest Illegally Parked Car")
    plt.xlabel("Distance (meters)")
    plt.ylabel("Frequency")
    plt.legend(title="Violation Type")
    plt.grid(True)
    plt.show()
            
    plt.figure(figsize=(10, 6))
    sns.kdeplot(odd_even_nov_mar['distance'], color='blue', label='ODD/EVEN PARKING', bw_adjust=0.7)  # Lower bandwidth

    # Add title, labels, and legend
    plt.title("Distribution of Distances Between Skipped Pickups and Nearest Illegally Parked Car")
    plt.xlabel("Distance (meters)")
    plt.ylabel("Density")
    plt.legend(title="Violation Type")
    plt.grid(True)
    plt.show()