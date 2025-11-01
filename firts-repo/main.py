import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import folium

df=pd.read_csv('ride_safety_dataset.csv')
LAT_COLUMN = 'Latitude'
LON_COLUMN = 'Longitude'
CITY_COLUMN = 'City'
df_city=df[df[CITY_COLUMN] == 'Delhi'].copy()
print(f"loaded {len(df_city)} records of delhi")
df_cleaned = df_city.dropna(subset=[LAT_COLUMN, LON_COLUMN])

rows_dropped = len(df_city) - len(df_cleaned)
print(f"Remaining records: {len(df_cleaned)}")
print("Converting to GeoDataFrame")
geometry = [Point(xy) for xy in zip(df_cleaned[LON_COLUMN], df_cleaned[LAT_COLUMN])]
gdf = gpd.GeoDataFrame(df_cleaned, geometry=geometry)
gdf.set_crs(epsg=4326, inplace=True)
print("\n--- GeoDataFrame created successfully! ---")

print("Running DBSCAN clustering...")
eps_met=1000
min_sample=2
earth_radius_m =6371000
eps_radians=eps_met/earth_radius_m
X = np.radians(gdf[[LAT_COLUMN, LON_COLUMN]].values)
db = DBSCAN(eps=eps_radians, min_samples=min_sample, algorithm='ball_tree', metric='haversine')
db.fit(X)

labels = db.labels_
gdf['cluster'] = labels
print("Clustering complete.")

clusters = len(set(labels)) - (1 if -1 in labels else 0)
noise = list(labels).count(-1)

print(f"\n--- Clustering Summary ---")
print(f"Estimated number of hotspots (clusters): {clusters}")
print(f"Number of isolated incidents (noise): {noise}")
print(f"Total points clustered: {len(gdf) - noise}")


print("\nHotspot sizes (top 10):")
print(gdf['cluster'].value_counts().head(10))

print("Creating map...")

gdf_noise = gdf[gdf['cluster'] == -1]
gdf_hotspots = gdf[gdf['cluster'] != -1]
map_center = [gdf[LAT_COLUMN].mean(), gdf[LON_COLUMN].mean()]
m = folium.Map(location=map_center, zoom_start=11)

for _, row in gdf_noise.iterrows():
    folium.CircleMarker(
        location=[row[LAT_COLUMN], row[LON_COLUMN]],
        radius=2,
        color='gray',
        fill=True,
        fill_color='gray',
        fill_opacity=0.5,
        popup=f"Noise Point<br>Crime: {row['Crime_Type']}"
    ).add_to(m)

cluster_ids = sorted(gdf_hotspots['cluster'].unique())
colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'lightred', 'beige', 'darkblue']
for i, cluster_id in enumerate(cluster_ids):
    
    cluster_points = gdf_hotspots[gdf_hotspots['cluster'] == cluster_id]
    
    
    cluster_color = colors[i % len(colors)] 
    
    for _, row in cluster_points.iterrows():
        folium.CircleMarker(
            location=[row[LAT_COLUMN], row[LON_COLUMN]],
            radius=5,
            color=cluster_color,
            fill=True,
            fill_color=cluster_color,
            fill_opacity=1.0,
            popup=f"Hotspot: {cluster_id}<br>Crime: {row['Crime_Type']}"
        ).add_to(m)
    
output_file = 'delhi_hotspots.html'
m.save(output_file)

print(f"\n--- Map created successfully! ---")
print(f"Open the file '{output_file}' in your browser to see the results.")