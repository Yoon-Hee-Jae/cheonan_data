import osmnx as ox
import pandas as pd
df_new = pd.read_csv('가로등위험도최종데이터.csv')
# 천안시 중심 좌표
place = "Cheonan, South Korea"
roads = ox.graph_from_place(place, network_type='drive')  # 도로망
gdf_roads = ox.graph_to_gdfs(roads, nodes=False, edges=True)
gdf_roads.to_file("cheonan_roads.geojson", driver="GeoJSON")

import geopandas as gpd
import matplotlib.pyplot as plt

# 도로망 불러오기
gdf_roads = gpd.read_file("cheonan_roads.geojson")

# 시각화
fig, ax = plt.subplots(figsize=(10,10))
gdf_roads.plot(ax=ax, linewidth=0.5, color="black")
plt.title("Cheonan Road Network", fontsize=15)
plt.show()

df_new.info()