import requests
import plotly.graph_objects as go
import json
import math
import pandas as pd
import numpy as np

# ------------------------------
# 1. API 키 & 헤더
# ------------------------------
API_KEY = "d222f0f01e3470ce2b8a863cc30b151e"
headers = {"Authorization": f"KakaoAK {API_KEY}"}

# ------------------------------
# 2. 출발지 & 도착지 (천안시 예시)
# ------------------------------
origin = (36.78794, 127.1289)
destination = (36.83828, 127.1485)

# ------------------------------
# 3. 가로등 데이터 로드
# ------------------------------
try:
    df_new = pd.read_csv("가로등위험도최종데이터.csv")
except FileNotFoundError:
    df_new = pd.DataFrame({
        '위도': [36.791, 36.805, 36.820, 36.835, 36.811, 36.825, 36.801],
        '경도': [127.130, 127.145, 127.140, 127.150, 127.135, 127.155, 127.125],
        '위험도(100점)': [95.0, 75.0, 55.0, 25.0, 85.0, 65.0, 45.0]
    })

max_risk_in_data = df_new['위험도(100점)'].max()

# ------------------------------
# 4. 함수: 거리 계산 및 위험도에 따른 색상 반환
# ------------------------------
def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

def get_risk_color(risk_score, max_value):
    if max_value == 0:
        return 'rgb(0, 0, 255)'

    normalized_score = risk_score / max_value
    processed_score = normalized_score ** 2
    red = int(255 * processed_score)
    blue = int(255 * (1 - processed_score))
    return f'rgb({red}, 0, {blue})'

# ------------------------------
# 5. 카카오 API 호출
# ------------------------------
url = "https://apis-navi.kakaomobility.com/v1/directions"
params = {
    "origin": f"{origin[1]},{origin[0]}",
    "destination": f"{destination[1]},{destination[0]}",
    "priority": "RECOMMEND",
    "alternatives": "true",
    "car_type": 1
}
res = requests.get(url, headers=headers, params=params)
data = res.json()
routes_to_show = data.get("routes", [])[:3]

# ------------------------------
# 6. Plotly 지도에 경로 구간별 색상 표시
# ------------------------------
fig = go.Figure()
num_segments = 10

for idx, route in enumerate(routes_to_show):
    coords_all = []
    for section in route["sections"]:
        for road in section["roads"]:
            coords_all.extend(road["vertexes"])
    
    lons = coords_all[0::2]
    lats = coords_all[1::2]
    
    total_points = len(lats)
    step = total_points / num_segments

    # 구간별 위험도 계산
    for i in range(num_segments):
        start_idx = math.floor(i * step)
        end_idx = math.floor((i + 1) * step) if i < num_segments - 1 else total_points - 1

        segment_lats = lats[start_idx:end_idx+1]
        segment_lons = lons[start_idx:end_idx+1]

        if not segment_lats:
            continue
        
        # ---------------------------------------------
        # 최적화된 위험도 계산 로직
        # ---------------------------------------------
        segment_center_lat = segment_lats[len(segment_lats)//2]
        segment_center_lon = segment_lons[len(segment_lons)//2]
        
        search_radius_m = 50
        max_risk = 0.0
        
        # 구간의 중간 지점을 기준으로 주변 가로등의 최고점 찾기
        for _, row in df_new.iterrows():
            lp_lat = row['위도']
            lp_lon = row['경도']
            distance = haversine(segment_center_lat, segment_center_lon, lp_lat, lp_lon)
            
            if distance <= search_radius_m:
                risk_score = row['위험도(100점)']
                if risk_score > max_risk:
                    max_risk = risk_score
        
        color = get_risk_color(max_risk, max_risk_in_data)
        
        fig.add_trace(go.Scattermapbox(
            lat=segment_lats,
            lon=segment_lons,
            mode="lines",
            line=dict(width=5, color=color),
            name=f"경로 {idx+1} 위험도",
            hoverinfo="none",
            showlegend=False
        ))

# ------------------------------
# 7. 지도 레이아웃 설정
# ------------------------------
fig.update_layout(
    mapbox=dict(
        style="open-street-map",
        center=dict(lat=origin[0], lon=origin[1]),
        zoom=13
    ),
    margin={"r":0,"t":0,"l":0,"b":0},
    title="경로별 구간 위험도 시각화"
)

fig.add_trace(go.Scattermapbox(lat=[None], lon=[None], mode='lines', line=dict(color='red', width=4), name='최고 위험' ))
fig.add_trace(go.Scattermapbox(lat=[None], lon=[None], mode='lines', line=dict(color='blue', width=4), name='최저 위험' ))

fig.show()