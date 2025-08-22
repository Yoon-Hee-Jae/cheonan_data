from shiny import App, ui, reactive
from shinywidgets import output_widget, render_widget
import pandas as pd
import math
import requests
import plotly.graph_objects as go
#
# ------------------------------
# 1. 샘플 가로등 위험도 데이터
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
# 2. 출발지/도착지 선택 리스트
# ------------------------------
places = {
    "천안시청": (36.78794, 127.1289),
    "백석동": (36.8000, 127.1400),
    "두정동": (36.8200, 127.1500),
    "불당동": (36.8300, 127.1600)
}

# ------------------------------
# 3. 거리 계산 및 위험도 색상 함수
# ------------------------------
def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    c = 2*math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

def get_risk_color(risk_score, max_value):
    if max_value == 0:
        return 'rgb(0,0,255)'
    normalized_score = risk_score / max_value
    processed_score = normalized_score ** 2
    red = int(255 * processed_score)
    blue = int(255 * (1 - processed_score))
    return f'rgb({red},0,{blue})'

# ------------------------------
# 4. UI
# ------------------------------
app_ui = ui.page_fluid(
    ui.h2("야간운전 위험구간 알림"),
    ui.row(
        ui.column(3,
            ui.input_select("origin", "출발지", list(places.keys()), selected="천안시청"),
            ui.input_select("destination", "도착지", list(places.keys()), selected="두정동"),
            ui.input_action_button("update_btn", "경로 확인")
        ),
        ui.column(9,
            output_widget("map_ui")
        )
    )
)

# ------------------------------
# 5. 서버
# ------------------------------
def server(input, output, session):

    API_KEY = "d222f0f01e3470ce2b8a863cc30b151e"  # 🔑 본인 카카오 REST API Key

    @reactive.Effect
    def update_map():
        input.update_btn()  # 버튼 클릭 트리거

        origin = places[input.origin()]
        destination = places[input.destination()]

        url = "https://apis-navi.kakaomobility.com/v1/directions"
        headers = {"Authorization": f"KakaoAK {API_KEY}"}
        params = {
            "origin": f"{origin[1]},{origin[0]}",
            "destination": f"{destination[1]},{destination[0]}",
            "priority": "RECOMMEND",
            "alternatives": "true",
            "car_type": 1
        }

        try:
            res = requests.get(url, headers=headers, params=params)
            data = res.json()
            routes_to_show = data.get("routes", [])[:3]
        except Exception as e:
            print("API 호출 실패:", e)
            routes_to_show = []

        # 지도 생성
        fig = go.Figure()
        num_segments = 10

        for idx, route in enumerate(routes_to_show):
            coords_all = []
            for section in route.get("sections", []):
                for road in section.get("roads", []):
                    coords_all.extend(road.get("vertexes", []))

            if not coords_all:
                continue

            lons = coords_all[0::2]
            lats = coords_all[1::2]
            total_points = len(lats)
            step = total_points / num_segments

            for i in range(num_segments):
                start_idx = math.floor(i*step)
                end_idx = math.floor((i+1)*step) if i<num_segments-1 else total_points-1

                segment_lats = lats[start_idx:end_idx+1]
                segment_lons = lons[start_idx:end_idx+1]

                if not segment_lats:
                    continue

                segment_center_lat = segment_lats[len(segment_lats)//2]
                segment_center_lon = segment_lons[len(segment_lons)//2]

                search_radius_m = 50
                max_risk = 0.0
                for _, row in df_new.iterrows():
                    distance = haversine(segment_center_lat, segment_center_lon, row['위도'], row['경도'])
                    if distance <= search_radius_m and row['위험도(100점)'] > max_risk:
                        max_risk = row['위험도(100점)']

                color = get_risk_color(max_risk, max_risk_in_data)

                fig.add_trace(go.Scattermapbox(
                    lat=segment_lats,
                    lon=segment_lons,
                    mode="lines",
                    line=dict(width=5, color=color),
                    name=f"경로 {idx+1} 위험도",
                    showlegend=False
                ))

        fig.update_layout(
            mapbox=dict(
                style="open-street-map",
                center=dict(lat=origin[0], lon=origin[1]),
                zoom=13
            ),
            margin=dict(r=0, t=0, l=0, b=0),
            title="경로별 구간 위험도 시각화"
        )

        @output
        @render_widget
        def map_ui():
            return fig

# ------------------------------
# 6. 앱 실행
# ------------------------------
app = App(app_ui, server)
