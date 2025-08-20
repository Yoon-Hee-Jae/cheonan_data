from shiny import App, ui, reactive, render
import pandas as pd
import plotly.graph_objects as go
import requests, math

# ------------------------
# 1) 데이터 로드
# ------------------------
df_lamp = pd.read_csv('가로등위험도최종데이터.csv')
df_cctv = pd.read_csv('cctv최종데이터.csv')
df_school = pd.read_csv('학교최종데이터.csv')
df_kickrani = pd.read_excel('kickrani.xlsx', header=1)
df_store = pd.read_csv('상권최종데이터.csv')
all_zone = pd.read_csv("all_zone.csv", encoding="cp949")

# 경로 위험도용 데이터
max_risk_in_data = df_lamp['위험도(100점)'].max()

# ------------------------
# 2) 공통 함수
# ------------------------
def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    c = 2*math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R*c

def get_risk_color(risk_score, max_value):
    if max_value == 0:
        return 'rgb(0,0,255)'
    normalized_score = risk_score / max_value
    processed_score = normalized_score**2
    red = int(255*processed_score)
    blue = int(255*(1-processed_score))
    return f'rgb({red},0,{blue})'

# ------------------------
# 3) UI
# ------------------------
app_ui = ui.page_fluid(
    ui.h1("안전한 야간운전 시각화", style="text-align:center; margin-bottom:30px;"),

    ui.row(
        ui.column(12,
            ui.input_radio_buttons(
                "page_select", "구분",
                choices=["위험요소", "위험구역 및 안전구역", "경로 위험도"],
                selected="위험요소"
            )
        )
    ),

    # page1
    ui.row(ui.column(12, ui.output_ui("page1_ui"))),
    # page2
    ui.row(ui.column(12, ui.output_ui("page2_ui"))),
    # page3
    ui.row(ui.column(12, ui.output_ui("page3_ui")))
)

# ------------------------
# 4) 서버
# ------------------------
def server(input, output, session):

    # ===== 페이지 선택 =====
    @reactive.Calc
    def page_selected():
        return input.page_select()

    # ------------------ page1: 위험요소 ------------------
    @output
    @render.ui
    def page1_ui():
        if page_selected() != "위험요소":
            return ui.HTML("")
        return ui.div(
            ui.row(
                ui.column(3,
                    ui.input_checkbox_group(
                        "facility","시설",
                        choices=["가로등","CCTV","학교","킥라니","상권"],
                        selected=["가로등"]
                    )
                ),
                ui.column(9, ui.output_ui("map_plot"))
            )
        )

    @reactive.Calc
    def selected_facilities():
        return input.facility()

    @output
    @render.ui
    def map_plot():
        fig = go.Figure()
        selected = selected_facilities()

        if "가로등" in selected:
            fig.add_trace(go.Scattermapbox(
                lat=df_lamp['위도'], lon=df_lamp['경도'],
                mode='markers',
                marker=dict(size=7, color='yellow', opacity=0.5),
                text=df_lamp['설치형태']+'<br>위험도:'+df_lamp['위험도(100점)'].astype(str),
                name='가로등'
            ))
        if "CCTV" in selected:
            fig.add_trace(go.Scattermapbox(
                lat=df_cctv['위도'], lon=df_cctv['경도'],
                mode='markers', marker=dict(size=10,color='green',opacity=0.6), name='CCTV'
            ))
        if "학교" in selected:
            fig.add_trace(go.Scattermapbox(
                lat=df_school['lat'], lon=df_school['lon'],
                mode='markers', marker=dict(size=10,color='purple',opacity=0.6),
                text=df_school['구분'], name='학교'
            ))
        if "킥라니" in selected:
            fig.add_trace(go.Scattermapbox(
                lat=df_kickrani['위도'], lon=df_kickrani['경도'],
                mode='markers', marker=dict(size=10,color='black',opacity=0.6),
                text=df_kickrani['주차가능 대수'].astype(str), name='킥라니'
            ))
        if "상권" in selected:
            fig.add_trace(go.Scattermapbox(
                lat=df_store['위도'], lon=df_store['경도'],
                mode='markers', marker=dict(size=7,color='orange',opacity=0.6),
                text=df_store['상호명'], name='상권'
            ))

        fig.update_layout(
            mapbox_style="open-street-map",
            mapbox_zoom=12,
            mapbox_center={"lat":df_lamp['위도'].mean(),"lon":df_lamp['경도'].mean()},
            height=800, margin={"r":0,"t":0,"l":0,"b":0},
            legend=dict(title="시설 종류",orientation="h")
        )
        return ui.HTML(fig.to_html(full_html=False))

    # ------------------ page2: 위험구역 및 안전구역 ------------------
    @output
    @render.ui
    def page2_ui():
        if page_selected() != "위험구역 및 안전구역":
            return ui.HTML("")
        return ui.div(
            ui.row(
                ui.column(3,
                    ui.input_checkbox_group(
                        "layers","구역",
                        choices=["가로등","위험구역","안전구역","사고다발지역"],
                        selected=["가로등"]
                    )
                ),
                ui.column(9, ui.output_ui("map_chan"))
            )
        )

    @reactive.Calc
    def layers_df():
        return (
            df_lamp[df_lamp["위험도(100점)"]>=55],
            df_lamp[df_lamp["위험도(100점)"]<=25]
        )

    @output
    @render.ui
    def map_chan():
        base = df_lamp
        df_위험구역, df_안전구역 = layers_df()
        selected = set(input.layers())

        fig = go.Figure()
        if "가로등" in selected:
            fig.add_trace(go.Scattermapbox(
                lat=base["위도"], lon=base["경도"],
                mode="markers", marker=dict(size=7,color="yellow",opacity=0.5),
                text=base["위험도(100점)"].astype(str), name="가로등 위치"
            ))
        if "위험구역" in selected:
            fig.add_trace(go.Scattermapbox(
                lat=df_위험구역["위도"], lon=df_위험구역["경도"],
                mode="markers", marker=dict(size=10,color="orange",opacity=0.6),
                text=df_위험구역["위험도(100점)"].astype(str), name="위험구역"
            ))
        if "안전구역" in selected:
            fig.add_trace(go.Scattermapbox(
                lat=df_안전구역["위도"], lon=df_안전구역["경도"],
                mode="markers", marker=dict(size=10,color="pink",opacity=0.6),
                name="안전구역"
            ))
        if "사고다발지역" in selected:
            fig.add_trace(go.Scattermapbox(
                lat=all_zone["위도"], lon=all_zone["경도"],
                mode="markers", marker=dict(size=10,color="red",opacity=0.6),
                text=all_zone["지점명"], name="사고다발구역"
            ))

        fig.update_layout(
            mapbox_style="open-street-map",
            mapbox_zoom=12,
            mapbox_center={"lat":base["위도"].mean(),"lon":base["경도"].mean()},
            height=800, margin={"r":0,"t":0,"l":0,"b":0},
            legend=dict(title="시설 종류",orientation="h")
        )
        return ui.HTML(fig.to_html(full_html=False))

    # ------------------ page3: 경로 위험도 ------------------
    @output
    @render.ui
    def page3_ui():
        if page_selected() != "경로 위험도":
            return ui.HTML("")

        # Kakao API 호출
        API_KEY = "d222f0f01e3470ce2b8a863cc30b151e"
        headers = {"Authorization": f"KakaoAK {API_KEY}"}
        origin = (36.78794, 127.1289)
        destination = (36.83828, 127.1485)

        url = "https://apis-navi.kakaomobility.com/v1/directions"
        params = {
            "origin": f"{origin[1]},{origin[0]}",
            "destination": f"{destination[1]},{destination[0]}",
            "priority":"RECOMMEND",
            "alternatives":"true","car_type":1
        }
        res = requests.get(url, headers=headers, params=params)
        data = res.json()
        routes_to_show = data.get("routes", [])[:3]

        fig = go.Figure()
        num_segments = 100

        for idx, route in enumerate(routes_to_show):
            coords_all = []
            for section in route["sections"]:
                for road in section["roads"]:
                    coords_all.extend(road["vertexes"])
            lons = coords_all[0::2]; lats = coords_all[1::2]
            total_points = len(lats)
            step = total_points/num_segments

            for i in range(num_segments):
                start_idx = math.floor(i*step)
                end_idx = math.floor((i+1)*step) if i<num_segments-1 else total_points-1
                seg_lats = lats[start_idx:end_idx+1]
                seg_lons = lons[start_idx:end_idx+1]
                if not seg_lats: continue

                max_risk = 0.0
                for j in range(len(seg_lats)):
                    rlat, rlon = seg_lats[j], seg_lons[j]
                    for _, row in df_lamp.iterrows():
                        dist = haversine(rlat, rlon, row['위도'], row['경도'])
                        if dist <= 50 and row['위험도(100점)'] > max_risk:
                            max_risk = row['위험도(100점)']

                color = get_risk_color(max_risk, max_risk_in_data)
                fig.add_trace(go.Scattermapbox(
                    lat=seg_lats, lon=seg_lons,
                    mode="lines", line=dict(width=5,color=color),
                    showlegend=False, hoverinfo="none"
                ))

        fig.update_layout(
            mapbox_style="open-street-map",
            mapbox_zoom=13,
            mapbox_center={"lat":origin[0],"lon":origin[1]},
            height=800, margin={"r":0,"t":0,"l":0,"b":0},
            title="경로별 구간 위험도 시각화"
        )
        fig.add_trace(go.Scattermapbox(lat=[None], lon=[None], mode="lines", line=dict(color="red", width=4), name="최고 위험"))
        fig.add_trace(go.Scattermapbox(lat=[None], lon=[None], mode="lines", line=dict(color="blue", width=4), name="최저 위험"))

        return ui.HTML(fig.to_html(full_html=False))

# ------------------------
# 5) App 실행
# ------------------------
app = App(app_ui, server)
