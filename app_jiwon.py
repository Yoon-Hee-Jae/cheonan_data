from shiny import App, ui, reactive, render
import pandas as pd
import plotly.graph_objects as go

# ------------------------
# 1) 데이터 로드
# ------------------------
df_lamp = pd.read_csv('가로등위험도최종데이터.csv')
df_cctv = pd.read_csv('cctv최종데이터.csv')
df_school = pd.read_csv('학교최종데이터.csv')
df_kickrani = pd.read_excel('kickrani.xlsx', header=1)
df_store = pd.read_csv('상권최종데이터.csv')
all_zone = pd.read_csv("all_zone.csv", encoding="cp949")

# ------------------------
# 2) UI
# ------------------------
app_ui = ui.page_fluid(
    ui.h1("안전한 야간운전 위험구역 시각화", style="text-align:center; margin-bottom:30px;"),
    
    # 페이지 선택 버튼
    ui.row(
        ui.column(12,
                  ui.input_radio_buttons("page_select", "구분", choices=["위험요소", "위험구역 및 안전구역"], selected="위험요소")
        )
    ),
    
    # 코드1 페이지
    ui.row(
        ui.column(12,
                  ui.output_ui("page1_ui")
        )
    ),
    
    # 코드2 페이지
    ui.row(
        ui.column(12,
                  ui.output_ui("page2_ui")
        )
    )
)

# ------------------------
# 3) 서버
# ------------------------
def server(input, output, session):

    # ===== 페이지 표시 제어 =====
    @reactive.Calc
    def page_selected():
        return input.page_select()

    # ===== 코드1(페이지1) UI =====
    @output
    @render.ui
    def page1_ui():
        if page_selected() != "위험요소":
            return ui.HTML("")  # 숨기기
        return ui.div(
            ui.row(
                ui.column(3,
                          ui.input_checkbox_group(
                              "facility",
                              "시설",
                              choices=["가로등", "CCTV", "학교", "킥라니", "상권"],
                              selected=["가로등"]
                          )
                ),
                ui.column(9,
                          ui.output_ui("map_plot")
                )
            )
        )

    # ===== 코드2(페이지2) UI =====
    @output
    @render.ui
    def page2_ui():
        if page_selected() != "위험구역 및 안전구역":
            return ui.HTML("")  # 숨기기
        return ui.div(
            ui.row(
                ui.column(3,
                          ui.input_checkbox_group(
                              "layers", "구역",
                              choices=["가로등", "위험구역", "안전구역", "사고다발지역"],
                              selected=["가로등"]
                          ),
                ),
                ui.column(9,
                          ui.output_ui("map_chan")
                )
            )
        )

    # ===== 코드1 서버 =====
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
                lat=df_lamp['위도'],
                lon=df_lamp['경도'],
                mode='markers',
                marker=dict(size=7, color='yellow', opacity=0.5),
                text=df_lamp['설치형태'] + '<br>위험도: ' + df_lamp['위험도(100점)'].astype(str),
                name='가로등'
            ))
        if "CCTV" in selected:
            fig.add_trace(go.Scattermapbox(
                lat=df_cctv['위도'],
                lon=df_cctv['경도'],
                mode='markers',
                marker=dict(size=10, color='green', opacity=0.6),
                name='CCTV'
            ))
        if "학교" in selected:
            fig.add_trace(go.Scattermapbox(
                lat=df_school['lat'],
                lon=df_school['lon'],
                mode='markers',
                marker=dict(size=10, color='purple', opacity=0.6),
                text=df_school['구분'],
                name='학교'
            ))
        if "킥라니" in selected:
            fig.add_trace(go.Scattermapbox(
                lat=df_kickrani['위도'],
                lon=df_kickrani['경도'],
                mode='markers',
                marker=dict(size=10, color='black', opacity=0.6),
                text=df_kickrani['주차가능 대수'].astype(str),
                name='킥라니'
            ))
        if "상권" in selected:
            fig.add_trace(go.Scattermapbox(
                lat=df_store['위도'],
                lon=df_store['경도'],
                mode='markers',
                marker=dict(size=7, color='orange', opacity=0.6),
                text=df_store['상호명'],
                name='상권'
            ))

        fig.update_layout(
            mapbox_style="open-street-map",
            mapbox_zoom=12,
            mapbox_center={"lat": df_lamp['위도'].mean(), "lon": df_lamp['경도'].mean()},
            height=800,
            margin={"r":0,"t":0,"l":0,"b":0},
            legend=dict(title="시설 종류", orientation="h")
        )
        return ui.HTML(fig.to_html(full_html=False))

    # ===== 코드2 서버 =====
    @reactive.Calc
    def layers_df():
        df_위험구역 = df_lamp[df_lamp["위험도(100점)"] >= 55]
        df_안전구역 = df_lamp[df_lamp["위험도(100점)"] <= 25]
        return df_위험구역, df_안전구역

    def guess_zoom(span_deg: float) -> int:
        if span_deg <= 0.01: return 15
        if span_deg <= 0.02: return 14
        if span_deg <= 0.05: return 13
        if span_deg <= 0.10: return 12
        if span_deg <= 0.20: return 11
        if span_deg <= 0.40: return 10
        return 9

    @output
    @render.ui
    def map_chan():
        # ROI 제거, 전체 데이터 기준
        base = df_lamp
        center_lat = base["위도"].mean()
        center_lon = base["경도"].mean()
        span = max(base["위도"].max() - base["위도"].min(), base["경도"].max() - base["경도"].min())
        zoom = guess_zoom(span)

        df_위험구역, df_안전구역 = layers_df()
        selected = set(input.layers())

        fig = go.Figure()
        if "가로등" in selected:
            fig.add_trace(go.Scattermapbox(
                lat=base["위도"],
                lon=base["경도"],
                mode="markers",
                marker=dict(size=7, color="yellow", opacity=0.5),
                text=base["설치형태"] + "<br>위험도: " + base["위험도(100점)"].astype(str),
                name="가로등 위치",
            ))
        if "위험구역" in selected:
            fig.add_trace(go.Scattermapbox(
                lat=df_위험구역["위도"],
                lon=df_위험구역["경도"],
                mode="markers",
                marker=dict(size=10, color="orange", opacity=0.6),
                text=df_위험구역["위험도(100점)"].astype(str),
                name="위험도 상위 60%",
            ))
        if "안전구역" in selected:
            fig.add_trace(go.Scattermapbox(
                lat=df_안전구역["위도"],
                lon=df_안전구역["경도"],
                mode="markers",
                marker=dict(size=10, color="pink", opacity=0.6),
                name="위험도 상위 40%",
            ))
        if "사고다발지역" in selected:
            fig.add_trace(go.Scattermapbox(
                lat=all_zone["위도"],
                lon=all_zone["경도"],
                mode="markers",
                marker=dict(size=10, color="red", opacity=0.6),
                text=all_zone["지점명"],
                name="사고다발구역",
            ))

        fig.update_layout(
            mapbox_style="open-street-map",
            mapbox_zoom=zoom,
            mapbox_center={"lat": center_lat, "lon": center_lon},
            height=800,
            margin={"r": 0, "t": 0, "l": 0, "b": 0},
            legend=dict(title="시설 종류", orientation="h")
        )
        return ui.HTML(fig.to_html(full_html=False))


# ------------------------
# 4) App 실행
# ------------------------
app = App(app_ui, server)
