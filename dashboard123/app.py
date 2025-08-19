from shiny import App, ui, reactive, render, Inputs, Outputs, Session
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

# ------------------------
# 데이터 로드
# ------------------------
df_lamp = pd.read_csv('가로등위험도최종데이터.csv')
df_cctv = pd.read_csv('cctv최종데이터.csv')
df_school = pd.read_csv('학교최종데이터.csv')
df_kickrani = pd.read_excel('kickrani.xlsx', header=1)
df_store = pd.read_csv('상권최종데이터.csv')

# ------------------------
# UI 정의
# ------------------------
app_ui = ui.page_fluid(
    ui.h1("안전한 야간운전을 위한 위험구역 시각화", style="text-align:center; margin-bottom:30px;"),
    ui.row(
        ui.column(3,
            ui.h3("표시할 시설 선택"),
            ui.input_checkbox_group(
                "facility",
                "시설",
                choices=["가로등", "CCTV", "학교", "킥라니", "상권"],
                selected=["가로등", "CCTV", "학교", "킥라니", "상권"]
            )
        ),
        ui.column(9,
            ui.output_ui("map_plot")  # Plotly HTML을 Shiny UI로 표시
        )
    )
)

# ------------------------
# Server 정의
# ------------------------
def server(input: Inputs, output: Outputs, session: Session):

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
                name='가로등 위치'
            ))

        if "CCTV" in selected:
            fig.add_trace(go.Scattermapbox(
                lat=df_cctv['위도'],
                lon=df_cctv['경도'],
                mode='markers',
                marker=dict(size=10, color='green', opacity=0.6),
                name='CCTV 위치'
            ))

        if "학교" in selected:
            fig.add_trace(go.Scattermapbox(
                lat=df_school['lat'],
                lon=df_school['lon'],
                mode='markers',
                marker=dict(size=10, color='purple', opacity=0.6),
                text=df_school['구분'],
                name='학교 위치'
            ))

        if "킥라니" in selected:
            fig.add_trace(go.Scattermapbox(
                lat=df_kickrani['위도'],
                lon=df_kickrani['경도'],
                mode='markers',
                marker=dict(size=10, color='black', opacity=0.6),
                text=df_kickrani['주차가능 대수'].astype(str),
                name='킥라니 주차장 위치'
            ))

        if "상권" in selected:
            fig.add_trace(go.Scattermapbox(
                lat=df_store['위도'],
                lon=df_store['경도'],
                mode='markers',
                marker=dict(size=10, color='blue', opacity=0.6),
                name='상권 위치'
            ))

        fig.update_layout(
            mapbox_style="open-street-map",
            mapbox_zoom=12,
            mapbox_center={"lat": df_lamp['위도'].mean(), "lon": df_lamp['경도'].mean()},
            height=800,
            margin={"r":0,"t":0,"l":0,"b":0},
            legend=dict(title="시설 종류", orientation="h")
        )

        # Plotly figure를 HTML로 변환해서 Shiny에 전달
        return ui.HTML(pio.to_html(fig, full_html=False))

# ------------------------
# App 실행
# ------------------------
app = App(app_ui, server)
