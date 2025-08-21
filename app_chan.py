from shiny import App, ui, reactive, render
from shinywidgets import output_widget, render_widget
import plotly.graph_objects as go
import pandas as pd

# ------------------------
# 1) 데이터 로드
# ------------------------
df_000 = pd.read_csv("가로등위험도최종데이터.csv")
df_000 = df_000.dropna(subset=["위도", "경도"])  # 좌표 결측 제거

all_zone = pd.read_csv("all_zone.csv", encoding="cp949")
all_zone = all_zone.dropna(subset=["위도", "경도"])  # 좌표 결측 제거

# ------------------------
# 2) UI
# ------------------------
app_ui = ui.page_fluid(
    ui.h2("안전한 야간운전을 위한 위험구역 시각화", style="text-align:center; margin: 10px 0 20px;"),
    ui.row(
        ui.column(3,
            ui.h4("표시 옵션"),
            ui.input_checkbox_group(
                "layers", "레이어 선택",
                choices={
                    "lamp": "가로등(전체/ROI) [yellow]",
                    "top60": "위험도 상위 60% (≥55) [orange]",
                    "top40": "위험도 상위 40% (≤25) [pink]",
                    "accident": "사고다발구역 [red]",   # ✅ 추가
                },
                selected=["lamp", "top60", "top40", "accident"]
            ),
            ui.hr(),
            ui.h4("시작 뷰(ROI)"),
            ui.input_numeric("lat_min", "LAT_MIN", 36.79),
            ui.input_numeric("lat_max", "LAT_MAX", 36.86),
            ui.input_numeric("lon_min", "LON_MIN", 127.09),
            ui.input_numeric("lon_max", "LON_MAX", 127.17),
            ui.help_text("ROI 내 데이터가 없으면 전체 데이터를 기준으로 중심을 계산합니다."),
        ),
        ui.column(9,
            output_widget("map")  # Plotly를 직접 위젯으로 렌더
        )
    )
)

# ------------------------
# 3) Server
# ------------------------
def server(input, output, session):

    # 위험/안전 구간 미리 분리
    @reactive.Calc
    def layers_df():
        df_top60 = df_000[df_000["위험도(100점)"] >= 55]
        df_top40 = df_000[df_000["위험도(100점)"] <= 25]
        return df_top60, df_top40

    # 간단 줌 추정 함수
    def guess_zoom(span_deg: float) -> int:
        if span_deg <= 0.01: return 15
        if span_deg <= 0.02: return 14
        if span_deg <= 0.05: return 13
        if span_deg <= 0.10: return 12
        if span_deg <= 0.20: return 11
        if span_deg <= 0.40: return 10
        return 9

    @render_widget
    def map():
        # ROI 입력값
        LAT_MIN = float(input.lat_min())
        LAT_MAX = float(input.lat_max())
        LON_MIN = float(input.lon_min())
        LON_MAX = float(input.lon_max())

        # ROI 데이터 (없으면 전체)
        df_roi = df_000[
            (df_000["위도"] >= LAT_MIN) & (df_000["위도"] <= LAT_MAX) &
            (df_000["경도"] >= LON_MIN) & (df_000["경도"] <= LON_MAX)
        ].copy()
        base = df_roi if not df_roi.empty else df_000

        # 중심/줌
        center_lat = base["위도"].mean()
        center_lon = base["경도"].mean()
        span = max(max(LAT_MAX - LAT_MIN, 1e-6), max(LON_MAX - LON_MIN, 1e-6))
        zoom = guess_zoom(span)

        # 레이어 준비
        df_top60, df_top40 = layers_df()
        selected = set(input.layers())

        # Figure
        fig = go.Figure()

        # 가로등 (yellow)
        if "lamp" in selected and not base.empty:
            fig.add_trace(go.Scattermapbox(
                lat=base["위도"],
                lon=base["경도"],
                mode="markers",
                marker=dict(size=7, color="yellow", opacity=0.5),
                text=base["설치형태"] + "<br>위험도: " + base["위험도(100점)"].astype(str),
                name="가로등 위치",
            ))

        # 위험도 상위 60% (orange)
        if "top60" in selected and not df_top60.empty:
            fig.add_trace(go.Scattermapbox(
                lat=df_top60["위도"],
                lon=df_top60["경도"],
                mode="markers",
                marker=dict(size=10, color="orange", opacity=0.6),
                text=df_top60["위험도(100점)"].astype(str),
                name="위험도 상위 60%",
            ))

        # 위험도 상위 40% (pink)
        if "top40" in selected and not df_top40.empty:
            fig.add_trace(go.Scattermapbox(
                lat=df_top40["위도"],
                lon=df_top40["경도"],
                mode="markers",
                marker=dict(size=10, color="pink", opacity=0.6),
                name="위험도 상위 40%",
            ))

        # 사고다발구역 (red) ✅ 추가
        if "accident" in selected and not all_zone.empty:
            fig.add_trace(go.Scattermapbox(
                lat=all_zone["위도"],
                lon=all_zone["경도"],
                mode="markers",
                marker=dict(size=10, color="red", opacity=0.6),
                text=all_zone["지점명"],
                name="사고다발구역",
            ))

        # 레이아웃
        fig.update_layout(
            mapbox_style="open-street-map",
            mapbox_zoom=zoom,
            mapbox_center={"lat": center_lat, "lon": center_lon},
            height=800,
            margin={"r": 0, "t": 0, "l": 0, "b": 0},
            legend=dict(title="시설 종류", orientation="h")
        )
        return fig

# ------------------------
# 4) App
# ------------------------
app = App(app_ui, server)