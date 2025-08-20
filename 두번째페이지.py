from shiny import App, ui, reactive, render
from shinywidgets import output_widget, render_widget
import pandas as pd
import plotly.graph_objects as go

# ------------------------
# 1) 데이터 로드
# ------------------------
df_lamp = pd.read_csv('가로등위험도최종데이터.csv')
df_lamp.info()
df_cctv = pd.read_csv('cctv최종데이터.csv')
df_school = pd.read_csv('학교최종데이터.csv')
df_kickrani = pd.read_excel('kickrani.xlsx', header=1)
df_store = pd.read_csv('상권최종데이터.csv')
all_zone = pd.read_csv("all_zone.csv", encoding="cp949")

# 좌표 결측 제거 (안전)
def _clean_latlon(df, lat, lon):
    df[lat] = pd.to_numeric(df[lat], errors="coerce")
    df[lon] = pd.to_numeric(df[lon], errors="coerce")
    return df.dropna(subset=[lat, lon])

df_lamp = _clean_latlon(df_lamp, "위도", "경도")
df_cctv = _clean_latlon(df_cctv, "위도", "경도")
df_store = _clean_latlon(df_store, "위도", "경도")
all_zone = _clean_latlon(all_zone, "위도", "경도")
df_school = df_school.rename(columns={"lat":"위도", "lon":"경도"})
df_school = _clean_latlon(df_school, "위도", "경도")
df_kickrani = _clean_latlon(df_kickrani, "위도", "경도")

# 지도 높이(Plotly는 10 이상 필요)
MAP_HEIGHT = 720

# ------------------------
# 2) UI
# ------------------------
BASE_STYLES = ui.tags.style("""
:root{ --bg:#f7f8fa; --card:#ffffff; --muted:#667085; --radius:14px; --shadow:0 10px 25px rgba(16,24,40,.06); }
html,body{ background:var(--bg); }
.header{background:#f7f8fa;border-bottom:1px solid #e9edf2;padding:12px 20px;margin-bottom:10px;}
.h1{margin:0;font-size:22px;font-weight:800}
.h2{margin:2px 0 0;color:var(--muted);font-size:13px}
.panel{background:var(--card);border-radius:var(--radius);box-shadow:var(--shadow);padding:16px;height:100%;}
.tabs-fix .shiny-input-radiogroup{display:flex;gap:14px;align-items:center;margin:0}
.tab-box{background:#fff;border-radius:12px;box-shadow:var(--shadow);padding:14px 20px;margin:20px 0;display:inline-flex;align-items:center;gap:14px;}
.map-card{background:#fff;border-radius:var(--radius);box-shadow:var(--shadow);padding:8px;overflow:hidden;}
""")

app_ui = ui.page_fluid(
    ui.head_content(BASE_STYLES),

    ui.div({"class":"header"},
        ui.div({"class":"h1"}, "안전한 야간운전 위험구역 시각화"),
        ui.div({"class":"h2"}, "천안시 야간 위험요소/구역을 한눈에 — 필요 레이어만 선택해 비교하세요.")
    ),

    ui.div({"class":"tabs-fix tab-box"},
        ui.input_radio_buttons(
            "page_select", None,
            choices=["위험요소", "위험구역 및 안전구역"],
            selected="위험요소"
        )
    ),

    # 페이지1
    ui.output_ui("page1_ui"),
    # 페이지2
    ui.output_ui("page2_ui"),
)

# ------------------------
# 3) 서버
# ------------------------
def server(input, output, session):

    # 줌 추정
    def guess_zoom(span_deg: float) -> int:
        if span_deg <= 0.01: return 15
        if span_deg <= 0.02: return 14
        if span_deg <= 0.05: return 13
        if span_deg <= 0.10: return 12
        if span_deg <= 0.20: return 11
        if span_deg <= 0.40: return 10
        return 9

    # 페이지 선택
    @reactive.Calc
    def page_selected():
        return input.page_select()

    # ---------------- 페이지 1 (위험요소) ----------------
    @output
    @render.ui
    def page1_ui():
        if page_selected() != "위험요소":
            return ui.HTML("")
        return ui.div(
            ui.row(
                ui.column(3,
                    ui.div({"class":"panel"},
                        ui.h4("표시 옵션"),
                        ui.input_checkbox_group(
                            "facility", "시설",
                            choices=["가로등", "CCTV", "학교", "킥라니", "상권"],
                            selected=["가로등"]
                        ),
                    )
                ),
                ui.column(9,
                    ui.div({"class":"map-card"},
                        # 위젯 출력 영역(Plotly)
                        output_widget("map_plot", height=f"{MAP_HEIGHT}px")
                    )
                )
            )
        )

    @output
    @render_widget
    def map_plot():
        # 중심/줌
        if len(df_lamp) == 0:
            center_lat, center_lon = 36.815, 127.113
            zoom = 12
        else:
            center_lat, center_lon = df_lamp["위도"].mean(), df_lamp["경도"].mean()
            span = max(df_lamp["위도"].max() - df_lamp["위도"].min(),
                       df_lamp["경도"].max() - df_lamp["경도"].min())
            zoom = guess_zoom(span)

        fig = go.Figure()
        selected = set(input.facility() if "facility" in input else [])

        if "가로등" in selected and len(df_lamp):
            fig.add_trace(go.Scattermapbox(
                lat=df_lamp['위도'], lon=df_lamp['경도'],
                mode='markers',
                marker=dict(size=7, color='#f6c945', opacity=0.55),
                text=(
                    "가로등"
                    + "<br>설치형태: " + df_lamp.get('설치형태', pd.Series(['-']*len(df_lamp))).astype(str)
                    + "<br>위험도: " + df_lamp.get('위험도(100점)', pd.Series(['-']*len(df_lamp))).astype(str)
                ),
                hovertemplate="%{text}<extra></extra>",
                name='가로등'
            ))
        if "CCTV" in selected and len(df_cctv):
            fig.add_trace(go.Scattermapbox(
                lat=df_cctv['위도'], lon=df_cctv['경도'],
                mode='markers', marker=dict(size=10, color='#22c55e', opacity=0.65),
                text="CCTV", hovertemplate="%{text}<extra></extra>", name='CCTV'
            ))
        if "학교" in selected and len(df_school):
            fig.add_trace(go.Scattermapbox(
                lat=df_school['위도'], lon=df_school['경도'],
                mode='markers', marker=dict(size=10, color='#8b5cf6', opacity=0.65),
                text=df_school.get('구분', '학교'), hovertemplate="%{text}<extra></extra>", name='학교'
            ))
        if "킥라니" in selected and len(df_kickrani):
            fig.add_trace(go.Scattermapbox(
                lat=df_kickrani['위도'], lon=df_kickrani['경도'],
                mode='markers', marker=dict(size=10, color='#111827', opacity=0.65),
                text="주차가능 대수: " + df_kickrani.get('주차가능 대수', pd.Series(['-']*len(df_kickrani))).astype(str),
                hovertemplate="%{text}<extra></extra>", name='킥라니'
            ))
        if "상권" in selected and len(df_store):
            fig.add_trace(go.Scattermapbox(
                lat=df_store['위도'], lon=df_store['경도'],
                mode='markers', marker=dict(size=8, color='#f59e0b', opacity=0.65),
                text=df_store.get('상호명', '상권'), hovertemplate="%{text}<extra></extra>", name='상권'
            ))

        fig.update_layout(
            mapbox_style="open-street-map",
            mapbox_zoom=zoom,
            mapbox_center={"lat": center_lat, "lon": center_lon},
            height=MAP_HEIGHT,
            margin={"r":0,"t":0,"l":0,"b":0},
            legend=dict(orientation="h", yanchor="bottom", y=0.01)
        )
        return fig

    # ---------------- 페이지 2 (위험구역/안전구역) ----------------
    @output
    @render.ui
    def page2_ui():
        if page_selected() != "위험구역 및 안전구역":
            return ui.HTML("")
        return ui.div(
            ui.row(
                ui.column(3,
                    ui.div({"class":"panel"},
                        ui.h4("표시 옵션"),
                        ui.input_checkbox_group(
                            "layers", "구역",
                            choices=["가로등", "위험구역", "안전구역", "사고다발지역"],
                            selected=["가로등"]
                        ),
                    )
                ),
                ui.column(9,
                    ui.div({"class":"map-card"},
                        output_widget("map_chan", height=f"{MAP_HEIGHT}px")
                    )
                )
            )
        )

    @output
    @render_widget
    def map_chan():
        base = df_lamp
        if len(base) == 0:
            center_lat, center_lon = 36.815, 127.113
            zoom = 12
        else:
            center_lat, center_lon = base["위도"].mean(), base["경도"].mean()
            span = max(base["위도"].max()-base["위도"].min(), base["경도"].max()-base["경도"].min())
            zoom = guess_zoom(span)

        df_risk = base[base.get("위험도(100점)", 0) >= 55]
        df_safe = base[base.get("위험도(100점)", 100) <= 25]
        selected = set(input.layers() if "layers" in input else [])

        fig = go.Figure()

        if "가로등" in selected and len(base):
            fig.add_trace(go.Scattermapbox(
                lat=base["위도"], lon=base["경도"],
                mode="markers", marker=dict(size=7, color="#f6c945", opacity=0.5),
                text=(
                    "가로등"
                    + "<br>설치형태: " + base.get("설치형태", pd.Series(['-']*len(base))).astype(str)
                    + "<br>위험도: " + base.get("위험도(100점)", pd.Series(['-']*len(base))).astype(str)
                ),
                hovertemplate="%{text}<extra></extra>", name="가로등"
            ))
        if "위험구역" in selected and len(df_risk):
            fig.add_trace(go.Scattermapbox(
                lat=df_risk["위도"], lon=df_risk["경도"],
                mode="markers", marker=dict(size=10, color="#fb923c", opacity=0.65),
                text="위험도: " + df_risk.get("위험도(100점)", pd.Series(['-']*len(df_risk))).astype(str),
                hovertemplate="%{text}<extra></extra>", name="위험구역(≥55)"
            ))
        if "안전구역" in selected and len(df_safe):
            fig.add_trace(go.Scattermapbox(
                lat=df_safe["위도"], lon=df_safe["경도"],
                mode="markers", marker=dict(size=10, color="#f472b6", opacity=0.65),
                text="위험도: " + df_safe.get("위험도(100점)", pd.Series(['-']*len(df_safe))).astype(str),
                hovertemplate="%{text}<extra></extra>", name="안전구역(≤25)"
            ))
        if "사고다발지역" in selected and len(all_zone):
            fig.add_trace(go.Scattermapbox(
                lat=all_zone["위도"], lon=all_zone["경도"],
                mode="markers", marker=dict(size=11, color="#ef4444", opacity=0.7),
                text=all_zone.get("지점명", "사고다발지역"),
                hovertemplate="%{text}<extra></extra>", name="사고다발지역"
            ))

        fig.update_layout(
            mapbox_style="open-street-map",
            mapbox_zoom=zoom,
            mapbox_center={"lat": center_lat, "lon": center_lon},
            height=MAP_HEIGHT,
            margin={"r":0,"t":0,"l":0,"b":0},
            legend=dict(orientation="h", yanchor="bottom", y=0.01)
        )
        return fig

# ------------------------
# 4) 앱 실행
# ------------------------
app = App(app_ui, server)
app.run()
