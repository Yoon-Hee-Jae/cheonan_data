from shiny import App, ui, reactive, render
from shinywidgets import output_widget, render_widget
import pandas as pd
import plotly.graph_objects as go

# ------------------------
# 1) 데이터 로드 (코드 내 예시 데이터로 대체)
# ------------------------
df_lamp = pd.read_csv('가로등위험도최종데이터.csv')
df_cctv = pd.read_csv('cctv최종데이터.csv')
df_school = pd.read_csv('학교최종데이터.csv')  # lat, lon 컬럼
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
# 학교는 lat/lon 컬럼명 다름
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
.small-label{color:#6b7280;font-size:12px;margin:2px 0 8px}
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
            choices=["가중치 튜닝", "위험요소", "위험구역 및 안전구역"],
            selected="위험요소" # 기본 선택 페이지 변경
        )
    ),

    # 페이지0: 가중치 튜닝(신규)
    ui.output_ui("page0_ui"),
    # 페이지1
    ui.output_ui("page1_ui"),
    # 페이지2
    ui.output_ui("page2_ui"),
)

# ------------------------
# 3) 서버
# ------------------------
def server(input, output, session):

    # ===== 공통 유틸 =====
    def guess_zoom(span_deg: float) -> int:
        if span_deg <= 0.01: return 15
        if span_deg <= 0.02: return 14
        if span_deg <= 0.05: return 13
        if span_deg <= 0.10: return 12
        if span_deg <= 0.20: return 11
        if span_deg <= 0.40: return 10
        return 9

    def robust_minmax(s: pd.Series) -> pd.Series:
        s = pd.to_numeric(s, errors="coerce")
        q_low, q_high = s.quantile(0.05), s.quantile(0.95)
        denom = (q_high - q_low)
        out = (s - q_low) / (denom if pd.notnull(denom) and denom != 0 else 1)
        return out.clip(0, 1).fillna(0)

    # 범례(legend) 공통 배치 옵션: 지도 안쪽 오른쪽 위(2,3 느낌)
    LEGEND_KW = dict(
        orientation="h",
        x=0.98, y=0.98,
        xanchor="right", yanchor="top",
        bgcolor="rgba(255,255,255,0.65)",
        bordercolor="rgba(0,0,0,0.25)",
        borderwidth=1,
        font=dict(size=11)
    )

    # 페이지 선택
    @reactive.Calc
    def page_selected():
        return input.page_select()

    # ====== 페이지0: 가중치 튜닝 ======
    # @output 데코레이터를 제거하고, 함수 정의를 server 함수 안으로 이동합니다.
    @render.ui
    def page0_ui():
        if page_selected() != "가중치 튜닝":
            return ui.HTML("")
        return ui.div(
            ui.row(
                ui.column(3,
                    ui.div({"class":"panel"},
                        ui.h4("가중치 설정 (합계 100으로 자동 보정)"),
                        ui.div({"class":"small-label"}, "모든 슬라이더 기본 1.0 (0.5~1.5, 0.25 간격)"),
                        ui.input_slider("w_lamps",   "근처 가로등수 가중치",      min=0.5, max=1.5, value=1.0, step=0.25),
                        ui.input_slider("w_cctv",    "근처 CCTV개수 가중치",      min=0.5, max=1.5, value=1.0, step=0.25),
                        ui.input_slider("w_sch_cnt", "주변 학교 수 가중치",       min=0.5, max=1.5, value=1.0, step=0.25),
                        ui.input_slider("w_sch_dst", "가장 가까운 학교 거리 가중치", min=0.5, max=1.5, value=1.0, step=0.25),
                        ui.input_slider("w_light",   "광원 등급 가중치",          min=0.5, max=1.5, value=1.0, step=0.25),
                        ui.input_slider("w_escoot",  "근처 킥라니주차장수 가중치", min=0.5, max=1.5, value=1.0, step=0.25),
                        ui.input_slider("w_store",   "상점_300m 가중치",          min=0.5, max=1.5, value=1.0, step=0.25),
                    )
                ),
                ui.column(9,
                    ui.div({"class":"map-card"},
                        output_widget("map_weight", height=f"{MAP_HEIGHT}px")
                    )
                )
            )
        )

    # 슬라이더로부터 정규화된 가중치 계산(합 100)
    @reactive.Calc
    def weights_norm():
        mult = {
            "lamps":   input.w_lamps(),
            "cctv":    input.w_cctv(),
            "sch_cnt": input.w_sch_cnt(),
            "sch_dst": input.w_sch_dst(),
            "light":   input.w_light(),
            "escoot":  input.w_escoot(),
            "store":   input.w_store(),
        }
        vals = pd.Series(mult, dtype="float64")
        total = float(vals.sum()) if float(vals.sum()) != 0 else 1.0
        norm = (vals / total) * 100.0
        return norm.to_dict()

    # 가중치 반영하여 위험도(100점) 계산한 DF
    @reactive.Calc
    def scored_df():
        if len(df_lamp) == 0:
            return df_lamp.copy()

        df_new = df_lamp.copy()

        # 항목별 위험도(0~1)
        r_lamps   = 1 - robust_minmax(df_new.get("근처 가로등수"))
        r_cctv    = 1 - robust_minmax(df_new.get("근처 CCTV개수"))
        r_sch_cnt =      robust_minmax(df_new.get("주변 학교 수"))
        r_sch_dst = 1 - robust_minmax(df_new.get("가장 가까운 학교와의 거리"))
        r_light   = 1 - (pd.to_numeric(df_new.get("광원 등급"), errors="coerce") / 5.0)
        r_light   = r_light.clip(0,1).fillna(0)
        r_escoot  =      robust_minmax(df_new.get("근처 킥라니주차장개수"))
        r_store   =      robust_minmax(df_new.get("상점_300m"))

        W = weights_norm()  # 합계 100

        # 최종 위험도(100점)
        df_new["위험도(100점)"] = (
            r_lamps   * W["lamps"]   +
            r_cctv    * W["cctv"]    +
            r_sch_cnt * W["sch_cnt"] +
            r_sch_dst * W["sch_dst"] +
            r_light   * W["light"]   +
            r_escoot  * W["escoot"]  +
            r_store   * W["store"]
        ).round(2)

        return df_new

    # @output 데코레이터를 제거하고, 함수 정의를 server 함수 안으로 이동합니다.
    @render_widget
    def map_weight():
        base = scored_df()

        # 중심/줌
        if len(base) == 0:
            center_lat, center_lon = 36.815, 127.113
            zoom = 12
        else:
            center_lat, center_lon = base["위도"].mean(), base["경도"].mean()
            span = max(base["위도"].max()-base["위도"].min(), base["경도"].max()-base["경도"].min())
            zoom = guess_zoom(span)

        # 구간 분류
        df_risk  = base[base["위험도(100점)"] >= 55]
        df_safe  = base[base["위험도(100점)"] <= 25]
        df_mid   = base[(base["위험도(100점)"] > 25) & (base["위험도(100점)"] < 55)]

        fig = go.Figure()

        # 중간구역
        if len(df_mid):
            fig.add_trace(go.Scattermapbox(
                lat=df_mid["위도"], lon=df_mid["경도"],
                mode="markers", marker=dict(size=8, color="#60a5fa", opacity=0.55),
                text=("중간구역"
                    + "<br>위험도: " + df_mid["위험도(100점)"].astype(str)
                    + "<br>설치형태: " + base.get("설치형태", pd.Series(['-']*len(base))).astype(str).reindex(df_mid.index).fillna("-").astype(str)
                ),
                hovertemplate="%{text}<extra></extra>", name="중간구역(25~54.99)"
            ))
        # 위험구역
        if len(df_risk):
            fig.add_trace(go.Scattermapbox(
                lat=df_risk["위도"], lon=df_risk["경도"],
                mode="markers", marker=dict(size=10, color="#fb923c", opacity=0.7),
                text=("위험구역"
                    + "<br>위험도: " + df_risk["위험도(100점)"].astype(str)
                    + "<br>설치형태: " + base.get("설치형태", pd.Series(['-']*len(base))).astype(str).reindex(df_risk.index).fillna("-").astype(str)
                ),
                hovertemplate="%{text}<extra></extra>", name="위험구역(≥55)"
            ))
        # 안전구역
        if len(df_safe):
            fig.add_trace(go.Scattermapbox(
                lat=df_safe["위도"], lon=df_safe["경도"],
                mode="markers", marker=dict(size=10, color="#f472b6", opacity=0.7),
                text=("안전구역"
                    + "<br>위험도: " + df_safe["위험도(100점)"].astype(str)
                    + "<br>설치형태: " + base.get("설치형태", pd.Series(['-']*len(base))).astype(str).reindex(df_safe.index).fillna("-").astype(str)
                ),
                hovertemplate="%{text}<extra></extra>", name="안전구역(≤25)"
            ))

        fig.update_layout(
            mapbox_style="carto-positron",
            mapbox_zoom=zoom,
            mapbox_center={"lat": center_lat, "lon": center_lon},
            height=MAP_HEIGHT,
            margin={"r":0,"t":0,"l":0,"b":0},
            legend=LEGEND_KW
        )
        return fig

    # ====== 페이지1: 기존 위험요소 + 신규 바 그래프 ======
    # @output 데코레이터를 제거하고, 함수 정의를 server 함수 안으로 이동합니다.
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
                        output_widget("map_plot", height=f"{MAP_HEIGHT}px"),
                    )
                )
            ),
            ui.row(
                ui.column(12,
                    ui.div({"class":"map-card"},
                        ui.h4("위험/안전 지역 위험요인 평균 비교"),
                        output_widget("risk_factors_bar_chart", height="400px"),
                    )
                )
            )
        )

    # 지도를 렌더링하는 함수
    # @output 데코레이터를 제거하고, 함수 정의를 server 함수 안으로 이동합니다.
    @render_widget
    def map_plot():
        # 중심/줌
        if len(df_lamp) == 0:
            center_lat, center_lon = 36.815, 127.113  # 천안 기본
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
            mapbox_style="carto-positron",  # 토큰 불필요
            mapbox_zoom=zoom,
            mapbox_center={"lat": center_lat, "lon": center_lon},
            height=MAP_HEIGHT,
            margin={"r":0,"t":0,"l":0,"b":0},
            legend=LEGEND_KW
        )
        return fig

    # 위험 요인을 막대 그래프로 렌더링하는 함수 (신규)
    # @output 데코레이터를 제거하고, 함수 정의를 server 함수 안으로 이동합니다.
    @render_widget
    def risk_factors_bar_chart():
        base = scored_df()

        # 위험도 55점 이상과 35점 이하 데이터를 필터링
        df_high_risk = base[base['위험도(100점)'] >= 55]
        df_low_risk = base[base['위험도(100점)'] <= 35]

        # 비교할 위험 요인 컬럼 목록
        factors = [
            '근처 가로등수', '근처 CCTV개수', '주변 학교 수',
            '가장 가까운 학교와의 거리', '광원 등급',
            '근처 킥라니주차장개수', '상점_300m'
        ]

        # 각 그룹의 평균값 계산
        if not df_high_risk.empty:
            high_risk_avg = df_high_risk[factors].mean().tolist()
        else:
            high_risk_avg = [0] * len(factors)

        if not df_low_risk.empty:
            low_risk_avg = df_low_risk[factors].mean().tolist()
        else:
            low_risk_avg = [0] * len(factors)

        fig = go.Figure()

        # 고위험 그룹 막대 추가
        fig.add_trace(go.Bar(
            x=factors,
            y=high_risk_avg,
            name='위험 지역 (≥55점)',
            marker_color='#fb923c'
        ))

        # 저위험 그룹 막대 추가
        fig.add_trace(go.Bar(
            x=factors,
            y=low_risk_avg,
            name='안전 지역 (≤35점)',
            marker_color='#f472b6'
        ))

        fig.update_layout(
            barmode='group',
            yaxis_title="평균 값",
            xaxis_tickangle=-45,
            height=400,
            margin={"r":10,"t":10,"l":10,"b":10},
            legend=dict(x=0.01, y=0.99, xanchor='left', yanchor='top'),
            # x축 라벨을 더 읽기 쉽게 변경
            xaxis=dict(
                tickvals=factors,
                ticktext=[
                    '가로등수', 'CCTV수', '학교 수',
                    '학교 거리', '광원 등급',
                    '킥라니수', '상점수'
                ]
            )
        )

        return fig

    # ====== 페이지2: 기존 위험구역/안전구역 ======
    # @output 데코레이터를 제거하고, 함수 정의를 server 함수 안으로 이동합니다.
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

    # @output 데코레이터를 제거하고, 함수 정의를 server 함수 안으로 이동합니다.
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
            mapbox_style="carto-positron",
            mapbox_zoom=zoom,
            mapbox_center={"lat": center_lat, "lon": center_lon},
            height=MAP_HEIGHT,
            margin={"r":0,"t":0,"l":0,"b":0},
            legend=LEGEND_KW
        )
        return fig

    # 이제 위에서 정의한 모든 함수들을 output에 할당합니다.
    output.page0_ui = page0_ui
    output.map_weight = map_weight
    output.page1_ui = page1_ui
    output.map_plot = map_plot
    output.risk_factors_bar_chart = risk_factors_bar_chart
    output.page2_ui = page2_ui
    output.map_chan = map_chan

# ------------------------
# 4) 앱 실행
# ------------------------
app = App(app_ui, server)