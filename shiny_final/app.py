from shiny import App, ui, reactive, render
from shinywidgets import output_widget, render_widget
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import requests
import math

# ------------------------
# 1) 데이터 로드
# ------------------------
df_lamp = pd.read_csv('가로등위험도최종데이터.csv')
df_cctv = pd.read_csv('cctv최종데이터.csv')
df_school = pd.read_csv('학교최종데이터.csv')  # lat, lon 컬럼
df_kickrani = pd.read_excel('kickrani.xlsx', header=1)
df_store = pd.read_csv('상권최종데이터.csv')
all_zone = pd.read_csv("all_zone.csv", encoding="cp949")

def _clean_latlon(df, lat, lon):
    df[lat] = pd.to_numeric(df[lat], errors="coerce")
    df[lon] = pd.to_numeric(df[lon], errors="coerce")
    return df.dropna(subset=[lat, lon])

df_lamp   = _clean_latlon(df_lamp, "위도", "경도")
df_cctv   = _clean_latlon(df_cctv, "위도", "경도")
df_store  = _clean_latlon(df_store, "위도", "경도")
all_zone  = _clean_latlon(all_zone, "위도", "경도")
df_school = df_school.rename(columns={"lat":"위도", "lon":"경도"})
df_school = _clean_latlon(df_school, "위도", "경도")
df_kickrani = _clean_latlon(df_kickrani, "위도", "경도")

MAP_HEIGHT = 720

# ------------------------
# 2) 팔레트/도움함수
# ------------------------
DASH_FONT = "Pretendard, Pretendard Variable, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Noto Sans KR', 'Apple SD Gothic Neo', 'Malgun Gothic', 'Helvetica Neue', Arial, sans-serif"

BASE_CLUSTER_COLORS = [
    "#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00",
    "#a65628", "#f781bf", "#999999", "#66c2a5", "#e78ac3",
    "#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e",
    "#8dd3c7", "#fb8072", "#80b1d3", "#b3de69", "#bc80bd"
]
SAFE_CLUSTER_COLORS = [
    "#10b981", "#06b6d4", "#3b82f6", "#14b8a6", "#0ea5e9",
    "#34d399", "#22d3ee", "#4ade80", "#2dd4bf", "#60a5fa"
]
OVERRIDE_POINT_COLORS = {9: "#ff1493"}

# ---- 요청: Top3 고정 팔레트 + 고정 이름
RISK_TOP3_COLORS = {1: "#D32F2F", 2: "#F57C00", 3: "#C2185B"}  # 위험 1~3위
SAFE_TOP3_COLORS = {1: "#388E3C", 2: "#7CB342", 3: "#009688"}  # 안전 1~3위
RISK_TOP3_NAMES  = {1: "주택존", 2: "상권존", 3: "대학존"}
SAFE_TOP3_NAMES  = {1: "마천루존", 2: "공터존", 3: "노피플존"}

def cluster_point_color(idx: int) -> str:
    if idx in OVERRIDE_POINT_COLORS:
        return OVERRIDE_POINT_COLORS[idx]
    return BASE_CLUSTER_COLORS[int(idx) % len(BASE_CLUSTER_COLORS)]

def cluster_safe_color(idx: int) -> str:
    return SAFE_CLUSTER_COLORS[int(idx) % len(SAFE_CLUSTER_COLORS)]

def hex_to_rgb(hex_color: str):
    hex_color = hex_color.lstrip("#")
    return int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)

def rgba_str(hex_color: str, alpha: float) -> str:
    r, g, b = hex_to_rgb(hex_color)
    return f"rgba({r},{g},{b},{alpha})"

# ---- 경로 페이지 유틸: 거리/색상 ----
def haversine(lat1, lon1, lat2, lon2):
    R = 6371000.0
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    c = 2*math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

def get_risk_color(risk_score, max_value):
    if not max_value or max_value == 0:
        return 'rgb(0,0,255)'
    normalized_score = max(0.0, min(1.0, float(risk_score)/float(max_value)))
    processed = normalized_score ** 2
    red = int(255 * processed)
    blue = int(255 * (1 - processed))
    return f"rgb({red},0,{blue})"

# ---- 출발/도착 Preset ----
ROUTE_PLACES = {
    "천안역": (36.8101, 127.1464),
    "천안시청": (36.8150, 127.1132),
    "백석대": (36.8383, 127.1485),
    "단국대": (36.8389, 127.1831)
}

# ------------------------
# 3) UI (다크모드 + 스타일)
# ------------------------
BASE_STYLES = ui.tags.style(f"""
:root{{
  --bg:#0b0f14; --card:#0f172a; --muted:#94a3b8; --radius:14px; --shadow:0 10px 25px rgba(0,0,0,.35);
  --text:#e2e8f0; --border:#334155; --chip-bg:#1f2937; --chip-text:#e2e8f0; --font:{DASH_FONT};
}}
html,body{{ background:var(--bg); color:var(--text); font-family:var(--font); }}
.header{{background:var(--bg);border-bottom:1px solid #1f2733;padding:12px 20px;margin-bottom:10px;}}
.h1{{margin:0;font-size:22px;font-weight:800; font-family:var(--font);}}
.h2{{margin:2px 0 0;color:var(--muted);font-size:13px}}
.map-card{{background:var(--card);border:1px solid var(--border);border-radius:var(--radius);box-shadow:var(--shadow);padding:12px;overflow:hidden;position:relative;}}
.small-label{{color:var(--muted);font-size:12px;margin:2px 0 8px}}

/* 네비게이션 */
.nav-bar{{display:flex; gap:16px; align-items:center; flex-wrap:wrap; margin:16px 0;}}
.nav-btn{{ 
  background:linear-gradient(180deg,#f06565,#e55353); color:#fff; 
  border-radius:12px; font-weight:800; height:44px; padding:0 18px; 
  border:2px solid transparent; box-shadow:0 6px 14px rgba(229,83,83,.25);
}}
.nav-btn:active{{ transform:translateY(1px); }}
.nav-btn:hover{{ filter:brightness(0.97); }}

/* 슬라이더 행 */
.scale-row{{
  display:flex; gap:14px; align-items:stretch; padding:8px 8px 12px 8px;
  overflow-x:auto; white-space:nowrap;
}}
.scale-card{{
  display:inline-block; white-space:normal;
  min-width:220px; max-width:240px;
  background:var(--card); border:1.5px solid var(--border); border-radius:10px;
  padding:8px 10px; box-shadow:0 4px 10px rgba(0,0,0,.25);
}}
.scale-card .shiny-input-container{{margin-bottom:0;}}
.scale-title{{font-weight:700; margin:2px 0 6px 0;}}

/* page0/page3: 지도 + 오른쪽 패널 */
.page0-wrap, .page3-wrap{{ display:flex; gap:16px; align-items:stretch; }}
.map-panel{{ flex:1 1 0%; min-width:0; }}
.filter-panel{{ width:320px; background:var(--card); border:1.5px solid var(--border); border-radius:10px; padding:12px; height:{MAP_HEIGHT}px; overflow:auto; }}

/* page2: 지도 + 카드 */
.page2-wrap{{ display:flex; gap:16px; align-items:stretch; }}
.rank-panel{{ width:340px; display:flex; flex-direction:column; gap:12px; max-height:720px; overflow:auto; }}

/* 카드 */
.rank-card{{
  position:relative; background:var(--card); border:1.5px solid var(--border);
  border-radius:12px; box-shadow:0 6px 16px rgba(0,0,0,.35);
  padding:10px;
}}
.rank-title{{ font-weight:800; font-size:14px; margin-bottom:4px; }}
.rank-score{{ font-size:22px; font-weight:900; }}
.rank-sub{{ color:var(--muted); font-size:12px; margin-top:6px; }}

.rank-badge{{
  position:absolute; right:10px; top:10px; padding:2px 8px; border-radius:9999px;
  font-weight:900; font-size:11px; color:#fff; opacity:.95;
}}
.badge-risk{{ background:#ef4444; }}
.badge-safe{{ background:#10b981; }}

.chips{{ display:flex; gap:6px; flex-wrap:wrap; margin-top:6px; }}
.chip{{ background:var(--chip-bg); color:var(--chip-text); border-radius:9999px; padding:2px 8px; font-size:12px; font-weight:700; }}

/* page3 전용: 색상 그라데이션 바 */
.legend-card{{ 
  margin-top:14px; border:1.5px solid var(--border); border-radius:10px; 
  padding:10px; 
}}
.legend-title{{ font-weight:800; margin-bottom:8px; }}
.legend-wrap{{ display:flex; align-items:center; gap:12px; }}
.legend-bar{{ 
  width:18px; height:180px; 
  background: linear-gradient(180deg, #ef4444 0%, #0000ff 100%);
  border-radius:10px; box-shadow: inset 0 0 0 1px rgba(255,255,255,.08);
}}
.legend-labels{{ 
  display:flex; flex-direction:column; justify-content:space-between; 
  height:180px; font-size:12px; color:var(--muted);
}}
.legend-labels .top{{ color:#ef4444; font-weight:800; }}
.legend-labels .bottom{{ color:#60a5fa; font-weight:800; }}

@media (max-width: 1200px){{
  .page0-wrap, .page3-wrap{{ flex-direction:column; }}
  .filter-panel{{ width:auto; height:auto; }}
  .page2-wrap{{ flex-direction:column; }}
  .rank-panel{{ width:auto; max-height:none; }}
}}

/* Plotly 컨테이너 가로폭 채우기 */
.map-panel .js-plotly-plot, .map-panel .plotly, .map-panel .plot-container {{
  width: 100% !important;
}}
""")

# ▶ Plotly 리사이즈 고정
RESIZE_FIX = ui.tags.script("""
(function(){
  function resizeAll(){
    try {
      if (window.Plotly) {
        document.querySelectorAll('.js-plotly-plot').forEach(function(el){
          try { Plotly.Plots.resize(el); } catch(e){}
        });
      }
    } catch(e){}
    try { window.dispatchEvent(new Event('resize')); } catch(e){}
  }
  document.addEventListener('click', function(ev){
    if (ev.target && ev.target.closest('.nav-btn')) {
      setTimeout(resizeAll, 60);
      setTimeout(resizeAll, 220);
    }
  }, true);
  const obs = new MutationObserver(function(muts){
    let need = false;
    muts.forEach(m=>{
      m.addedNodes.forEach(n=>{
        if (n.nodeType===1 && (n.classList.contains('js-plotly-plot') || n.querySelector?.('.js-plotly-plot'))) need = true;
      });
    });
    if (need){ setTimeout(resizeAll, 80); }
  });
  obs.observe(document.body, {childList:true, subtree:true});
  window.addEventListener('load', function(){ setTimeout(resizeAll, 120); });
})();
""")

app_ui = ui.page_fluid(
    ui.head_content(BASE_STYLES, RESIZE_FIX),

    ui.div({"class":"header"},
        ui.div({"class":"h1"}, "안전한 야간운전을 위한 위험구간 알림서비스"),
        ui.div({"class":"h2"}, "🚗 🚙 🛻 🚐 🚕 🚖 🚓 🚔 🚑 🚒 🚌 🚎 🚍 🚛 🚚 🚜 🏎️ 🚘 🚖 🚔 🚍 🚙 🚗")
    ),

    # 네비게이션
    ui.div({"class":"nav-bar"},
        ui.input_action_button("nav_home",  "안전 등급 지도", class_="nav-btn"),
        ui.input_action_button("nav_feat",  "시설위치",   class_="nav-btn"),
        ui.input_action_button("nav_zone",  "위험구역 및 안전구역", class_="nav-btn"),
        ui.input_action_button("nav_route", "야간운전 위험구간 알림", class_="nav-btn"),
    ),

    ui.output_ui("page0_ui"),
    ui.output_ui("page1_ui"),
    ui.output_ui("page2_ui"),
    ui.output_ui("page3_ui"),
)

# ------------------------
# 4) 서버
# ------------------------
def server(input, output, session):

    # --- 페이지 전환 ---
    current_page = reactive.Value("가중치 튜닝")

    @reactive.Effect
    @reactive.event(input.nav_home)
    def _go_home(): current_page.set("가중치 튜닝")

    @reactive.Effect
    @reactive.event(input.nav_feat)
    def _go_feat(): current_page.set("위험요소")

    @reactive.Effect
    @reactive.event(input.nav_zone)
    def _go_zone(): current_page.set("위험구역 및 안전구역")

    @reactive.Effect
    @reactive.event(input.nav_route)
    def _go_route(): current_page.set("야간운전 위험구간 알림")

    @reactive.Calc
    def page_selected():
        return current_page.get()

    # ---- 유틸 ----
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

    # ---- 가중치(합 100 정규화) ----
    @reactive.Calc
    def weights_norm():
        def getv(name, default=1.0):
            return getattr(input, name)() if name in input else default
        mult = {
            "lamps":  getv("w_lamps"),
            "cctv":   getv("w_cctv"),
            "sch_cnt":getv("w_sch_cnt"),
            "sch_dst":getv("w_sch_dst"),
            "light":  getv("w_light"),
            "escoot": getv("w_escoot"),
            "store":  getv("w_store"),
        }
        vals = pd.Series(mult, dtype="float64")
        total = float(vals.sum()) if float(vals.sum()) != 0 else 1.0
        norm = (vals / total) * 100.0
        return norm.to_dict()

    # ---- 점수 계산 ----
    @reactive.Calc
    def scored_df():
        if len(df_lamp) == 0:
            return df_lamp.copy()

        df_new = df_lamp.copy()
        r_lamps   = 1 - robust_minmax(df_new.get("근처 가로등수"))
        r_cctv    = 1 - robust_minmax(df_new.get("근처 CCTV개수"))
        r_sch_cnt =      robust_minmax(df_new.get("주변 학교 수"))
        r_sch_dst = 1 - robust_minmax(df_new.get("가장 가까운 학교와의 거리"))
        r_light   = 1 - (pd.to_numeric(df_new.get("광원 등급"), errors="coerce") / 5.0)
        r_light   = r_light.clip(0,1).fillna(0)
        r_escoot  =      robust_minmax(df_new.get("근처 킥라니주차장개수"))
        r_store   =      robust_minmax(df_new.get("상점_300m"))

        W = weights_norm()
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

    # ===== KMeans(+실루엣) 일반화: 정렬방향 선택 가능 =====
    def cluster_kmeans_auto(df_in: pd.DataFrame, k_min=5, k_max=10, sort_ascending=False, random_state=42):
        df_in = df_in.copy()
        if len(df_in) < 2:
            df_in["cluster"] = 0
            cen = pd.DataFrame({"cluster":[0], "위도":[df_in["위도"].mean()], "경도":[df_in["경도"].mean()]})
            summary = pd.DataFrame([{"cluster":0,"avg":df_in["위험도(100점)"].mean(),"n":len(df_in)}])
            return df_in, cen, 1, summary, {0:1}

        # 좌표 투영
        XY = None
        transformer_inv = None
        try:
            from pyproj import Transformer
            transformer_fwd = Transformer.from_crs(4326, 5179, always_xy=True)
            transformer_inv = Transformer.from_crs(5179, 4326, always_xy=True)
            x, y = transformer_fwd.transform(df_in["경도"].values, df_in["위도"].values)
            XY = np.c_[x, y]
        except Exception:
            lat0_deg = float(df_in["위도"].mean()); lat0 = np.deg2rad(lat0_deg)
            x = (df_in["경도"].values - float(df_in["경도"].mean())) * np.cos(lat0) * 111_320.0
            y = (df_in["위도"].values - lat0_deg) * 110_540.0
            XY = np.c_[x, y]

        n = len(df_in)
        k_candidates = [k for k in range(k_min, k_max+1) if k < n]
        if not k_candidates:
            k_candidates = [max(2, min(3, n-1))]

        best_k, best_score, best_labels, best_km = None, -1, None, None
        for k in k_candidates:
            km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
            labels = km.fit_predict(XY)
            if len(set(labels)) < 2:
                continue
            s = silhouette_score(XY, labels)
            if s > best_score:
                best_k, best_score, best_labels, best_km = k, s, labels, km

        if best_labels is None:
            k = k_candidates[0]
            best_km = KMeans(n_clusters=k, n_init=10, random_state=random_state).fit(XY)
            best_labels = best_km.labels_
            best_k = k

        df_in["cluster"] = best_labels

        centers_xy = best_km.cluster_centers_
        if transformer_inv is not None:
            cen_lon, cen_lat = transformer_inv.transform(centers_xy[:,0], centers_xy[:,1])
        else:
            lon0 = float(df_in["경도"].mean()); lat0_deg = float(df_in["위도"].mean()); lat0 = np.deg2rad(lat0_deg)
            cen_lon = lon0 + (centers_xy[:,0] / (np.cos(lat0) * 111_320.0))
            cen_lat = lat0_deg + (centers_xy[:,1] / 110_540.0)

        centroids = pd.DataFrame({"cluster": range(best_k), "위도": cen_lat, "경도": cen_lon})

        summary = (
            df_in.groupby("cluster")["위험도(100점)"]
            .agg(["mean","count"]).round({"mean":1})
            .rename(columns={"mean":"avg","count":"n"})
            .reset_index()
            .sort_values("avg", ascending=sort_ascending)
            .reset_index(drop=True)
        )
        rank_map = {int(row.cluster): i+1 for i, row in summary.iterrows()}
        return df_in, centroids, best_k, summary, rank_map

    # ===== Concave Hull(알파셰이프) → 폴리곤 =====
    def concave_polygons(lons, lats, alpha=None):
        lons = np.asarray(lons); lats = np.asarray(lats)
        if len(lons) == 0:
            return []
        try:
            from pyproj import Transformer
            tx_fwd = Transformer.from_crs(4326, 5179, always_xy=True)
            tx_inv = Transformer.from_crs(5179, 4326, always_xy=True)
            x, y = tx_fwd.transform(lons, lats)
            pts = np.column_stack([x, y])

            if len(pts) <= 2:
                return []

            try:
                import alphashape
                alpha = alphashape.optimizealpha(pts) if alpha is None else alpha
                ashape = alphashape.alphashape(pts, alpha)
                if ashape.is_empty:
                    return []
                geoms = []
                if ashape.geom_type == "Polygon":
                    geoms = [ashape]
                elif ashape.geom_type == "MultiPolygon":
                    geoms = list(ashape.geoms)
                else:
                    geoms = [ashape.buffer(20.0)]
                out = []
                for g in geoms:
                    ex_x, ex_y = g.exterior.xy
                    lon, lat = tx_inv.transform(ex_x, ex_y)
                    out.append((list(lon), list(lat)))
                return out
            except Exception:
                from scipy.spatial import ConvexHull
                hull = ConvexHull(pts)
                order = list(hull.vertices) + [hull.vertices[0]]
                lon, lat = tx_inv.transform(pts[order,0], pts[order,1])
                return [(lon.tolist(), lat.tolist())]
        except Exception:
            return []

    # ===== 페이지0: 가중치 튜닝 =====
    MW_OPTIONS = [
        ("중간구역 (blue)", "중간구역"),
        ("위험구역 (orange)", "위험구역"),
        ("안전구역 (pink)", "안전구역"),
        ("사고다발지역 (red)", "사고다발지역"),
    ]

    @output
    @render.ui
    def page0_ui():
        if page_selected() != "가중치 튜닝":
            return ui.HTML("")
        return ui.div(
            ui.div({"class":"map-card"},
                ui.div({"class":"scale-row"},
                    ui.div({"class":"scale-card"}, ui.input_slider("w_lamps",   ui.tags.div({"class":"scale-title"},"근처 가로등수"),   0, 1, 1.0, step=0.25)),
                    ui.div({"class":"scale-card"}, ui.input_slider("w_cctv",    ui.tags.div({"class":"scale-title"},"근처 CCTV개수"),   0, 1, 1.0, step=0.25)),
                    ui.div({"class":"scale-card"}, ui.input_slider("w_sch_cnt", ui.tags.div({"class":"scale-title"},"주변 학교 수"),    0, 1, 1.0, step=0.25)),
                    ui.div({"class":"scale-card"}, ui.input_slider("w_sch_dst", ui.tags.div({"class":"scale-title"},"가까운 학교 거리"), 0, 1, 1.0, step=0.25)),
                    ui.div({"class":"scale-card"}, ui.input_slider("w_light",   ui.tags.div({"class":"scale-title"},"광원 등급"),       0, 1, 1.0, step=0.25)),
                    ui.div({"class":"scale-card"}, ui.input_slider("w_escoot",  ui.tags.div({"class":"scale-title"},"킥보드주차장"),     0, 1, 1.0, step=0.25)),
                    ui.div({"class":"scale-card"}, ui.input_slider("w_store",   ui.tags.div({"class":"scale-title"},"상점_300m"),       0, 1, 1.0, step=0.25)),
                ),
                ui.div({"class":"page0-wrap"},
                    ui.div({"class":"map-panel"}, output_widget("map_weight", height=f"{MAP_HEIGHT}px")),
                    ui.div({"class":"filter-panel"},
                        ui.div({"class":"small-label"}, "표시할 범주 (기본: 모두 ON)"),
                        ui.input_checkbox_group(
                            "mw_layers", None,
                            choices=[lab for (lab, _val) in MW_OPTIONS],
                            selected=[lab for (lab, _val) in MW_OPTIONS]
                        )
                    )
                )
            )
        )

    @output
    @render_widget
    def map_weight():
        base = scored_df()
        if len(base) == 0:
            center_lat, center_lon, zoom = 36.815, 127.113, 12
        else:
            center_lat, center_lon = base["위도"].mean(), base["경도"].mean()
            span = max(base["위도"].max()-base["위도"].min(), base["경도"].max()-base["경도"].min())
            zoom = guess_zoom(span)

        df_risk = base[base["위험도(100점)"] >= 55]
        df_safe = base[base["위험도(100점)"] <= 25]
        df_mid  = base[(base["위험도(100점)"] > 25) & (base["위험도(100점)"] < 55)]
        설치형태_base = base.get("설치형태", pd.Series(["-"]*len(base), index=base.index)).astype(str)

        labels_selected = set(input.mw_layers() if "mw_layers" in input else [lab for (lab,_v) in MW_OPTIONS])
        selected = set(val for (lab, val) in MW_OPTIONS if lab in labels_selected)

        fig = go.Figure()
        if "중간구역" in selected and len(df_mid):
            fig.add_trace(go.Scattermapbox(
                lat=df_mid["위도"], lon=df_mid["경도"], mode="markers",
                marker=dict(size=9, color="#60a5fa", opacity=0.6),
                text=("중간구역"
                      + "<br>설치형태: " + 설치형태_base.reindex(df_mid.index, fill_value="-")
                      + "<br>위험도: " + df_mid["위험도(100점)"].astype(str)),
                hovertemplate="%{text}<extra></extra>", name="중간구역"
            ))
        if "위험구역" in selected and len(df_risk):
            fig.add_trace(go.Scattermapbox(
                lat=df_risk["위도"], lon=df_risk["경도"], mode="markers",
                marker=dict(size=11, color="#fb923c", opacity=0.75),
                text=("위험구역"
                      + "<br>설치형태: " + 설치형태_base.reindex(df_risk.index, fill_value="-")
                      + "<br>위험도: " + df_risk["위험도(100점)"].astype(str)),
                hovertemplate="%{text}<extra></extra>", name="위험구역"
            ))
        if "안전구역" in selected and len(df_safe):
            fig.add_trace(go.Scattermapbox(
                lat=df_safe["위도"], lon=df_safe["경도"], mode="markers",
                marker=dict(size=11, color="#f472b6", opacity=0.75),
                text=("안전구역"
                      + "<br>설치형태: " + 설치형태_base.reindex(df_safe.index, fill_value="-")
                      + "<br>위험도: " + df_safe["위험도(100점)"].astype(str)),
                hovertemplate="%{text}<extra></extra>", name="안전구역"
            ))
        if "사고다발지역" in selected and len(all_zone):
            fig.add_trace(go.Scattermapbox(
                lat=all_zone["위도"], lon=all_zone["경도"],
                mode="markers", marker=dict(size=12, color="#ef4444", opacity=0.85),
                text=all_zone.get("지점명", "사고다발지역"),
                hovertemplate="%{text}<extra></extra>", name="사고다발지역"
            ))

        fig.update_layout(
            mapbox_style="carto-positron",
            mapbox_zoom=zoom,
            mapbox_center={"lat": center_lat, "lon": center_lon},
            height=MAP_HEIGHT, margin={"r":0,"t":0,"l":0,"b":0},
            legend=dict(orientation="h", y=0.01, x=0.01, font=dict(size=18, color="#ffffff", family=DASH_FONT)),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family=DASH_FONT, color="#ffffff"),
            autosize=True, uirevision="keep"
        )
        return fig

    # ===== 페이지1: 위험요소 =====
    @output
    @render.ui
    def page1_ui():
        if page_selected() != "위험요소":
            return ui.HTML("")
        return ui.div(ui.div({"class":"map-card"}, output_widget("map_plot", height=f"{MAP_HEIGHT}px")))

    @output
    @render_widget
    def map_plot():
        if len(df_lamp) == 0:
            center_lat, center_lon, zoom = 36.815, 127.113, 12
        else:
            center_lat, center_lon = df_lamp["위도"].mean(), df_lamp["경도"].mean()
            span = max(df_lamp["위도"].max()-df_lamp["위도"].min(), df_lamp["경도"].max()-df_lamp["경도"].min())
            zoom = guess_zoom(span)

        fig = go.Figure()
        fig.add_trace(go.Scattermapbox(
            lat=df_lamp['위도'], lon=df_lamp['경도'],
            mode='markers', marker=dict(size=7, color='#f6c945', opacity=0.55),
            text=("가로등"
                  + "<br>설치형태: " + df_lamp.get('설치형태', pd.Series(['-']*len(df_lamp))).astype(str)
                  + "<br>위험도: " + df_lamp.get('위험도(100점)', pd.Series(['-']*len(df_lamp))).astype(str)),
            hovertemplate="%{text}<extra></extra>", name='가로등', visible=True
        ))
        fig.add_trace(go.Scattermapbox(
            lat=df_cctv['위도'], lon=df_cctv['경도'],
            mode='markers', marker=dict(size=10, color='#22c55e', opacity=0.65),
            text="CCTV", hovertemplate="%{text}<extra></extra>", name='CCTV', visible=False
        ))
        fig.add_trace(go.Scattermapbox(
            lat=df_school['위도'], lon=df_school['경도'],
            mode='markers', marker=dict(size=10, color='#8b5cf6', opacity=0.65),
            text=df_school.get('구분', '학교'), hovertemplate="%{text}<extra></extra>", name='학교', visible=False
        ))
        fig.add_trace(go.Scattermapbox(
            lat=df_kickrani['위도'], lon=df_kickrani['경도'],
            mode='markers', marker=dict(size=10, color='#111827', opacity=0.7),
            text="킥보드주차장 주차가능 대수: " + df_kickrani.get('주차가능 대수', pd.Series(['-']*len(df_kickrani))).astype(str),
            hovertemplate="%{text}<extra></extra>", name='킥보드주차장', visible=False
        ))
        fig.add_trace(go.Scattermapbox(
            lat=df_store['위도'], lon=df_store['경도'],
            mode='markers', marker=dict(size=8, color='#0ea5e9', opacity=0.75),
            text=df_store.get('상호명', '상권'), hovertemplate="%{text}<extra></extra>", name='상권', visible=False
        ))

        fig.update_layout(
            updatemenus=[
                dict(
                    type="buttons", direction="right",
                    x=0.01, y=0.99, xanchor="left", yanchor="top",
                    showactive=False,
                    bgcolor="rgba(17,24,39,.9)", font=dict(color="#e2e8f0"),
                    bordercolor="#334155", borderwidth=1, pad={"r":6,"t":4,"l":6,"b":4},
                    buttons=[
                        dict(label="전체 켜기", method="update", args=[{"visible":[True, True, True, True, True]}]),
                        dict(label="전체 끄기", method="update", args=[{"visible":[False, False, False, False, False]}]),
                    ],
                ),
                dict(
                    type="buttons", direction="down",
                    x=0.01, y=0.90, xanchor="left", yanchor="top",
                    showactive=False,
                    bgcolor="rgba(17,24,39,.9)", font=dict(color="#e2e8f0"),
                    bordercolor="#334155", borderwidth=1, pad={"r":6,"t":4,"l":6,"b":4},
                    buttons=[
                        dict(label="가로등",        method="restyle", args=[{"visible": True},  [0]], args2=[{"visible": False}, [0]]),
                        dict(label="CCTV",         method="restyle", args=[{"visible": True},  [1]], args2=[{"visible": False}, [1]]),
                        dict(label="학교",         method="restyle", args=[{"visible": True},  [2]], args2=[{"visible": False}, [2]]),
                        dict(label="킥보드주차장",   method="restyle", args=[{"visible": True},  [3]], args2=[{"visible": False}, [3]]),
                        dict(label="상권",         method="restyle", args=[{"visible": True},  [4]], args2=[{"visible": False}, [4]]),
                    ],
                ),
            ],
            legend=dict(
                title_text="레이어", orientation="h",
                yanchor="bottom", y=0.01, xanchor="left", x=0.01,
                font=dict(color="#e2e8f0", family=DASH_FONT),
                itemclick="toggle", itemdoubleclick="toggleothers"
            ),
            mapbox_style="carto-positron",
            mapbox_zoom=zoom,
            mapbox_center={"lat": center_lat, "lon": center_lon},
            height=MAP_HEIGHT, margin={"r":0,"t":0,"l":0,"b":0},
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e2e8f0", family=DASH_FONT),
            autosize=True, uirevision="keep"
        )
        return fig

    # ===== 클러스터 계산: 위험 Top3 + 안전 Top3 (rank_map 포함) =====
    @reactive.Calc
    def cluster_info():
        base = scored_df()
        if len(base) == 0:
            return {"ok": False}

        df_risk = base[base["위험도(100점)"] >= 55].copy()
        df_safe = base[base["위험도(100점)"] <= 25].copy()

        # 위험: 평균 높은 순 → 상위 3개
        if len(df_risk) >= 2:
            risk_lab, risk_cen, risk_k, risk_sum, risk_rank = cluster_kmeans_auto(
                df_risk, k_min=5, k_max=10, sort_ascending=False
            )
            risk_top3 = set(risk_sum.head(3)["cluster"].astype(int).tolist())
        else:
            risk_lab, risk_cen, risk_sum, risk_rank, risk_top3 = None, None, pd.DataFrame(), {}, set()

        # 안전: 평균 낮은 순 → 상위 3개
        if len(df_safe) >= 2:
            safe_lab, safe_cen, safe_k, safe_sum, safe_rank = cluster_kmeans_auto(
                df_safe, k_min=5, k_max=10, sort_ascending=True
            )
            safe_top3 = set(safe_sum.head(3)["cluster"].astype(int).tolist())
        else:
            safe_lab, safe_cen, safe_sum, safe_rank, safe_top3 = None, None, pd.DataFrame(), {}, set()

        return {
            "ok": True,
            "base": base,
            "risk": {"df_lab": risk_lab, "centroids": risk_cen, "summary": risk_sum, "top3": risk_top3, "rank_map": risk_rank},
            "safe": {"df_lab": safe_lab, "centroids": safe_cen, "summary": safe_sum, "top3": safe_top3, "rank_map": safe_rank},
        }

    # ===== 페이지2: 위험구역 및 안전구역 =====
    @output
    @render.ui
    def page2_ui():
        if page_selected() != "위험구역 및 안전구역":
            return ui.HTML("")
        return ui.div(
            ui.div({"class":"map-card"},
                ui.div({"class":"page2-wrap"},
                    ui.div({"class":"map-panel"}, output_widget("map_chan", height=f"{MAP_HEIGHT}px")),
                    ui.div({"class":"rank-panel"}, ui.output_ui("rankcards"))
                )
            )
        )

    @output
    @render_widget
    def map_chan():
        info = cluster_info()
        base = scored_df()

        if not info.get("ok", False) or len(base) == 0:
            if len(base) == 0:
                center_lat, center_lon, zoom = 36.815, 127.113, 12
            else:
                center_lat, center_lon = base["위도"].mean(), base["경도"].mean()
                span = max(base["위도"].max()-base["위도"].min(), base["경도"].max()-base["경도"].min())
                zoom = guess_zoom(span)
            fig = go.Figure()
            fig.update_layout(
                mapbox_style="carto-positron",
                mapbox_zoom=zoom, mapbox_center={"lat": center_lat, "lon": center_lon},
                height=MAP_HEIGHT, margin={"r":0,"t":0,"l":0,"b":0},
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                autosize=True, uirevision="keep"
            )
            return fig

        center_lat, center_lon = base["위도"].mean(), base["경도"].mean()
        span = max(base["위도"].max()-base["위도"].min(), base["경도"].max()-base["경도"].min())
        zoom = guess_zoom(span)

        fig = go.Figure()

        # (변경) 배경 가로등 — 더 진하게
        fig.add_trace(go.Scattermapbox(
            lat=base["위도"], lon=base["경도"], mode="markers",
            marker=dict(size=8, color="#f6c945", opacity=0.60),
            text=("가로등"
                  + "<br>설치형태: " + base.get("설치형태", pd.Series(['-']*len(base))).astype(str)
                  + "<br>위험도: " + base.get("위험도(100점)", pd.Series(['-']*len(base))).astype(str)),
            hovertemplate="%{text}<extra></extra>", name="가로등(전체)"
        ))
        t_idx = 1
        risk_indices, safe_indices = [], []

        # === 위험 Top3: 점+영역+라벨 ===
        r = info["risk"]
        if r["df_lab"] is not None and len(r["df_lab"]):
            cent = r["centroids"] if r["centroids"] is not None else pd.DataFrame()
            for c, g in r["df_lab"].groupby("cluster"):
                c = int(c)
                if c not in r["top3"]:
                    continue
                rank = r["rank_map"].get(c)  # 1,2,3,...
                col  = RISK_TOP3_COLORS.get(rank, "#D32F2F")
                label_name = RISK_TOP3_NAMES.get(rank, f"위험 {rank}위")

                # 포인트
                fig.add_trace(go.Scattermapbox(
                    lat=g["위도"], lon=g["경도"], mode="markers",
                    marker=dict(size=10, opacity=0.9, color=col),
                    name=f"위험 {rank}위 · {label_name}",
                    text=(f"{label_name}"
                          + "<br>설치형태: " + g["설치형태"].astype(str)
                          + "<br>위험도: " + g["위험도(100점)"].astype(str)),
                    hovertemplate="%{text}<extra></extra>", visible=True
                ))
                risk_indices.append(t_idx); t_idx += 1

                # 영역
                polys = concave_polygons(g["경도"].values, g["위도"].values, alpha=None)
                for (plon, plat) in polys:
                    fig.add_trace(go.Scattermapbox(
                        lon=plon, lat=plat, mode="lines", fill="toself",
                        line=dict(width=2, color=col), fillcolor=rgba_str(col, 0.22),
                        name=f"{label_name} 영역", hoverinfo="skip", visible=True
                    ))
                    risk_indices.append(t_idx); t_idx += 1

                # 라벨(중심)
                row = cent[cent["cluster"]==c]
                if len(row):
                    cy, cx = float(row["위도"].iloc[0]), float(row["경도"].iloc[0])
                    fig.add_trace(go.Scattermapbox(
                        lat=[cy], lon=[cx], mode="text",
                        text=[label_name],
                        textfont=dict(size=18, family=DASH_FONT, color="#111111"),
                        hoverinfo="skip", showlegend=False, visible=True
                    ))
                    risk_indices.append(t_idx); t_idx += 1

        # === 안전 Top3: 점+영역+라벨 ===
        s = info["safe"]
        if s["df_lab"] is not None and len(s["df_lab"]):
            cent = s["centroids"] if s["centroids"] is not None else pd.DataFrame()
            for c, g in s["df_lab"].groupby("cluster"):
                c = int(c)
                if c not in s["top3"]:
                    continue
                rank = s["rank_map"].get(c)
                col  = SAFE_TOP3_COLORS.get(rank, "#388E3C")
                label_name = SAFE_TOP3_NAMES.get(rank, f"안전 {rank}위")

                # 포인트
                fig.add_trace(go.Scattermapbox(
                    lat=g["위도"], lon=g["경도"], mode="markers",
                    marker=dict(size=10, opacity=0.9, color=col),
                    name=f"안전 {rank}위 · {label_name}",
                    text=(f"{label_name}"
                          + "<br>설치형태: " + g["설치형태"].astype(str)
                          + "<br>위험도: " + g["위험도(100점)"].astype(str)),
                    hovertemplate="%{text}<extra></extra>", visible=True
                ))
                safe_indices.append(t_idx); t_idx += 1

                # 영역
                polys = concave_polygons(g["경도"].values, g["위도"].values, alpha=None)
                for (plon, plat) in polys:
                    fig.add_trace(go.Scattermapbox(
                        lon=plon, lat=plat, mode="lines", fill="toself",
                        line=dict(width=2, color=col), fillcolor=rgba_str(col, 0.22),
                        name=f"{label_name} 영역", hoverinfo="skip", visible=True
                    ))
                    safe_indices.append(t_idx); t_idx += 1

                # 라벨(중심)
                row = cent[cent["cluster"]==c]
                if len(row):
                    cy, cx = float(row["위도"].iloc[0]), float(row["경도"].iloc[0])
                    fig.add_trace(go.Scattermapbox(
                        lat=[cy], lon=[cx], mode="text",
                        text=[label_name],
                        textfont=dict(size=18, family=DASH_FONT, color="#111111"),
                        hoverinfo="skip", showlegend=False, visible=True
                    ))
                    safe_indices.append(t_idx); t_idx += 1

        all_indices = [0] + risk_indices + safe_indices

        # 토글
        fig.update_layout(
            updatemenus=[
                dict(
                    type="buttons", direction="right",
                    x=0.01, y=0.99, xanchor="left", yanchor="top",
                    showactive=False,
                    bgcolor="rgba(17,24,39,.9)", font=dict(color="#e2e8f0", family=DASH_FONT),
                    bordercolor="#334155", borderwidth=1, pad={"r":6,"t":4,"l":6,"b":4},
                    buttons=[
                        dict(label="전체 켜기", method="restyle", args=[{"visible": True},  all_indices]),
                        dict(label="전체 끄기", method="restyle", args=[{"visible": False}, all_indices]),
                    ],
                ),
                dict(
                    type="buttons", direction="down",
                    x=0.01, y=0.90, xanchor="left", yanchor="top",
                    showactive=False,
                    bgcolor="rgba(17,24,39,.9)", font=dict(color="#e2e8f0", family=DASH_FONT),
                    bordercolor="#334155", borderwidth=1, pad={"r":6,"t":4,"l":6,"b":4},
                    buttons=[
                        dict(label="가로등",    method="restyle", args=[{"visible": True},  [0]], args2=[{"visible": False}, [0]]),
                        dict(label="위험 Top3", method="restyle", args=[{"visible": True},  risk_indices], args2=[{"visible": False}, risk_indices]),
                        dict(label="안전 Top3", method="restyle", args=[{"visible": True},  safe_indices], args2=[{"visible": False}, safe_indices]),
                    ],
                ),
            ],
            legend=dict(title_text="레이어", orientation="h", yanchor="bottom", y=0.01, xanchor="left", x=0.01),
            mapbox_style="carto-positron",
            mapbox_zoom=zoom, mapbox_center={"lat": center_lat, "lon": center_lon},
            height=MAP_HEIGHT, margin={"r":0,"t":0,"l":0,"b":0},
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            autosize=True, uirevision="keep"
        )
        return fig

    # 우측 카드: 위험 Top3 + 안전 Top3 (총 6장, 이름 치환)
    @output
    @render.ui
    def rankcards():
        info = cluster_info()
        if not info.get("ok", False):
            return ui.div(
                ui.div({"class":"rank-card"},
                    ui.div({"class":"rank-title"}, "클러스터 순위"),
                    ui.div({"class":"rank-sub"}, "표시할 데이터가 부족합니다.")
                )
            )

        cards = []
        # 위험 Top3
        rsum = info["risk"]["summary"]; rrank = info["risk"]["rank_map"]
        if rsum is not None and len(rsum):
            top = rsum.head(3).reset_index(drop=True)
            for _, row in top.iterrows():
                cidx = int(row["cluster"]); rank = rrank.get(cidx, 0)
                name = RISK_TOP3_NAMES.get(rank, f"위험 {rank}위")
                score = float(row["avg"]); n = int(row["n"])
                cards.append(
                    ui.div({"class":"rank-card"},
                        ui.div({"class":"rank-badge badge-risk"}, f"위험 {rank}위"),
                        ui.div({"class":"rank-title"}, f"{name}"),
                        ui.div({"class":"rank-score"}, f"{score:.1f}점"),
                        ui.div({"class":"rank-sub"}, f"표본 {n}개 · 평균 위험도"),
                        ui.div({"class":"chips"},
                            ui.span({"class":"chip"}, f"n={n}"),
                            ui.span({"class":"chip"}, f"rank={rank}")
                        )
                    )
                )
        # 안전 Top3
        ssum = info["safe"]["summary"]; srank = info["safe"]["rank_map"]
        if ssum is not None and len(ssum):
            top = ssum.head(3).reset_index(drop=True)
            for _, row in top.iterrows():
                cidx = int(row["cluster"]); rank = srank.get(cidx, 0)
                name = SAFE_TOP3_NAMES.get(rank, f"안전 {rank}위")
                score = float(row["avg"]); n = int(row["n"])
                cards.append(
                    ui.div({"class":"rank-card"},
                        ui.div({"class":"rank-badge badge-safe"}, f"안전 {rank}위"),
                        ui.div({"class":"rank-title"}, f"{name}"),
                        ui.div({"class":"rank-score"}, f"{score:.1f}점"),
                        ui.div({"class":"rank-sub"}, "표본 {}개 · 평균 위험도(낮을수록 안전)".format(n)),
                        ui.div({"class":"chips"},
                            ui.span({"class":"chip"}, f"n={n}"),
                            ui.span({"class":"chip"}, f"rank={rank}")
                        )
                    )
                )
        return ui.div(*cards)

    # ===== 페이지3: 야간운전 위험구간 알림 =====
    KAKAO_API_KEY = "d222f0f01e3470ce2b8a863cc30b151e"  # 예시 키

    @output
    @render.ui
    def page3_ui():
        if page_selected() != "야간운전 위험구간 알림":
            return ui.HTML("")
        return ui.div(
            ui.div({"class":"map-card"},
                ui.div({"class":"page3-wrap"},
                    ui.div({"class":"map-panel"}, output_widget("map_route", height=f"{MAP_HEIGHT}px")),
                    ui.div({"class":"filter-panel"},
                        ui.div({"class":"small-label"}, "출발지/도착지 선택"),
                        ui.input_select("route_origin", "출발지", list(ROUTE_PLACES.keys()), selected="천안역"),
                        ui.input_select("route_dest",   "도착지", list(ROUTE_PLACES.keys()), selected="단국대"),
                        ui.input_action_button("route_btn", "경로 확인", class_="nav-btn"),
                        # 구간 색상 기준 그라데이션 바
                        ui.div({"class":"legend-card"},
                            ui.div({"class":"legend-title"}, "구간 색상 기준"),
                            ui.div({"class":"legend-wrap"},
                                ui.div({"class":"legend-bar"}),
                                ui.div({"class":"legend-labels"},
                                    ui.div({"class":"top"}, "위험 (빨강)"),
                                    ui.div({"class":"bottom"}, "안전 (파랑)")
                                )
                            )
                        )
                    )
                )
            )
        )

    @output
    @render_widget
    def map_route():
        # 입력 의존성 (버튼 + 셀렉트 변경 시 갱신)
        _ = input.route_origin(); _ = input.route_dest(); _ = input.route_btn()

        df_route = scored_df()
        max_risk_all = float(df_route["위험도(100점)"].max()) if "위험도(100점)" in df_route else 0.0

        origin = ROUTE_PLACES[input.route_origin()]
        dest   = ROUTE_PLACES[input.route_dest()]

        # 카카오 경로 API
        routes_to_show = []
        try:
            url = "https://apis-navi.kakaomobility.com/v1/directions"
            headers = {"Authorization": f"KakaoAK {KAKAO_API_KEY}"}
            params = {
                "origin": f"{origin[1]},{origin[0]}",
                "destination": f"{dest[1]},{dest[0]}",
                "priority": "RECOMMEND",
                "alternatives": "true",
                "car_type": 1
            }
            res = requests.get(url, headers=headers, params=params, timeout=10)
            data = res.json() if res.ok else {}
            routes_to_show = data.get("routes", [])[:3]
        except Exception as e:
            print("카카오 경로 API 실패:", e)

        fig = go.Figure()
        num_segments = 10

        if not routes_to_show:
            fig.update_layout(
                mapbox=dict(style="carto-positron", center=dict(lat=origin[0], lon=origin[1]), zoom=13),
                margin=dict(r=0, t=0, l=0, b=0),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                uirevision="keep", height=MAP_HEIGHT
            )
            return fig

        # 경로별 세그먼트 색상
        for ridx, route in enumerate(routes_to_show, start=1):
            coords_all = []
            for section in route.get("sections", []):
                for road in section.get("roads", []):
                    coords_all.extend(road.get("vertexes", []))

            if not coords_all:
                continue

            lons = coords_all[0::2]
            lats = coords_all[1::2]
            total_points = len(lats)
            if total_points < 2:
                continue

            step = total_points / num_segments

            for i in range(num_segments):
                start_idx = math.floor(i*step)
                end_idx = math.floor((i+1)*step) if i < num_segments-1 else total_points-1

                segment_lats = lats[start_idx:end_idx+1]
                segment_lons = lons[start_idx:end_idx+1]
                if not segment_lats:
                    continue

                c_lat = segment_lats[len(segment_lats)//2]
                c_lon = segment_lons[len(segment_lons)//2]
                search_radius_m = 50.0

                max_risk = 0.0
                for _, row in df_route.iterrows():
                    d = haversine(c_lat, c_lon, float(row["위도"]), float(row["경도"]))
                    if d <= search_radius_m:
                        score = float(row.get("위험도(100점)", 0.0))
                        if score > max_risk:
                            max_risk = score

                color = get_risk_color(max_risk, max_risk_all)

                fig.add_trace(go.Scattermapbox(
                    lat=segment_lats,
                    lon=segment_lons,
                    mode="lines",
                    line=dict(width=6, color=color),
                    name=f"경로 {ridx}",
                    showlegend=False
                ))

        # 출발/도착 마커
        fig.add_trace(go.Scattermapbox(
            lat=[origin[0], dest[0]],
            lon=[origin[1], dest[1]],
            mode="markers+text",
            marker=dict(size=12, color="#111827"),
            text=["출발", "도착"],
            textposition="top center",
            showlegend=False,
            hoverinfo="skip"
        ))

        fig.update_layout(
            mapbox=dict(style="carto-positron", center=dict(lat=origin[0], lon=origin[1]), zoom=13),
            margin=dict(r=0, t=0, l=0, b=0),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            uirevision="keep", height=MAP_HEIGHT
        )
        return fig

# ------------------------
# 5) 앱 실행
# ------------------------
app = App(app_ui, server)
