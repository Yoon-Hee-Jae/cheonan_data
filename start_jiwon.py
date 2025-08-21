import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
from tqdm import tqdm
from sklearn.neighbors import BallTree
from sklearn.preprocessing import MinMaxScaler


# 윈도우: Malgun Gothic / 맥: AppleGothic
plt.rc('font', family='Malgun Gothic')  # 또는 'AppleGothic'
plt.rc('axes', unicode_minus=False)     # 마이너스 기호 깨짐 방지

# 데이터 불러오기
df = pd.read_excel('main_dataset.xlsx') # 교통사고 데이터
df2 = pd.read_excel('소통통계 (1).xlsx') # 교통량 데이터 2024년1월 ~ 2024년 12월

############################################################################################

# 데이터 전처리

############################################################################################

# 가로등 시각화
df3 = pd.read_csv('충청남도 천안시_가로등 현황_20240729.csv') # 가로등 데이터
df3.info() # 결측치 존재 특히, 도로묭주소에 결측치 10000개 존재
df3['설치형태'].unique() # ['LED', 'CML', 'CDM', '나트륨', '메탈', '삼파장', 'CPO', 'CCTV', '써치등']
df3['설치형태'].value_counts()
# 설치형태 cctv 삭제
df3[df3['설치형태']=='CCTV'].index
df3 = df3.drop(index=3628).reset_index(drop=True)

# cctv 시각화
df4 = pd.read_csv('충청남도 천안시_교통정보 CCTV_20220922.csv', encoding='cp949') 
# CCTV의 수가 94개로 너무 적음
df4_1 = df4[['위도','경도']]
# 과속카메라 추가
cctv_2 = pd.read_csv('cctv_2.csv')
cctv_2 = cctv_2[['위도','경도']]
cctv = pd.concat([df4_1,cctv_2], ignore_index=True)
# cctv.to_csv("cctv최종데이터.csv")
# 데이터 재점검 필요

# 학교 
df5 = pd.read_csv('충청남도_학교 현황_20240807.csv', encoding='cp949')
df5.info() # 1243개 결측치 0게
#주소 천안만 필터링
df5['학교_천안포함'] = df5['주소'].str.contains('천안')
df_천안 = df5[df5['학교_천안포함'] == True]
df_천안.info()  #250개
# 학생수가 0명인 학교는 제외
df_천안[df_천안['학생수(명)']==0].shape # 54개
# 학생수가 0인 행의 인덱스 저장
null_index = df_천안[df_천안['학생수(명)']==0].index
df_천안 = df_천안[~df_천안.index.isin(null_index)].reset_index()
# df_천안.to_csv('학교최종데이터.csv')
df5.info()

# 주소를 좌표로 변환하는 함수

KAKAO_KEY = 'd222f0f01e3470ce2b8a863cc30b151e'

def get_lat_lon_by_keyword(keyword):
    url = 'https://dapi.kakao.com/v2/local/search/keyword.json'
    headers = {"Authorization": f"KakaoAK {KAKAO_KEY}"}
    params = {'query': keyword, 'size': 1}
    res = requests.get(url, headers=headers, params=params).json()
    if res['documents']:
        return float(res['documents'][0]['y']), float(res['documents'][0]['x'])
    return None, None

# 주소 컬럼 활용
df_천안['lat'], df_천안['lon'] = zip(*[
    get_lat_lon_by_keyword(addr) for addr in tqdm(df_천안['주소'])
])

# 킥라니 주차장 
df6 = pd.read_excel('kickrani.xlsx')
df6.info()
# 2번쨰 행부터 불러오기
df6 = pd.read_excel("kickrani.xlsx", header=1)


# 이륜차 사고 다발 지역
df7 = pd.read_csv('motorcycle.csv', encoding='cp949')


df7 = df7[df7['시도시군구명'].str.contains('천안시')]
df7['구분'] = '이륜차 사고다발지역'

# 보행자 사고 다발 지역
df8 = pd.read_csv('pedstrians.csv', encoding='cp949')
df8 = df8[df8['시도시군구명'].str.contains('천안시')]
df8['구분'] = '보행자 사고다발지역'

# 음주운전 사고 다발 지역
df9 = pd.read_csv('drunk.csv',encoding='cp949')
df9 = df9[df9['시도시군구명'].str.contains('천안시')]
df9['구분'] = '음주운전 사고다발지역'

# 화물차 사고 다발 지역
df10 =  pd.read_csv('truck.csv',encoding='cp949')
df10 = df10[df10['시도시군구명'].str.contains('천안시')]
df10['구분'] = '화물차 사고다발지역'

# 사고다발지역 데이터프레임 생성
danger_zone = pd.concat([df7, df8], ignore_index=True)
danger_zone = pd.concat([danger_zone, df9], ignore_index=True)
danger_zone = pd.concat([danger_zone, df10], ignore_index=True)

danger_zone['사고다발지id'] = danger_zone['사고다발지id'].astype(str)
danger_zone['연도'] = danger_zone['사고다발지id'].str.extract(r'^(\d{4})')
danger_zone['연도'] = danger_zone['연도'].astype(int)
danger_zone = danger_zone[danger_zone['연도']>=2021].reset_index(drop=True)
danger_zone = danger_zone.sort_values('연도').reset_index(drop=True)

# danger_zone.to_csv('danger_zone.csv', index=False, encoding='cp949')

# 영찬 지원누나 사고 다발 구간 합치기
danger_jiwon = pd.read_csv('jiwon_danger_zone.csv',encoding='cp949')
danger_youngchan = pd.read_csv('youngchan_danger_zone.csv')
danger_youngchan=danger_youngchan.drop('법규위반', axis=1)

danger_zone = pd.concat([danger_zone, danger_youngchan], ignore_index=True)
danger_zone = pd.concat([danger_zone, danger_jiwon], ignore_index=True)

# danger_zone.to_csv('all_zone.csv', index=False, encoding='cp949') # 사고다발구간 최종 데이터프레임
#######################################################################

############################################################################################
# 반경 300m 내 가로등 개수추가

# 가로등 데이터프레임에서 위도 경도 설치형태만 남기기
df3 = df3[['위도','경도','설치형태']]

# 위경도를 라디안으로 변환
coords = np.radians(df3[['위도', '경도']].values)

# BallTree 생성 (haversine 거리 사용)
tree = BallTree(coords, metric='haversine')

# 검색 반경 (300m → km → 라디안)
radius = 0.3 / 6371.0  # 지구 반지름 6371km

# 각 좌표에 대해 반경 내 포인트 개수 검색
counts = tree.query_radius(coords, r=radius, count_only=True)

# 결과 저장 (자기 자신 포함이므로 -1)
df3['근처 가로등수'] = counts - 1

#################################################################################################
# 반경 300m 내 cctv 수 추가

# 위도·경도 → 라디안 변환
lamp_coords_rad = np.radians(df3[['위도', '경도']].values)   # 가로등 좌표
cctv_coords_rad = np.radians(cctv[['위도', '경도']].values)   # CCTV 좌표

# CCTV 좌표로 BallTree 생성
tree_cctv = BallTree(cctv_coords_rad, metric='haversine')

# 반경 300m (0.3km) → 라디안 변환
EARTH_RADIUS_KM = 6371.0
radius_rad = 0.3 / EARTH_RADIUS_KM

# 각 가로등 좌표에서 반경 내 CCTV 개수 구하기
cctv_counts = tree_cctv.query_radius(lamp_coords_rad, r=radius_rad, count_only=True)

# 결과를 '근처 CCTV개수' 열로 추가
df3['근처 CCTV개수'] = cctv_counts

# 확인
print(df3.head())
df3.describe()

#############################################################################
# 반경 300m 내 킥보드 주차장
df6 = df6[['위도','경도']]
df6

# 위도·경도 → 라디안 변환
lamp_coords_rad = np.radians(df3[['위도', '경도']].values)   # 가로등 좌표
cctv_coords_rad = np.radians(df6[['위도', '경도']].values)   # CCTV 좌표

# CCTV 좌표로 BallTree 생성
tree_cctv = BallTree(cctv_coords_rad, metric='haversine')

# 반경 300m (0.3km) → 라디안 변환
EARTH_RADIUS_KM = 6371.0
radius_rad = 0.3 / EARTH_RADIUS_KM

# 각 가로등 좌표에서 반경 내 CCTV 개수 구하기
cctv_counts = tree_cctv.query_radius(lamp_coords_rad, r=radius_rad, count_only=True)

# 결과를 '근처 CCTV개수' 열로 추가
df3['근처 킥라니주차장개수'] = cctv_counts

# 확인
print(df3.head())
df3.describe()

##################################################################################
# 지원 누나
jiwon = pd.read_csv('accident_traffic.csv',encoding='cp949')
jiwon = jiwon[['300m내_사고다발지역_개수','가장가까운_사고다발지역_거리(m)']]
df3 = pd.concat([df3, jiwon], axis=1)
# 영찬
youngchan = pd.read_csv('youngchan.csv')
youngchan = youngchan[['주변 학교 수','가장 가까운 학교와의 거리','광원 등급']]
df3 = pd.concat([df3, youngchan], axis=1)

# df3.to_csv('main_data_0814.csv', index=False) # 최종 데이터프레임

df3.info()
#########################################################3
## 위험도 점수 산출  -- 영천이꺼

# 영찬이 점수
df_000 = pd.read_csv('충청남도 천안시_가로등_위험도_20240729.csv')
df_000['위험도(100점)']

plt.figure(figsize=(8,5))
plt.hist(df_000['위험도(100점)'], bins=20, color='orange', edgecolor='black')
plt.title('가로등 위험도 점수 분포')
plt.xlabel('위험도 점수')
plt.ylabel('가로등 개수')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show() # 168개
##########################################################################################################

import geopandas as gpd
from shapely.geometry import Point, Polygon

# 1. 가로등 데이터 → GeoDataFrame 변환
gdf = gpd.GeoDataFrame(
    df_000,
    geometry=gpd.points_from_xy(df_000["경도"], df_000["위도"]),
    crs="EPSG:4326"
)

# 2. 사각형 좌표 (예시: 4개 꼭짓점)
# 반드시 순서대로 이어지도록 작성해야 함
rect_coords = [
    (36.78382, 127.0941),  # 왼쪽 아래
    (36.78655, 127.1684),  # 오른쪽 아래
    (36.84505, 127.1645),  # 오른쪽 위
    (36.84310, 127.1072),  # 왼쪽 위
    (36.78382, 127.0941)   # 다시 시작점 닫기
]

rectangle = Polygon(rect_coords)

# 3. 사각형 안에 있는 가로등만 필터링
filtered = gdf[gdf.within(rectangle)]

print(filtered)
############################여기서부터 df_filtered

lat_min, lat_max = 36.78382, 36.84505
lon_min, lon_max = 127.0941, 127.1684

df_filtered = df_000[
    (df_000["위도"] >= lat_min) & (df_000["위도"] <= lat_max) &
    (df_000["경도"] >= lon_min) & (df_000["경도"] <= lon_max)
]
df_filtered = df_filtered.reset_index(drop=True)

print(len(df_filtered))

# 60점 이상
df_top60 = df_filtered[df_filtered['위험도(100점)'] >= 60]

# 40점 이하
df_top40 = df_filtered[df_filtered['위험도(100점)'] <= 40 ]

fig = go.Figure()

# 가로등 전체 위치
fig.add_trace(go.Scattermap(
    lat=df_filtered['위도'],
    lon=df_filtered['경도'],
    mode='markers',
    marker=dict(
        size=7,
        color='yellow',
        opacity=0.5
    ),
    text=df_filtered['설치형태'] + '<br>위험도: ' + df_filtered['위험도(100점)'].astype(str),
    name='가로등 위치'
))

# 위험도 상위 60%
fig.add_trace(go.Scattermap(
    lat=df_top60['위도'],
    lon=df_top60['경도'],
    mode='markers',
    marker=dict(size=10, color='orange', opacity=0.6),
    name='위험도 상위 60%'
))

# 위험도 상위 40%
fig.add_trace(go.Scattermap(
    lat=df_top40['위도'],
    lon=df_top40['경도'],
    mode='markers',
    marker=dict(size=10, color='pink', opacity=0.6),
    name='위험도 상위 40%'
))

#######################
# 3) CCTV
fig.add_trace(go.Scattermap(
    lat=df4['위도'],
    lon=df4['경도'],
    mode='markers',
    marker=dict(size=10, color='green', opacity=0.6),
    text=df4['설치위치주소'],
    name='CCTV 위치'
))

# 4) 학교
fig.add_trace(go.Scattermap(
    lat=df_천안['lat'],
    lon=df_천안['lon'],
    mode='markers',
    marker=dict(size=10, color='purple', opacity=0.6),
    text=df_천안['구분'],
    name='학교 위치'
))

# 5) 킥라니
df6 = pd.read_excel('kickrani.xlsx',header=1)
fig.add_trace(go.Scattermap(
    lat=df6['위도'],
    lon=df6['경도'],
    mode='markers',
    marker=dict(size=10, color='black', opacity=0.6),
    text=df6['주차가능 대수'].astype(str),
    name='킥라니 주차장 위치'
))

# 6) 사고다발구역
all_zone = pd.read_csv('all_zone.csv',encoding='cp949')
fig.add_trace(go.Scattermap(
    lat=all_zone['위도'],
    lon=all_zone['경도'],
    mode='markers',
    marker=dict(size=10, color='red', opacity=0.6),
    text=all_zone['지점명'],
    name='사고다발구역'
))
############################3

# 지도 설정
fig.update_layout(
    map=dict(
        style="open-street-map",
        center=dict(lat=df_000['위도'].mean(), lon=df_000['경도'].mean()),
        zoom=13   # 숫자 높일수록 확대됨 (기존 11 → 13 정도)
    )
)

fig.show()

# df_filtered.to_csv('final.csv', index=False)
#####################################################################

# 0) 작업 대상 DF
df_new = df_filtered.copy()
df_new = df_new.drop('위험도(100점)',axis=1)
# 1) (이상치 완화) 1~99% 분위 기반 0~1 정규화
def robust_minmax(s: pd.Series, q_low=0.01, q_high=0.99):
    s = pd.to_numeric(s, errors="coerce")
    lo, hi = s.quantile(q_low), s.quantile(q_high)
    if pd.isna(lo) or pd.isna(hi) or hi == lo:
        return pd.Series(0.0, index=s.index)  # 분산이 없으면 0
    n = (s - lo) / (hi - lo)
    return n.clip(0, 1)
####################################

# 상권 위험도 반영
df_store = pd.read_csv("소상공인시장진흥공단_상가(상권)정보_충남_202506.csv")
df_store = df_store[df_store['시군구명'].str.contains("천안", na=False)].reset_index(drop=True)
df_store = df_store[(df_store['상권업종대분류명']=="음식")|(df_store['상권업종대분류명']=="숙박")]
df_store.reset_index(drop=True,inplace=True)
# df_store.to_csv("상권최종데이터.csv")
import requests
import pandas as pd
import time

# 카카오 REST API 키
KAKAO_API_KEY = 'd222f0f01e3470ce2b8a863cc30b151e'

# 카카오 API 호출 함수
def get_lat_lon_by_keyword(keyword):
    url = 'https://dapi.kakao.com/v2/local/search/keyword.json'
    headers = {"Authorization": f"KakaoAK {KAKAO_KEY}"}
    params = {'query': keyword, 'size': 1}  # 가장 가까운 1개 결과
    try:
        res = requests.get(url, headers=headers, params=params).json()
        if res['documents']:
            return float(res['documents'][0]['y']), float(res['documents'][0]['x'])
    except:
        pass
    return None, None

# tqdm으로 진행률 확인하며 lat/lon 생성
df_store['lat'], df_store['lon'] = zip(*[
    get_lat_lon_by_keyword(addr) for addr in tqdm(df_store['지번주소'])
])
df_store.info()
df_store.to_csv("상점위도경도.csv")

def haversine(lat1, lon1, lat2, lon2):
    """
    두 위도/경도 좌표 사이 거리 계산 (m 단위)
    """
    R = 6371000  # 지구 반지름 (m)
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)

    a = np.sin(delta_phi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(delta_lambda/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

    return R * c  # 거리(m)

# 결과 저장할 리스트
store_count = []

for idx, row in df_new.iterrows():
    lat_lamp = row['위도']
    lon_lamp = row['경도']

    # df_store 각 상점과 거리 계산
    distances = haversine(lat_lamp, lon_lamp, df_store['lat'].values, df_store['lon'].values)
    
    # 300m 이내 상점 수
    count = np.sum(distances <= 300)
    store_count.append(count)
# 새로운 열 추가
df_new['상점_300m'] = store_count

##################################

# 2) 항목별 위험도(0~1)
r_lamps   = 1 - robust_minmax(df_new["근처 가로등수"])                 # 많을수록 안전
r_cctv    = 1 - robust_minmax(df_new["근처 CCTV개수"])                # 많을수록 안전
r_sch_cnt = robust_minmax(df_new["주변 학교 수"])                # 많을수록 위험
r_sch_dst = 1 - robust_minmax(df_new["가장 가까운 학교와의 거리"])      # 가까울수록 위험
r_light   = 1 - (pd.to_numeric(df_new["광원 등급"], errors="coerce") / 5.0)  # 등급 높을수록 안전
r_escoot  = robust_minmax(df_new["근처 킥라니주차장개수"])         # 많을수록 위험
r_store = robust_minmax(df_new["상점_300m"])
# 3) 가중치(총 100점, 6개 항목 동일)
W = {
    "lamps":   100 / 7,  # 약 16.6667
    "cctv":    100 / 7,
    "sch_cnt": 100 / 7,
    "sch_dst": 100 / 7,
    "light":   100 / 7,
    "escoot": 100/7,
    "store" : 100/7
}

# 4) 최종 위험도 계산
df_new["위험도(100점)"] = (
    r_lamps   * W["lamps"]   +
    r_cctv    * W["cctv"]    +
    r_sch_cnt * W["sch_cnt"] +
    r_sch_dst * W["sch_dst"] +
    r_light   * W["light"] +
    r_escoot  * W["escoot"] +
    r_store * W['store']
).round(2)

sns.histplot(df_new['위험도(100점)'], bins=30, kde=True)  # kde=True: 밀도선 추가
plt.title("위도 히스토그램")
plt.show()
df_new.info()
# 60점 이상
df_top60 = df_new[df_new['위험도(100점)'] >= 55]
df_top60
# 40점 이하
df_top40 = df_new[(df_new['위험도(100점)'] <= 25) ]
# df_new.to_csv("가로등위험도최종데이터.csv")
fig = go.Figure()

# 가로등 전체 위치
fig.add_trace(go.Scattermap(
    lat=df_new['위도'],
    lon=df_new['경도'],
    mode='markers',
    marker=dict(
        size=7,
        color='yellow',
        opacity=0.5
    ),
    text=df_new['설치형태'] + '<br>위험도: ' + df_new['위험도(100점)'].astype(str),
    name='가로등 위치'
))

# 위험도 상위 60%
fig.add_trace(go.Scattermap(
    lat=df_top60['위도'],
    lon=df_top60['경도'],
    mode='markers',
    marker=dict(size=10, color='orange', opacity=0.6),
    text=df_top60['위험도(100점)'],
    name='위험도 상위 60%'
))

# 위험도 상위 40%
fig.add_trace(go.Scattermap(
    lat=df_top40['위도'],
    lon=df_top40['경도'],
    mode='markers',
    marker=dict(size=10, color='pink', opacity=0.6),
    name='위험도 상위 40%'
))

#######################
# 3) CCTV
fig.add_trace(go.Scattermap(
    lat=df4['위도'],
    lon=df4['경도'],
    mode='markers',
    marker=dict(size=10, color='green', opacity=0.6),
    text=df4['설치위치주소'],
    name='CCTV 위치'
))

# 4) 학교
fig.add_trace(go.Scattermap(
    lat=df_천안['lat'],
    lon=df_천안['lon'],
    mode='markers',
    marker=dict(size=10, color='purple', opacity=0.6),
    text=df_천안['구분'],
    name='학교 위치'
))

# 6) 사고다발구역
all_zone = pd.read_csv('all_zone.csv',encoding='cp949')
fig.add_trace(go.Scattermap(
    lat=all_zone['위도'],
    lon=all_zone['경도'],
    mode='markers',
    marker=dict(size=10, color='red', opacity=0.6),
    text=all_zone['지점명'],
    name='사고다발구역'
))
############################3

# 지도 설정
fig.update_layout(
    map=dict(
        style="open-street-map",
        center=dict(lat=df_000['위도'].mean(), lon=df_000['경도'].mean()),
        zoom=13   # 숫자 높일수록 확대됨 (기존 11 → 13 정도)
    )
)

fig.show()





#위험구역 특징 분석 (막대그래프 + 파이차트 / 한장짜리 대시보드)
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from adjustText import adjust_text
df_000 = df_new.copy()
df_000.info()
# ✅ 1. 분석 변수 구분
count_cols = ["근처 가로등수","근처 CCTV개수","근처 킥라니주차장개수","300m내_사고다발지역_개수","주변 학교 수"]
dist_cols = ["가장가까운_사고다발지역_거리(m)","가장 가까운 학교와의 거리"]
cat_col = "광원 등급"

# ✅ 2. 위험/안전 구역 분리
df_risk = df_000[df_000["위험도(100점)"] >= 60]
df_safe = df_000[df_000["위험도(100점)"] <= 30]

# ✅ 평균 + 95% 신뢰구간 계산 함수
def mean_ci(df, cols, confidence=0.95):
    means = df[cols].mean()
    ci = []
    for col in cols:
        n = df[col].count()
        if n > 1:
            se = stats.sem(df[col], nan_policy='omit')
            h = se * stats.t.ppf((1 + confidence) / 2., n-1)
        else:
            h = 0
        ci.append(h)
    return means, ci

# 📊 평균/CI 계산
risk_mean_count, risk_ci_count = mean_ci(df_risk, count_cols)
safe_mean_count, safe_ci_count = mean_ci(df_safe, count_cols)
risk_mean_dist, risk_ci_dist = mean_ci(df_risk, dist_cols)
safe_mean_dist, safe_ci_dist = mean_ci(df_safe, dist_cols)

# 📊 광원등급 분포
risk_light = df_risk[cat_col].value_counts(normalize=True) * 100
safe_light = df_safe[cat_col].value_counts(normalize=True) * 100

# =============================
# ✅ Figure 한 장짜리 대시보드
# =============================
fig, axes = plt.subplots(2, 2, figsize=(20,14))
axes = axes.flatten()

# (1) 개수형 변수 그래프
x = np.arange(len(count_cols))
width = 0.35
axes[0].bar(x - width/2, risk_mean_count, width, yerr=risk_ci_count, capsize=5, label='위험구역', color='red', alpha=0.7)
axes[0].bar(x + width/2, safe_mean_count, width, yerr=safe_ci_count, capsize=5, label='안전구역', color='green', alpha=0.7)

# 숫자 라벨
for i, v in enumerate(risk_mean_count):
    axes[0].text(i - width/2, v + max(risk_ci_count[i],0.5), f"{v:.1f}", ha='center', va='bottom', fontsize=9, color='red')
for i, v in enumerate(safe_mean_count):
    axes[0].text(i + width/2, v + max(safe_ci_count[i],0.5), f"{v:.1f}", ha='center', va='bottom', fontsize=9, color='green')

axes[0].set_xticks(x)
axes[0].set_xticklabels(count_cols, rotation=45)
axes[0].set_ylabel("평균값 (개수)")
axes[0].set_title("개수 변수 비교 (95% CI)")
axes[0].legend()
axes[0].grid(axis='y', linestyle='--', alpha=0.7)

# (2) 거리형 변수 그래프
x = np.arange(len(dist_cols))
axes[1].bar(x - width/2, risk_mean_dist, width, yerr=risk_ci_dist, capsize=5, label='위험구역', color='red', alpha=0.7)
axes[1].bar(x + width/2, safe_mean_dist, width, yerr=safe_ci_dist, capsize=5, label='안전구역', color='green', alpha=0.7)

# 숫자 라벨
for i, v in enumerate(risk_mean_dist):
    axes[1].text(i - width/2, v + max(risk_ci_dist[i],1), f"{v:.0f}", ha='center', va='bottom', fontsize=9, color='red')
for i, v in enumerate(safe_mean_dist):
    axes[1].text(i + width/2, v + max(safe_ci_dist[i],1), f"{v:.0f}", ha='center', va='bottom', fontsize=9, color='green')

axes[1].set_xticks(x)
axes[1].set_xticklabels(dist_cols, rotation=45)
axes[1].set_ylabel("평균값 (m)")
axes[1].set_title("거리 변수 비교 (95% CI)")
axes[1].legend()
axes[1].grid(axis='y', linestyle='--', alpha=0.7)

# (3) 위험구역 광원등급 파이차트
wedges, _ = axes[2].pie(
    risk_light, startangle=90,
    colors=plt.cm.Reds(np.linspace(0.3, 0.8, len(risk_light)))
)
axes[2].set_title("위험구역 - 광원등급 분포")

for i, p in enumerate(wedges):
    value = risk_light.iloc[i]
    ang = (p.theta2 - p.theta1)/2. + p.theta1
    y = np.sin(np.deg2rad(ang))
    x = np.cos(np.deg2rad(ang))
    if value <= 5:  # 5% 이하만 화살표로 밖으로
        axes[2].annotate(f"{risk_light.index[i]}: {value:.1f}%",
                         xy=(x*0.7, y*0.7), xytext=(x*1.2, y*1.2),
                         arrowprops=dict(arrowstyle="->", color='black'),
                         ha='center', va='center')
    else:  # 나머지는 wedge 안쪽
        axes[2].text(0.7*x, 0.7*y, f"{value:.1f}%", ha='center', va='center', fontsize=9)

axes[2].legend(wedges, risk_light.index, title="광원등급", loc="best")

# (4) 안전구역 광원등급 파이차트
wedges, _ = axes[3].pie(
    safe_light, startangle=90,
    colors=plt.cm.Greens(np.linspace(0.3, 0.8, len(safe_light)))
)
axes[3].set_title("안전구역 - 광원등급 분포")

for i, p in enumerate(wedges):
    value = safe_light.iloc[i]
    ang = (p.theta2 - p.theta1)/2. + p.theta1
    y = np.sin(np.deg2rad(ang))
    x = np.cos(np.deg2rad(ang))

    if value <= 2:  # 2% 이하(수정)만 화살표로 밖으로
        # 겹치지 않게 특정 등급만 xytext 조정
        offset_multiplier = 1.2
        if safe_light.index[i] == '2등급':
            xytext = (x*offset_multiplier, y*offset_multiplier + 0.1)
        elif safe_light.index[i] == '3등급':
            xytext = (x*offset_multiplier, y*offset_multiplier - 0.1)
        else:
            xytext = (x*offset_multiplier, y*offset_multiplier)

        axes[3].annotate(f"{safe_light.index[i]}: {value:.1f}%",
                         xy=(x*0.7, y*0.7), xytext=xytext,
                         arrowprops=dict(arrowstyle="->", color='black'),
                         ha='center', va='center')
    else:  # 나머지는 wedge 안쪽
        axes[3].text(0.7*x, 0.7*y, f"{value:.1f}%", ha='center', va='center', fontsize=9)

axes[3].legend(wedges, safe_light.index, title="광원등급", loc="best")

plt.tight_layout()
plt.show()


######## 지도만들기 수정 ########
import osmnx as ox

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

