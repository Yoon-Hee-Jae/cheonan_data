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

# 데이터 전처리 - 0
# df2 데이터 12개월치 평균
df2_group = df2.groupby(['도로명','구간명']).mean().reset_index()
df2_group.info()

# 데이터 전처리-1

# '구간명' 시작지점, 도착지점으로 구분
# 정규표현식 사용해서 시작지점과 도착지점 칼럼 새로 추가
df2_group[['시작지점','도착지점']] = df2_group['구간명'].str.split(' -> ',expand=True)
# 두 지점 모두 65개의 지점이 있으며 완전 동일함
len(df2_group['시작지점'].unique()) # 65
len(df2_group['도착지점'].unique()) # 65
df2_group['시작지점'].unique()
df2_group['도착지점'].unique()
# 값이 동일한지 확인
arr1 = ['동극섬유', '배방1교차로', '이마트앞교차로서측', '새말사거리', '신방삼거리', '장재2교차로',
       '청룡지하차도', '청삼교차로', '방죽안오거리', '천안역앞교차로', '동서고가교동측', '동서고가교서측',
       '서부대로사거리', '시민문화여성회관사거리', '은총교', '인쇄창사거리', '대우1차아파트107동',
       '대전충남양돈농협신두정지점', '두정역삼거리', '성성고가차도', '카페베네천안두정점', '천안로사거리',
       '터미널사거리', '석문교차로', '입장교차로', '천안IC앞', 'KB국민은행천안백석종합금융센터', '구상골사거리',
       '백석사거리', '백석요양원', '봉정사거리', '운동장사거리', '교보사거리', '버들육거리', '고속철도사거리',
       '불당아이파크아파트', '불당행복주유소', '성성2교차로', '시청앞사거리', '업성동삼거리', '북부고가교',
       '천고사거리', '한올수예', '쌍용동사거리', '일봉산사거리', '두정지하차도사거리', '북부지하차도',
       '손수남황태전문점', '쌍용삼거리', 'IBK기업은행천안쌍용지점', '산내들유치원', '삼일원앙아파트101동',
       '음봉로교차로', '용연마을삼거리', '역말오거리', '쌍용지하차도앞교차로', '천안지하차도', '구성삼거리',
       '남천안IC', '대림한내아파트', '도로원점삼거리', '수헐교차로', '충무로사거리', '삼룡사거리', '충절오거리']

arr2 = ['이마트앞교차로서측', '동극섬유', '배방1교차로', '신방삼거리', '청룡지하차도', '새말사거리',
       '장재2교차로', '천안역앞교차로', '방죽안오거리', '동서고가교서측', '동서고가교동측', '시민문화여성회관사거리',
       '은총교', '서부대로사거리', '인쇄창사거리', '대전충남양돈농협신두정지점', '두정역삼거리',
       '대우1차아파트107동', '카페베네천안두정점', '성성고가차도', '터미널사거리', '천안IC앞', '천안로사거리',
       '입장교차로', '석문교차로', '백석요양원', '운동장사거리', '백석사거리', '봉정사거리', '구상골사거리',
       'KB국민은행천안백석종합금융센터', '버들육거리', '교보사거리', '불당아이파크아파트', '고속철도사거리',
       '시청앞사거리', '업성동삼거리', '성성2교차로', '불당행복주유소', '북부고가교', '천고사거리', '한올수예',
       '음봉로교차로', '용연마을삼거리', '쌍용동사거리', '일봉산사거리', '충무로사거리', '두정지하차도사거리',
       '북부지하차도', '손수남황태전문점', '쌍용삼거리', 'IBK기업은행천안쌍용지점', '삼일원앙아파트101동',
       '산내들유치원', '청삼교차로', '역말오거리', '쌍용지하차도앞교차로', '천안지하차도', '도로원점삼거리',
       '구성삼거리', '수헐교차로', '대림한내아파트', '남천안IC', '충절오거리', '삼룡사거리']

# 집합(set)으로 변환하여 비교
set1 = set(arr1)
set2 = set(arr2)

if set1 == set2:
    print("두 배열에 포함된 값들은 완전히 동일합니다.")
else:
    print("두 배열에 포함된 값들이 다릅니다.")
    print("arr1에는 있지만 arr2에는 없는 값:", set1 - set2)
    print("arr2에는 있지만 arr1에는 없는 값:", set2 - set1)


# 데이터 전처리 - 2 상세주소 입력 + 위도 경도 위치 열 추가

# 카카오 api 불러와서 사용
# 사용전에 홈페이지 들어가서 사용 설정 상태 on으로 설정
import requests

KAKAO_KEY = 'd222f0f01e3470ce2b8a863cc30b151e'

place_names = ['이마트앞교차로서측', '동극섬유', '배방1교차로', '신방삼거리', '청룡지하차도', '새말사거리',
       '장재2교차로', '천안역앞교차로', '방죽안오거리', '동서고가교서측', '동서고가교동측', '시민문화여성회관사거리',
       '은총교', '서부대로사거리', '인쇄창사거리', '대전충남양돈농협신두정지점', '두정역삼거리',
       '대우1차아파트107동', '카페베네천안두정점', '성성고가차도', '터미널사거리', '천안IC앞', '천안로사거리',
       '입장교차로', '석문교차로', '백석요양원', '운동장사거리', '백석사거리', '봉정사거리', '구상골사거리',
       'KB국민은행천안백석종합금융센터', '버들육거리', '교보사거리', '불당아이파크아파트', '고속철도사거리',
       '시청앞사거리', '업성동삼거리', '성성2교차로', '불당행복주유소', '북부고가교', '천고사거리', '한올수예',
       '음봉로교차로', '용연마을삼거리', '쌍용동사거리', '일봉산사거리', '충무로사거리', '두정지하차도사거리',
       '북부지하차도', '손수남황태전문점', '쌍용삼거리', 'IBK기업은행천안쌍용지점', '삼일원앙아파트101동',
       '산내들유치원', '청삼교차로', '역말오거리', '쌍용지하차도앞교차로', '천안지하차도', '도로원점삼거리',
       '구성삼거리', '수헐교차로', '대림한내아파트', '남천안IC', '충절오거리', '삼룡사거리']
len(place_names)

def get_location_by_keyword(keyword):
    url = 'https://dapi.kakao.com/v2/local/search/keyword.json'
    headers = {"Authorization": f"KakaoAK {KAKAO_KEY}"}
    params = {'query': keyword, 'size': 5}
    res = requests.get(url, headers=headers, params=params)
    if res.status_code != 200:
        print(f"Error: {res.status_code}")
        return None
    data = res.json()
    if data['documents']:
        for doc in data['documents']:
            print(f"Name: {doc['place_name']}")
            print(f"Address: {doc.get('road_address_name') or doc.get('address_name')}")
            print(f"Lat,Lon: {doc['y']}, {doc['x']}")
            print('---')
        return data['documents'][0]  # 첫 번째 결과 반환
    else:
        print("검색 결과가 없습니다.")
        return None

results = {}
for place in place_names:
    addr = get_location_by_keyword(place)
    results[place] = addr
    print(f"{place} => {addr}")

results # results에 {장소명: 주소} 저장
type(results)
results['배방1교차로']['address_name']

# 상세주소 열 추가
results.get('배방1교차로')

def get_detailed_address(place):
    info = results.get(place)
    if info:
        return info['address_name']  # 지번 주소만 반환
    else:
        return None

# 새 컬럼 추가
df2_group['시작지점_상세주소'] = df2_group['시작지점'].apply(get_detailed_address)
df2_group['도착지점_상세주소'] = df2_group['도착지점'].apply(get_detailed_address)
df2_group['도착지점_상세주소']

# 위도 경도 열 추가 총 4개 ( 시작지점xy, 도착지점xy)

def get_x(place):
    info = results.get(place)
    if info:
        return info.get('x')
    return None

def get_y(place):
    info = results.get(place)
    if info:
        return info.get('y')
    return None

df2_group['시작지점_x'] = df2_group['시작지점'].apply(get_x)
df2_group['시작지점_y'] = df2_group['시작지점'].apply(get_y)
df2_group['도착지점_x'] = df2_group['도착지점'].apply(get_x)
df2_group['도착지점_y'] = df2_group['도착지점'].apply(get_y)

# 위도, 경도 컬럼 타입 확인
print(df2_group['시작지점_y'].dtype)
print(df2_group['시작지점_x'].dtype)

# 숫자형으로 강제 변환 (변환 불가능한 값은 NaN 처리)
df2_group['시작지점_y'] = pd.to_numeric(df2_group['시작지점_y'], errors='coerce')
df2_group['시작지점_x'] = pd.to_numeric(df2_group['시작지점_x'], errors='coerce')

df2_group.info()

# 채워넣을 좌표 사전
coords = {
    "이마트앞교차로서측": (36.795903, 127.125890),
    "동서고가교서측": (36.825046, 127.148013),
    "동서고가교동측": (36.824900, 127.149997),
    "시민문화여성회관사거리": (36.827078, 127.135256),
    "두정지하차도사거리": (36.837068, 127.151585),
    "손수남황태전문점": (36.838993, 127.135390)
}

# 시작지점 Null 값 채우기
for name, (lat, lon) in coords.items():
    mask = (df2_group["시작지점"] == name)
    df2_group.loc[mask, "시작지점_y"] = df2_group.loc[mask, "시작지점_y"].fillna(lat)
    df2_group.loc[mask, "시작지점_x"] = df2_group.loc[mask, "시작지점_x"].fillna(lon)

# 도착지점 Null 값 채우기
for name, (lat, lon) in coords.items():
    mask = (df2_group["도착지점"] == name)
    df2_group.loc[mask, "도착지점_y"] = df2_group.loc[mask, "도착지점_y"].fillna(lat)
    df2_group.loc[mask, "도착지점_x"] = df2_group.loc[mask, "도착지점_x"].fillna(lon)

df2_group[df2_group['시작지점']=='이마트앞교차로서측']

df2_group.isnull().sum()
df2_group[df2_group['시작지점_x'].isnull()].index
df2_group[df2_group['시작지점_y'].isnull()].index
df2_group[df2_group['도착지점_x'].isnull()].index
df2_group[df2_group['도착지점_y'].isnull()].index

null_index = df2_group[df2_group['시작지점_x'].isnull()].index.union(
    df2_group[df2_group['도착지점_y'].isnull()].index
)

df2_group.shape

df2_group = df2_group[~df2_group.index.isin(null_index)].reset_index()

df2_group.isnull().sum()

len(df2_group['도착지점'].unique())

df2_group

############################################################################################

import plotly.express as px

# 가로등 시각화
df3 = pd.read_csv('충청남도 천안시_가로등 현황_20240729.csv') # 가로등 데이터
df3.info() # 결측치 존재 특히, 도로묭주소에 결측치 10000개 존재
df3['설치형태'].unique() # ['LED', 'CML', 'CDM', '나트륨', '메탈', '삼파장', 'CPO', 'CCTV', '써치등']
df3['설치형태'].value_counts()
# 설치형태 이상치 제거 필요
df3[df3['설치형태']=='CCTV']
df3[df3['설치형태']=='나트륨']
df3[df3['설치형태']=='메탈']
df3[df3['설치형태']=='삼파장']

# 시작지점 + 가로등 시각화
import plotly.graph_objects as go

# cctv 시각화
df4 = pd.read_csv('충청남도 천안시_교통정보 CCTV_20220922.csv', encoding='cp949') 
# CCTV의 수가 94개로 너무 적음
df4_1 = df4[['위도','경도']]
# 과속카메라 추가
cctv_2 = pd.read_csv('cctv_2.csv')
cctv_2 = cctv_2[['위도','경도']]
cctv = pd.concat([df4_1,cctv_2], ignore_index=True)
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

df5.info()

# 주소를 좌표로 변환하는 함수
import requests
from tqdm import tqdm

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


df70 = df_000[df_000['위험도(100점)']>=70]
df_000.sort_values('위험도(100점)',inplace=True)
df_000 = df_000.reset_index(drop=True)

##########################################################################################################

# 가로등 위험도만 시각화 진행

# 60점 이상
df_top60 = df_000[df_000['위험도(100점)'] >= 60]

# 40점 이하
df_top40 = df_000[(df_000['위험도(100점)'] >= 30) & (df_000['위험도(100점)'] <= 40) ]

fig = go.Figure()

# 가로등 전체 위치
fig.add_trace(go.Scattermap(
    lat=df_000['위도'],
    lon=df_000['경도'],
    mode='markers',
    marker=dict(
        size=7,
        color='yellow',
        opacity=0.5
    ),
    text=df_000['설치형태'] + '<br>위험도: ' + df_000['위험도(100점)'].astype(str),
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

# 지도 설정
fig.update_layout(
    map=dict(
        style="open-street-map",
        center=dict(lat=df_000['위도'].mean(), lon=df_000['경도'].mean()),
        zoom=13   # 숫자 높일수록 확대됨 (기존 11 → 13 정도)
    )
)

fig.show()

# 가로등 위치 표시
fig = go.Figure()

fig.add_trace(go.Scattermapbox(
    lat=df_000['위도'],
    lon=df_000['경도'],
    mode='markers',
    marker=dict(
        size=7,
        color='yellow',  # 색상 적용
        opacity=0.5
        # colorbar 제거
    ),
    text=df_000['설치형태'] + '<br>위험도: ' + df_000['위험도(100점)'].astype(str),
    name='가로등 위치'
))

# 3) CCTV
fig.add_trace(go.Scattermapbox(
    lat=df4['위도'],
    lon=df4['경도'],
    mode='markers',
    marker=dict(size=10, color='green', opacity=0.6),
    text=df4['설치위치주소'],
    name='CCTV 위치'
))

# 4) 학교
fig.add_trace(go.Scattermapbox(
    lat=df_천안['lat'],
    lon=df_천안['lon'],
    mode='markers',
    marker=dict(size=10, color='purple', opacity=0.6),
    text=df_천안['구분'],
    name='학교 위치'
))

# 5) 킥라니
df6 = pd.read_excel('kickrani.xlsx',header=1)
fig.add_trace(go.Scattermapbox(
    lat=df6['위도'],
    lon=df6['경도'],
    mode='markers',
    marker=dict(size=10, color='black', opacity=0.6),
    text=df6['주차가능 대수'].astype(str),
    name='킥라니 주차장 위치'
))

# 6) 사고다발구역
all_zone = pd.read_csv('all_zone.csv',encoding='cp949')
fig.add_trace(go.Scattermapbox(
    lat=all_zone['위도'],
    lon=all_zone['경도'],
    mode='markers',
    marker=dict(size=10, color='gray', opacity=0.6),
    text=all_zone['지점명'],
    name='사고다발구역'
))

# 레이아웃
fig.update_layout(
    mapbox_style="open-street-map",
    mapbox_zoom=12,
    mapbox_center={"lat": df3['위도'].mean(), "lon": df3['경도'].mean()},
    height=800,
    margin={"r":0,"t":0,"l":0,"b":0}
)

fig.show()

# 천안시 지도 시각화
center_lat = 36.8195
center_lon = 127.1135

fig = go.Figure(go.Scattermapbox(
    mode="markers",
    lat=[center_lat],
    lon=[center_lon],
    marker=dict(size=10, color="red"),
))

fig.update_layout(
    mapbox_style="open-street-map",
    mapbox_center={"lat": center_lat, "lon": center_lon},
    mapbox_zoom=12,
    height=800,
    margin={"r":0,"t":0,"l":0,"b":0}
)

fig.show()

##################################################################################
#위험구역 특징
df_000.info()
danger_zone = df_000[df_000["위험도(100점)"] >= 60]

# 관심 변수 목록
cols = [
    "근처 가로등수",
    "근처 CCTV개수",
    "근처 킥라니주차장개수",
    "300m내_사고다발지역_개수",
    "가장가까운_사고다발지역_거리(m)",
    "주변 학교 수",
    "가장 가까운 학교와의 거리",
    "광원 등급"
]

# 위험구역 특징 요약
danger_features = danger_zone[cols].describe().T
danger_features[["mean","50%","min","max"]]
