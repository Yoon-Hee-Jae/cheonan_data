import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker

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
df4.info()
df4
# CCTV의 수가 94개로 너무 적음
# 데이터 재점검 필요

# 학교 
df5 = pd.read_csv('충청남도_학교 현황_20240807.csv', encoding='cp949')
df5.info() # 1243개 결측치 0게
# 학생수가 0명인 학교는 제외
df5[df5['학생수(명)']==0].shape # 54개
# 학생수가 0인 행의 인덱스 저장
null_index = df5[df5['학생수(명)']==0].index
df5 = df5[~df5.index.isin(null_index)].reset_index()

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
df5['lat'], df5['lon'] = zip(*[
    get_lat_lon_by_keyword(addr) for addr in tqdm(df5['주소'])
])
# 오래 걸려서 데이터프레임으로 저장
df5.to_csv("df5_school.csv", index=False, encoding="utf-8-sig")

# 킥라니 주차장 
df6 = pd.read_excel('kickrani.xlsx')
df6.info()
# 2번쨰 행부터 불러오기
df6 = pd.read_excel("kickrani.xlsx", header=1)
df6.info()

# 이륜차 사고 다발 지역
df7 = pd.read_csv('motorcycle.csv', encoding='cp949')
df7.info()
df7.head()

df7 = df7[df7['시도시군구명'].str.contains('천안시')]
df7['구분'] = '이륜차 사고다발지역'
df7.head()

# 보행자 사고 다발 지역
df8 = pd.read_csv('pedstrians.csv', encoding='cp949')
df8.info()
df8.head()
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
danger_zone['구분'].unique()
danger_zone.info()
danger_zone['사고다발지id'] = danger_zone['사고다발지id'].astype(str)
danger_zone['연도'] = danger_zone['사고다발지id'].str.extract(r'^(\d{4})')
danger_zone['연도'] = danger_zone['연도'].astype(int)
danger_zone = danger_zone[danger_zone['연도']>=2021].reset_index(drop=True)
danger_zone = danger_zone.sort_values('연도').reset_index(drop=True)
danger_zone
danger_zone.to_csv('danger_zone.csv', index=False, encoding='cp949')
danger_zone[danger_zone['연도']==2024]

danger_zone
# 영찬 지원누나 사고 다발 구간 합치기
danger_jiwon = pd.read_csv('jiwon_danger_zone.csv',encoding='cp949')
danger_youngchan = pd.read_csv('youngchan_danger_zone.csv')
danger_jiwon.columns
danger_youngchan.columns
danger_zone.columns
danger_youngchan=danger_youngchan.drop('법규위반', axis=1)

danger_zone = pd.concat([danger_zone, danger_youngchan], ignore_index=True)
danger_zone = pd.concat([danger_zone, danger_jiwon], ignore_index=True)
danger_zone.info()
danger_2024 = danger_zone[danger_zone['연도'] == 2024]
danger_2024.shape # 54개 사고 다발 구간
danger_2024['구분'].value_counts() 
danger_zone.to_csv('all_zone.csv', index=False, encoding='cp949')











#######################################################################
# 시각화
fig = go.Figure()

# 1) df2_group 산점도 추가 (기존 시각화 데이터)
fig.add_trace(go.Scattermapbox(
    lat=df2_group['시작지점_y'],
    lon=df2_group['시작지점_x'],
    mode='markers',
    marker=dict(size=20, color='blue'),
    text=df2_group['시작지점'],
    name='시작지점'
))

# 2) df3 가로등 위치 추가
fig.add_trace(go.Scattermapbox(
    lat=df3['위도'],
    lon=df3['경도'],
    mode='markers',
    marker=dict(size=7, color='pink', opacity=0.6),
    text=df3['설치형태'],  # 마우스 올리면 관리기관 표시
    name='가로등 위치'
))

# 3) cctv 위치 시각화
fig.add_trace(go.Scattermapbox(
    lat=df4['위도'],
    lon=df4['경도'],
    mode='markers',
    marker=dict(size=10, color='green', opacity=0.6),
    text=df4['설치위치주소'],  # 마우스 올리면 관리기관 표시
    name='CCTV 위치'
))

# 4) school 위치 시각화
fig.add_trace(go.Scattermapbox(
    lat=df5['lat'],
    lon=df5['lon'],
    mode='markers',
    marker=dict(size=10, color='purple', opacity=0.6),
    text=df5['구분'],  # 마우스 올리면 관리기관 표시
    name='학교 위치'
))

# 5) 킥라니 위치 시각화
fig.add_trace(go.Scattermapbox(
    lat=df6['위도'],
    lon=df6['경도'],
    mode='markers',
    marker=dict(size=10, color='black', opacity=0.6),
    text=df6['주차가능 대수'],  # 마우스 올리면 관리기관 표시
    name='킥라니 주차장 위치'
))



# 레이아웃 설정
fig.update_layout(
    mapbox_style="open-street-map",
    mapbox_zoom=12,
    mapbox_center={
        "lat": df2_group['시작지점_y'].mean(),
        "lon": df2_group['시작지점_x'].mean()
    },
    height=800,
    margin={"r":0,"t":0,"l":0,"b":0}
)

fig.show()

