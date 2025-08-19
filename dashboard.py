#Shiny Dashboard code 1

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


KAKAO_KEY = 'd222f0f01e3470ce2b8a863cc30b151e'

def get_lat_lon_by_keyword(keyword):
    url = 'https://dapi.kakao.com/v2/local/search/keyword.json'
    headers = {"Authorization": f"KakaoAK {KAKAO_KEY}"}
    params = {'query': keyword, 'size': 1}
    res = requests.get(url, headers=headers, params=params).json()
    if res['documents']:
        return float(res['documents'][0]['y']), float(res['documents'][0]['x'])
    return None, None

import pandas as pd
import plotly.graph_objects as go
df_000 = pd.read_csv('충청남도 천안시_가로등_위험도_20240729.csv')
df4 = pd.read_csv('충청남도 천안시_교통정보 CCTV_20220922.csv', encoding='cp949') 
df6 = pd.read_excel('kickrani.xlsx', header=1)
df5 = pd.read_csv('충청남도_학교 현황_20240807.csv', encoding='cp949')
df5['학교_천안포함'] = df5['주소'].str.contains('천안')
df_천안 = df5[df5['학교_천안포함'] == True]
df_천안.info()  #250개
# 학생수가 0명인 학교는 제외
df_천안[df_천안['학생수(명)']==0].shape # 54개
# 학생수가 0인 행의 인덱스 저장
null_index = df_천안[df_천안['학생수(명)']==0].index
df_천안 = df_천안[~df_천안.index.isin(null_index)].reset_index()

fig = go.Figure()
df_천안['lat'], df_천안['lon'] = zip(*[
    get_lat_lon_by_keyword(addr) for addr in tqdm(df_천안['주소'])
])


# 가로등
fig.add_trace(go.Scattermapbox(
    lat=df_000['위도'],
    lon=df_000['경도'],
    mode='markers',
    marker=dict(size=7, color='yellow', opacity=0.5),
    text=df_000['설치형태'] + '<br>위험도: ' + df_000['위험도(100점)'].astype(str),
    name='가로등 위치'
))

# CCTV
fig.add_trace(go.Scattermapbox(
    lat=df4['위도'],
    lon=df4['경도'],
    mode='markers',
    marker=dict(size=10, color='green', opacity=0.6),
    text=df4['설치위치주소'],
    name='CCTV 위치'
))

# 학교
fig.add_trace(go.Scattermapbox(
    lat=df_천안['lat'],
    lon=df_천안['lon'],
    mode='markers',
    marker=dict(size=10, color='purple', opacity=0.6),
    text=df_천안['구분'],
    name='학교 위치'
))

# 킥라니
fig.add_trace(go.Scattermapbox(
    lat=df6['위도'],
    lon=df6['경도'],
    mode='markers',
    marker=dict(size=10, color='black', opacity=0.6),
    text=df6['주차가능 대수'].astype(str),
    name='킥라니 주차장 위치'
))

# 레이아웃
fig.update_layout(
    mapbox_style="open-street-map",
    mapbox_zoom=12,
    mapbox_center={"lat": df_000['위도'].mean(), "lon": df_000['경도'].mean()},
    height=800,
    margin={"r":0,"t":0,"l":0,"b":0},
    legend=dict(title="시설 종류", orientation="h")
)

fig.show()
