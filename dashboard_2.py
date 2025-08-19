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

df4 = pd.read_csv('cctv최종데이터.csv') 
df6 = pd.read_excel('kickrani.xlsx', header=1)
df5 = pd.read_csv('학교최종데이터.csv')
df_000 = pd.read_csv('가로등위험도최종데이터.csv')

fig = go.Figure()



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
    name='CCTV 위치'
))

# 학교
fig.add_trace(go.Scattermapbox(
    lat=df5['lat'],
    lon=df5['lon'],
    mode='markers',
    marker=dict(size=10, color='purple', opacity=0.6),
    text=df5['구분'],
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
