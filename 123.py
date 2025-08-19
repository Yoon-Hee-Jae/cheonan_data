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

df_000.info()
