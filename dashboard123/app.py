import dash
from dash import dcc, html, Output, Input
import pandas as pd
import plotly.graph_objects as go

# 데이터 불러오기
df_000 = pd.read_csv('가로등위험도최종데이터.csv')
df4 = pd.read_csv('cctv최종데이터.csv') 
df5 = pd.read_csv('학교최종데이터.csv')
df6 = pd.read_excel('kickrani.xlsx', header=1)

# Dash 앱 초기화
app = dash.Dash(__name__)

# 앱 레이아웃
app.layout = html.Div([
    html.H2("천안시 위험지역 지도"),
    html.Div([
        html.Label("표시할 시설 선택:"),
        dcc.Checklist(
            id='layer_checklist',
            options=[
                {'label': '가로등', 'value': 'lamps'},
                {'label': 'CCTV', 'value': 'cctv'},
                {'label': '학교', 'value': 'school'},
                {'label': '킥라니 주차장', 'value': 'escoot'}
            ],
            value=['lamps', 'cctv'],  # 기본 선택
            inline=True
        )
    ], style={'margin-bottom': '20px'}),
    
    dcc.Graph(id='map_graph', style={'height':'800px'})
])

# 콜백 함수: 체크박스 선택에 따라 지도 갱신
@app.callback(
    Output('map_graph', 'figure'),
    Input('layer_checklist', 'value')
)
def update_map(selected_layers):
    fig = go.Figure()

    if 'lamps' in selected_layers:
        fig.add_trace(go.Scattermapbox(
            lat=df_000['위도'],
            lon=df_000['경도'],
            mode='markers',
            marker=dict(size=7, color='yellow', opacity=0.5),
            text=df_000['설치형태'] + '<br>위험도: ' + df_000['위험도(100점)'].astype(str),
            name='가로등 위치'
        ))

    if 'cctv' in selected_layers:
        fig.add_trace(go.Scattermapbox(
            lat=df4['위도'],
            lon=df4['경도'],
            mode='markers',
            marker=dict(size=10, color='green', opacity=0.6),
            name='CCTV 위치'
        ))

    if 'school' in selected_layers:
        fig.add_trace(go.Scattermapbox(
            lat=df5['lat'],
            lon=df5['lon'],
            mode='markers',
            marker=dict(size=10, color='purple', opacity=0.6),
            text=df5['구분'],
            name='학교 위치'
        ))

    if 'escoot' in selected_layers:
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
        margin={"r":0,"t":0,"l":0,"b":0},
        legend=dict(title="시설 종류", orientation="h")
    )

    return fig

# 앱 실행
if __name__ == '__main__':
    app.run_server(debug=True)
