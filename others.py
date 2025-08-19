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