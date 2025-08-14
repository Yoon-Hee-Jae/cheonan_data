import pandas as pd

df1 = pd.read_excel('main_dataset.xlsx')
df2 = pd.read_excel('traffic_data.xlsx')

df1.info()
df2.info()

df1["도로형태"].unique()
df1["사고유형"].unique()
df1_crossroad = df1[df1["도로형태"].str.contains("교차로", na=False)].dropna()
df1_crossroad

import matplotlib.pyplot as plt
from matplotlib import rcParams

# 폰트 설정
rcParams['font.family'] = 'Malgun Gothic'
rcParams['axes.unicode_minus'] = False

# '교차로' 데이터 필터링
df1_crossroad = df1[df1["도로형태"].str.contains("교차로", na=False)]

# 사고유형별 건수 집계
accident_counts = df1_crossroad["사고유형"].value_counts()

# 색상 리스트 생성 (최댓값만 빨강, 나머지는 하늘색)
colors = ['red' if i == accident_counts.idxmax() else 'skyblue'
          for i in accident_counts.index]

# 시각화
plt.figure(figsize=(10, 6))
accident_counts.plot(kind='bar', color=colors)
plt.title("교차로 사고유형별 건수")
plt.xlabel("사고유형")
plt.ylabel("사고 건수")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
from matplotlib import rcParams

# 한글 폰트 설정 (Windows 기준)
rcParams['font.family'] = 'Malgun Gothic'
rcParams['axes.unicode_minus'] = False

# '교차로' 데이터 필터링
df1_crossroad = df1[df1["도로형태"].str.contains("교차로", na=False)]

# 법규위반별 건수 집계
law_violation_counts = df1_crossroad["법규위반"].value_counts()

# 색상 리스트 생성 (기본 파랑, 최댓값은 빨강)
colors = ['red' if i == law_violation_counts.idxmax() else 'skyblue'
          for i in law_violation_counts.index]

# 시각화
plt.figure(figsize=(10, 6))
law_violation_counts.plot(kind='bar', color=colors)
plt.title("교차로 법규위반별 건수")
plt.xlabel("법규위반")
plt.ylabel("사고 건수")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
from matplotlib import rcParams

# 한글 폰트(Windows 기준) & 마이너스 기호 설정
rcParams['font.family'] = 'Malgun Gothic'
rcParams['axes.unicode_minus'] = False

# 1) 결측치 제거
df_clean = df1.dropna(subset=["도로형태"]).copy()

# 2) '교차로' / '단일로' / 그 외('기타')로 분류
def map_category(x: str) -> str:
    if "교차로" in x:
        return "교차로"
    elif "단일로" in x:
        return "단일로"
    else:
        return "기타"

df_clean["도로분류"] = df_clean["도로형태"].apply(map_category)

# 3) 비율(%) 계산
order = ["교차로", "단일로", "기타"]
ratio = (df_clean["도로분류"]
         .value_counts(normalize=True)
         .reindex(order)
         .fillna(0) * 100)

# 4) 파이차트 시각화
plt.figure(figsize=(6, 6))
plt.pie(ratio,
        labels=ratio.index,
        autopct="%.1f%%",
        startangle=90,
        colors=["#FF9999", "#99CCFF", "#99FF99"])
plt.title("도로형태 분류 비율(%) - 결측치 제거 후")
plt.show()

import matplotlib.pyplot as plt
from matplotlib import rcParams

# 한글 폰트 설정 (Windows 기준)
rcParams['font.family'] = 'Malgun Gothic'
rcParams['axes.unicode_minus'] = False

# 1) 도로형태, 사고유형 결측치 제거
df_clean = df1.dropna(subset=["도로형태", "사고유형"]).copy()

# 2) 도로형태 단순 분류
def map_category(x: str) -> str:
    if "교차로" in x:
        return "교차로"
    elif "단일로" in x:
        return "단일로"
    else:
        return "기타"

df_clean["도로분류"] = df_clean["도로형태"].apply(map_category)

# 3) 파이차트 비교를 위한 데이터 준비
road_types = ["교차로", "단일로"]
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

for ax, road in zip(axes, road_types):
    df_subset = df_clean[df_clean["도로분류"] == road]
    total_count = len(df_subset)
    car_to_car_count = df_subset["사고유형"].str.contains("차대차", na=False).sum()
    other_count = total_count - car_to_car_count

    sizes = [car_to_car_count, other_count]
    labels = ["차대차", "기타"]
    colors = ["#FF9999", "#99CCFF"]

    ax.pie(sizes, labels=labels, autopct="%.1f%%", startangle=90, colors=colors)
    ax.set_title(f"{road} 사고유형 비율")

plt.suptitle("교차로 vs 단일로의 '차대차' 사고 비율 비교", fontsize=14)
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
from matplotlib import rcParams
import pandas as pd
import numpy as np

# 한글 폰트(Windows 기준) & 마이너스 기호 설정
rcParams['font.family'] = 'Malgun Gothic'
rcParams['axes.unicode_minus'] = False

# 1) 전체 데이터에서 결측치 제거
df_all = df1.dropna().copy()

# 2) '교차로' 포함 행만 필터 + 결측치 제거
df_cross = df1[df1['도로형태'].str.contains('교차로', na=False)].dropna().copy()

# 3) 법규위반별 비율(%) 계산
all_ratio = (df_all['법규위반'].value_counts(normalize=True) * 100)
cross_ratio = (df_cross['법규위반'].value_counts(normalize=True) * 100)

# 4) 항목 통일(전체와 교차로의 카테고리 합집합으로 정렬)
cats = list(all_ratio.sort_values(ascending=False).index)  # 전체 비중 기준으로 정렬
for k in cross_ratio.index:
    if k not in cats:
        cats.append(k)

all_ratio = all_ratio.reindex(cats).fillna(0)
cross_ratio = cross_ratio.reindex(cats).fillna(0)

# 5) 그룹 막대그래프 그리기
x = np.arange(len(cats))
width = 0.38

fig, ax = plt.subplots(figsize=(12, 6))
bars1 = ax.bar(x - width/2, all_ratio.values, width, label='전체', alpha=0.9)
bars2 = ax.bar(x + width/2, cross_ratio.values, width, label='교차로', alpha=0.9)

ax.set_title("법규위반별 비율(%) — 결측치 제외: 전체 vs 교차로")
ax.set_xlabel("법규위반")
ax.set_ylabel("비율(%)")
ax.set_xticks(x, cats, rotation=45, ha='right')
ax.set_ylim(0, max(all_ratio.max(), cross_ratio.max()) * 1.15)  # 여유 공간
ax.legend()

# 6) 막대 위 퍼센트 라벨 표시
def add_labels(bars):
    for b in bars:
        h = b.get_height()
        ax.annotate(f"{h:.1f}%",
                    (b.get_x() + b.get_width()/2, h),
                    ha='center', va='bottom', fontsize=9, rotation=0)

add_labels(bars1)
add_labels(bars2)

plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
from matplotlib import rcParams

# 한글 폰트 설정 (Windows 기준)
rcParams['font.family'] = 'Malgun Gothic'
rcParams['axes.unicode_minus'] = False

# 1) '교차로' 포함 데이터 필터링
df_cross = df1[df1['도로형태'].str.contains('교차로', na=False)].copy()

# 2) 합계 집계
injury_sums = df_cross[['사망자수', '중상자수', '경상자수']].sum()

# 3) 시각화
plt.figure(figsize=(6, 5))
bars = plt.bar(injury_sums.index, injury_sums.values, color=['#FF6666', '#FFCC66', '#66B2FF'])

plt.title("교차로 사고 피해 정도 합계")
plt.xlabel("피해 정도")
plt.ylabel("인원 수")

# 막대 위 라벨 표시
for bar in bars:
    height = bar.get_height()
    plt.annotate(f"{int(height)}",
                 xy=(bar.get_x() + bar.get_width()/2, height),
                 ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
from matplotlib import rcParams

# 한글 폰트 설정
rcParams['font.family'] = 'Malgun Gothic'
rcParams['axes.unicode_minus'] = False

# 1) 주야, 사망자수 결측치 제거
df_clean = df1.dropna(subset=["주야", "사망자수"]).copy()

# 2) '야간' 데이터만 필터링
df_night = df_clean[df_clean["주야"] == "야간"]

# 3) 사망자수 1 이상 vs 0
count_death = (df_night["사망자수"] >= 1).sum()
count_no_death = (df_night["사망자수"] == 0).sum()

# 4) 파이차트 시각화
labels = ["사망자수 1 이상", "사망자수 0"]
sizes = [count_death, count_no_death]
colors = ["#FF6666", "#99CCFF"]

plt.figure(figsize=(6, 6))
plt.pie(sizes, labels=labels, autopct="%.1f%%", startangle=90, colors=colors)
plt.title("야간 사고 중 사망자수 1 이상 비율")
plt.show()

import matplotlib.pyplot as plt
from matplotlib import rcParams

# 한글 폰트 설정
rcParams['font.family'] = 'Malgun Gothic'
rcParams['axes.unicode_minus'] = False

# 1) 주야, 사망자수 결측치 제거
df_clean = df1.dropna(subset=["주야", "사망자수"]).copy()

# 2) 사망자수 조건에 따라 건수 계산
count_death = (df_clean["사망자수"] >= 1).sum()
count_no_death = (df_clean["사망자수"] == 0).sum()

# 3) 파이차트 시각화
labels = ["사망자수 1 이상", "사망자수 0"]
sizes = [count_death, count_no_death]
colors = ["#FF6666", "#99CCFF"]

plt.figure(figsize=(6, 6))
plt.pie(sizes, labels=labels, autopct="%.1f%%", startangle=90, colors=colors)
plt.title("전체 사고 중 사망자수 1 이상 비율")
plt.show()