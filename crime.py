import pandas as pd
from sklearn.cluster import KMeans
import streamlit as st
import requests

# Streamlit 앱 제목
st.title('범죄 데이터 분석 앱')

# Google Drive 파일 URL에서 데이터 로드
url = 'https://docs.google.com/uc?export=download&id=10amnGP2QDd8byQJHp_gk1c8y0OGRGvP3&confirm=t'
df = pd.read_csv(url)

# 데이터 전처리
df.columns = df.iloc[0]  # 첫 행을 컬럼으로 설정
df = df[1:]  # 데이터 시작
df.reset_index(drop=True, inplace=True)

# 숫자 데이터를 정리
numerical_data = df.iloc[:, 2:].apply(pd.to_numeric, errors='coerce').fillna(0)

# 장소 데이터와 범죄 데이터 분리
location_data = df.iloc[:, :2].copy()
location_data.columns = ["대분류", "장소"]

# 총 범죄 건수 및 발생 확률 계산
location_data["총범죄건수"] = numerical_data.sum(axis=1).values
total_crimes = location_data["총범죄건수"].sum()
location_data["발생확률(%)"] = (location_data["총범죄건수"] / total_crimes) * 100

# K-Means 클러스터링
kmeans = KMeans(n_clusters=3, random_state=42)
location_data["클러스터"] = kmeans.fit_predict(numerical_data)

# Streamlit 사용자 입력 처리
user_input = st.text_input("목적지를 입력해주세요:")

if user_input:
    # 입력된 장소 검색
    result = location_data[location_data["장소"].str.contains(user_input, case=False, na=False)]
    
    if not result.empty:
        st.write("### 입력된 장소의 범죄 통계")
        st.write(result[["대분류", "장소", "총범죄건수", "발생확률(%)"]])
    else:
        st.error("입력한 장소는 데이터에 없습니다.")
