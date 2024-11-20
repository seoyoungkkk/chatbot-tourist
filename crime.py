pip install pandas scikit-learn streamlit requests
streamlit run crime.py
import pandas as pd
from sklearn.cluster import KMeans
import streamlit as st
import requests

# Streamlit 앱 제목
st.title('범죄 데이터 분석 앱')

# Google Drive 파일 URL
file_url = 'https://docs.google.com/uc?export=download&id=10amnGP2QDd8byQJHp_gk1c8y0OGRGvP3'
file_path = 'crime_data.csv'

# 파일 다운로드
response = requests.get(file_url)
with open(file_path, 'wb') as f:
    f.write(response.content)

# 데이터 로드
data = pd.read_csv(file_path)

# 데이터 전처리
data.columns = data.iloc[0]  # 첫 행을 컬럼으로 설정
data = data[1:]  # 데이터 시작
numerical_data = data.iloc[:, 2:].apply(pd.to_numeric, errors='coerce').fillna(0)

location_data = data.iloc[:, :2].copy()
location_data.columns = ["대분류", "장소"]

crime_data = numerical_data
location_data["총범죄건수"] = crime_data.sum(axis=1).values
total_crimes = location_data["총범죄건수"].sum()
location_data["발생확률(%)"] = (location_data["총범죄건수"] / total_crimes) * 100

# K-Means 클러스터링
kmeans = KMeans(n_clusters=3, random_state=42)
location_data["클러스터"] = kmeans.fit_predict(crime_data)

# 사용자 입력
user_input = st.text_input("목적지를 입력해주세요:")

if user_input:
    # 입력된 장소에 대한 정보 검색
    result = location_data[location_data["장소"].str.contains(user_input, case=False, na=False)]
    if not result.empty:
        st.write("### 입력된 장소의 범죄 통계")
        st.write(result[["대분류", "장소", "총범죄건수", "발생확률(%)"]])
    else:
        st.error("입력한 장소는 데이터에 없습니다.")
