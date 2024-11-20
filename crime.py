pip install pandas matplotlib scikit-learn gdown
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import gdown

# Google Drive 파일 다운로드
file_url = 'https://docs.google.com/uc?export=download&id=10amnGP2QDd8byQJHp_gk1c8y0OGRGvP3'
file_path = 'crime_data.csv'

# gdown으로 파일 다운로드
gdown.download(file_url, file_path, quiet=False)

# 데이터 불러오기
data = pd.read_csv(file_path)

# 데이터 전처리
data.columns = data.iloc[0]  # 첫 행을 컬럼으로 설정
data = data[1:]  # 데이터 시작
numerical_data = data.iloc[:, 2:].apply(pd.to_numeric, errors='coerce').fillna(0)

location_data = data.iloc[:, :2]  # 첫 두 열을 장소 정보로 사용
location_data.columns = ["대분류", "장소"]

crime_data = numerical_data  # 나머지 열은 범죄 데이터
location_data["총범죄건수"] = crime_data.sum(axis=1).values  # 총 범죄 건수 계산

# 총 범죄 대비 발생 확률 계산
total_crimes = location_data["총범죄건수"].sum()
location_data["발생확률(%)"] = (location_data["총범죄건수"] / total_crimes) * 100

# K-Means 클러스터링
kmeans = KMeans(n_clusters=3, random_state=42)
location_data["클러스터"] = kmeans.fit_predict(crime_data)

# 클러스터링 시각화
plt.figure(figsize=(10, 6))
plt.scatter(location_data.index, location_data["총범죄건수"], c=location_data["클러스터"], cmap='viridis')
plt.colorbar(label='Cluster')
plt.xlabel('지역 Index')
plt.ylabel('총범죄건수')
plt.title('K-Means 클러스터링 결과')
plt.show()

# 장소별 결과 출력
print(location_data)
