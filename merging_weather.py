import pandas as pd

# ---------------------------------------------------------
# 데이터 불러오기
# ---------------------------------------------------------
file_path = '기온데이터.csv'  

print("데이터 불러오는 중...")

try:
    df = pd.read_csv(file_path, encoding='cp949')
except:
    df = pd.read_csv(file_path, encoding='utf-8')


df.rename(columns={'일시': 'Timestamp', '기온(°C)': 'Temperature'}, inplace=True)

# ---------------------------------------------------------
# 24시간 형식 시간 변환
# ---------------------------------------------------------

df['Timestamp'] = pd.to_datetime(df['Timestamp'])


df.set_index('Timestamp', inplace=True)
df.sort_index(inplace=True)


df = df[~df.index.duplicated(keep='first')]
df = df['2023-01-01':'2024-11-30']

print(f"   -> 원본 데이터 샘플 (시간 확인용):\n{df.head(3)}")
print(f"   -> 원본 데이터 개수: {len(df)}개")

# ---------------------------------------------------------
# 2초 단위 선형 보간 (Linear Interpolation)
# ---------------------------------------------------------
print("3. 2초 단위 선형 보간")


df_interpolated = df.resample('2s').asfreq().interpolate(method='linear')

# ---------------------------------------------------------
# 결과 저장
# ---------------------------------------------------------
print("4. 결과 저장 중...")


output_name = 'weather_2023_2024_2s_linear.parquet'
df_interpolated.to_parquet(output_name)



print(f"\n[완료] 총 {len(df_interpolated):,}개의 데이터가 생성되었습니다.")
print(f"저장된 파일: {output_name}")
print("\n--- 결과 미리보기 (앞부분) ---")

print(df_interpolated.head(10))
