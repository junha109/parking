import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import warnings


warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'Malgun Gothic' 
plt.rcParams['axes.unicode_minus'] = False

# ================================================================================
# 경로 및 컬럼명
# ================================================================================
INPUT_ROOT_DIR = r"Z:\SamsungSTF\Processed_Data\Parking\Classified_final"
PARQUET_FILE_PATH = r"Z:\SamsungSTF\Processed_Data\Parking\code\weather_2023_2024_linear.parquet"
SAVE_IMAGE_PATH = r'Segment_Mean_Scatter3.png'

# CSV 내 컬럼명 
BMS_TIME_COL = 'time'          
BMS_TEMP_COL = 'ext_temp' 

# 기상청 Parquet 컬럼명 
WEATHER_TIME_COL = 'Timestamp'      
WEATHER_TEMP_COL = 'Temperature'

# ================================================================================
# BMS 파일 정보 추출 함수 
# ================================================================================
def get_bms_segment_stats(file_path):
    try:
        # 파일 경로에서 상태(Charging/Rest) 판단
        category = 'Rest'
        if 'Charging' in file_path or 'charging' in os.path.basename(file_path):
            category = 'Charging'

        # CSV 읽기 
        try: df = pd.read_csv(file_path, usecols=[BMS_TIME_COL, BMS_TEMP_COL], encoding='utf-8')
        except: df = pd.read_csv(file_path, usecols=[BMS_TIME_COL, BMS_TEMP_COL], encoding='cp949')
        
        if df.empty: return None

        # 데이터 전처리
        df[BMS_TIME_COL] = pd.to_datetime(df[BMS_TIME_COL], errors='coerce')
        df[BMS_TEMP_COL] = pd.to_numeric(df[BMS_TEMP_COL], errors='coerce')
        df = df.dropna()

        if len(df) < 2: return None

 
        # -----------------------------------------------------------
        
        # 세그먼트 전체 평균
        bms_mean = df[BMS_TEMP_COL].mean()
        
        # 시간 범위 추출
        start_time = df[BMS_TIME_COL].min()
        end_time = df[BMS_TIME_COL].max()

        return {
            'file_name': os.path.basename(file_path),
            'category': category,
            'start_time': start_time,
            'end_time': end_time,
            'bms_mean': bms_mean
        }

    except Exception:
        return None

# ================================================================================
# 메인 실행
# ================================================================================
def main():
    # ---------------------------------------------------------
    # BMS 파일 정보 수집 
    # ---------------------------------------------------------
    files_to_process = []
    for root, dirs, files in os.walk(INPUT_ROOT_DIR):
        for f in files:
            if f.lower().endswith('.csv'):
                files_to_process.append(os.path.join(root, f))
    
    if not files_to_process:
        print("CSV 파일이 없음")
        return

    print(f" BMS 파일 {len(files_to_process):,}개 ")
    
    bms_results = []
    max_workers = max(1, os.cpu_count() - 2)
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(get_bms_segment_stats, f) for f in files_to_process]
        for future in tqdm(as_completed(futures), total=len(files_to_process), desc="BMS Stats"):
            res = future.result()
            if res:
                bms_results.append(res)
    
    df_summary = pd.DataFrame(bms_results)
    print(f" BMS 요약 완료: {len(df_summary):,}개 ")

    # ---------------------------------------------------------
    # 기상청 데이터 로드 및 매칭
    # ---------------------------------------------------------
    if not os.path.exists(PARQUET_FILE_PATH):
        print(f" 기상청 파일을 찾을 수 없음: {PARQUET_FILE_PATH}")
        return

    print(f" 기상청 데이터 로드 중")
    df_weather = pd.read_parquet(PARQUET_FILE_PATH)
    
    # 시간 인덱스 설정 
    df_weather = pd.read_parquet(PARQUET_FILE_PATH)
    
    # 인덱스 설정
    if WEATHER_TIME_COL in df_weather.columns:
        df_weather[WEATHER_TIME_COL] = pd.to_datetime(df_weather[WEATHER_TIME_COL])
        df_weather = df_weather.set_index(WEATHER_TIME_COL)
    
    df_weather = df_weather.sort_index()

    
    weather_means = []
    

    for idx, row in tqdm(df_summary.iterrows(), total=len(df_summary), desc="Weather Matching"):
        try:
            start = row['start_time']
            end = row['end_time']
            
            # 해당 시간 구간의 기상청 데이터 Slicing, loc으로 범위 선택 후 mean
            subset = df_weather.loc[start:end, WEATHER_TEMP_COL]
            
            if subset.empty:
                weather_means.append(np.nan)
            else:
                weather_means.append(subset.mean())
                
        except KeyError: # 해당 시간이 기상청 데이터 범위 밖인 경우
            weather_means.append(np.nan)

    df_summary['weather_mean'] = weather_means
    
    # 결측치 제거 (기상청 데이터 없는 구간)
    df_plot = df_summary.dropna(subset=['weather_mean'])
    print(f" 최종 {len(df_plot):,}개 포인트 시각화")

    # ---------------------------------------------------------
    # Scatter Plot 그리기
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 10))
    
    # Charging과 Rest를 다른 색으로 표시 
    sns.scatterplot(
        data=df_plot,
        x='weather_mean',
        y='bms_mean',
        hue='category',     
        style='category',   
        alpha=0.3,          
        s=15,               
        palette={'Charging': 'red', 'Rest': 'blue'} 
    )
    
    # 기준선 (y=x)
    min_val = min(df_plot['weather_mean'].min(), df_plot['bms_mean'].min())
    max_val = max(df_plot['weather_mean'].max(), df_plot['bms_mean'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', label='Reference (y=x)', alpha=0.5)

    plt.title('세그먼트별 평균 온도 비교 (BMS vs 기상청)', fontsize=16)
    plt.xlabel('기상청 평균 기온 (°C)', fontsize=14)
    plt.ylabel('BMS 외기온도 평균 (°C)', fontsize=14)
    plt.legend(title='주차 상태')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.savefig(SAVE_IMAGE_PATH, dpi=300)
    print(f" 그래프 저장 완료: {SAVE_IMAGE_PATH}")
    plt.show()

if __name__ == "__main__":
    main()