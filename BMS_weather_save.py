import os
import pandas as pd
import numpy as np
import glob
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

# ================================================================================
# 경로 및 환경 설정
# ================================================================================
WEATHER_PATH = r"Z:\SamsungSTF\Processed_Data\Parking\code\weather_2023_2024_linear.parquet"
BMS_ROOT_PATH = r"Z:\SamsungSTF\Processed_Data\Parking\Classified_final"
TARGET_VEHICLES = ['EV6', 'Ioniq5']
OUTPUT_CSV = "BMS_Weather_Temp_Average_lin.csv"
TIME_FORMAT = '%Y-%m-%d %H:%M:%S'

def process_task(file_info):
    file_path, vehicle, status = file_info
    try:
        # csv 읽기 (필요한 컬럼만 고속 로드)
        df = pd.read_csv(file_path, usecols=['time', 'ext_temp'], engine='c', low_memory=False)
        df['time'] = pd.to_datetime(df['time'], format=TIME_FORMAT)
        df = df.dropna().sort_values('time')
        if df.empty: return None

        times = df['time'].values.astype('datetime64[s]').astype(np.int64)
        temps = df['ext_temp'].values
        
        # 기본 산술 합계 (2초 간격 데이터는 선형보간 없이 그대로 사용)
        sum_temp = np.sum(temps)
        total_count = len(temps)
        
        # 1시간 이상의 결측치 구간만 선형 보간 가중치 추가
        dt = np.diff(times)
        big_gap_mask = dt >= 3600 # 1시간 이상 차이 나는 인덱스 추출
        
        if np.any(big_gap_mask):
            gap_durations = dt[big_gap_mask]
            # 결측 구간 양 끝점의 평균 온도 계산
            gap_avg_temps = (temps[:-1][big_gap_mask] + temps[1:][big_gap_mask]) / 2.0
            
            # 2초 간격을 기준으로 '가상 데이터'가 몇 개 들어갈지 계산 (virtual_n = (결측시간 / 2초) - 1)
            virtual_counts = (gap_durations / 2.0) - 1
            
            # 가상 데이터들의 온도 합과 개수를 기존 합계에 추가
            sum_temp += np.sum(gap_avg_temps * virtual_counts)
            total_count += np.sum(virtual_counts)

        # 최종 평균 계산 (산술 평균 + 결측 구간 보간분)
        bms_avg = sum_temp / total_count

        return (df['time'].iloc[0], df['time'].iloc[-1], vehicle, status, bms_avg, os.path.basename(file_path))
    except:
        return None

def main():
    # 기상 데이터 로드
    print("[*] 기상 데이터 로딩 중...")
    df_weather = pd.read_parquet(WEATHER_PATH)
    df_weather.index = pd.to_datetime(df_weather.index)
    df_weather = df_weather.sort_index().rename(columns={'Temperature': 'weather_temp'})

    # 파일 목록 수집 및 중복 제거
    raw_file_tasks = []
    print("[*] 전체 파일 목록 수집 중...")
    for vehicle in TARGET_VEHICLES:
        for status in ['Rest', 'Charging']:
            path = os.path.join(BMS_ROOT_PATH, vehicle, "**", "**", status, "*.csv")
            for f in glob.iglob(path, recursive=True):
                raw_file_tasks.append((os.path.abspath(f), vehicle, status))

    # 중복된 경로 제거 set을 이용해 고유한 (경로, 차종, 상태)만 남김
    file_tasks = list(set(raw_file_tasks))
    
    duplicate_count = len(raw_file_tasks) - len(file_tasks)
    print(f"총 {len(raw_file_tasks):,}개 발견")
    if duplicate_count > 0:
        print(f"{duplicate_count:,}개의 중복 경로를 제외")
    print(f"최종 분석 대상 {len(file_tasks):,}개")

    # 병렬 처리 실행 
    bms_results = []
    with ProcessPoolExecutor(max_workers=8) as executor: 
        for res in tqdm(executor.map(process_task, file_tasks, chunksize=200), total=len(file_tasks), desc="BMS Analysis"):
            if res: bms_results.append(res)

    # 기상 매칭
    print("기상 데이터 매칭 중...")
    final_list = []
    for r in tqdm(bms_results, desc="Weather Matching"):
        rel_w = df_weather.loc[r[0]:r[1]]
        w_avg = rel_w['weather_temp'].mean() if not rel_w.empty else np.nan
        final_list.append({
            'start_time': r[0], 'ext_temp_avg': r[4], 'weather_temp_avg': w_avg,
            'temp_diff': r[4] - w_avg, 'status': r[3], 'vehicle': r[2], 'file_name': r[5]
        })

    #  결과 저장
    df_final = pd.DataFrame(final_list).dropna(subset=['weather_temp_avg'])
    # 저장 전 마지막으로 한번 더 중복 체크 
    df_final = df_final.drop_duplicates(subset=['file_name'], keep='first')
    
    df_final.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    print(f"\n완료 (최종 {len(df_final):,}행)")

if __name__ == "__main__":
    main()