import pandas as pd
import numpy as np
import os
import glob
from scipy.stats import linregress
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# ==========================================

target_root_folder = r"Z:\SamsungSTF\Processed_Data\Parking\Classified"
output_file_name = 'BMS_Segment_Analysis_New_Metric.csv'

COL_TIME = 'time'
COL_MOD_TEMP = 'mod_avg_temp'
COL_EXT_TEMP = 'ext_temp'

# ==========================================
# 분석 함수
# ==========================================
def process_segment(file_path):
    try:
        p = Path(file_path)
        
        # 데이터 추출
        terminal_id = p.parent.parent.parent.name
        vehicle_model = p.parent.parent.parent.parent.name
        
        # CSV 읽기 
        try: df = pd.read_csv(file_path, encoding='utf-8')
        except:
            try: df = pd.read_csv(file_path, encoding='cp949')
            except: return 0 
        
        # 필수 컬럼 확인
        if not {COL_MOD_TEMP, COL_EXT_TEMP, COL_TIME}.issubset(df.columns):
            return None
        if len(df) < 2:
            return None

        # ------------------------------------------------
        # 시간 파싱 
        # ------------------------------------------------
        df[COL_TIME] = pd.to_datetime(df[COL_TIME], errors='coerce')
        df = df.dropna(subset=[COL_TIME])
        
        if df.empty or len(df) < 2:
            return None
            
        start_time = df[COL_TIME].iloc[0]
        end_time = df[COL_TIME].iloc[-1]
        
        # 주차 시간 (초 단위)
        duration_seconds = (end_time - start_time).total_seconds()
        
        if duration_seconds <= 0:
            return None

        # ------------------------------------------------
        # Time-Normalized NM Metric 
        # ------------------------------------------------
        temps = df[COL_MOD_TEMP]
        
        tv = temps.diff().abs().sum()
        net = abs(temps.iloc[-1] - temps.iloc[0])
        
        # (TV - Net) / 주차 시간(분)
        nm_metric = (tv - net) / duration_seconds * 60

        # ------------------------------------------------
        # 기타 통계
        # ------------------------------------------------
        mean_mod = df[COL_MOD_TEMP].mean()
        mean_ext = df[COL_EXT_TEMP].mean()
        last_mod = df[COL_MOD_TEMP].iloc[-1]
        last_ext = df[COL_EXT_TEMP].iloc[-1]

        x = (df[COL_TIME] - start_time).dt.total_seconds().values
        slope, intercept, r_val, p_val, std_err = linregress(x, temps.values)

        return {
            '차종': vehicle_model,
            '단말기 번호': terminal_id,
            'Non_Monotonic_Metric': nm_metric,
            'TV': tv,
            'Net': net,
            'Duration_Sec': duration_seconds,
            'Data_Count': len(df),
            'Mean_Mod_Temp': mean_mod,
            'Mean_Ext_Temp': mean_ext,
            'Slope_Mod_Temp': slope,
            'Last_Mod_Temp': last_mod,
            'Last_Ext_Temp': last_ext,
            'File_Path': str(p)
        }

    except Exception:
        return None

# ==========================================
# 메인 실행 루프 (병렬 처리)
# ==========================================
def main():
    print(f"경로 검색 : {target_root_folder}")
    csv_files = glob.glob(os.path.join(target_root_folder, "**", "*.csv"), recursive=True)
    print(f"총 {len(csv_files):,}개")

    if not csv_files:
        print("파일이 없음")
        return

    results = []
    
    max_workers = max(1, os.cpu_count() - 2) 
    print(f"병렬 처리 시작 (Workers: {max_workers})")

    # 병렬 처리 실행
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_segment, f): f for f in csv_files}
        
        # tqdm으로 진행 상황 표시 
        for future in tqdm(as_completed(futures), total=len(csv_files), desc="Processing", unit="file"):
            try:
                res = future.result()
                if res:
                    results.append(res)
            except Exception as e:
                pass

    # ==========================================
    # 결과 저장
    # ==========================================
    print("\n결과 집계 중")
    if results:
        result_df = pd.DataFrame(results)
        
        cols_order = [
            '차종', '단말기 번호', 'Non_Monotonic_Metric', 
            'TV', 'Net', 'Duration_Sec', 'Data_Count',
            'Mean_Mod_Temp', 'Mean_Ext_Temp', 
            'Slope_Mod_Temp', 'Last_Mod_Temp', 'Last_Ext_Temp', 
            'File_Path'
        ]
        
        # 존재하는 컬럼만 선택
        final_cols = [c for c in cols_order if c in result_df.columns]
        result_df = result_df[final_cols]
        
        # NM Metric 기준 내림차순 정렬
        result_df = result_df.sort_values(by='Non_Monotonic_Metric', ascending=False)
        
        result_df.to_csv(output_file_name, index=False, encoding='utf-8-sig')
        print(f"\총 {len(results):,}개 데이터가 '{output_file_name}'에 저장완료")
    else:
        print("\n분석 결과가 없음")

if __name__ == '__main__':
    main()