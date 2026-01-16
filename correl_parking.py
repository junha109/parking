import os
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings

# 경고 무시
warnings.filterwarnings('ignore')


# ==============================================================================
ROOT_DIR = r"Z:\SamsungSTF\Processed_Data\Parking\segment"
OUTPUT_DIR = r"Z:\SamsungSTF\Processed_Data\Parking"

OUTPUT_FILENAME = "Terminal_Temp_Correlation_Weighted_Result.csv"

try:
    from GS_vehicle_dict import vehicle_dict
except ImportError:
    print(" GS_vehicle_dict.py를 찾을 수 없습니다.")
    exit()

# ==============================================================================
# 시간 포맷 자동 감지
# ==============================================================================
def parse_time_universal(df, time_col='time'):
    try:
        df[time_col] = df[time_col].astype(str).str.replace(r'\s+', ' ', regex=True).str.strip()
        df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
    except Exception as e:
        print(f"   파싱 중 오류 발생: {e}")
    return df

# ==============================================================================
# 가중 평균 계산 
# ==============================================================================
def calculate_weighted_avg(group):
    """
    차종별 가중 평균 상관계수 계산
    """
    summary = {}
    
    # 1. 전체(Total) 가중 평균
    valid_total = group.dropna(subset=['Total_Corr'])
    valid_total = valid_total[valid_total['Total_Count'] > 0]
    
    if not valid_total.empty:
        weighted_sum = (valid_total['Total_Corr'] * valid_total['Total_Count']).sum()
        total_n = valid_total['Total_Count'].sum()
        summary['Total_Corr'] = weighted_sum / total_n
    else:
        summary['Total_Corr'] = None
    
    # Total Count 합계
    summary['Total_Count'] = group['Total_Count'].sum()

    # 2. 시즌별 가중 평균
    season_cols = [c for c in group.columns if any(x in c for x in ['Winter', 'Spring', 'Summer', 'Autumn'])]
    season_names = [c for c in season_cols if not c.endswith('_Count')]
    
    for season in season_names:
        corr_col = season
        count_col = f"{season}_Count"
        
        if count_col in group.columns:
            valid = group.dropna(subset=[corr_col])
            valid = valid[valid[count_col] > 0]
            
            if not valid.empty:
                weighted_sum = (valid[corr_col] * valid[count_col]).sum()
                total_n = valid[count_col].sum()
                summary[corr_col] = weighted_sum / total_n
            else:
                summary[corr_col] = None
            
            # 시즌별 Count 합계
            summary[count_col] = group[count_col].sum()
                
    return pd.Series(summary)


# ==============================================================================
def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    result_data = []
    print(" 상관관계 분석 시작...")

    # 시즌 정의
    seasons = {
        'Winter (12~2)': [12, 1, 2],
        'Spring (3~5)': [3, 4, 5],
        'Summer (6~8)': [6, 7, 8],
        'Autumn (9~11)': [9, 10, 11]
    }

    for car_model, terminals in vehicle_dict.items():
        for terminal_id in terminals:
            print(f"\nProcessing: {car_model} - {terminal_id}")

            search_pattern = os.path.join(ROOT_DIR, car_model, terminal_id, '**', '*.csv')
            csv_files = glob.glob(search_pattern, recursive=True)

            if not csv_files: continue

            dfs = []
            for file in tqdm(csv_files, desc=f"  Loading", leave=False):
                try:
                    temp_df = pd.read_csv(file, usecols=['time', 'ext_temp', 'mod_avg_temp'])
                    dfs.append(temp_df)
                except: continue
            
            if not dfs: continue

            full_df = pd.concat(dfs, ignore_index=True)
            full_df = parse_time_universal(full_df, 'time')
            full_df = full_df.dropna(subset=['time', 'ext_temp', 'mod_avg_temp'])
            
            if full_df.empty: continue

            full_df['month'] = full_df['time'].dt.month

            # 전체 기간 상관관계
            total_corr = full_df['ext_temp'].corr(full_df['mod_avg_temp'])
            total_count = len(full_df)
            
            row_result = {
                'Car_Model': car_model,
                'Terminal_ID': terminal_id,
                'Total_Corr': round(total_corr, 4) if not pd.isna(total_corr) else None,
                'Total_Count': total_count
            }

            # 시즌별 상관관계
            for season_name, months in seasons.items():
                season_df = full_df[full_df['month'].isin(months)]
                count_val = len(season_df)
                
                # 데이터가 3개 초과일 때만 계산
                if count_val > 3:
                    corr_val = season_df['ext_temp'].corr(season_df['mod_avg_temp'])
                    if not pd.isna(corr_val):
                        row_result[season_name] = round(corr_val, 4)
                        row_result[f"{season_name}_Count"] = count_val
                    else:
                        row_result[season_name] = None
                        row_result[f"{season_name}_Count"] = count_val
                else:
                    row_result[season_name] = None
                    row_result[f"{season_name}_Count"] = count_val

            result_data.append(row_result)

    # -------------------------------------------------------
    # CSV 병합
    # -------------------------------------------------------
    if result_data:

        detail_df = pd.DataFrame(result_data)
        
        # 컬럼 정렬
        base_cols = ['Car_Model', 'Terminal_ID', 'Total_Corr', 'Total_Count']
        season_cols = []
        for s in seasons.keys():
            season_cols.append(s)
            season_cols.append(f"{s}_Count")
        
        final_cols = base_cols + [c for c in season_cols if c in detail_df.columns]
        detail_df = detail_df[final_cols]

        print("-" * 50)
        print("차종별 가중 평균 계산...")
        
        summary_df = detail_df.groupby('Car_Model').apply(calculate_weighted_avg).reset_index()
        
        summary_df['Terminal_ID'] = "[Weighted_Avg_Result]"
        

        summary_df = summary_df[detail_df.columns]


        final_df = pd.concat([detail_df, summary_df], ignore_index=True)
        
        # 3-4. CSV 저장
        save_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
        final_df.to_csv(save_path, index=False, encoding='utf-8-sig')
        
        print("\n" + "="*50)
        print("="*50)
        print(summary_df[['Car_Model', 'Terminal_ID', 'Total_Corr', 'Total_Count']])

    else:
        print("\n 데이터가 없음")

if __name__ == "__main__":

    main()
