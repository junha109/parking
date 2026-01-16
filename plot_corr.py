import os
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm

# 경고 무시
warnings.filterwarnings('ignore')

# ==============================================================================
# 경로 및 시각화 설정
# ==============================================================================
ROOT_DIR = r"Z:\SamsungSTF\Processed_Data\Parking\segment"
OUTPUT_DIR = r"Z:\SamsungSTF\Processed_Data\Parking\plot"

# 폰트 설정
try:
    font_path = "C:/Windows/Fonts/malgun.ttf"
    font_name = fm.FontProperties(fname=font_path).get_name()
    plt.rc('font', family=font_name)
except:
    print("맑은 고딕 폰트를 찾을 수 없어 기본 폰트 사용")

plt.rc('axes', unicode_minus=False)

# 폰트 사이즈 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rc('font', size=45)          
plt.rc('axes', titlesize=45)     
plt.rc('axes', labelsize=45)    
plt.rc('xtick', labelsize=35)    
plt.rc('ytick', labelsize=35)       
plt.rc('figure', titlesize=50) 
plt.rc('legend', fontsize=35)

try:
    from GS_vehicle_dict import vehicle_dict
except ImportError:
    print("GS_vehicle_dict.py를 찾을 수 없음")
    vehicle_dict = {}


# ==============================================================================
def parse_time_universal(df, time_col='time'):
    try:
        df[time_col] = df[time_col].astype(str).str.replace(r'\s+', ' ', regex=True).str.strip()
        df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
    except Exception:
        pass
    return df

def find_car_model_by_terminal(target_terminal, v_dict):
    if v_dict:
        for car_model, terminals in v_dict.items():
            if target_terminal in terminals:
                return car_model
    return "Unknown_Car"

def load_single_vehicle_data(car_model, terminal_id):
    print(f"데이터 로딩 시작: {car_model} - {terminal_id}")
    search_pattern = os.path.join(ROOT_DIR, car_model, terminal_id, '**', '*.csv')
    csv_files = glob.glob(search_pattern, recursive=True)

    if not csv_files:
        print(f"파일 없음: {search_pattern}")
        return pd.DataFrame()

    dfs = []
    for file in tqdm(csv_files, desc="Loading", leave=False):
        try:
            temp_df = pd.read_csv(file, usecols=['time', 'ext_temp', 'mod_avg_temp'])
            dfs.append(temp_df)
        except:
            continue
    
    if not dfs: return pd.DataFrame()

    full_df = pd.concat(dfs, ignore_index=True)
    full_df = parse_time_universal(full_df, 'time')
    full_df = full_df.dropna(subset=['time', 'ext_temp', 'mod_avg_temp'])
    full_df['month'] = full_df['time'].dt.month
    
    print(f"로드 완료: {len(full_df):,} 건")
    return full_df

# ==============================================================================
# 시각화 메인 함수 
# ==============================================================================
def plot_final_visualization(df, car_model, terminal_id):
    if df.empty: return

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print("그래프 그리는 중...")

    total_corr = df['ext_temp'].corr(df['mod_avg_temp'])
    total_n = len(df)

    # 1. 캔버스 생성
    fig, ax = plt.subplots(figsize=(28, 14)) 

    # 2. 산점도
    sns.regplot(data=df, x='ext_temp', y='mod_avg_temp', ax=ax,
                scatter_kws={'s': 50, 'alpha': 0.5, 'color': 'royalblue', 'edgecolor': 'none'}, 
                line_kws={'color': 'crimson', 'linewidth': 6}) 

    # 3. 텍스트 박스
    info_text = f'Total Corr: {total_corr:.3f}\nTotal N: {total_n:,}'
    bbox_props = dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.9)
    
    ax.text(0.95, 0.15, info_text, transform=ax.transAxes, fontsize=40,
            verticalalignment='center', horizontalalignment='right', bbox=bbox_props)

    # 4. 타이틀 및 축
    ax.set_title(f"[{car_model} - {terminal_id}] 외기온도 Vs 모듈평균온도 상관관계 (Total Corr: {total_corr:.3f})", 
                 fontweight='bold', pad=40) 
    
    ax.set_xlabel('외기 온도 (°C)', fontweight='bold', labelpad=20)
    ax.set_ylabel('모듈 평균 온도 (°C)', fontweight='bold', labelpad=20)
    ax.grid(True, linestyle='--', alpha=0.5)

    # 5. 저장 및 메모리 해제 
    plt.tight_layout()

    save_filename = f"{car_model}_{terminal_id}_Total_Correlation_Plot_Clean.png"
    save_path = os.path.join(OUTPUT_DIR, save_filename)
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close() 
    
    print(f"\n[완료] 그래프 저장됨: {save_path}")

# ==============================================================================
# 실행
# ==============================================================================
if __name__ == "__main__":
    

    TARGET_TERMINAL_ID = "01241248850"


    print(f"Target Terminal: {TARGET_TERMINAL_ID}")
    
    target_car = find_car_model_by_terminal(TARGET_TERMINAL_ID, vehicle_dict)
    print(f"Car Model: {target_car}")

    df_res = load_single_vehicle_data(target_car, TARGET_TERMINAL_ID)

    if not df_res.empty:
        plot_final_visualization(df_res, target_car, TARGET_TERMINAL_ID)
    else:

        print("데이터가 없습니다.")
