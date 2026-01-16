import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from tqdm import tqdm
import warnings
import numpy as np
from collections import defaultdict
from datetime import timedelta

# 경고 메시지 무시
warnings.filterwarnings('ignore')

# ==============================================================================
TARGET_TERMINAL_ID = '01241228144'  # 테스트할 단말기 번호
ROOT_DIR = r"Z:\SamsungSTF\Processed_Data\Parking\segment" 
OUTPUT_DIR = r"Z:\SamsungSTF\Processed_Data\Parking\plot\Weekly"


plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

#  폰트 사이즈
plt.rcParams['font.size'] = 20           
plt.rcParams['axes.titlesize'] = 30      # 제목 (차종-단말기)
plt.rcParams['axes.labelsize'] = 24      # 축 이름 (Time, Voltage...)

#  Grid 글자 크기 
plt.rcParams['xtick.labelsize'] = 24     # X축 눈금 (날짜)
plt.rcParams['ytick.labelsize'] = 24     # Y축 눈금 (숫자)

# ==============================================================================
# 차종 자동 검색
# ==============================================================================
try:
    from GS_vehicle_dict import vehicle_dict
    
    found_car_model = None
    for car_model, terminal_list in vehicle_dict.items():
        if TARGET_TERMINAL_ID in terminal_list:
            found_car_model = car_model
            break
    
    if found_car_model:
        print(f" 단말기 확인 완료: {TARGET_TERMINAL_ID}")
        print(f" 매칭된 차종: {found_car_model}")
        TEST_CAR_MODEL = found_car_model
    else:
        print(f" 오류: 딕셔너리에서 단말기 번호({TARGET_TERMINAL_ID})를 찾을 수 없습니다.")
        exit()

except ImportError:
    print(" [오류] GS_vehicle_dict.py 파일을 찾을 수 없습니다.")
    exit()

# ==============================================================================
# 보조 함수
# ==============================================================================
def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

def parse_time_column(df, time_col='time'):
    try:
        df[time_col] = pd.to_datetime(df[time_col], format='%Y-%m-%d %I:%M:%S %p')
    except Exception:
        df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
    return df

def get_week_start_sunday(date_obj):
    """ 입력된 날짜가 포함된 주의 '일요일' 날짜 반환 """
    if pd.isna(date_obj): return None
    days_to_subtract = (date_obj.weekday() + 1) % 7
    sunday = date_obj - timedelta(days=days_to_subtract)
    return sunday.replace(hour=0, minute=0, second=0, microsecond=0)

def add_avg_text(ax, avg_ext, avg_mod):
    text_str = f"Avg Ext: {avg_ext:.1f}°C\nAvg Mod: {avg_mod:.1f}°C"
    props = dict(boxstyle='round', facecolor='white', alpha=0.7, edgecolor='gray')
    ax.text(0.99, 0.95, text_str, transform=ax.transAxes, fontsize=20,
            verticalalignment='top', horizontalalignment='right', bbox=props, color='black')

def set_xlim_dynamic(ax, week_start_date, data_max_time):
    """ X축 범위 설정 (데이터가 토요일 넘어가면 자동 확장) """
    week_end_sat = week_start_date + timedelta(days=6, hours=23, minutes=59, seconds=59)
    if data_max_time > week_end_sat:
        final_end = data_max_time + timedelta(hours=1)
    else:
        final_end = week_end_sat
    ax.set_xlim(week_start_date, final_end)

# ==============================================================================
# [함수] Plotting
# ==============================================================================
def plot_temp_correlation(df, save_path, file_name, avg_ext, avg_mod):
    if df.empty: return
    plt.figure(figsize=(14, 12))
    plt.scatter(df['ext_temp'], df['mod_avg_temp'], color='blue', alpha=0.5, s=5)
    
    title_str = f"{TEST_CAR_MODEL} - {TARGET_TERMINAL_ID}\n({file_name})"
    plt.title(title_str, pad=20)
    
    plt.xlabel('External Temp (°C)')
    plt.ylabel('Module Avg Temp (°C)')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    ax = plt.gca()
    add_avg_text(ax, avg_ext, avg_mod)
    
    plt.tight_layout()
    plt.savefig(f"{save_path}/{file_name}_temp.png")
    plt.close()

def plot_time_individual(df, save_path, file_name, week_start, avg_ext, avg_mod):
    if df.empty: return
    time_data = df['time']
    data_max_time = time_data.max()
    
    fig, ax = plt.subplots(5, 1, figsize=(26, 26), sharex=True)
    marker_size = 4
    
    for a in ax:
        set_xlim_dynamic(a, week_start, data_max_time)


    title_str = f"{TEST_CAR_MODEL} - {TARGET_TERMINAL_ID}\n({file_name})"
    ax[0].set_title(title_str, pad=30)

    # 1. Current
    ax[0].scatter(time_data, df['pack_current'], color='red', s=marker_size)
    ax[0].set_ylabel('Current (A)')
    ax[0].grid(True)

    # 2. Voltage
    ax[1].scatter(time_data, df['pack_volt'], color='green', s=marker_size)
    ax[1].set_ylabel('Voltage (V)')
    ax[1].grid(True)

    # 3. SOC
    ax[2].scatter(time_data, df['soc'], color='blue', s=marker_size)
    ax[2].set_ylabel('SOC (%)')
    ax[2].grid(True)

    # 4. Ext Temp
    ax[3].scatter(time_data, df['ext_temp'], color='orange', s=marker_size)
    ax[3].set_ylabel('Ext Temp (°C)')
    ax[3].grid(True)
    add_avg_text(ax[3], avg_ext, avg_mod)

    # 5. Mod Avg Temp
    ax[4].scatter(time_data, df['mod_avg_temp'], color='purple', s=marker_size)
    ax[4].set_ylabel('Mod Avg Temp (°C)')
    ax[4].set_xlabel('Time')
    ax[4].grid(True)
    add_avg_text(ax[4], avg_ext, avg_mod)

    ax[4].xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax[4].xaxis.set_major_formatter(mdates.DateFormatter('%m-%d\n(%a)'))
    plt.xticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(f"{save_path}/{file_name}_individual.png")
    plt.close(fig)

def plot_time_combined(df, save_path, file_name, week_start, avg_ext, avg_mod):
    if df.empty: return
    time_data = df['time']
    data_max_time = time_data.max()

    fig, ax = plt.subplots(2, 1, figsize=(26, 18), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    
    for a in ax:
        set_xlim_dynamic(a, week_start, data_max_time)

    # [수정] 타이틀 변경: 차종 - 단말기번호
    title_str = f"{TEST_CAR_MODEL} - {TARGET_TERMINAL_ID}\n({file_name})"
    ax[0].set_title(title_str, pad=30)

    # Row 1
    host = ax[0]
    par1 = host.twinx()
    par2 = host.twinx()
    par2.spines["right"].set_position(("axes", 1.12)) 

    s_size = 1.5
    host.scatter(time_data, df['pack_current'], color='red', s=s_size)
    par1.scatter(time_data, df['pack_volt'], color='green', s=s_size)
    par2.scatter(time_data, df['soc'], color='blue', s=s_size)

    host.set_ylabel("Current (A)", color='red')
    par1.set_ylabel("Voltage (V)", color='green')
    par2.set_ylabel("SOC (%)", color='blue')

    host.tick_params(axis='y', labelcolor='red')
    par1.tick_params(axis='y', labelcolor='green')
    par2.tick_params(axis='y', labelcolor='blue')

    c_min, c_max = df['pack_current'].min(), df['pack_current'].max()
    host.set_ylim(c_min - 10 if pd.notna(c_min) else -10, c_max + 10 if pd.notna(c_max) else 10)
    par2.set_ylim(0, 100)
    host.grid(True, linestyle='--', alpha=0.6)

    # Row 2
    ax[1].scatter(time_data, df['ext_temp'], color='orange', s=s_size)
    ax[1].scatter(time_data, df['mod_avg_temp'], color='purple', s=s_size)
    
    ax[1].set_ylabel("Temp (°C)")
    ax[1].set_xlabel("Time")
    ax[1].grid(True, linestyle='--', alpha=0.6)
    
    add_avg_text(ax[1], avg_ext, avg_mod)

    ax[1].xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax[1].xaxis.set_major_formatter(mdates.DateFormatter('%m-%d\n(%a)'))
    plt.xticks(rotation=0)

    plt.tight_layout()
    plt.savefig(f"{save_path}/{file_name}_combined.png")
    plt.close(fig)

# ==============================================================================
# [메인 실행 로직]
# ==============================================================================
def main():
    print(f"\n Start Processing (Title Changed): {TEST_CAR_MODEL} - {TARGET_TERMINAL_ID}")
    
    terminal_path = os.path.join(ROOT_DIR, TEST_CAR_MODEL, TARGET_TERMINAL_ID)
    if not os.path.exists(terminal_path):
        print(f" [오류] 경로가 존재하지 않습니다.")
        return

    # 1. 파일 스캔 및 '주(Week)'별 분류
    files_by_week = defaultdict(list)
    all_files = []
    month_folders = sorted([d for d in os.listdir(terminal_path) if d.isdigit() and len(d)==6])
    
    for month in month_folders:
        csvs = glob.glob(os.path.join(terminal_path, month, '*.csv'))
        all_files.extend(csvs)

    if not all_files:
        print(" [오류] 처리할 파일이 없습니다.")
        return

    print(f" 총 {len(all_files):,}개의 세그먼트 파일을 스캔하여 주(Week)별로 분류합니다...")
    
    for f_path in tqdm(all_files, desc="Grouping Files"):
        try:
            df_head = pd.read_csv(f_path, nrows=1, usecols=['time'])
            if df_head.empty: continue
            
            start_time = pd.to_datetime(df_head['time'].iloc[0])
            week_sunday = get_week_start_sunday(start_time)
            
            if week_sunday:
                files_by_week[week_sunday].append(f_path)
        except:
            continue

    # 2. 주별 시각화
    sorted_weeks = sorted(files_by_week.keys())
    
    base_save_path = os.path.join(OUTPUT_DIR, TEST_CAR_MODEL, TARGET_TERMINAL_ID)
    dir_1 = os.path.join(base_save_path, "1_Temp_Correlation")
    dir_2 = os.path.join(base_save_path, "2_Time_Individual")
    dir_3 = os.path.join(base_save_path, "3_Time_Combined")
    create_directory(dir_1); create_directory(dir_2); create_directory(dir_3)
    
    print(f" 결과 저장 경로: {base_save_path}")
    print(f" 총 {len(sorted_weeks)}주(Week)의 데이터를 시각화합니다.")

    use_cols = ['time', 'ext_temp', 'mod_avg_temp', 'pack_current', 'pack_volt', 'soc']

    for week_start in tqdm(sorted_weeks, desc="Plotting Weeks"):
        file_list = files_by_week[week_start]
        if not file_list: continue
        
        week_dfs = []
        for f in file_list:
            try:
                header = pd.read_csv(f, nrows=0).columns
                cols = [c for c in use_cols if c in header]
                if 'time' not in cols: continue
                chunk = pd.read_csv(f, usecols=cols)
                week_dfs.append(chunk)
            except: continue
        
        if not week_dfs: continue
        week_df = pd.concat(week_dfs, ignore_index=True)
        
        week_df = parse_time_column(week_df)
        week_df = week_df.dropna(subset=['time']).sort_values('time').reset_index(drop=True)
        
        for c in ['ext_temp', 'mod_avg_temp', 'pack_current', 'pack_volt', 'soc']:
            if c in week_df.columns:
                week_df[c] = pd.to_numeric(week_df[c], errors='coerce')
        
        if week_df.empty: continue

        week_end = week_start + timedelta(days=6)
        file_name = f"{week_start.strftime('%Y%m%d')}_{week_end.strftime('%Y%m%d')}"
        
        avg_ext = week_df['ext_temp'].mean()
        avg_mod = week_df['mod_avg_temp'].mean()
        
        try:
            plot_temp_correlation(week_df, dir_1, file_name, avg_ext, avg_mod)
            plot_time_individual(week_df, dir_2, file_name, week_start, avg_ext, avg_mod)
            plot_time_combined(week_df, dir_3, file_name, week_start, avg_ext, avg_mod)
        except Exception as e:
            print(f" Error plotting {file_name}: {e}")

    print("\n 작업 완료!")

if __name__ == "__main__":
    main()