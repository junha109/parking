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
# 대상 단말기 리스트 및 경로 설정
# ==============================================================================
TARGET_TERMINAL_LIST = [
'01241248850' , '01241227999'
]

ROOT_DIR = r"Z:\SamsungSTF\Processed_Data\Parking\segment" 
OUTPUT_DIR = r"Z:\SamsungSTF\Processed_Data\Parking\plot\Weekly"

# ==============================================================================
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 20           
plt.rcParams['axes.titlesize'] = 45      
plt.rcParams['axes.labelsize'] = 40      
plt.rcParams['xtick.labelsize'] = 35     
plt.rcParams['ytick.labelsize'] = 35     

# ==============================================================================
# [함수] 보조 함수
# ==============================================================================
def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)


def parse_time_universal(df, time_col='time'):
    try:
        df[time_col] = df[time_col].astype(str).str.replace(r'\s+', ' ', regex=True).str.strip()
        df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
    except:
        pass
    return df

def get_week_start_sunday(date_obj):
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
    week_end_sat = week_start_date + timedelta(days=6, hours=23, minutes=59, seconds=59)
    if data_max_time > week_end_sat:
        final_end = data_max_time + timedelta(hours=1)
    else:
        final_end = week_end_sat
    ax.set_xlim(week_start_date, final_end)

# ==============================================================================
# [함수] Plotting 
# ==============================================================================
def plot_temp_correlation(df, save_path, file_name, avg_ext, avg_mod, car_model, term_id):
    if df.empty: return
    plt.figure(figsize=(14, 12))
    plt.scatter(df['ext_temp'], df['mod_avg_temp'], color='blue', alpha=0.5, s=8)
    
    title_str = f"{car_model} - {term_id}\n({file_name})"
    plt.title(title_str, pad=20)
    
    plt.xlabel('External Temp (°C)')
    plt.ylabel('Module Avg Temp (°C)')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    ax = plt.gca()
    add_avg_text(ax, avg_ext, avg_mod)
    
    plt.tight_layout()
    plt.savefig(f"{save_path}/{file_name}_temp.png")
    plt.close()

def plot_time_individual(df, save_path, file_name, week_start, avg_ext, avg_mod, car_model, term_id):
    if df.empty: return
    time_data = df['time']
    data_max_time = time_data.max()
    
    fig, ax = plt.subplots(5, 1, figsize=(26, 26), sharex=True)
    

    marker_size = 15
    
    for a in ax:
        set_xlim_dynamic(a, week_start, data_max_time)

    title_str = f"{car_model} - {term_id}\n({file_name})"
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

def plot_time_combined(df, save_path, file_name, week_start, avg_ext, avg_mod, car_model, term_id):
    if df.empty: return
    time_data = df['time']
    data_max_time = time_data.max()

    fig, ax = plt.subplots(2, 1, figsize=(26, 18), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    
    for a in ax:
        set_xlim_dynamic(a, week_start, data_max_time)

    title_str = f"{car_model} - {term_id}\n({file_name})"
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
# 메인 처리 루프
# ==============================================================================
def main():
    try:
        from GS_vehicle_dict import vehicle_dict
    except ImportError:
        print(" [오류] GS_vehicle_dict.py 파일을 찾을 수 없습니다.")
        exit()

    print(f"총 {len(TARGET_TERMINAL_LIST)}개의 단말기에 대해 시각화를 시작합니다.")

    for current_terminal_id in TARGET_TERMINAL_LIST:
        print("\n" + "="*60)
        print(f"▶ 처리 시작: {current_terminal_id}")
        print("="*60)

        found_car_model = None
        for car_model, terminal_list in vehicle_dict.items():
            if current_terminal_id in terminal_list:
                found_car_model = car_model
                break
        
        if not found_car_model:
            print(f" [Skip] 단말기 번호({current_terminal_id})를 vehicle_dict에서 찾을 수 없습니다.")
            continue

        terminal_path = os.path.join(ROOT_DIR, found_car_model, current_terminal_id)
        if not os.path.exists(terminal_path):
            print(f" [Skip] 데이터 폴더 없음: {terminal_path}")
            continue

        files_by_week = defaultdict(list)
        all_files = []
        month_folders = sorted([d for d in os.listdir(terminal_path) if d.isdigit() and len(d)==6])
        
        for month in month_folders:
            csvs = glob.glob(os.path.join(terminal_path, month, '*.csv'))
            all_files.extend(csvs)

        if not all_files:
            print(" [Skip] CSV 파일 없음")
            continue

        print(f" - 총 {len(all_files):,}개 파일 분류 중...")
        
        for f_path in tqdm(all_files, desc="Grouping", leave=False):
            try:
                df_head = pd.read_csv(f_path, nrows=1, usecols=['time'])
                if df_head.empty: continue
                

                time_val = df_head['time'].iloc[0]
                if isinstance(time_val, str):
                    time_val = time_val.replace(' PM', '').replace(' AM', '')
                start_time = pd.to_datetime(time_val, errors='coerce')
                
                week_sunday = get_week_start_sunday(start_time)
                if week_sunday:
                    files_by_week[week_sunday].append(f_path)
            except: continue

        sorted_weeks = sorted(files_by_week.keys())
        
        base_save_path = os.path.join(OUTPUT_DIR, found_car_model, current_terminal_id)
        dir_1 = os.path.join(base_save_path, "1_Temp_Correlation")
        dir_2 = os.path.join(base_save_path, "2_Time_Individual")
        dir_3 = os.path.join(base_save_path, "3_Time_Combined")
        create_directory(dir_1); create_directory(dir_2); create_directory(dir_3)
        
        print(f" - 저장 경로: {base_save_path}")

        use_cols = ['time', 'ext_temp', 'mod_avg_temp', 'pack_current', 'pack_volt', 'soc']

        for week_start in tqdm(sorted_weeks, desc=f"Plotting ({current_terminal_id})"):
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
            
            week_df = parse_time_universal(week_df)
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
                plot_temp_correlation(week_df, dir_1, file_name, avg_ext, avg_mod, found_car_model, current_terminal_id)
                plot_time_individual(week_df, dir_2, file_name, week_start, avg_ext, avg_mod, found_car_model, current_terminal_id)
                plot_time_combined(week_df, dir_3, file_name, week_start, avg_ext, avg_mod, found_car_model, current_terminal_id)
            except Exception as e:
                pass

    print("\n" + "="*60)
    print("모든 단말기 처리 완료!")
    print("="*60)

if __name__ == "__main__":
    main()