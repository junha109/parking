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
from matplotlib.lines import Line2D

# 경고 메시지 무시
warnings.filterwarnings('ignore')

# ==============================================================================
# 대상 단말기 및 경로
# ==============================================================================
TARGET_TERMINAL_LIST = [
 '01241248850'
]

ROOT_DIR = r"Z:\SamsungSTF\Processed_Data\Parking\Classified2"
OUTPUT_DIR = r"Z:\SamsungSTF\Processed_Data\Parking\plot\Weekly"

# 기상청 데이터 경로
WEATHER_FILE_PATH = r"Z:\SamsungSTF\Processed_Data\Parking\code\weather_2023_2024_linear.parquet"

# ==============================================================================
#  그래프 스타일
# ==============================================================================
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 20
plt.rcParams['axes.titlesize'] = 35
plt.rcParams['axes.labelsize'] = 35
plt.rcParams['xtick.labelsize'] = 25
plt.rcParams['ytick.labelsize'] = 25

# Y축 라벨 위치 고정
FIXED_Y_LABEL_POS = -0.04

# ==============================================================================
#  데이터 로드 및 전처리
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

def determine_state_by_data(df):
    """
    세그먼트 내 SOC 변화량(종료값 - 시작값)이 +5 이상이면 Charging,
    그렇지 않으면 Rest로 판단.
    """
    if df.empty or 'soc' not in df.columns:
        return 'Rest'
    
    # 결측치 제외한 유효 SOC 데이터만 추출
    valid_soc = df['soc'].dropna()
    
    if valid_soc.empty:
        return 'Rest'
    
    # 시작 SOC와 종료 SOC 비교
    soc_start = valid_soc.iloc[0]
    soc_end = valid_soc.iloc[-1]
    
    delta_soc = soc_end - soc_start
    
    # SOC가 5% 이상 증가했으면 충전으로 판단
    if delta_soc >= 5:
        return 'Charging'
    else:
        return 'Rest'

def load_weather_data(path):
    print(f" 기상청 데이터 로드 시도: {path}")
    if not os.path.exists(path):
        print(f" 파일이 존재하지 않습니다: {path}")
        return pd.DataFrame()
    
    try:
        df = pd.read_parquet(path)
        
        # 인덱스 리셋
        df = df.reset_index()

        rename_map = {}
        for col in df.columns:
            if col.lower() == 'timestamp': rename_map[col] = 'time'
            elif col.lower() == 'temperature': rename_map[col] = 'weather_temp'
        
        if rename_map: df.rename(columns=rename_map, inplace=True)
        
        if 'time' not in df.columns:
            return pd.DataFrame()
        
        # Unix Timestamp 변환
        if pd.api.types.is_numeric_dtype(df['time']):
            df['time'] = pd.to_datetime(df['time'], unit='ms')
        else:
            df = parse_time_universal(df, 'time')

        # Timezone 제거
        if pd.api.types.is_datetime64_any_dtype(df['time']):
            if df['time'].dt.tz is not None:
                df['time'] = df['time'].dt.tz_localize(None)
        
        return df.sort_values('time')

    except Exception as e:
        print(f" 기상청 로드 오류: {e}")
        return pd.DataFrame()


def add_avg_text(ax, avg_ext, avg_mod, avg_weather, method="Seg Avg"):
    text_str = f"[{method}]\nAvg Ext: {avg_ext:.1f}°C\nAvg Mod: {avg_mod:.1f}°C\nAvg Weather: {avg_weather:.1f}°C"
    props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray')
    ax.text(0.99, 0.95, text_str, transform=ax.transAxes, fontsize=20,
            verticalalignment='top', horizontalalignment='right', bbox=props, color='black')

def set_xlim_dynamic(ax, week_start_date, data_max_time):
    week_end_sat = week_start_date + timedelta(days=6, hours=23, minutes=59, seconds=59)
    final_end = max(data_max_time, week_end_sat)
    ax.set_xlim(week_start_date, final_end)

# ==============================================================================
#  Plotting Logic
# ==============================================================================

def plot_time_individual(segment_data_list, weather_df, save_path, file_name, week_start, avg_ext, avg_mod, avg_weather, car_model, term_id):
    if not segment_data_list: return
    
    full_df = pd.concat([item[0] for item in segment_data_list], ignore_index=True)
    data_max_time = full_df['time'].max()
    
    week_weather = pd.DataFrame()
    if not weather_df.empty:
        w_end = max(data_max_time, week_start + timedelta(days=7))
        week_weather = weather_df[(weather_df['time'] >= week_start) & (weather_df['time'] <= w_end)]

    fig, ax = plt.subplots(5, 1, figsize=(26, 26), sharex=True)
    
    for a in ax:
        set_xlim_dynamic(a, week_start, data_max_time)

    title_str = f"{car_model} - {term_id}\n({file_name})"

    ax[0].set_title(title_str, pad=40)

    # 색상 매핑
    color_map = {'Charging': 'tab:red', 'Rest': 'tab:blue'}

    # [1~3번] Current, Voltage, SOC
    for seg_df, state in segment_data_list:
        t = seg_df['time']
        c = color_map.get(state, 'black')
        
        ax[0].plot(t, seg_df['pack_current'], color=c, linewidth=2)
        ax[1].plot(t, seg_df['pack_volt'], color=c, linewidth=2)
        ax[2].plot(t, seg_df['soc'], color=c, linewidth=2)

    ax[0].set_ylabel('Current (A)'); ax[0].yaxis.set_label_coords(FIXED_Y_LABEL_POS, 0.5); ax[0].grid(True)
    ax[1].set_ylabel('Voltage (V)'); ax[1].yaxis.set_label_coords(FIXED_Y_LABEL_POS, 0.5); ax[1].grid(True)
    ax[2].set_ylabel('SOC (%)');     ax[2].yaxis.set_label_coords(FIXED_Y_LABEL_POS, 0.5); ax[2].grid(True)

    # [4번] BMS Ext Temp
    if not week_weather.empty:
        ax[3].plot(week_weather['time'], week_weather['weather_temp'], 
                   color='grey', linestyle='--', linewidth=2, label='Weather (KMA)', alpha=0.7)
    
    for seg_df, state in segment_data_list:
        t = seg_df['time']
        c = color_map.get(state, 'black')
        ax[3].plot(t, seg_df['ext_temp'], color=c, linewidth=2)

    ax[3].set_ylabel('Ext Temp (°C)')
    ax[3].yaxis.set_label_coords(FIXED_Y_LABEL_POS, 0.5)
    ax[3].grid(True)
    # avg_weather 전달
    add_avg_text(ax[3], avg_ext, avg_mod, avg_weather, method="Seg Avg")

    # [5번] BMS Mod Avg Temp
    if not week_weather.empty:
        ax[4].plot(week_weather['time'], week_weather['weather_temp'], 
                   color='grey', linestyle='--', linewidth=2, label='Weather (KMA)', alpha=0.7)
    
    for seg_df, state in segment_data_list:
        t = seg_df['time']
        c = color_map.get(state, 'black')
        ax[4].plot(t, seg_df['mod_avg_temp'], color=c, linewidth=2)

    ax[4].set_ylabel('Mod Avg Temp (°C)')
    ax[4].yaxis.set_label_coords(FIXED_Y_LABEL_POS, 0.5)
    ax[4].set_xlabel('Time')
    ax[4].grid(True)
    # avg_weather 전달
    add_avg_text(ax[4], avg_ext, avg_mod, avg_weather, method="Seg Avg")

    ax[4].xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax[4].xaxis.set_major_formatter(mdates.DateFormatter('%m-%d\n(%a)'))

    # 범례 설정
    legend_elements = [
        Line2D([0], [0], color='grey', lw=2, linestyle='--', label='Weather (KMA)'),
        Line2D([0], [0], color='tab:red', lw=2, label='Charging'),
        Line2D([0], [0], color='tab:blue', lw=2, label='Rest')
    ]
    
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.95), ncol=3, fontsize=25)
    
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(f"{save_path}/{file_name}_individual.png")
    plt.close(fig)

# ==============================================================================
# 실행 로직
# ==============================================================================
def main():
    try:
        from GS_vehicle_dict import vehicle_dict
    except ImportError:
        print(" GS_vehicle_dict.py 파일 없음")
        exit()

    print(" 기상청 데이터 로딩 중")
    weather_df = load_weather_data(WEATHER_FILE_PATH)
    print(" 기상청 데이터 로딩 완료")

    print(f"총 {len(TARGET_TERMINAL_LIST)}개 시각화")

    for current_terminal_id in TARGET_TERMINAL_LIST:
        print("\n" + "="*60)
        print(f" 처리 시작: {current_terminal_id}")
        print("="*60)

        found_car_model = None
        for car_model, terminal_list in vehicle_dict.items():
            if current_terminal_id in terminal_list:
                found_car_model = car_model
                break
        
        if not found_car_model:
            print(f" 미등록 단말기: {current_terminal_id}")
            continue

        terminal_path = os.path.join(ROOT_DIR, found_car_model, current_terminal_id)
        if not os.path.exists(terminal_path):
            print(f" 경로 없음: {terminal_path}")
            continue

        files_by_week = defaultdict(list)
        all_files = []
        month_folders = sorted([d for d in os.listdir(terminal_path) if d.isdigit() and len(d)==6])
        
        for month in month_folders:
            search_pattern = os.path.join(terminal_path, month, '**', '*.csv')
            csvs = glob.glob(search_pattern, recursive=True)
            all_files.extend(csvs)

        if not all_files:
            print(" CSV 파일 없음")
            continue
        
        # 파일들을 주차(Sunday) 기준으로 그룹핑
        for f_path in tqdm(all_files, desc="Grouping", leave=False):
            try:
                df_head = pd.read_csv(f_path, nrows=1, usecols=['time'])
                if df_head.empty: continue
                
                time_val = str(df_head['time'].iloc[0]).replace(' PM', '').replace(' AM', '')
                start_time = pd.to_datetime(time_val, errors='coerce')
                
                week_sunday = get_week_start_sunday(start_time)
                if week_sunday:
                    files_by_week[week_sunday].append(f_path)
            except: continue

        sorted_weeks = sorted(files_by_week.keys())
        
        base_save_path = os.path.join(OUTPUT_DIR, found_car_model, current_terminal_id)
        dir_2 = os.path.join(base_save_path, "2_Time_Individual")
        create_directory(dir_2)
        
        use_cols = ['time', 'ext_temp', 'mod_avg_temp', 'pack_current', 'pack_volt', 'soc']

        for week_start in tqdm(sorted_weeks, desc=f"Plotting ({current_terminal_id})"):
            file_list = files_by_week[week_start]
            if not file_list: continue
            
            week_segment_data = [] 
            seg_ext_means = []
            seg_mod_means = []
            seg_weather_means = [] #  기상청 온도 평균 저장용 리스트

            for f in file_list:
                try:
                    # 데이터 읽기
                    header = pd.read_csv(f, nrows=0).columns
                    cols = [c for c in use_cols if c in header]
                    if 'time' not in cols: continue
                    chunk = pd.read_csv(f, usecols=cols)
                    
                    chunk = parse_time_universal(chunk)
                    chunk = chunk.dropna(subset=['time']).sort_values('time').reset_index(drop=True)
                    
                    if chunk.empty: continue

                    for c in ['ext_temp', 'mod_avg_temp', 'pack_current', 'pack_volt', 'soc']:
                        if c in chunk.columns:
                            chunk[c] = pd.to_numeric(chunk[c], errors='coerce')
                    
                    # 평균 계산 (BMS)
                    if 'ext_temp' in chunk.columns:
                        m_ext = chunk['ext_temp'].mean()
                        if not pd.isna(m_ext): seg_ext_means.append(m_ext)
                    
                    if 'mod_avg_temp' in chunk.columns:
                        m_mod = chunk['mod_avg_temp'].mean()
                        if not pd.isna(m_mod): seg_mod_means.append(m_mod)

                    #해당 세그먼트 기간 동안의 Weather 평균 계산
                    if not weather_df.empty:
                        chunk_start = chunk['time'].min()
                        chunk_end = chunk['time'].max()
                        
                        # 해당 시간대 필터링
                        mask = (weather_df['time'] >= chunk_start) & (weather_df['time'] <= chunk_end)
                        weather_slice = weather_df.loc[mask, 'weather_temp']
                        
                        if not weather_slice.empty:
                            w_mean = weather_slice.mean()
                            if not pd.isna(w_mean): seg_weather_means.append(w_mean)

                    state = determine_state_by_data(chunk)

                    # 리스트에 저장 (데이터, 상태)
                    week_segment_data.append((chunk, state))

                except: continue
            
            if not week_segment_data: continue
            
            # 최종 평균 계산
            final_avg_ext = np.mean(seg_ext_means) if seg_ext_means else 0.0
            final_avg_mod = np.mean(seg_mod_means) if seg_mod_means else 0.0
            # 기상청 최종 평균 계산
            final_avg_weather = np.mean(seg_weather_means) if seg_weather_means else 0.0

            week_end = week_start + timedelta(days=6)
            file_name = f"{week_start.strftime('%Y%m%d')}_{week_end.strftime('%Y%m%d')}"
            
            try:
                plot_time_individual(week_segment_data, weather_df, dir_2, file_name, week_start, 
                                     final_avg_ext, final_avg_mod, final_avg_weather, 
                                     found_car_model, current_terminal_id)
            except Exception as e:
                print(f" [Error] Plotting fail ({file_name}): {e}")
                pass

    print("\n" + "="*60)
    print("파일 처리 완료")
    print("="*60)

if __name__ == "__main__":
    main()