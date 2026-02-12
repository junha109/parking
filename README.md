# parking

Non_Monotonic.py 
: BMS 온도 데이터의 비단조성(NM) 지표를 계산하여 세그먼트별 온도 변화의 특성을 기록

plot_BMS-KMA.py 
: BMS 외기 온도와 기상청 관측 기온을 매칭하여 주차 상태(충전/Rest)별 상관관계를 산점도로 시각화

plot_weekly_mlt.py 
: 특정 단말기의 전류, 전압, SOC, 온도(ext,mod,weather)를 주 단위로 묶어 시계열 그래프로 생성

preprocessing_prk.py 
: speed와 Pack_current 조건을 기준으로 전체 데이터에서 주차 세그먼트를 분리하고 모듈 평균 온도를 계산하여 저장

merging_weather.py 
: 기상 데이터를 2초 단위로 선형 보간 처리하여 Parquet 형식으로 저장

arrange.py 
: 파일 내부의 실제 시간을 확인하여 폴더 경로를 재배치하고 파일명을 순차적으로 정리

check.py 
: EV6 및 Ioniq5의 Charging 세그먼트 내에 chrg_cable_conn 신호(0)가 포함되었는지 확인

check_mod_temp.py 
: SOC 변화 및 케이블 신호 기반 충전 분류, 배터리 모듈 온도 0도 발생 로그 기록

Chrg_Rest_Sep.py 
: Charging 세그먼트에서 실제 케이블 연결 전후의 Rest (10분 이상) 구간을 Rest 파일로 분리

classfied_ChargRest.py 
: SOC 상승량(5% 이상)을 기준으로 주차 데이터를 Charging과 Rest 폴더로 분류

deviation.py 
: 단말기별로 0도를 제외한 배터리 모듈 온도 간의 최대 편차를 분석

faulty_file.py 
: CSV에 명시된 불량 데이터 파일들을 일괄 삭제

filttering_ext_outlier.py 
: 외기 온도(ext_temp)가 모두 0인 이상치 파일을 탐색하고 단말기별 불량률 통계 생성

BMS_Weather_save 
: ext_temp와 기온을 세그먼트 파일별로 시간가중평균 계산 후 기록 + temp_diff(ext_temp-weather)기록

Clustering_Fianl 
: BMS_Weather_save 실행 후 output으로 클러스터링 (K 여러개 동시 실행 가능) + 결과 plot (scatter[ext vs Weather], box plot[temp_diff]) + 세그먼트가 속한 클러스터 기록

Clustering_monthly
: 월별로 실내/외 주차 유저로 클러스터링 

Usre_Categorization
: Clustering 후 유저가 속하고 있는 클러스터의 수를 기반으로 실내/실외/애매 주차 유저인지 판단
