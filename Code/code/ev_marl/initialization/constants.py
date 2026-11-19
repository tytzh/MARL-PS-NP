import os
from pathlib import Path

path_main = str(Path(__file__).resolve().parents[1])
path_data = path_main + "/0_Data/"
path_case = path_main + "/0_Data_cases/"
adr_Load_nonEV = path_data + "Load_NonEV_pu.mat"

scale_nonEV = 350
step_interval = 30
scores_average_window = 100
rate_CS_b = 10
rate_CS_c = 100
rate_PT_b = rate_CS_b
rate_PT_c = rate_CS_c
cap_trans = 500
