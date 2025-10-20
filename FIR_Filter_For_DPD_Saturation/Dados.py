import matplotlib.pyplot as plt
import numpy as np
import scipy as py
from scipy.interpolate import interp1d
import pandas as pd
from Funções import separa_variaveis, envoltoria, saturacao_soma, PAPR
# OBTENDO DADOS PARA TRABALHO -------------------------------------------------------------

url_sinal_LTE = 'sinal_LTE.csv'
url_sinal_wifi = 'sinal_wifi.csv'

projeto_LTE = pd.read_csv(url_sinal_LTE)
projeto_wifi = pd.read_csv(url_sinal_wifi)

tempo_LTE_original = np.array(projeto_LTE.iloc[:, 0], dtype=float)
parte_imaginaria_LTE = np.array(projeto_LTE.iloc[:, 2], dtype=float)
parte_real_LTE = np.array(projeto_LTE.iloc[:, 1], dtype=float)

tempo_wifi_original = np.array(projeto_wifi.iloc[:, 0], dtype=float)
parte_imaginaria_wifi = np.array(projeto_wifi.iloc[:, 2], dtype=float)
parte_real_wifi = np.array(projeto_wifi.iloc[:, 1], dtype=float)



fs = 120e6
interp_real_LTE = interp1d(tempo_LTE_original, parte_real_LTE, kind='linear')
interp_imag_LTE = interp1d(tempo_LTE_original, parte_imaginaria_LTE, kind='linear')
tempo_reamostrado_LTE = np.linspace(tempo_LTE_original[0], tempo_LTE_original[-1],int(fs * tempo_LTE_original[-1]))

sinal_real_reamostrado_LTE = interp_real_LTE(tempo_reamostrado_LTE)
sinal_imag_reamostrado_LTE = interp_imag_LTE(tempo_reamostrado_LTE)


interp_real_wifi = interp1d(tempo_wifi_original, parte_real_wifi, kind='linear')
interp_imag_wifi = interp1d(tempo_wifi_original, parte_imaginaria_wifi, kind='linear')
tempo_reamostrado_wifi = np.linspace(tempo_wifi_original[0], tempo_wifi_original[-1],int(fs * tempo_wifi_original[-1]))

sinal_real_reamostrado_wifi = interp_real_wifi(tempo_reamostrado_wifi)
sinal_imag_reamostrado_wifi = interp_imag_wifi(tempo_reamostrado_wifi)


sinal_wifi1 = separa_variaveis(sinal_real_reamostrado_wifi, sinal_imag_reamostrado_wifi)
sinal_LTE = separa_variaveis(sinal_real_reamostrado_LTE, sinal_imag_reamostrado_LTE)

# CONCATENAR SINAL WIFI
r = int(len(sinal_LTE) / len(sinal_wifi1)) + 1
sinal_wifi = np.tile(sinal_wifi1, r)
sinal_wifi = sinal_wifi[0:len(sinal_LTE)]

#___________________________________________________________________________________________________________
# DADOS DO TRABALHO ---------------------------------------------------------------------------------------

f_amostragem = 120e6 # Frequência de amostragem do projeto
f_transicao1 = 20e6 # Banda de transição
f_transicao2 = 20e6 # Banda de transição
fc_normalizada1 = f_transicao1 / (f_amostragem / 2)

#___________________________________________________________________________________________________________
# APLICANDO A SATURAÇÃO -----------------------------------------------------------------------------------
L = 1.25
delta_w = ((2 * np.pi * 3.5e9) - (2 * np.pi * 2.4e9)) / 2
xn = envoltoria(sinal_wifi, sinal_LTE, f_amostragem, delta_w)
sinal_wifi_saturado, sinal_LTE_saturado = saturacao_soma(xn, L, sinal_wifi, sinal_LTE)
#======================================================================================================