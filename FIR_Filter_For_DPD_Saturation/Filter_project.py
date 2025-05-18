import pandas as pd
import numpy as np
import scipy as py
import matplotlib.ticker as ticker
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from Funções import separa_variaveis, envoltoria, saturacao_soma, PAPR

# OBTENDO DADOS PARA TRABALHO -------------------------------------------------------------

url_sinal_LTE = 'sinal_LTE.csv'
url_sinal_wifi = 'sinal_wifi.csv'
#save_path = "xxx\xxxx\xxxx"

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

sigma = 0.001 # valor do ripple
f_amostragem = 120e6 # Frequência de amostragem do projeto
f_transicao1 = 20e6 # Banda de transição
f_transicao2 = 20e6 # Banda de transição
fc_normalizada1 = f_transicao1 / (f_amostragem / 2)

#___________________________________________________________________________________________________________
# APLICANDO A SATURAÇÃO -----------------------------------------------------------------------------------
L = 1.25
delta_w = ((2 * np.pi * 3.5e9) - (2 * np.pi * 2.4e9)) / 2
xn = envoltoria(sinal_wifi, sinal_LTE, f_amostragem, delta_w)
x1c_s, x2c_s = saturacao_soma(xn, L, sinal_wifi, sinal_LTE)

# plt.subplots()
# plt.xlabel("Time (μs)")
# plt.ylabel("Amplitude (V)")
# plt.xlim([50e-6, 60e-6])
# plt.plot(tempo_reamostrado_LTE, abs(sinal_wifi), label="Input signal")
# plt.plot(tempo_reamostrado_LTE, abs(x1c_s), "--", label="Saturated signal", color='red')
# plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x * 1e6:.0f}'))
# plt.legend()
# plt.grid()
# plt.show()

# plt.subplots()
# # plt.title("Soma sinal 2")
# plt.xlabel("Time (μs)")
# plt.ylabel("Amplitude (V)")
# plt.xlim([0, 0.8e-5])
# plt.plot(tempo_reamostrado_LTE, abs(sinal_LTE), label="Input signal")
# plt.plot(tempo_reamostrado_LTE, abs(x2c_s), "--", label="Saturated signal", color='red')
# plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x * 1e6:.0f}'))
# plt.grid()
# plt.legend()

xn2 = envoltoria(x1c_s, x2c_s, f_amostragem, delta_w)
# plt.subplots()
# #plt.title("Sum")
# plt.xlabel("Time (μs)")
# plt.ylabel("Amplitude (V)")
# plt.xlim([0, 0.8e-5])
# plt.plot(tempo_reamostrado_LTE, abs(xn), label="Input Signal")
# plt.plot(tempo_reamostrado_LTE, abs(xn2), "--", label="Saturated Signal", color='red')
# plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x * 1e6:.0f}'))
# plt.grid()
# plt.legend()
#-------------------------------------------------------------------------------------------
#DEFININDO JANELA TRIANGULAR

coeficientes1 = py.signal.firwin(51, fc_normalizada1, window="triang")
coeficientes2 = py.signal.firwin(81, fc_normalizada1, window="triang")
sinal_filtrado1 = py.signal.lfilter(coeficientes1, 1, x1c_s)
sinal_filtrado2 = py.signal.lfilter(coeficientes2, 1, x2c_s)
atraso1 = (len(coeficientes1) - 1) // 2
atraso2 = (len(coeficientes2) - 1) // 2
sinal_filtrado1_corrigido = np.roll(sinal_filtrado1, -atraso1)
sinal_filtrado2_corrigido = np.roll(sinal_filtrado2, -atraso2)
sinal_LTE_triangular = sinal_filtrado2_corrigido
sinal_wifi_triangular = sinal_filtrado1_corrigido[:len(sinal_wifi1)]

#DEFININDO JANELA HANNING

coeficientes1 = py.signal.firwin(51, fc_normalizada1, window="hann")
coeficientes2 = py.signal.firwin(81, fc_normalizada1, window="hann")
sinal_filtrado1 = py.signal.lfilter(coeficientes1, 1, x1c_s)
sinal_filtrado2 = py.signal.lfilter(coeficientes2, 1, x2c_s)
atraso1 = (len(coeficientes1) - 1) // 2
atraso2 = (len(coeficientes2) - 1) // 2
sinal_filtrado1_corrigido = np.roll(sinal_filtrado1, -atraso1)
sinal_filtrado2_corrigido = np.roll(sinal_filtrado2, -atraso2)
sinal_LTE_hann = sinal_filtrado2_corrigido
sinal_wifi_hann = sinal_filtrado1_corrigido[:len(sinal_wifi1)]
xf = envoltoria(sinal_filtrado1_corrigido, sinal_filtrado2_corrigido, f_amostragem, delta_w)
x1f_s, x2f_s = saturacao_soma(xf, L, sinal_filtrado1_corrigido, sinal_filtrado2_corrigido)
xf2 = envoltoria(x1f_s, x2f_s, f_amostragem, delta_w)

# plt.subplots()
# #plt.title("Sum")
# plt.xlabel("Time (μs)")
# plt.ylabel("Amplitude (V)")
# plt.xlim([0.25e-5, 0.35e-5])
# plt.ylim([1, 1.6])
# plt.plot(tempo_reamostrado_LTE, abs(xn), label="Input Signal")
# plt.plot(tempo_reamostrado_LTE, abs(xn2), "--", label="Saturated Signal", color='red')
# plt.plot(tempo_reamostrado_LTE, abs(xf2), "--", label="Filtered Signal", color='black')
# plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x * 1e6:.1f}'))
# plt.grid()
# plt.legend()



#DEFININDO JANELA BLACKMAN

coeficientes1 = py.signal.firwin(51, fc_normalizada1, window="blackman")
coeficientes2 = py.signal.firwin(81, fc_normalizada1, window="blackman")
sinal_filtrado1 = py.signal.lfilter(coeficientes1, 1, x1c_s)
sinal_filtrado2 = py.signal.lfilter(coeficientes2, 1, x2c_s)
atraso1 = (len(coeficientes1) - 1) // 2
atraso2 = (len(coeficientes2) - 1) // 2
sinal_filtrado1_corrigido = np.roll(sinal_filtrado1, -atraso1)
sinal_filtrado2_corrigido = np.roll(sinal_filtrado2, -atraso2)
sinal_LTE_blackman = sinal_filtrado2_corrigido
sinal_wifi_blackman = sinal_filtrado1_corrigido[:len(sinal_wifi1)]

#DEFININDO JANELA HAMMING

coeficientes1 = py.signal.firwin(51, fc_normalizada1, window="hamming")
coeficientes2 = py.signal.firwin(81, fc_normalizada1, window="hamming")
sinal_filtrado1 = py.signal.lfilter(coeficientes1, 1, x1c_s)
sinal_filtrado2 = py.signal.lfilter(coeficientes2, 1, x2c_s)
atraso1 = (len(coeficientes1) - 1) // 2
atraso2 = (len(coeficientes2) - 1) // 2
sinal_filtrado1_corrigido = np.roll(sinal_filtrado1, -atraso1)
sinal_filtrado2_corrigido = np.roll(sinal_filtrado2, -atraso2)
sinal_LTE_hamming = sinal_filtrado2_corrigido
sinal_wifi_hamming = sinal_filtrado1_corrigido[:len(sinal_wifi1)]

#DEFININDO JANELA BARTLETT

coeficientes1 = py.signal.firwin(51, fc_normalizada1, window="bartlett")
coeficientes2 = py.signal.firwin(81, fc_normalizada1, window="bartlett")
sinal_filtrado1 = py.signal.lfilter(coeficientes1, 1, x1c_s)
sinal_filtrado2 = py.signal.lfilter(coeficientes2, 1, x2c_s)
atraso1 = (len(coeficientes1) - 1) // 2
atraso2 = (len(coeficientes2) - 1) // 2
sinal_filtrado1_corrigido = np.roll(sinal_filtrado1, -atraso1)
sinal_filtrado2_corrigido = np.roll(sinal_filtrado2, -atraso2)
sinal_LTE_bartlett = sinal_filtrado2_corrigido
sinal_wifi_bartlett = sinal_filtrado1_corrigido[:len(sinal_wifi1)]

#DEFININDO JANELA LANCZOS

coeficientes1 = py.signal.firwin(51, fc_normalizada1, window="lanczos")
coeficientes2 = py.signal.firwin(81, fc_normalizada1, window="lanczos")
sinal_filtrado1 = py.signal.lfilter(coeficientes1, 1, x1c_s)
sinal_filtrado2 = py.signal.lfilter(coeficientes2, 1, x2c_s)
atraso1 = (len(coeficientes1) - 1) // 2
atraso2 = (len(coeficientes2) - 1) // 2
sinal_filtrado1_corrigido = np.roll(sinal_filtrado1, -atraso1)
sinal_filtrado2_corrigido = np.roll(sinal_filtrado2, -atraso2)
sinal_LTE_lanczos = sinal_filtrado2_corrigido
sinal_wifi_lanczos = sinal_filtrado1_corrigido[:len(sinal_wifi1)]
#--------------------------------------------------------------------------------------------

# #PAPR__________________________________________________________________________________________________
# print("SEM SATURAÇÃO")
# #sinal wifi
# print(f"PAPR do sinal wifi : {PAPR(sinal_wifi)} dB ")
# #sinal LTE
# print(f"PAPR do sinal LTE : {PAPR(sinal_LTE)} dB ")
# # envoltória
# print(f"PAPR da envoltória : {PAPR(xn)} dB ")

# print("COM SATURAÇÃO")
# #sinal wifi
# print(f"PAPR do sinal wifi : {PAPR(x1c_s)} dB ")
# #sinal LTE
# print(f"PAPR do sinal LTE : {PAPR(x2c_s)} dB ")
# print(f"PAPR da envoltória: {PAPR(xn2)} dB ")

# print("FILTRADO")
# #sinal wifi
# print(f"PAPR do sinal wifi : {PAPR(sinal_filtrado1_corrigido)} dB ")
# #sinal LTE
# print(f"PAPR do sinal LTE : {PAPR(sinal_filtrado2_corrigido)} dB ")
# print(f"PAPR da envoltória : {PAPR(xf)} dB ")

# print(f"Maximo valor no vetor de entrada wifi: {max(abs(sinal_wifi1))}")
# print(f"Maximo valor no vetor de entrada LTE: {max(abs(sinal_LTE))}")
# print(f"Maximo valor no vetor de envoltoria : {max(abs(xn))}")
# print(f"Maximo valor no vetor de envoltoria : {max(abs(xn2))}")
# print(f"Maximo valor no vetor de envoltoria : {max(abs(xf))}")

# print(f"O maior valor de amplitude para o sinal saturado wifi: {max(abs(x1c_s))}")
# print(f"O maior valor de amplitude para o sinal saturado LTE: {max(abs(x2c_s))}")
# print(f"O maior valor de amplitude para a envoltoria saturada: {max(abs(xn2))}")
# print("")
# print(f"O maior valor de amplitude para o sinal  wifi: {max(abs(sinal_wifi))}")
# print(f"O maior valor de amplitude para o sinal  LTE: {max(abs(sinal_LTE))}")
# print(f"O maior valor de amplitude para a envoltória : {max(abs(xn))}")
# print("")
# print(f"O maior valor de amplitude para o sinal  wifi filtrado: {max(abs(sinal_filtrado1_corrigido))}")
# print(f"O maior valor de amplitude para o sinal  LTE filtrado: {max(abs(sinal_filtrado2_corrigido))}")
# print(f"O maior valor de amplitude para a envoltória filtrada : {max(abs(xf))}")

#-------------------------------------------------------------------------------------------
# SALVAR DOCUMENTOS

# np.savetxt(save_path + "1entrada_real_Wifi.csv", sinal_wifi.real, delimiter=",")
# np.savetxt(save_path + "1entrada_imag_Wifi.csv", sinal_wifi.imag, delimiter=",")
# np.savetxt(save_path + "1entrada_real_LTE.csv", sinal_LTE.real, delimiter=",")
# np.savetxt(save_path + "1entrada_imag_LTE.csv", sinal_LTE.imag, delimiter=",")
#
# np.savetxt(save_path + "2saturado_real_Wifi.csv", x1c_s.real, delimiter=",")
# np.savetxt(save_path + "2saturado_imag_Wifi.csv", x1c_s.imag, delimiter=",")
# np.savetxt(save_path + "2saturado_real_LTE.csv", x2c_s.real, delimiter=",")
# np.savetxt(save_path + "2saturado_imag_LTE.csv", x2c_s.imag, delimiter=",")
#
# np.savetxt(save_path + "3filtrado_real_Wifi_triangular.csv", sinal_wifi_triangular.real, delimiter=",")
# np.savetxt(save_path + "3filtrado_imag_Wifi_triangular.csv", sinal_wifi_triangular.imag, delimiter=",")
# np.savetxt(save_path + "3filtrado_real_LTE_triangular.csv", sinal_LTE_triangular.real, delimiter=",")
# np.savetxt(save_path + "3filtrado_imag_LTE_triangular.csv", sinal_LTE_triangular.imag, delimiter=",")
#
# np.savetxt(save_path + "4filtrado_real_Wifi_hann.csv", sinal_wifi_hann.real, delimiter=",")
# np.savetxt(save_path + "4filtrado_imag_Wifi_hann.csv", sinal_wifi_hann.imag, delimiter=",")
# np.savetxt(save_path + "4filtrado_real_LTE_hann.csv", sinal_LTE_hann.real, delimiter=",")
# np.savetxt(save_path + "4filtrado_imag_LTE_hann.csv", sinal_LTE_hann.imag, delimiter=",")
#
# np.savetxt(save_path + "5filtrado_real_Wifi_blackman.csv", sinal_wifi_blackman.real, delimiter=",")
# np.savetxt(save_path + "5filtrado_imag_Wifi_blackman.csv", sinal_wifi_blackman.imag, delimiter=",")
# np.savetxt(save_path + "5filtrado_real_LTE_blackman.csv", sinal_LTE_blackman.real, delimiter=",")
# np.savetxt(save_path + "5filtrado_imag_LTE_blackman.csv", sinal_LTE_blackman.imag, delimiter=",")
#
# np.savetxt(save_path + "6filtrado_real_Wifi_hamming.csv", sinal_wifi_hamming.real, delimiter=",")
# np.savetxt(save_path + "6filtrado_imag_Wifi_hamming.csv", sinal_wifi_hamming.imag, delimiter=",")
# np.savetxt(save_path + "6filtrado_real_LTE_hamming.csv", sinal_LTE_hamming.real, delimiter=",")
# np.savetxt(save_path + "6filtrado_imag_LTE_hamming.csv", sinal_LTE_hamming.imag, delimiter=",")
#
# np.savetxt(save_path + "7filtrado_real_Wifi_bartlett.csv", sinal_wifi_bartlett.real, delimiter=",")
# np.savetxt(save_path + "7filtrado_imag_Wifi_bartlett.csv", sinal_wifi_bartlett.imag, delimiter=",")
# np.savetxt(save_path + "7filtrado_real_LTE_bartlett.csv", sinal_LTE_bartlett.real, delimiter=",")
# np.savetxt(save_path + "7filtrado_imag_LTE_bartlett.csv", sinal_LTE_bartlett.imag, delimiter=",")
#
# np.savetxt(save_path + "8filtrado_real_Wifi_lanczos.csv", sinal_wifi_lanczos.real, delimiter=",")
# np.savetxt(save_path + "8filtrado_imag_Wifi_lanczos.csv", sinal_wifi_lanczos.imag, delimiter=",")
# np.savetxt(save_path + "8filtrado_real_LTE_lanczos.csv", sinal_LTE_lanczos.real, delimiter=",")
# np.savetxt(save_path + "8filtrado_imag_LTE_lanczos.csv", sinal_LTE_lanczos.imag, delimiter=",")