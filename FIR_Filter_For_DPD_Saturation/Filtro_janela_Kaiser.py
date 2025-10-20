import math
import numpy as np
import pandas as pd
import scipy as py
from scipy.special import i0
import matplotlib.pyplot as plt
from Funções import juntar, envoltoria, saturacao_soma



# TRAZENDO ARQUIVOS DE REFERÊNCIA-------------------------------------------------------------------------------
loc_LTE = "sinal_LTE.csv"
loc_Wifi = "sinal_wifi.csv"
save_path = "/home/clayson/Área de trabalho/Projetos/python/FIR_Filter_For_DPD_Saturation/arquivos_salvos/IC2/Kaiser/"
sinal_LTE = pd.read_csv(loc_LTE)
sinal_Wifi = pd.read_csv(loc_Wifi)

# EXTRAÇÃO DOS DADOS DOS ARQUIVOS (tempo| n° reais| n° imaginários)

tempo_wifi = np.array(sinal_Wifi.iloc[:, 0], dtype=float)
real_wifi = np.array(sinal_Wifi.iloc[:, 1], dtype=float)
imag_wifi = np.array(sinal_Wifi.iloc[:, 2], dtype=float)

tempo_LTE = np.array(sinal_LTE.iloc[:, 0], dtype=float)
real_LTE = np.array(sinal_LTE.iloc[:, 1], dtype=float)
imag_LTE = np.array(sinal_LTE.iloc[:, 2], dtype=float)

# INTERPOLAÇÃO DO SINAL ---------------------------------------------------------------------

interpolacao_LTE_real = py.interpolate.interp1d(tempo_LTE, real_LTE)
interpolacao_LTE_imag = py.interpolate.interp1d(tempo_LTE, imag_LTE)

interpolacao_Wifi_real = py.interpolate.interp1d(tempo_wifi, real_wifi)
interpolacao_Wifi_imag = py.interpolate.interp1d(tempo_wifi, imag_wifi)

# REAMOSTRAGEM DOS SINAIS ---------------------------------------------------------------------

fs = 120e6

tempo_reamostrado_LTE = np.linspace(tempo_LTE[0], tempo_LTE[-1],int(fs * tempo_LTE[-1]))
tempo_reamostrado_wifi = np.linspace(tempo_wifi[0], tempo_wifi[-1],int(fs * tempo_wifi[-1]))

sinal_LTE_reamostrado_real = interpolacao_LTE_real(tempo_reamostrado_LTE)
sinal_LTE_reamostrado_imag = interpolacao_LTE_imag(tempo_reamostrado_LTE)

sinal_Wifi_reamostrado_real = interpolacao_Wifi_real(tempo_reamostrado_wifi)
sinal_Wifi_reamostrado_imag = interpolacao_Wifi_imag(tempo_reamostrado_wifi)


# JUNTAR PARTES REAIS E IMAGINARIAS --------------------------------------------------------------

vetor_wifi = juntar(sinal_Wifi_reamostrado_real, sinal_Wifi_reamostrado_imag)
vetor_LTE = juntar(sinal_LTE_reamostrado_real, sinal_LTE_reamostrado_imag)

# REPLICAR O VETOR WIFI PARA TER O MESMO TAMANHO QUE O VETOR LTE -----------------------------------------

x = int(len(vetor_LTE) / len(vetor_wifi)) + 1
vetor_wifi_replicado = np.tile(vetor_wifi, x)[:len(vetor_LTE)]

# DADOS DO TRABALHO -------------------------------------------------------------------------------------

L = 1.25 # Limiar de corte para a saturação
delta_w = ((2 * np.pi * 3.5e9) - (2 * np.pi * 2.4e9)) / 2


# Aplicação da envoltória e saturação dos sinais ---------------------------------------------------------

xn = envoltoria(vetor_wifi_replicado, vetor_LTE, fs, delta_w)
x1c_s, x2c_s = saturacao_soma(xn, L, vetor_wifi_replicado, vetor_LTE)




# Plotagem dos sinais saturados --------------------------------------------------------------------------

# xn2 = envoltoria(x1c_s, x2c_s, fs, delta_w)
# plt.subplots()
# # plt.title("Soma")
# plt.xlabel("Tempo (μs)")
# plt.ylabel("Amplitude (V)")
# plt.xlim([0, 0.8e-5])
# plt.plot(tempo_reamostrado_LTE, abs(xn), label="Sinais de entrada")
# plt.plot(tempo_reamostrado_LTE, abs(xn2), "--", label="Sinal saturado", color='red')
# #plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x * 1e6:.0f}'))
# plt.grid()
# plt.legend()
# plt.show()

# DEFININDO OS PARAMETROS PARA A ORDEM DO FILTRO -----------------------------------------------------------

f_trans_LTE = 19.5e6
f_norm_LTE = f_trans_LTE / (fs / 2)
f_trans_wifi = 15.25e6
f_norm_wifi = f_trans_wifi / (fs / 2)
K_hanning = 3.3

N_LTE = math.ceil(K_hanning / f_norm_LTE) # número de coeficientes
N_wifi = math.ceil(K_hanning / f_norm_wifi)# número de coeficientes

print("(HANNING)n° de coeficientes LTE: ", N_LTE)
print("(HANNING)n° de coeficientes wifi: ", N_wifi)


# APLICAÇÃO DO FILTRO HANNING --------------------------------------------------------------------------------
coeficientes1 = py.signal.firwin(N_wifi, f_norm_wifi, window="hann")
coeficientes2 = py.signal.firwin(N_wifi, f_norm_LTE, window="hann")
sinal_filtrado1 = py.signal.lfilter(coeficientes1, 1, x1c_s)
sinal_filtrado2 = py.signal.lfilter(coeficientes2, 1, x2c_s)
atraso1 = (len(coeficientes1) - 1) // 2
atraso2 = (len(coeficientes2) - 1) // 2
sinal_filtrado1_corrigido = np.roll(sinal_filtrado1, -atraso1)
sinal_filtrado2_corrigido = np.roll(sinal_filtrado2, -atraso2)
sinal_LTE_hann = sinal_filtrado2_corrigido
sinal_wifi_hann = sinal_filtrado1_corrigido[:len(vetor_wifi_replicado)]


# APLICAÇÃO DO FILTRO KAISAR ----------------------------------------------------------------------------------

ripple = 0.05
atenuacao = -20 * np.log10(ripple)

def beta(A):

    if A > 50:
        valor_beta = 0.1102 * (A - 8.7)
    elif 21 <= A <= 50:
        valor_beta = (0.5842 * ((A - 21) ** 0.4)) + (0.07886 * (A - 21))
    else:
        valor_beta = 0.0

    return  valor_beta

# Ordem do filtro kaiser
frequencia_transicao = 35e6
omega_delta = 2 * np.pi * (frequencia_transicao / fs)
M = np.ceil((atenuacao - 8) / (2.285 * omega_delta))
N = int(M + 1)

n = np.arange(0, N) # indice
m = M // 2 # valor central

hd_LTE = np.sinc(2 * f_trans_LTE / fs * (n - m))
hd_wifi = np.sinc(2 * f_trans_wifi / fs * (n - m))

w = i0(beta(atenuacao) * np.sqrt(1 - ((2 * n / M) - 1)**2)) / i0(beta(atenuacao))

h_LTE = hd_LTE * w #coeficientes
h_wifi = hd_wifi * w #coeficientes

print("(KAISAR)n° de coeficientes LTE: ", len(h_LTE))
print("(KAISAR)n° de coeficientes wifi: ", len(h_wifi))

h_LTE = h_LTE / np.sum(h_LTE) # Foi preciso pois o sinal estava sendo amplificado
h_wifi = h_wifi / np.sum(h_wifi) # Foi preciso pois o sinal estava sendo amplificado


y_LTE = py.signal.lfilter(h_LTE,1,x2c_s) # aplica o FIR no sinal complexo
y_wifi = py.signal.lfilter(h_wifi,1,x1c_s) # aplica o FIR no sinal complexo

atraso1 = (len(h_wifi) - 1) // 2
atraso2 = (len(h_LTE) - 1) // 2
sinal_filtrado1_kaisar = np.roll(y_wifi, -atraso1)
sinal_filtrado2_kaisar = np.roll(y_LTE, -atraso2)
sinal_LTE_kaiser = sinal_filtrado2_kaisar
sinal_wifi_kaiser = sinal_filtrado1_kaisar[:len(vetor_wifi_replicado)]


sinal_wifi_para_envoltoria = sinal_filtrado1_kaisar
envoltoria_sinal_final = envoltoria(vetor_LTE, sinal_wifi_para_envoltoria, fs, delta_w)
envoltoria_sinal_saturado = envoltoria(x1c_s, x2c_s, fs, delta_w)
esf1, esf2 = saturacao_soma(envoltoria_sinal_final, L, sinal_wifi_para_envoltoria, sinal_LTE_kaiser)
xf2 = envoltoria(esf1, esf2, fs, delta_w)


vetor_de_diferenca_kaiser = np.maximum((abs(xf2 )- abs(envoltoria_sinal_saturado)), 0)

fig, ax = plt.subplots(2,1, figsize=(10,6))
ax[0].set_xlabel("Time (μs)")
ax[0].set_ylabel("Amplitude (V)")
# ax[0].set_xlim([0.25e-5, 0.35e-5])
ax[0].set_xlim([0.6e-5, 1.3e-5])
# ax[0].xlim([0.25e-5, 0.7e-5])
# ax[0].ylim([1, 1.6])
ax[0].plot(tempo_reamostrado_LTE, abs(xn), label="Input Signal")
ax[0].plot(tempo_reamostrado_LTE, abs(envoltoria_sinal_saturado), "--", label="Saturated Signal", color='red')
ax[0].plot(tempo_reamostrado_LTE, abs(xf2), "--", label="Filtered Signal", color='black')
ax[0].grid()
ax[0].legend()

ax[1].stem(tempo_reamostrado_LTE,vetor_de_diferenca_kaiser,
    label=(
        f"Média = {np.mean(abs(vetor_de_diferenca_kaiser)):.3e}"
    )
)

# ax[1].set_xlim([0.25e-5, 0.35e-5])
ax[1].set_xlim([0.6e-5, 1.3e-5])
ax[1].set_ylim([0, 0.4])
ax[1].grid()
ax[1].legend()
plt.tight_layout()
plt.show()





























# SALVAR DOCUMENTOS --------------------------------------------------------------------------------------------

# np.savetxt(save_path + "1entrada_real_Wifi.csv", vetor_wifi_replicado.real, delimiter=",")
# np.savetxt(save_path + "1entrada_imag_Wifi.csv", vetor_wifi_replicado.imag, delimiter=",")
# np.savetxt(save_path + "1entrada_real_LTE.csv", vetor_LTE.real, delimiter=",")
# np.savetxt(save_path + "1entrada_imag_LTE.csv", vetor_LTE.imag, delimiter=",")
#
# np.savetxt(save_path + "2saturado_real_Wifi.csv", x1c_s.real, delimiter=",")
# np.savetxt(save_path + "2saturado_imag_Wifi.csv", x1c_s.imag, delimiter=",")
# np.savetxt(save_path + "2saturado_real_LTE.csv", x2c_s.real, delimiter=",")
# np.savetxt(save_path + "2saturado_imag_LTE.csv", x2c_s.imag, delimiter=",")
#
# np.savetxt(save_path + "4filtrado_real_Wifi_hann.csv", sinal_wifi_hann.real, delimiter=",")
# np.savetxt(save_path + "4filtrado_imag_Wifi_hann.csv", sinal_wifi_hann.imag, delimiter=",")
# np.savetxt(save_path + "4filtrado_real_LTE_hann.csv", sinal_LTE_hann.real, delimiter=",")
# np.savetxt(save_path + "4filtrado_imag_LTE_hann.csv", sinal_LTE_hann.imag, delimiter=",")

np.savetxt(save_path + "filtrado_real_Wifi_kaiser.csv", sinal_wifi_kaiser.real, delimiter=",")
np.savetxt(save_path + "filtrado_imag_Wifi_kaiser.csv", sinal_wifi_kaiser.imag, delimiter=",")
np.savetxt(save_path + "filtrado_real_LTE_kaiser.csv", sinal_LTE_kaiser.real, delimiter=",")
np.savetxt(save_path + "filtrado_imag_LTE_kaiser.csv", sinal_LTE_kaiser.imag, delimiter=",")











