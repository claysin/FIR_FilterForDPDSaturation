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

# === Parâmetros ===========================================
Fs = 120e6
F_corte = 20e6
f_normalizada = F_corte / (Fs / 2)
ordem = 11
numero_de_coeficientes = ordem + 1
pontos = np.zeros(ordem, dtype=complex)



# APLICANDO A SATURAÇÃO -----------------------------------------------------------------------------------
L = 1.25
delta_w = ((2 * np.pi * 3.5e9) - (2 * np.pi * 2.4e9)) / 2
xn = envoltoria(sinal_wifi, sinal_LTE, Fs, delta_w)
x1c_s, x2c_s = saturacao_soma(xn, L, sinal_wifi, sinal_LTE)

#===========================================================
Nfft = 1024
vetor_comprimento = np.linspace(-np.pi, np.pi, Nfft)
frequencia_corte = f_normalizada * np.pi
H_d_ideal = np.zeros(Nfft)

for chave, valor in enumerate(vetor_comprimento):
    if np.abs(valor) <= frequencia_corte:
        H_d_ideal[chave] = 1


#===============================================================================================
#======================= RESPOSTA AO IMPULSO ===================================================
#===============================================================================================
h_t = np.fft.ifft(np.fft.ifftshift(H_d_ideal))
h_t = np.fft.ifftshift(h_t)

inicio = Nfft // 2 - ordem // 2
fim = inicio + numero_de_coeficientes
h_d = np.real(h_t[inicio:fim])

#================== PLOTAGEM DO FILTRO NO TEMPO E FREQUÊNCIA =============================

# fig, axs = plt.subplots(2, 1, figsize=(12, 6))
#
# # --- Resposta ao impulso (tempo) ---
# axs[0].stem(np.arange(numero_de_coeficientes) - (ordem // 2), h_d, linefmt='k-', markerfmt='ko', basefmt='k-')
# axs[0].grid(True, which='both')
# axs[0].set_xlabel(r'$n$')
# axs[0].set_ylabel(r'$h[n]$')
# axs[0].set_title(f"Resposta ao Impulso (Filtro FIR), Ordem={ordem}")
#
# # --- Resposta em frequência ---
# H_d_fft = np.fft.fft(h_d, Nfft)
# H_d_fft = np.fft.fftshift(H_d_fft)
#
# axs[1].plot(vetor_comprimento, np.abs(H_d_fft), lw=2, color='black', label='FIR Projetado')
# axs[1].plot(vetor_comprimento, H_d_ideal, ls='--', lw=2, color="red", label='Ideal')
# axs[1].grid(True, which='both')
# axs[1].margins(0.01)
# axs[1].set_xlabel(r'$\omega$ [rad/amostra]')
# axs[1].set_ylabel(r'$|H(e^{j\omega})|$')
# axs[1].set_title("Resposta em Frequência")
# axs[1].legend()
#
# plt.tight_layout()
# plt.show()


#============================== APLICAÇÃO DO FILTRO ======================================================

y_LTE_teste = py.signal.lfilter(h_d,1,x2c_s) # aplica o FIR no sinal complexo
y_wifi_teste = py.signal.lfilter(h_d,1,x1c_s) # aplica o FIR no sinal complexo

# COMPENSAR ATRADO DA CONVOLUÇÃO
atraso_um = (len(h_d) - 1) // 2
sinal_filtrado1 = np.roll(y_wifi_teste, -atraso_um)
sinal_filtrado2 = np.roll(y_LTE_teste, -atraso_um)

# RESULTADO FINAL ------------------------------------------------------------------------------------
sinal_LTE = sinal_filtrado2
sinal_wifi = sinal_filtrado1[:len(sinal_wifi1)]

# REAMOSTRAR PARA A FREQUÊNCIA ORIGINAL
interp_real_LTE_2 = interp1d(tempo_reamostrado_LTE, sinal_LTE.real, kind='linear')
interp_imag_LTE_2 = interp1d(tempo_reamostrado_LTE, sinal_LTE.imag, kind='linear')
sinal_real_LTE = interp_real_LTE_2(tempo_LTE_original)
sinal_imag_LTE = interp_imag_LTE_2(tempo_LTE_original)

interp_real_wifi_2 = interp1d(tempo_reamostrado_wifi, sinal_wifi.real, kind='linear')
interp_imag_wifi_2 = interp1d(tempo_reamostrado_wifi, sinal_wifi.imag, kind='linear')
sinal_real_wifi = interp_real_wifi_2(tempo_wifi_original)
sinal_imag_wifi = interp_imag_wifi_2(tempo_wifi_original)

# RESULTADO FINAL PARA SIMULAR NO CADENCE
cadence_sinal_wifi = separa_variaveis(sinal_real_wifi, sinal_imag_wifi)
cadence_sinal_LTE = separa_variaveis(sinal_real_LTE, sinal_imag_LTE)




#======================= SALVAR ================================================================
save_path = "/home/clayson/Área de trabalho/Projetos/python/FIR_Filter_For_DPD_Saturation/arquivos_salvos/IC2/janelaRetangular_frequencia/"

# np.savetxt(save_path + "IC2_filtrado_real_Wifi.csv", sinal_wifi.real, delimiter=",")
# np.savetxt(save_path + "IC2_filtrado_imag_Wifi.csv", sinal_wifi.imag, delimiter=",")
# np.savetxt(save_path + "IC2_filtrado_real_LTE.csv", sinal_LTE.real, delimiter=",")
# np.savetxt(save_path + "IC2_filtrado_imag_LTE.csv", sinal_LTE.imag, delimiter=",")

#========================== EXPORTAR PARA O CADENCE =============================================
save_path2 = "/home/clayson/Área de trabalho/Projetos/python/FIR_Filter_For_DPD_Saturation/arquivos_salvos/IC2/janelaRetangular_frequencia/Cadence/"

dados_LTE_real = np.column_stack((tempo_LTE_original, cadence_sinal_LTE.real))
dados_LTE_imag = np.column_stack((tempo_LTE_original, cadence_sinal_LTE.imag))
dados_wifi_real = np.column_stack((tempo_wifi_original, cadence_sinal_wifi.real))
dados_wifi_imag = np.column_stack((tempo_wifi_original, cadence_sinal_wifi.imag))

# np.savetxt(save_path2 + "cadence_wifi_real.pwl", dados_wifi_real, delimiter="  ")
# np.savetxt(save_path2 + "cadence_wifi_imag.pwl", dados_wifi_imag, delimiter="  ")
# np.savetxt(save_path2 + "cadence_LTE_real.pwl", dados_LTE_real, delimiter="  ")
# np.savetxt(save_path2 + "cadence_LTE_imag.pwl", dados_LTE_imag, delimiter="  ")



