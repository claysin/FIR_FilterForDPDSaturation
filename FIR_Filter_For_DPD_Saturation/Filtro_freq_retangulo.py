import matplotlib.pyplot as plt
import numpy as np
import scipy as py
from scipy.interpolate import interp1d
import pandas as pd
from Funções import separa_variaveis, envoltoria, saturacao_soma
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

# =================================== Parâmetros =======================================================
Fs = 120e6
F_corte = 20e6
f_normalizada = F_corte / Fs
ordem_wifi = 13
ordem_LTE = 13
numero_de_coeficientes_wifi = ordem_wifi + 1
numero_de_coeficientes_LTE = ordem_LTE + 1
pontos_wifi = np.zeros(ordem_wifi, dtype=complex)
pontos_LTE = np.zeros(ordem_LTE, dtype=complex)



# APLICANDO A SATURAÇÃO -----------------------------------------------------------------------------------
L = 1.25
delta_w = ((2 * np.pi * 3.5e9) - (2 * np.pi * 2.4e9)) / 2
xn = envoltoria(sinal_wifi, sinal_LTE, Fs, delta_w) # Envoltória sem saturação
x1c_s, x2c_s = saturacao_soma(xn, L, sinal_wifi, sinal_LTE)

envoltoria_sinal_saturado = envoltoria(x1c_s, x2c_s, Fs, delta_w)

#===========================================================
Nfft = 1024
vetor_comprimento = np.linspace(-np.pi, np.pi, Nfft)
# frequencia_corte = f_normalizada * (2 * np.pi)

frequencia_corte_LTE = 1.6
frequencia_corte_wifi = 1

H_d_ideal_wifi = np.zeros(Nfft)
H_d_ideal_LTE = np.zeros(Nfft)

for chave, valor in enumerate(vetor_comprimento):
    if np.abs(valor) <= frequencia_corte_wifi:
        H_d_ideal_wifi[chave] = 1

for chave, valor in enumerate(vetor_comprimento):
    if np.abs(valor) <= frequencia_corte_LTE:
        H_d_ideal_LTE[chave] = 1


#===============================================================================================
#======================= RESPOSTA AO IMPULSO ===================================================
#===============================================================================================
h_t_wifi = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(H_d_ideal_wifi)))
h_t_LTE = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(H_d_ideal_LTE)))


inicio_wifi = Nfft // 2 - ordem_wifi // 2
fim_wifi = inicio_wifi + numero_de_coeficientes_wifi
h_d_wifi = np.real(h_t_wifi[inicio_wifi:fim_wifi])

inicio_LTE = Nfft // 2 - ordem_LTE // 2
fim_LTE = inicio_LTE + numero_de_coeficientes_LTE
h_d_LTE = np.real(h_t_LTE[inicio_LTE:fim_LTE])

#================== PLOTAGEM DO FILTRO NO TEMPO E FREQUÊNCIA =============================

# fig, axs = plt.subplots(2, 1, figsize=(12, 6))

# --- Resposta ao impulso (tempo) ---
# axs[0].stem(np.arange(numero_de_coeficientes_wifi) - (ordem_wifi // 2), h_d_wifi, linefmt='k-', markerfmt='ko', basefmt='k-')
# axs[0].grid(True, which='both')
# axs[0].set_xlabel(r'$n$')
# axs[0].set_ylabel(r'$h[n]$')
# axs[0].set_title(f"Resposta ao Impulso (Filtro FIR), Ordem={ordem_wifi}")
#
# # --- Resposta em frequência ---
# H_d_fft = np.fft.fft(h_d_wifi, Nfft)
# H_d_fft = np.fft.fftshift(H_d_fft)
#
# axs[1].plot(vetor_comprimento, np.abs(H_d_fft), lw=2, color='black', label='FIR Projetado')
# axs[1].plot(vetor_comprimento, H_d_ideal_wifi, ls='--', lw=2, color="red", label='Ideal')
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

y_LTE_teste = py.signal.lfilter(h_d_LTE,1,x2c_s) # aplica o FIR no sinal complexo
y_wifi_teste = py.signal.lfilter(h_d_wifi,1,x1c_s) # aplica o FIR no sinal complexo

# COMPENSAR ATRADO DA CONVOLUÇÃO
atraso_wifi = (len(h_d_wifi) - 1) // 2
atraso_LTE = (len(h_d_LTE) - 1) // 2
sinal_filtrado1 = np.roll(y_wifi_teste, -atraso_wifi)
sinal_filtrado2 = np.roll(y_LTE_teste, -atraso_LTE)

# RESULTADO FINAL ------------------------------------------------------------------------------------
sinal_LTE = sinal_filtrado2
sinal_wifi = sinal_filtrado1[:len(sinal_wifi1)]
sinal_wifi_para_envoltoria = sinal_filtrado1

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

envoltoria_sinal_final = envoltoria(sinal_LTE, sinal_wifi_para_envoltoria, Fs, delta_w)
esf1, esf2 = saturacao_soma(envoltoria_sinal_final, L, sinal_wifi_para_envoltoria, sinal_LTE)
xf2 = envoltoria(esf1, esf2, Fs, delta_w)
vetor_de_diferenca = np.maximum((abs(xf2 ) - abs(envoltoria_sinal_saturado)), 0)




# fig, ax = plt.subplots(2,1, figsize=(10,6))
# ax[0].set_xlabel("Time (μs)")
# ax[0].set_ylabel("Amplitude (V)")
# # ax[0].set_xlim([0.25e-5, 0.35e-5])
# ax[0].set_xlim([0.6e-5, 1.3e-5])
# # ax[0].xlim([0.25e-5, 0.7e-5])
# # ax[0].ylim([1, 1.6])
# ax[0].plot(tempo_reamostrado_LTE, abs(xn), label="Input Signal")
# ax[0].plot(tempo_reamostrado_LTE, abs(envoltoria_sinal_saturado), "--", label="Saturated Signal", color='red')
# ax[0].plot(tempo_reamostrado_LTE, abs(xf2), "--", label="Filtered Signal", color='black')
# ax[0].grid()
# ax[0].legend()
#
# ax[1].stem(tempo_reamostrado_LTE,vetor_de_diferenca,
#     label=(
#         f"Média = {np.mean(abs(vetor_de_diferenca)):.3e}"
#     )
# )
#
# # ax[1].set_xlim([0.25e-5, 0.35e-5])
# ax[1].set_xlim([0.6e-5, 1.3e-5])
# ax[1].set_ylim([0, 0.4])
# ax[1].grid()
# ax[1].legend()
# plt.tight_layout()
# plt.show()
#
# import matplotlib.pyplot as plt
# from matplotlib.ticker import MaxNLocator
#
# plt.figure(figsize=(8, 5))
# plt.hist(vetor_de_diferenca, bins=30, color="skyblue", edgecolor="black", alpha=0.7)
#
# plt.title("Histograma das Diferenças", fontsize=14, fontweight="bold")
# plt.xlabel("Valores", fontsize=12)
# plt.ylabel("Frequência", fontsize=12)
#
# # Mais divisões no eixo X
# ax = plt.gca()
# ax.xaxis.set_major_locator(MaxNLocator(nbins=15))  # aumenta número de steps
#
# plt.grid(axis="y", linestyle="--", alpha=0.6)
# plt.tight_layout()
# plt.show()

parametro = vetor_de_diferenca.copy()
print(parametro)



#======================= SALVAR ================================================================
# save_path = "/home/clayson/Área de trabalho/Projetos/python/FIR_Filter_For_DPD_Saturation/arquivos_salvos/IC2/janelaRetangular_frequencia/"
#
# np.savetxt(save_path + "IC2_filtrado_real_Wifi.csv", sinal_wifi.real, delimiter=",")
# np.savetxt(save_path + "IC2_filtrado_imag_Wifi.csv", sinal_wifi.imag, delimiter=",")
# np.savetxt(save_path + "IC2_filtrado_real_LTE.csv", sinal_LTE.real, delimiter=",")
# np.savetxt(save_path + "IC2_filtrado_imag_LTE.csv", sinal_LTE.imag, delimiter=",")



#========================== EXPORTAR PARA O CADENCE =============================================
# save_path2 = "/home/clayson/Área de trabalho/Projetos/python/FIR_Filter_For_DPD_Saturation/arquivos_salvos/IC2/janelaRetangular_frequencia/Cadence/"
#
# dados_LTE_real = np.column_stack((tempo_LTE_original, cadence_sinal_LTE.real))
# dados_LTE_imag = np.column_stack((tempo_LTE_original, cadence_sinal_LTE.imag))
# dados_wifi_real = np.column_stack((tempo_wifi_original, cadence_sinal_wifi.real))
# dados_wifi_imag = np.column_stack((tempo_wifi_original, cadence_sinal_wifi.imag))
#
# np.savetxt(save_path2 + "cadence_wifi_real.pwl", dados_wifi_real, delimiter="  ", fmt="%.5e")
# np.savetxt(save_path2 + "cadence_wifi_imag.pwl", dados_wifi_imag, delimiter="  ", fmt="%.5e")
# np.savetxt(save_path2 + "cadence_LTE_real.pwl", dados_LTE_real, delimiter="  ", fmt="%.5e")
# np.savetxt(save_path2 + "cadence_LTE_imag.pwl", dados_LTE_imag, delimiter="  ", fmt="%.5e")



