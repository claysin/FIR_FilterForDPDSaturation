import os
import numpy as np
import scipy as py
import matplotlib.pyplot as plt
from Funções import load_and_validate_mask, envoltoria, saturacao_soma
from Dados import sinal_wifi, sinal_LTE, tempo_reamostrado_LTE

path = "/home/clayson/Área de trabalho/Projetos/python/FIR_Filter_For_DPD_Saturation/arquivos_salvos/IC2/janelaRetangular_frequencia"

freq_mask_LTE, power_mask_LTE = load_and_validate_mask(os.path.join(path, "mask_lte_20M.csv"))
freq_mask_wifi, power_mask_wifi = load_and_validate_mask(os.path.join(path, "mask_wlan_20M.csv"))
mask_LTE_amplitude = 10 ** (power_mask_LTE / 20)
mask_wifi_amplitude = 10 ** (power_mask_wifi / 20)
normalizado_LTE = mask_LTE_amplitude / np.max(mask_LTE_amplitude)
normalizado_wifi = mask_wifi_amplitude / np.max(mask_wifi_amplitude)


# plt.figure()
# plt.plot(freq_mask_LTE, normalizado_LTE)
# plt.title("Máscara LTE normalizada (primeiro)")
# plt.xlabel("Frequência [Hz]")
# plt.ylabel("Amplitude normalizada")
# plt.grid(True)
# plt.show()


resposta_ao_impulso_LTE = np.real(np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(normalizado_LTE))))[1:]
resposta_ao_impulso_wifi = np.real(np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(normalizado_wifi))))[1:]




# # --- Estilo geral
# plt.style.use("seaborn-v0_8-whitegrid")
# fig, axes = plt.subplots(1, 2, figsize=(12, 5))  # 1 linha, 2 colunas
#
# # --- Gráfico LTE
# axes[0].stem(resposta_ao_impulso_LTE, linefmt='C1-', markerfmt='C1o', basefmt=" ")
# axes[0].set_title("Resposta ao Impulso — LTE", fontsize=13, fontweight="bold")
# axes[0].set_xlabel("Amostras", fontsize=11)
# axes[0].set_ylabel("Amplitude", fontsize=11)
# axes[0].grid(True, linestyle=":", linewidth=0.8)
#
# # --- Gráfico Wi-Fi
# axes[1].stem(resposta_ao_impulso_wifi, linefmt='C0-', markerfmt='C0s', basefmt=" ")
# axes[1].set_title("Resposta ao Impulso — Wi-Fi", fontsize=13, fontweight="bold")
# axes[1].set_xlabel("Amostras", fontsize=11)
# axes[1].set_ylabel("Amplitude", fontsize=11)
# axes[1].grid(True, linestyle=":", linewidth=0.8)
#
# plt.savefig("/home/clayson/Área de trabalho/Projetos/Latex/template_semicro_2025/Template_SeMicro_2025/Figuras/resposta_ao_impulso.pdf")
#
#
# # --- Ajustes gerais
# plt.tight_layout()
# plt.show()



# APLICANDO A SATURAÇÃO -----------------------------------------------------------------------------------
L = 1.25
Fs = 120e6
delta_w = ((2 * np.pi * 3.5e9) - (2 * np.pi * 2.4e9)) / 2
xn = envoltoria(sinal_wifi, sinal_LTE, Fs, delta_w)
x1c_s, x2c_s = saturacao_soma(xn, L, sinal_wifi, sinal_LTE)

envoltoria_sinal_saturado = envoltoria(x1c_s, x2c_s, Fs, delta_w)
#==========================================================================================================
# Aplicando Filtro ========================================================================================

filtro_aplicado_LTE = py.signal.lfilter(resposta_ao_impulso_LTE, 1, x2c_s)
filtro_aplicado_wifi = py.signal.lfilter(resposta_ao_impulso_wifi, 1, x1c_s)

# COMPENSAR ATRASO DA CONVOLUÇÃO
atraso_wifi = (len(resposta_ao_impulso_wifi) - 1) // 2
atraso_LTE = (len(resposta_ao_impulso_LTE) - 1) // 2
sinal_filtrado1 = np.roll(filtro_aplicado_wifi, -atraso_wifi)
sinal_filtrado2 = np.roll(filtro_aplicado_LTE, -atraso_LTE)

# RESULTADO FINAL ------------------------------------------------------------------------------------
sinal_LTE = sinal_filtrado2
sinal_wifi = sinal_filtrado1[:12960]
sinal_wifi_para_envoltoria = sinal_filtrado1

envoltoria_sinal_final = envoltoria(sinal_LTE, sinal_wifi_para_envoltoria, Fs, delta_w)
esf1, esf2 = saturacao_soma(envoltoria_sinal_final, L, sinal_wifi_para_envoltoria, sinal_LTE)
xf2 = envoltoria(esf1, esf2, Fs, delta_w)

vetor_de_diferenca = np.maximum((np.abs(xf2) - np.abs(envoltoria_sinal_saturado)) ** 2, 0)

plt.style.use("seaborn-v0_8-whitegrid")
fig, ax = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

ax[0].plot(tempo_reamostrado_LTE, abs(xn), label="Envoltória pré-saturação", color='blue', linewidth=1.5)
ax[0].plot(tempo_reamostrado_LTE, abs(envoltoria_sinal_saturado), "--", label="Envoltória saturada", color='red', linewidth=1.5)
ax[0].plot(tempo_reamostrado_LTE, abs(xf2), "--", label="Envoltória filtrada", color='black', linewidth=1.5)

ax[0].set_ylabel("Amplitude (V)", fontsize=12)
ax[0].set_xlim([0.25e-5, 0.35e-5])
ax[0].grid(True, linestyle='--', alpha=0.7)
ax[0].legend(fontsize=10)

container = ax[1].stem(
    tempo_reamostrado_LTE, vetor_de_diferenca,
    linefmt='-', markerfmt='o', basefmt='k-',
    label=f"Média = {np.mean(abs(vetor_de_diferenca)):.3e}"
)

container.stemlines.set_color('dimgray')
container.markerline.set_color('dimgray')



ax[1].set_xlabel("Tempo (μs)", fontsize=12)
ax[1].set_ylabel("Diferença", fontsize=12)
ax[1].set_xlim([0.25e-5, 0.35e-5])
ax[1].set_ylim([0, 0.25])
ax[1].grid(True, linestyle='--', alpha=0.7)
ax[1].legend(fontsize=10)

plt.tight_layout()

# plt.savefig("/home/clayson/Área de trabalho/Projetos/Latex/template_semicro_2025/Template_SeMicro_2025/Figuras/compara_envoltoria.pdf")

plt.show()





# #======================= SALVAR ================================================================
# save_path = "/home/clayson/Área de trabalho/Projetos/python/FIR_Filter_For_DPD_Saturation/arquivos_salvos/IC2/Janela_da_mascara/"
#
# np.savetxt(save_path + "IC2_filtrado_real_Wifi.csv", sinal_wifi.real, delimiter=",")
# np.savetxt(save_path + "IC2_filtrado_imag_Wifi.csv", sinal_wifi.imag, delimiter=",")
# np.savetxt(save_path + "IC2_filtrado_real_LTE.csv", sinal_LTE.real, delimiter=",")
# np.savetxt(save_path + "IC2_filtrado_imag_LTE.csv", sinal_LTE.imag, delimiter=",")
