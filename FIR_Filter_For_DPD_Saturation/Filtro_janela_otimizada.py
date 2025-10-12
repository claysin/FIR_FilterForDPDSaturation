import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from Dados import sinal_LTE_saturado, sinal_wifi_saturado, tempo_reamostrado_wifi, tempo_reamostrado_LTE, sinal_LTE, sinal_wifi, xn,  saturacao_soma, tempo_LTE_original, tempo_wifi_original
from Funções import envoltoria, separa_variaveis, PAPR

#========================= DADOS DO FILTRO NO TEMPO ======================================================
dados = pd.read_csv("Coeficientes_otimizados_mascara.csv", header=None)
coeficientes_otimizados = np.array(dados.iloc[:, 0])
Fs = 120e6
L = 1.25
delta_w = ((2 * np.pi * 3.5e9) - (2 * np.pi * 2.4e9)) / 2

# plt.style.use("seaborn-v0_8-whitegrid")
# plt.figure()
# plt.xlabel("Amostras", fontsize=11)
# plt.ylabel("Amplitude", fontsize=11)
# plt.grid(True, linestyle=":", linewidth=0.8)
# container = plt.stem(coeficientes_otimizados, linefmt='-', markerfmt='o', basefmt='k-')

# container.stemlines.set_color('dimgray')
# container.markerline.set_color('dimgray')
# # plt.savefig("/home/clayson/Área de trabalho/Projetos/Latex/template_semicro_2025/Template_SeMicro_2025/Figuras/coeficientes_otimizados.pdf")
# plt.show()

envoltoria_sinal_saturado = envoltoria(sinal_wifi_saturado, sinal_LTE_saturado, Fs, delta_w)
a1, a2 = saturacao_soma(envoltoria_sinal_saturado, L, sinal_wifi_saturado, sinal_LTE_saturado)
xf1 = envoltoria(a1, a2, Fs, delta_w)

#==================== APLICAÇÃO DO FILTRO ================================================================
sinal_wifi_filtrado = np.convolve(coeficientes_otimizados, sinal_wifi_saturado)[:len(tempo_reamostrado_LTE)]
sinal_lte_filtrado = np.convolve(coeficientes_otimizados, sinal_LTE_saturado)[:len(tempo_reamostrado_LTE)]


# ===================CORREÇÃO DO ATRASO =================================================================
sinal_wifi_corrigido = (len(coeficientes_otimizados) - 1) // 2
sinal_lte_corrigido = (len(coeficientes_otimizados) - 1) // 2
sinal_wifi_otm = np.roll(sinal_wifi_filtrado, -sinal_wifi_corrigido)
sinal_lte = np.roll(sinal_lte_filtrado, -sinal_lte_corrigido)

envoltoria_sinal_otimizado = envoltoria(sinal_wifi_otm, sinal_lte, Fs, delta_w)


# ================== PLOTS ==================
# xn = envoltoria(sinal_wifi, sinal_LTE, Fs, delta_w)
# vetor_de_diferenca = np.maximum((abs(xf2) - abs(envoltoria_sinal_saturado)), 0)
#
#
# plt.style.use("seaborn-v0_8-whitegrid")
# fig, ax = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
#
# ax[0].plot(tempo_reamostrado_LTE, abs(xn), label="Envoltória pré-saturação", color='blue', linewidth=1.5)
# ax[0].plot(tempo_reamostrado_LTE, abs(envoltoria_sinal_saturado), "--", label="Envoltória saturada", color='red', linewidth=1.5)
# ax[0].plot(tempo_reamostrado_LTE, abs(xf2), "--", label="Envoltória filtrada", color='black', linewidth=1.5)
#
# ax[0].set_ylabel("Amplitude (V)", fontsize=12)
# ax[0].set_xlim([0.25e-5, 0.35e-5])
# ax[0].grid(True, linestyle='--', alpha=0.7)
# ax[0].legend(fontsize=10)
#
# container = ax[1].stem(
#     tempo_reamostrado_LTE, vetor_de_diferenca,
#     linefmt='-', markerfmt='o', basefmt='k-',
#     label=f"Média = {np.mean(abs(vetor_de_diferenca)):.3e}"
# )
#
# container.stemlines.set_color('dimgray')
# container.markerline.set_color('dimgray')
#
#
#
# ax[1].set_xlabel("Tempo (μs)", fontsize=12)
# ax[1].set_ylabel("Diferença", fontsize=12)
# ax[1].set_xlim([0.25e-5, 0.35e-5])
# ax[1].set_ylim([0, 0.25])
# ax[1].grid(True, linestyle='--', alpha=0.7)
# ax[1].legend(fontsize=10)
#
# plt.tight_layout()
# # plt.savefig("/home/clayson/Área de trabalho/Projetos/Latex/template_semicro_2025/Template_SeMicro_2025/Figuras/compara_envoltoria_otimizada.pdf")
# plt.show()



# save_path = "/home/clayson/Área de trabalho/Projetos/python/FIR_Filter_For_DPD_Saturation/arquivos_salvos/IC2/filtro_otimizado/"
#
# np.savetxt(save_path + "IC2_filtrado_real_Wifi.csv", sinal_wifi.real, delimiter=",")
# np.savetxt(save_path + "IC2_filtrado_imag_Wifi.csv", sinal_wifi.imag, delimiter=",")
# np.savetxt(save_path + "IC2_filtrado_real_LTE.csv", sinal_lte.real, delimiter=",")
# np.savetxt(save_path + "IC2_filtrado_imag_LTE.csv", sinal_lte.imag, delimiter=",")








# REAMOSTRAR PARA A FREQUÊNCIA ORIGINAL ====================================================================
sinal_LTE_otm = sinal_lte
sinal_wifi2 = sinal_wifi[:12960]

interp_real_LTE_2 = interp1d(tempo_reamostrado_LTE, sinal_LTE_otm.real, kind='linear')
interp_imag_LTE_2 = interp1d(tempo_reamostrado_LTE, sinal_LTE_otm.imag, kind='linear')
sinal_real_LTE = interp_real_LTE_2(tempo_LTE_original)
sinal_imag_LTE = interp_imag_LTE_2(tempo_LTE_original)

interp_real_wifi_2 = interp1d(tempo_reamostrado_wifi, sinal_wifi2.real, kind='linear')
interp_imag_wifi_2 = interp1d(tempo_reamostrado_wifi, sinal_wifi2.imag, kind='linear')
sinal_real_wifi = interp_real_wifi_2(tempo_wifi_original)
sinal_imag_wifi = interp_imag_wifi_2(tempo_wifi_original)

# RESULTADO FINAL PARA SIMULAR NO CADENCE
cadence_sinal_wifi = separa_variaveis(sinal_real_wifi, sinal_imag_wifi)
cadence_sinal_LTE = separa_variaveis(sinal_real_LTE, sinal_imag_LTE)

#========================== EXPORTAR PARA O CADENCE =============================================
# save_path2 = "/home/clayson/Área de trabalho/Projetos/python/FIR_Filter_For_DPD_Saturation/arquivos_salvos/IC2/filtro_otimizado/cadence/"
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


# RESULTADOS ================================================================================================
# print("Número de coeficientes: ", len(coeficientes_otimizados))
print("Sinal original-------------------------------------------------------------")
print(f"PAPR canal 1 (sem filtro): {PAPR(abs(sinal_wifi2)):.2f} dB")
print(f"PAPR canal 2 (sem filtro): {PAPR(abs(sinal_LTE)):.2f} dB")
print(f"PAPR da envoltória original(sem filtro): {PAPR(abs(xn)):.2f} dB")

print("Sinal saturado -------------------------------------------------------------")
print(f"PAPR canal 1 (saturado): {PAPR(abs(sinal_wifi_saturado)):.2f} dB")
print(f"PAPR canal 2 (saturado): {PAPR(abs(sinal_LTE_saturado)):.2f} dB")
print(f"PAPR da envoltória saturado(saturado): {PAPR(abs(envoltoria_sinal_saturado)):.2f} dB")

print("Sinal filtrado-------------------------------------------------------------")
print(f"PAPR canal 1 (com filtro): {PAPR(abs(sinal_wifi_otm)):.2f} dB")
print(f"PAPR canal 2 (com filtro): {PAPR(abs(sinal_lte)):.2f} dB")
print(f"PAPR da envoltória saturado(com filtro): {PAPR(abs(envoltoria_sinal_otimizado)):.2f} dB")


# print(f"Potência média envoltória saturada (sem filtro): {np.mean(np.abs(envoltoria_sinal_saturado)**2):.2f} W")
# print(f"Potência média envoltória saturada (com filtro): {np.mean(np.abs(envoltoria_sinal_otimizado)**2):.2f} W")



# PLOT FIGURAS DE CADA CANAL ORIGINAL, SATURADA, FILTRADA =================================================
plt.style.use("seaborn-v0_8-whitegrid")
fig, axs = plt.subplots(2,1, figsize=(12,7), sharex=True)
axs[0].set_title("WiFi")
axs[0].plot(tempo_reamostrado_LTE, abs(sinal_wifi), color="blue", lw=1.5, label="Sinal pré-saturação")
axs[0].plot(tempo_reamostrado_LTE, abs(sinal_wifi_saturado), "--" ,color="red", lw=1.5, label="Sinal saturado")
axs[0].plot(tempo_reamostrado_LTE, abs(sinal_wifi_otm), "--", color="black", lw=1.5, label="Sinal filtrado")
axs[0].set_xlim([0.25e-5, 0.5e-5])
axs[0].set_ylim([0.2, 1.2])
axs[0].set_ylabel("Amplitude", fontsize=12)
axs[0].grid(True, linestyle='--', alpha=0.7)
axs[0].legend(fontsize=10)


axs[1].set_title("LTE")
axs[1].plot(tempo_reamostrado_LTE, abs(sinal_LTE), color="blue", lw=1.5, label="Sinal pré-saturação")
axs[1].plot(tempo_reamostrado_LTE, abs(sinal_LTE_saturado), "--" ,color="red", lw=1.5, label="Sinal saturado")
axs[1].plot(tempo_reamostrado_LTE, abs(sinal_lte), "--", color="black", lw=1.5, label="Sinal filtrado")
axs[1].set_xlim([0.25e-5, 0.5e-5])
axs[1].set_ylim([0, 1.6])
axs[1].set_ylabel("Amplitude", fontsize=12)
axs[1].set_xlabel("Tempo (μs)", fontsize=12)
axs[1].grid(True, linestyle='--', alpha=0.7)
axs[1].legend(fontsize=10)


plt.tight_layout()
plt.show()

#============================================================================================================