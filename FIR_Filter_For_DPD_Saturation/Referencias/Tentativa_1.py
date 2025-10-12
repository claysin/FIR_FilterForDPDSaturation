import os
import numpy as np
import scipy as py
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from Funções import load_and_validate_mask, envoltoria, saturacao_soma
from Dados import sinal_wifi, sinal_LTE, tempo_reamostrado_LTE

# ================== CARREGAR MÁSCARAS ==================
path = "/arquivos_salvos/IC2/janelaRetangular_frequencia"

freq_mask_LTE, power_mask_LTE = load_and_validate_mask(os.path.join(path, "mask_lte_20M.csv"))
freq_mask_wifi, power_mask_wifi = load_and_validate_mask(os.path.join(path, "mask_wlan_20M.csv"))

mask_LTE_amplitude = 10 ** (power_mask_LTE / 20)
mask_wifi_amplitude = 10 ** (power_mask_wifi / 20)

normalizado_LTE = mask_LTE_amplitude / np.max(mask_LTE_amplitude)
normalizado_wifi = mask_wifi_amplitude / np.max(mask_wifi_amplitude)

# Resposta ao impulso inicial (via IFFT da máscara)
resposta_ao_impulso_LTE = np.real(np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(normalizado_LTE))))[1:]
resposta_ao_impulso_wifi = np.real(np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(normalizado_wifi))))[1:]

# ================== SATURAÇÃO ==================
L = 1.25
Fs = 120e6
delta_w = ((2 * np.pi * 3.5e9) - (2 * np.pi * 2.4e9)) / 2
xn = envoltoria(sinal_wifi, sinal_LTE, Fs, delta_w)
x1c_s, x2c_s = saturacao_soma(xn, L, sinal_wifi, sinal_LTE)

envoltoria_sinal_saturado = envoltoria(x1c_s, x2c_s, Fs, delta_w)

# ================== FUNÇÃO OBJETIVO ==================
def funcao_objetivo(coeficientes, sinal_wifi, sinal_LTE, Fs, delta_w, L, envoltoria_sinal_saturado):
    # aplica o filtro FIR nos dois sinais
    filtro_aplicado_LTE = py.signal.lfilter(coeficientes, 1, sinal_LTE)
    filtro_aplicado_wifi = py.signal.lfilter(coeficientes, 1, sinal_wifi)

    # compensar atraso
    atraso = (len(coeficientes) - 1) // 2
    sinal_filtrado_wifi = np.roll(filtro_aplicado_wifi, -atraso)
    sinal_filtrado_LTE = np.roll(filtro_aplicado_LTE, -atraso)

    # reconstruir envoltória
    envoltoria_sinal_final = envoltoria(sinal_filtrado_LTE, sinal_filtrado_wifi, Fs, delta_w)
    esf1, esf2 = saturacao_soma(envoltoria_sinal_final, L, sinal_filtrado_wifi, sinal_filtrado_LTE)
    xf2 = envoltoria(esf1, esf2, Fs, delta_w)

    # métrica = diferença média entre envoltória filtrada e saturada
    vetor_de_diferenca = np.maximum((abs(xf2) - abs(envoltoria_sinal_saturado)), 0)
    return np.mean(np.abs(vetor_de_diferenca))

# ================== OTIMIZAÇÃO ==================
# usar os primeiros 31 coeficientes como chute inicial
coef_iniciais = resposta_ao_impulso_wifi[:9]

resultado = minimize(
    funcao_objetivo,
    coef_iniciais,
    args=(sinal_wifi, sinal_LTE, Fs, delta_w, L, envoltoria_sinal_saturado),
    method="Powell"  # pode trocar por 'Nelder-Mead', 'BFGS' etc.
)

coef_otimizados = resultado.x
print("Coeficientes do filtro otimizados:", coef_otimizados)
np.savetxt("Coeficientes_otimizados_mascara.csv", coef_otimizados, delimiter=",")

# ================== APLICAR FILTRO OTIMIZADO ==================
filtro_aplicado_LTE = py.signal.lfilter(coef_otimizados, 1, x2c_s)
filtro_aplicado_wifi = py.signal.lfilter(coef_otimizados, 1, x1c_s)

# compensar atraso
atraso_wifi = (len(coef_otimizados) - 1) // 2
atraso_LTE = (len(coef_otimizados) - 1) // 2
sinal_filtrado1 = np.roll(filtro_aplicado_wifi, -atraso_wifi)
sinal_filtrado2 = np.roll(filtro_aplicado_LTE, -atraso_LTE)

# resultado final
sinal_LTE = sinal_filtrado2
sinal_wifi = sinal_filtrado1[:12960]
sinal_wifi_para_envoltoria = sinal_filtrado1

envoltoria_sinal_final = envoltoria(sinal_LTE, sinal_wifi_para_envoltoria, Fs, delta_w)
esf1, esf2 = saturacao_soma(envoltoria_sinal_final, L, sinal_wifi_para_envoltoria, sinal_LTE)
xf2 = envoltoria(esf1, esf2, Fs, delta_w)

vetor_de_diferenca = np.maximum((abs(xf2) - abs(envoltoria_sinal_saturado)), 0)

# ================== PLOTS ==================
fig, ax = plt.subplots(2, 1, figsize=(10, 6))
ax[0].plot(tempo_reamostrado_LTE, abs(xn), label="Input Signal")
ax[0].plot(tempo_reamostrado_LTE, abs(envoltoria_sinal_saturado), "--", label="Saturated Signal", color='red')
ax[0].plot(tempo_reamostrado_LTE, abs(xf2), "--", label="Filtered Signal (Optimized)", color='black')
ax[0].set_xlabel("Time (μs)")
ax[0].set_ylabel("Amplitude (V)")
ax[0].set_xlim([0.25e-5, 0.35e-5])
ax[0].grid()
ax[0].legend()

ax[1].stem(tempo_reamostrado_LTE, vetor_de_diferenca,
           label=(f"Média = {np.mean(abs(vetor_de_diferenca)):.3e}"))
ax[1].set_xlim([0.25e-5, 0.35e-5])
ax[1].set_ylim([0, 0.4])
ax[1].grid()
ax[1].legend()

plt.tight_layout()
plt.show()


