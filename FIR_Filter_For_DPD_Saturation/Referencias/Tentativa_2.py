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
def metade_para_filtro(h_half):
    """Cria filtro simétrico (fase linear) a partir de metade dos coeficientes."""
    return np.concatenate([h_half, h_half[-2::-1]])  # [h0,h1,...,hn,...,h1,h0]

def funcao_objetivo(h_half, sinal_wifi_local, sinal_LTE_local, env_sat_local, Fs, delta_w, L, lambda_gain=1e3):
    h = metade_para_filtro(h_half)
    # aplicar nos sinais
    filt_LTE = py.signal.lfilter(h, 1, sinal_LTE_local)
    filt_wifi = py.signal.lfilter(h, 1, sinal_wifi_local)
    atraso = (len(h) - 1) // 2
    filt_LTE = np.roll(filt_LTE, -atraso)
    filt_wifi = np.roll(filt_wifi, -atraso)

    # reconstruir envoltória
    env_final = envoltoria(filt_LTE, filt_wifi, Fs, delta_w)
    esf1, esf2 = saturacao_soma(env_final, L, filt_wifi, filt_LTE)
    xf2 = envoltoria(esf1, esf2, Fs, delta_w)

    # truncar para comparação
    xf2 = xf2[:len(env_sat_local)]

    # MSE complexa
    mse = np.mean(np.abs(xf2 - env_sat_local) ** 2)

    # penalização de ganho DC
    gain_dc = np.sum(h)
    penalty = lambda_gain * (gain_dc - 1.0) ** 2

    return mse + penalty

# ================== OTIMIZAÇÃO ==================
# usar trecho reduzido para acelerar
N_test = 2000
sinal_wifi_reduz = sinal_wifi[:N_test]
sinal_LTE_reduz = sinal_LTE[:N_test]
env_sat_reduz = envoltoria_sinal_saturado[:N_test]

# filtro inicial: pegar 9 coeficientes do centro
target_len = 9
half_len = (target_len + 1) // 2
center = len(resposta_ao_impulso_wifi) // 2
start = center - half_len
h_init_full = resposta_ao_impulso_wifi[start:start + target_len]
h_init_half = h_init_full[:half_len]

# otimização
bnds = [(-2.0, 2.0)] * len(h_init_half)
res = minimize(
    funcao_objetivo,
    h_init_half,
    args=(sinal_wifi_reduz, sinal_LTE_reduz, env_sat_reduz, Fs, delta_w, L, 1e3),
    method="L-BFGS-B",
    bounds=bnds,
    options={"maxiter": 200, "ftol": 1e-8}
)

h_half_opt = res.x
h_opt = metade_para_filtro(h_half_opt)
print("Success:", res.success, "Message:", res.message)
print("Filtro otimizado (h_opt):", h_opt)

# ================== APLICAR FILTRO OTIMIZADO AOS SINAIS COMPLETOS ==================
filtro_aplicado_LTE = py.signal.lfilter(h_opt, 1, x2c_s)
filtro_aplicado_wifi = py.signal.lfilter(h_opt, 1, x1c_s)

# compensar atraso
atraso_wifi = (len(h_opt) - 1) // 2
atraso_LTE = (len(h_opt) - 1) // 2
sinal_filtrado1 = np.roll(filtro_aplicado_wifi, -atraso_wifi)
sinal_filtrado2 = np.roll(filtro_aplicado_LTE, -atraso_LTE)

# resultado final
sinal_LTE = sinal_filtrado2
sinal_wifi = sinal_filtrado1[:12960]
sinal_wifi_para_envoltoria = sinal_filtrado1

envoltoria_sinal_final = envoltoria(sinal_LTE, sinal_wifi_para_envoltoria, Fs, delta_w)
esf1, esf2 = saturacao_soma(envoltoria_sinal_final, L, sinal_wifi_para_envoltoria, sinal_LTE)
xf2 = envoltoria(esf1, esf2, Fs, delta_w)

vetor_de_diferenca = np.abs(xf2 - envoltoria_sinal_saturado[:len(xf2)])

# ================== PLOTS ==================
fig, ax = plt.subplots(2, 1, figsize=(10, 6))
ax[0].plot(tempo_reamostrado_LTE, abs(xn), label="Input Signal")
ax[0].plot(tempo_reamostrado_LTE, abs(envoltoria_sinal_saturado), "--", label="Saturated Signal", color="red")
ax[0].plot(tempo_reamostrado_LTE[:len(xf2)], abs(xf2), "--", label="Filtered Signal (Optimized)", color="black")
ax[0].set_xlabel("Time (μs)")
ax[0].set_ylabel("Amplitude (V)")
ax[0].set_xlim([0.25e-5, 0.35e-5])
ax[0].grid()
ax[0].legend()

ax[1].stem(tempo_reamostrado_LTE[:len(vetor_de_diferenca)], vetor_de_diferenca,
           label=(f"Média = {np.mean(vetor_de_diferenca):.3e}"))
ax[1].set_xlim([0.25e-5, 0.35e-5])
ax[1].set_ylim([0, 0.4])
ax[1].grid()
ax[1].legend()

plt.tight_layout()
plt.show()

