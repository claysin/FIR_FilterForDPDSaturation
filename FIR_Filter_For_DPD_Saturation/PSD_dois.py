import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import get_window
from Funções import load_and_validate_mask, temp_to_freq

# Caminho base
path = "/home/clayson/Área de trabalho/Projetos/python/FIR_Filter_For_DPD_Saturation/arquivos_salvos/IC2/filtro_otimizado/"

# =============== LTE ===============
try:
    s1re = np.loadtxt(os.path.join(path, "1entrada_real_LTE.csv"))
    s1im = np.loadtxt(os.path.join(path, "1entrada_imag_LTE.csv"))
    s2re = np.loadtxt(os.path.join(path, "2saturado_real_LTE.csv"))
    s2im = np.loadtxt(os.path.join(path, "2saturado_imag_LTE.csv"))
    s3re = np.loadtxt(os.path.join(path, "IC2_filtrado_real_LTE.csv"))
    s3im = np.loadtxt(os.path.join(path, "IC2_filtrado_imag_LTE.csv"))

    s1 = s1re + 1j * s1im
    s2 = s2re + 1j * s2im
    s3 = s3re + 1j * s3im

    s1 = s1[:5000]
    s2 = s2[:5000]
    s3 = s3[:5000]

    tempo = np.arange(1, 301) * (1 / 120)

    plt.figure()
    plt.title('LTE - Sinal no tempo')
    plt.xlabel('Tempo (µs)')
    plt.ylabel('Amplitude (V)')
    plt.plot(tempo, np.abs(s1[:300]), 'k', linewidth=2)
    plt.plot(tempo, np.abs(s2[:300]), linewidth=2)
    plt.plot(tempo, np.abs(s3[:300]), 'r', linewidth=2)
    plt.legend(['Entrada', 'Saturado', 'Filtrado'])
    plt.grid()

    # Parâmetros
    repetitions = 5
    pontos_para_media = 2
    Band = 20e6
    redBanda = 0.1
    janela = 1

    x1_LTE, y1_LTE, *_ = temp_to_freq(s1, 120e6, repetitions, pontos_para_media, Band, redBanda, janela)
    x2_LTE, y2_LTE, *_ = temp_to_freq(s2, 120e6, repetitions, pontos_para_media, Band, redBanda, janela)
    x3_LTE, y3_LTE, *_ = temp_to_freq(s3, 120e6, repetitions, pontos_para_media, Band, redBanda, janela)

    freq_mask_LTE, power_mask_LTE = load_and_validate_mask(os.path.join(path, "mask_lte_20M.csv"))

except Exception as e:
    print(f"Error processing LTE data: {e}")
    x1_LTE = y1_LTE = x2_LTE = y2_LTE = x3_LTE = y3_LTE = freq_mask_LTE = power_mask_LTE = None


# =============== WIFI ===============
try:
    s1re = np.loadtxt(os.path.join(path, "1entrada_real_Wifi.csv"))
    s1im = np.loadtxt(os.path.join(path, "1entrada_imag_Wifi.csv"))
    s2re = np.loadtxt(os.path.join(path, "2saturado_real_Wifi.csv"))
    s2im = np.loadtxt(os.path.join(path, "2saturado_imag_Wifi.csv"))
    s3re = np.loadtxt(os.path.join(path, "IC2_filtrado_real_Wifi.csv"))
    s3im = np.loadtxt(os.path.join(path, "IC2_filtrado_imag_Wifi.csv"))

    s1 = s1re + 1j * s1im
    s2 = s2re + 1j * s2im
    s3 = s3re + 1j * s3im

    s1 = s1[:5000]
    s2 = s2[:5000]
    s3 = s3[:5000]

    plt.figure()
    plt.title('WiFi - Sinal no tempo')
    plt.xlabel('Tempo (µs)')
    plt.ylabel('Amplitude (V)')
    plt.plot(tempo, np.abs(s1[1000:1300]), 'k', linewidth=2)
    plt.plot(tempo, np.abs(s2[1000:1300]), linewidth=2)
    plt.plot(tempo, np.abs(s3[1000:1300]), 'r', linewidth=2)
    plt.legend(['Entrada', 'Saturado', 'Filtrado'])
    plt.grid()

    x1_WIFI, y1_WIFI, *_ = temp_to_freq(s1, 120e6, repetitions, pontos_para_media, Band, redBanda, janela)
    x2_WIFI, y2_WIFI, *_ = temp_to_freq(s2, 120e6, repetitions, pontos_para_media, Band, redBanda, janela)
    x3_WIFI, y3_WIFI, *_ = temp_to_freq(s3, 120e6, repetitions, pontos_para_media, Band, redBanda, janela)

    freq_mask_WIFI, power_mask_WIFI = load_and_validate_mask(os.path.join(path, "mask_wlan_20M.csv"))

except Exception as e:
    print(f"Error processing WiFi data: {e}")
    x1_WIFI = y1_WIFI = x2_WIFI = y2_WIFI = x3_WIFI = y3_WIFI = freq_mask_WIFI = power_mask_WIFI = None


# =============== GRÁFICOS DE PSD LADO A LADO ===============
fig, axs = plt.subplots(1, 2, figsize=(14, 5))

# --- LTE ---
if x1_LTE is not None:
    axs[0].plot(x1_LTE / 1e6, y1_LTE - np.max(y1_LTE), 'k', linewidth=2)
    axs[0].plot(x2_LTE / 1e6, y2_LTE - np.max(y2_LTE), 'r', linewidth=2)
    axs[0].plot(x3_LTE / 1e6, y3_LTE - np.max(y3_LTE), linewidth=2)

    if freq_mask_LTE is not None and power_mask_LTE is not None:
        axs[0].plot(freq_mask_LTE / 1e6, power_mask_LTE - np.max(power_mask_LTE), 'y', linewidth=2)
        legend_labels_LTE = ['Entrada', 'Saturado', 'Filtrado', 'Máscara']
    else:
        legend_labels_LTE = ['Entrada', 'Saturado', 'Filtrado']

    axs[0].legend(legend_labels_LTE)
    axs[0].set_title("LTE")
    axs[0].set_xlabel("Frequência (MHz)")
    axs[0].set_ylabel("PSD (dBm/Hz)")
    axs[0].set_xlim([-40, 40])
    axs[0].set_ylim([-80, 0])
    axs[0].grid(True)

# --- WiFi ---
if x1_WIFI is not None:
    axs[1].plot(x1_WIFI / 1e6, y1_WIFI - np.max(y1_WIFI), 'k', linewidth=2)
    axs[1].plot(x2_WIFI / 1e6, y2_WIFI - np.max(y2_WIFI), 'r', linewidth=2)
    axs[1].plot(x3_WIFI / 1e6, y3_WIFI - np.max(y3_WIFI), linewidth=2)

    if freq_mask_WIFI is not None and power_mask_WIFI is not None:
        axs[1].plot(freq_mask_WIFI / 1e6, power_mask_WIFI - np.max(power_mask_WIFI), 'y', linewidth=2)
        legend_labels_WIFI = ['Entrada', 'Saturado', 'Filtrado', 'Máscara']
    else:
        legend_labels_WIFI = ['Entrada', 'Saturado', 'Filtrado']

    axs[1].legend(legend_labels_WIFI)
    axs[1].set_title("WiFi")
    axs[1].set_xlabel("Frequência (MHz)")
    axs[1].set_ylabel("PSD (dBm/Hz)")
    axs[1].set_xlim([-25, 25])
    axs[1].set_ylim([-80, 0])
    axs[1].grid(True)

plt.tight_layout()
fig.savefig("/home/clayson/Área de trabalho/Projetos/Latex/template_semicro_2025/Template_SeMicro_2025/Figuras/psd.pdf")

plt.show()
