import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import get_window
from Funções import load_and_validate_mask, temp_to_freq

# Caminho base
# path = "/home/clayson/Área de trabalho/Projetos/python/FIR_Filter_For_DPD_Saturation/arquivos_salvos/IC2/janelaRetangular_frequencia"
# path = "/home/clayson/Área de trabalho/Projetos/python/FIR_Filter_For_DPD_Saturation/arquivos_salvos/IC2/Janela_da_mascara"
# Kaiser
# path = "/home/clayson/Área de trabalho/Projetos/python/FIR_Filter_For_DPD_Saturation/arquivos_salvos/IC2/Kaiser"

# Otimizado
path = "/home/clayson/Área de trabalho/Projetos/python/FIR_Filter_For_DPD_Saturation/arquivos_salvos/IC2/filtro_otimizado/"







# ================= LTE =================
try:
    s1re = np.loadtxt(os.path.join(path, "1entrada_real_LTE.csv"))
    s1im = np.loadtxt(os.path.join(path, "1entrada_imag_LTE.csv"))
    s2re = np.loadtxt(os.path.join(path, "2saturado_real_LTE.csv"))
    s2im = np.loadtxt(os.path.join(path, "2saturado_imag_LTE.csv"))
    s3re = np.loadtxt(os.path.join(path, "IC2_filtrado_real_LTE.csv"))
    s3im = np.loadtxt(os.path.join(path, "IC2_filtrado_imag_LTE.csv"))
    # s3re = np.loadtxt(os.path.join(path, "filtrado_real_LTE_kaiser.csv"))
    # s3im = np.loadtxt(os.path.join(path, "filtrado_imag_LTE_kaiser.csv"))

    s1 = s1re + 1j * s1im
    s2 = s2re + 1j * s2im
    s3 = s3re + 1j * s3im

    s1 = s1[:5000]
    s2 = s2[:5000]
    s3 = s3[:5000]

    tempo = np.arange(1, 301) * (1 / 120)

    plt.figure()
    plt.title('LTE')
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

    x1, y1, ACPR_low1, ACPR_upper1, ACPR_mean1 = temp_to_freq(s1, 120e6, repetitions, pontos_para_media, Band, redBanda,
                                                              janela)
    x2, y2, ACPR_low2, ACPR_upper2, ACPR_mean2 = temp_to_freq(s2, 120e6, repetitions, pontos_para_media, Band, redBanda,
                                                              janela)
    x3, y3, ACPR_low3, ACPR_upper3, ACPR_mean3 = temp_to_freq(s3, 120e6, repetitions, pontos_para_media, Band, redBanda,
                                                              janela)


    freq_mask, power_mask = load_and_validate_mask(os.path.join(path, "mask_lte_20M.csv"))

    plt.figure()
    plt.plot(x1 / 1e6, y1 - np.max(y1), 'k', linewidth=2)
    plt.plot(x2 / 1e6, y2 - np.max(y2), 'r', linewidth=2)
    plt.plot(x3 / 1e6, y3 - np.max(y3), linewidth=2)

    # Only plot mask if data was loaded successfully
    if freq_mask is not None and power_mask is not None:
        plt.plot(freq_mask / 1e6, power_mask - np.max(power_mask), 'y', linewidth=2)
        legend_labels = ['Entrada', 'Saturado', 'Filtrado', 'Mask']
    else:
        print("Warning: LTE mask data could not be loaded or contains invalid data")
        legend_labels = ['Entrada', 'Saturado', 'Filtrado']

    plt.legend(legend_labels, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title("LTE")
    plt.xlabel("Frequência (MHz)")
    plt.ylabel("PSD (dBm/Hz)")
    plt.xlim([-40, 40])
    plt.ylim([-80, 0])
    plt.grid()

except Exception as e:
    print(f"Error processing LTE data: {e}")

# ================= WIFI =================
try:
    s1re = np.loadtxt(os.path.join(path, "1entrada_real_Wifi.csv"))
    s1im = np.loadtxt(os.path.join(path, "1entrada_imag_Wifi.csv"))
    s2re = np.loadtxt(os.path.join(path, "2saturado_real_Wifi.csv"))
    s2im = np.loadtxt(os.path.join(path, "2saturado_imag_Wifi.csv"))
    s3re = np.loadtxt(os.path.join(path, "IC2_filtrado_real_Wifi.csv"))
    s3im = np.loadtxt(os.path.join(path, "IC2_filtrado_imag_Wifi.csv"))
    # s3re = np.loadtxt(os.path.join(path, "filtrado_real_Wifi_kaiser.csv"))
    # s3im = np.loadtxt(os.path.join(path, "filtrado_imag_Wifi_kaiser.csv"))

    s1 = s1re + 1j * s1im
    s2 = s2re + 1j * s2im
    s3 = s3re + 1j * s3im

    s1 = s1[:5000]
    s2 = s2[:5000]
    s3 = s3[:5000]

    plt.figure()
    plt.title('WiFi')
    plt.xlabel('Tempo (µs)')
    plt.ylabel('Amplitude (V)')
    plt.plot(tempo, np.abs(s1[1000:1300]), 'k', linewidth=2)
    plt.plot(tempo, np.abs(s2[1000:1300]), linewidth=2)
    plt.plot(tempo, np.abs(s3[1000:1300]), 'r', linewidth=2)
    plt.legend(['Entrada', 'Saturado', 'Filtrado'])
    plt.grid()

    x1, y1, ACPR_low1, ACPR_upper1, ACPR_mean1 = temp_to_freq(s1, 120e6, repetitions, pontos_para_media, Band, redBanda,
                                                              janela)
    x2, y2, ACPR_low2, ACPR_upper2, ACPR_mean2 = temp_to_freq(s2, 120e6, repetitions, pontos_para_media, Band, redBanda,
                                                              janela)
    x3, y3, ACPR_low3, ACPR_upper3, ACPR_mean3 = temp_to_freq(s3, 120e6, repetitions, pontos_para_media, Band, redBanda,
                                                              janela)

    # Load and validate mask data
    freq_mask, power_mask = load_and_validate_mask(os.path.join(path, "mask_wlan_20M.csv"))

    plt.figure()
    plt.plot(x1 / 1e6, y1 - np.max(y1), 'k')
    plt.plot(x2 / 1e6, y2 - np.max(y2), 'r', linewidth=2)
    plt.plot(x3 / 1e6, y3 - np.max(y3), linewidth=2)

    # Only plot mask if data was loaded successfully
    if freq_mask is not None and power_mask is not None:
        plt.plot(freq_mask / 1e6, power_mask - np.max(power_mask), 'y', linewidth=2)
        legend_labels = ['Entrada', 'Saturado', 'Filtrado', 'Mask']
    else:
        print("Warning: WiFi mask data could not be loaded or contains invalid data")
        legend_labels = ['Entrada', 'Saturado', 'Filtrado']

    plt.legend(legend_labels, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title("WiFi")
    plt.xlabel("Frequência (MHz)")
    plt.ylabel("PSD (dBm/Hz)")
    plt.xlim([-40, 40])
    plt.ylim([-80, 0])
    plt.grid()

except Exception as e:
    print(f"Error processing WiFi data: {e}")

plt.show()