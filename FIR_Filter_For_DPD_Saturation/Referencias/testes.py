import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import get_window
from Funções import load_and_validate_mask, temp_to_freq

# Caminho base
path = "/arquivos_salvos/IC2/filtro_otimizado/"

# Carregar máscaras
freq_lte, mascara_lte = load_and_validate_mask(os.path.join(path, "mask_lte_20M.csv"))
freq_wifi, mascara_wifi = load_and_validate_mask(os.path.join(path, "mask_wlan_20M.csv"))

# Estilo do gráfico
plt.style.use("seaborn-v0_8-whitegrid")

plt.figure(figsize=(10, 6))

# Plot das máscaras
plt.plot(freq_wifi, mascara_wifi, linestyle="--", color="#1f77b4", linewidth=2.2, label="Máscara Wi-Fi")
plt.plot(freq_lte, mascara_lte, linestyle="-", color="#ff7f0e", linewidth=2.2, label="Máscara LTE")

# Título e rótulos
plt.xlabel("Frequência (MHz)", fontsize=12)
plt.ylabel("Magnitude (dB)", fontsize=12)

# Grade e legenda
plt.grid(True, which="both", linestyle=":", linewidth=0.8)
plt.legend(fontsize=11, loc="best", frameon=True)
plt.tight_layout()

plt.savefig("/home/clayson/Área de trabalho/Projetos/Latex/template_semicro_2025/Template_SeMicro_2025/Figuras/Máscaras_unidas.pdf")
# Exibir
plt.show()
