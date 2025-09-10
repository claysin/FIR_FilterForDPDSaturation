import numpy as np
import scipy as py
import matplotlib.pyplot as plt

# Parâmetros
M = 46
Nfft = 65536

janelas = {
    "retangular": np.ones(M),
    "hanning": py.signal.windows.hann(M),
    "hamming": py.signal.windows.hamming(M),
    "bartlett":py.signal.windows.bartlett(M)
}

for chave, valor in janelas.items():
    w = janelas[chave]
    w = w / np.sum(w)
    janelas[chave] = w

# Janela retangular normalizada
w_retangular = janelas["retangular"]
w_hanning = janelas["hanning"]
w_hamming = janelas["hamming"]
w_bartlett = janelas["bartlett"]

# FFT
W_retangular = np.fft.fft(w_retangular, n=Nfft)[:Nfft//2]
W_hanning = np.fft.fft(w_hanning, n=Nfft)[:Nfft//2]
W_hamming = np.fft.fft(w_hamming, n=Nfft)[:Nfft//2]
W_bartlett = np.fft.fft(w_bartlett, n=Nfft)[:Nfft//2]

eps = 1e-8
mag_retangular = np.maximum(np.abs(W_retangular), eps)
mag_hanning = np.maximum(np.abs(W_hanning), eps)
mag_hamming = np.maximum(np.abs(W_hamming), eps)
mag_bartlett = np.maximum(np.abs(W_bartlett), eps)

W_dB = {
    "retangular": 20 * np.log10(mag_retangular),
    "hanning": 20 * np.log10(mag_hanning),
    "hamming": 20 * np.log10(mag_hamming),
    "bartlett": 20 * np.log10(mag_bartlett)

}
for chave, valor in W_dB.items():
    valor -= np.max(valor)
    W_dB[chave] = np.maximum(valor, -180)

freq_norm = np.linspace(0, np.pi, Nfft//2)

#--------------------------------------------------------------------------------------------------------------------
#plotagem
fig, axs = plt.subplots(2, 2, figsize=(12, 9), sharex=True, sharey=True)
xtick_locs = np.linspace(0, np.pi, 5)
xtick_labels = [r'$0$', r'$0.25\pi$', r'$0.5\pi$', r'$0.75\pi$', r'$\pi$']
plt.xticks(xtick_locs, xtick_labels)

for ax, (nome, dados_db) in zip(axs.flatten(), W_dB.items()):
    ax.plot(freq_norm, dados_db, color="blue", linewidth=2.0)
    ax.set_title(f"Janela {nome.capitalize()} (M = {M})")
    ax.grid()

fig.supxlabel("Frequência Normalizada ($\omega$)")
fig.supylabel("Magnitude (dB)")

plt.tight_layout(rect=[0, 0, 1, 0.95])

plt.show()
