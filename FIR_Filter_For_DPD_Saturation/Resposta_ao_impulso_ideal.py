import numpy as np
import scipy as py
import matplotlib.pyplot as plt


M = 10
Nfft = 65536

freq = np.linspace(-np.pi, np.pi, Nfft)

plt.figure()
plt.title("Resposta em frequência de uma janela retangular")
x_ticks = np.linspace(-np.pi, np.pi, 9)
x_labels = [r'-$\pi$', r'$-0.75\pi$', r'-$0.5\pi$', r'-$0.25\pi$' ,r'0', r'$0.25\pi$', r'$0.5\pi$', r'$0.75\pi$', r'$\pi$']
plt.xticks(x_ticks, x_labels)

for n in range(1, 4):
    rect = (np.ones(M * n) / (M*n))
    rect_freq = py.fft.fftshift(py.fft.fft(rect, n=Nfft))
    rect_freq = np.real(rect_freq)
    plt.plot(freq, rect_freq, linewidth=2, label={n*M})

plt.grid()
plt.legend()
plt.show()
