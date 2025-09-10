import matplotlib.pyplot as plt
import numpy as np

# === Parâmetros ===
Fs = 120e6
F_corte = 20e6
f_normalizada = F_corte / (Fs / 2)
ordem = 9
num_coef = ordem + 1  # Número de coeficientes do filtro

# Usar um número maior de pontos para melhor resolução
N_fft = 1024

# Criar vetor de frequência normalizada de -π a π
vetor_w = np.linspace(-np.pi, np.pi, N_fft)
H_d = np.zeros(N_fft)

# Frequência de corte normalizada em radianos
w_corte = f_normalizada * np.pi

# Criar resposta em frequência desejada (passa-baixas ideal)
for i, w in enumerate(vetor_w):
    if np.abs(w) <= w_corte:
        H_d[i] = 1

#===============================================================================================
#======================= RESPOSTA AO IMPULSO POR IFFT =======================================
#===============================================================================================

# Calcular resposta ao impulso usando IFFT
h_temp = np.fft.ifft(np.fft.ifftshift(H_d))
h_temp = np.fft.fftshift(h_temp)  # Centralizar

# Extrair apenas os coeficient10es necessários (ordem+1 coeficientes)
# Pegar os coeficientes centrais
inicio = N_fft//2 - ordem//2
fim = inicio + num_coef
h_d = np.real(h_temp[inicio:fim])

# Índices do filtro
n = np.arange(num_coef)

# Calcular a resposta em frequência do filtro projetado
H_real = np.fft.fft(h_d, N_fft)
H_real = np.fft.fftshift(H_real)

#===============================================================================================
#======================= PLOTAGEM ============================================================
#===============================================================================================

fig, axs = plt.subplots(2, 1, figsize=(12, 6))

# Resposta em frequência desejada vs real
axs[0].plot(vetor_w, H_d, 'r-', linewidth=2, label='Ideal')
axs[0].plot(vetor_w, np.abs(H_real), 'b--', linewidth=1.5, label=f'FIR ({num_coef} coef.)')
axs[0].set_title(f"Resposta em Frequência - Ordem {ordem}")
axs[0].grid(True, which="both", alpha=0.3)
axs[0].set_xlabel("ω [rad/amostra]")
axs[0].set_ylabel("Magnitude")
axs[0].legend()
axs[0].set_ylim(-0.1, 1.1)
axs[0].axvline(x=w_corte, color='k', linestyle=':', alpha=0.7, label='Freq. de corte')
axs[0].axvline(x=-w_corte, color='k', linestyle=':', alpha=0.7)

# Resposta ao impulso
axs[1].stem(n, h_d, linefmt='b-', markerfmt='bo', basefmt='b-')
axs[1].set_title("Resposta ao Impulso h[n]")
axs[1].grid(True, which="both", alpha=0.3)
axs[1].set_xlabel("n (amostras)")
axs[1].set_ylabel("h[n]")

plt.tight_layout()
plt.show()

# Informações do filtro
print(f"\nParâmetros do Filtro:")
print(f"Fs = {Fs/1e6:.0f} MHz")
print(f"Fc = {F_corte/1e6:.0f} MHz")
print(f"Frequência normalizada: {f_normalizada:.4f}")
print(f"Ordem: {ordem}")
print(f"ω_corte = {w_corte:.4f} rad/amostra")
print(f"Número de coeficientes: {num_coef}")

# Verificar se a resposta ao impulso é simétrica (como deve ser para filtro linear phase)
print(f"\nVerificação de simetria:")
print(f"h[0] = {h_d[0]:.6f}, h[{num_coef-1}] = {h_d[num_coef-1]:.6f}")
print(f"h[1] = {h_d[1]:.6f}, h[{num_coef-2}] = {h_d[num_coef-2]:.6f}")
print(f"Centro: h[{num_coef//2}] = {h_d[num_coef//2]:.6f}")