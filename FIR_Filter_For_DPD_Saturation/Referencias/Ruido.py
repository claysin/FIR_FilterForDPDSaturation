import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# =============================================================================
# 1. GERANDO RUÍDO GAUSSIANO BÁSICO
# =============================================================================

# Parâmetros do ruído
n_samples = 1000  # Número de amostras
mean = 0  # Média (mu)
std_dev = 1  # Desvio padrão (sigma)

# Método 1: Usando numpy.random.normal()
noise_np = np.random.normal(mean, std_dev, n_samples)

# Método 2: Usando numpy.random.randn() (média 0, desvio 1)
noise_randn = np.random.randn(n_samples)

# Método 3: Usando numpy.random.default_rng() (mais moderno)
rng = np.random.default_rng(seed=42)  # seed para reprodutibilidade
noise_rng = rng.normal(mean, std_dev, n_samples)

# =============================================================================
# 2. VISUALIZAÇÃO DO RUÍDO GAUSSIANO
# =============================================================================

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Histograma
axes[0, 0].hist(noise_np, bins=50, density=True, alpha=0.7, color='blue')
axes[0, 0].set_title('Histograma do Ruído Gaussiano')
axes[0, 0].set_xlabel('Valor')
axes[0, 0].set_ylabel('Densidade')

# Adicionar curva teórica gaussiana
x = np.linspace(-4, 4, 100)
y = stats.norm.pdf(x, mean, std_dev)
axes[0, 0].plot(x, y, 'r-', linewidth=2, label=f'Gaussiana teórica (μ={mean}, σ={std_dev})')
axes[0, 0].legend()

# Série temporal
axes[0, 1].plot(noise_np[:200])  # Plotar apenas as primeiras 200 amostras
axes[0, 1].set_title('Série Temporal do Ruído')
axes[0, 1].set_xlabel('Amostra')
axes[0, 1].set_ylabel('Amplitude')

# Q-Q plot para verificar normalidade
stats.probplot(noise_np, dist="norm", plot=axes[1, 0])
axes[1, 0].set_title('Q-Q Plot (Teste de Normalidade)')

# Função de autocorrelação
from numpy import correlate


def autocorrelation(x, max_lag=50):
    n = len(x)
    x = x - np.mean(x)
    autocorr = correlate(x, x, mode='full')
    autocorr = autocorr[n - 1:]
    autocorr = autocorr / autocorr[0]  # Normalizar
    return autocorr[:max_lag + 1]


lags = range(51)
autocorr = autocorrelation(noise_np, 50)
axes[1, 1].plot(lags, autocorr)
axes[1, 1].set_title('Função de Autocorrelação')
axes[1, 1].set_xlabel('Lag')
axes[1, 1].set_ylabel('Autocorrelação')
axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()

# =============================================================================
# 3. DIFERENTES TIPOS DE RUÍDO GAUSSIANO
# =============================================================================

# Ruído branco gaussiano com diferentes variâncias
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

variances = [0.5, 1.0, 2.0, 4.0]
colors = ['blue', 'green', 'red', 'orange']

for i, (var, color) in enumerate(zip(variances, colors)):
    noise = np.random.normal(0, np.sqrt(var), 500)

    row = i // 2
    col = i % 2

    axes[row, col].plot(noise, color=color, alpha=0.7)
    axes[row, col].set_title(f'Ruído Gaussiano (σ² = {var})')
    axes[row, col].set_xlabel('Amostra')
    axes[row, col].set_ylabel('Amplitude')
    axes[row, col].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# =============================================================================
# 4. ADICIONANDO RUÍDO A UM SINAL
# =============================================================================

# Criar um sinal limpo (senoidal)
t = np.linspace(0, 2 * np.pi, 1000)
signal_clean = np.sin(2 * t) + 0.5 * np.sin(5 * t)


# Adicionar ruído com diferentes níveis (SNR - Signal-to-Noise Ratio)
def add_gaussian_noise(signal, snr_db):
    """
    Adiciona ruído gaussiano a um sinal com SNR especificada

    Parameters:
    signal: sinal original
    snr_db: relação sinal-ruído em decibéis

    Returns:
    sinal com ruído adicionado
    """
    signal_power = np.mean(signal ** 2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = np.random.normal(0, np.sqrt(noise_power), len(signal))

    return signal + noise


# Diferentes níveis de SNR
snr_levels = [30, 20, 10, 0]  # dB

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

for i, snr in enumerate(snr_levels):
    noisy_signal = add_gaussian_noise(signal_clean, snr)

    row = i // 2
    col = i % 2

    axes[row, col].plot(t[:200], signal_clean[:200], 'b-', linewidth=2, label='Sinal limpo')
    axes[row, col].plot(t[:200], noisy_signal[:200], 'r-', alpha=0.7, label=f'Com ruído (SNR={snr}dB)')
    axes[row, col].set_title(f'Sinal com Ruído Gaussiano (SNR = {snr} dB)')
    axes[row, col].set_xlabel('Tempo')
    axes[row, col].set_ylabel('Amplitude')
    axes[row, col].legend()
    axes[row, col].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# =============================================================================
# 5. ESTATÍSTICAS DO RUÍDO GERADO
# =============================================================================

print("=== ESTATÍSTICAS DO RUÍDO GAUSSIANO ===")
print(f"Número de amostras: {len(noise_np)}")
print(f"Média: {np.mean(noise_np):.4f} (esperado: {mean})")
print(f"Desvio padrão: {np.std(noise_np):.4f} (esperado: {std_dev})")
print(f"Variância: {np.var(noise_np):.4f} (esperado: {std_dev ** 2})")
print(f"Valor mínimo: {np.min(noise_np):.4f}")
print(f"Valor máximo: {np.max(noise_np):.4f}")

# Teste de normalidade (Shapiro-Wilk)
from scipy.stats import shapiro

stat, p_value = shapiro(noise_np[:1000])  # Máximo 5000 amostras para o teste
print(f"\nTeste de Shapiro-Wilk:")
print(f"Estatística: {stat:.4f}")
print(f"P-valor: {p_value:.4f}")
if p_value > 0.05:
    print("O ruído segue distribuição normal (p > 0.05)")
else:
    print("O ruído NÃO segue distribuição normal (p <= 0.05)")


# =============================================================================
# 6. FUNÇÃO UTILITÁRIA PARA GERAR RUÍDO CUSTOMIZADO
# =============================================================================

def generate_gaussian_noise(n_samples, mean=0, std=1, seed=None):
    """
    Gera ruído gaussiano personalizado

    Parameters:
    n_samples: número de amostras
    mean: média da distribuição
    std: desvio padrão da distribuição
    seed: semente para reprodutibilidade

    Returns:
    array com ruído gaussiano
    """
    if seed is not None:
        np.random.seed(seed)

    return np.random.normal(mean, std, n_samples)


# Exemplo de uso da função
custom_noise = generate_gaussian_noise(1000, mean=2, std=0.5, seed=123)
print(f"\n=== RUÍDO CUSTOMIZADO ===")
print(f"Média: {np.mean(custom_noise):.4f}")
print(f"Desvio padrão: {np.std(custom_noise):.4f}")