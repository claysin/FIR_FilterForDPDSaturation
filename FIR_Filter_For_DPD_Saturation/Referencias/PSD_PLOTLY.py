import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.subplots as sp
from scipy.signal import get_window

# Caminho base
path = "/home/clayson/Área de trabalho/Projetos/python/FIR_Filter_For_DPD_Saturation/arquivos_salvos/IC2/janelaRetangular_frequencia"


def temp_to_freq(sig, fs, repetitions, pontos_para_media, Band, redBanda, janela):
    """
    Converte sinal no tempo para frequência (PSD) e calcula ACPR.
    """
    # Tamanho da FFT
    N = 2 ** int(np.floor(np.log2(len(sig) / repetitions)))
    resolution = fs / N

    # Seleção da janela
    janela_map = {
        1: "hann",
        2: "hamming",
        3: ("kaiser", 14),
        4: "blackman",
        5: "bartlett",
        6: "boxcar",
        7: ("chebwin", 100),
        8: "hann",
        9: "triang"
    }
    if janela in janela_map:
        w = get_window(janela_map[janela], N)
    else:
        raise ValueError("Janela inválida!")

    # FFT por repetições
    Y = []
    for var1 in range(repetitions):
        start = var1 * N
        end = (var1 + 1) * N
        x_win = w * sig[start:end]
        Y.append(np.fft.fftshift(np.fft.fft(x_win, N) / N))
    Y = np.array(Y).T

    # Potência média
    if repetitions > 1:
        Z = np.mean((np.abs(Y) ** 2) / 2 / 50, axis=1)
    else:
        Z = (np.abs(Y) ** 2) / 2 / 50

    # Eixo de frequência
    a = np.arange(0, resolution * N, resolution) - resolution * N / 2

    # Média por pontos
    p = pontos_para_media
    x = []
    y_abs = []
    for v in range(len(a) // p):
        x.append(np.mean(a[v * p:(v + 1) * p]))
        y_abs.append(np.mean(Z[v * p:(v + 1) * p]))
    x = np.array(x)
    y_abs = np.array(y_abs)

    # Converter para dBm/Hz
    y_db = 10 * np.log10(y_abs / 1e-3)

    # Máscaras de banda
    bL = (x < -(1 + redBanda) * Band / 2) & (x > -1.5 * (1 - redBanda) * Band)
    bM = (x > -(1 - redBanda) * Band / 2) & (x < (1 - redBanda) * Band / 2)
    bU = (x > (1 + redBanda) * Band / 2) & (x < 1.5 * (1 - redBanda) * Band)

    # ACPR
    ACPR_low_abs = np.sum(y_abs[bL]) / np.sum(y_abs[bM])
    ACPR_upper_abs = np.sum(y_abs[bU]) / np.sum(y_abs[bM])
    ACPR_low = 10 * np.log10(ACPR_low_abs)
    ACPR_upper = 10 * np.log10(ACPR_upper_abs)
    ACPR_mean = 10 * np.log10(np.mean([ACPR_low_abs, ACPR_upper_abs]))

    return x, y_db, ACPR_low, ACPR_upper, ACPR_mean


def load_and_validate_mask(filename):
    """
    Load mask file and ensure numeric data
    """
    try:
        m = pd.read_csv(filename, header=None)
        try:
            m[0] = pd.to_numeric(m[0], errors='coerce')
            m[1] = pd.to_numeric(m[1], errors='coerce')
            m = m.dropna()
            freq_mask = m[0].values
            power_mask = m[1].values
            return freq_mask, power_mask
        except Exception as e:
            print(f"Error converting mask data to numeric: {e}")
            return None, None
    except Exception as e:
        print(f"Error loading mask file {filename}: {e}")
        return None, None


def create_time_domain_plot(s1, s2, s3, tempo, title, start_idx=0, end_idx=300):
    """
    Cria gráfico interativo no domínio do tempo
    """
    fig = go.Figure()

    # Ajustar índices para o tempo
    time_slice = tempo[:end_idx - start_idx]

    fig.add_trace(go.Scatter(
        x=time_slice,
        y=np.abs(s1[start_idx:end_idx]),
        mode='lines',
        name='Entrada',
        line=dict(color='black', width=2),
        hovertemplate='<b>Entrada</b><br>Tempo: %{x:.3f} µs<br>Amplitude: %{y:.4f} V<extra></extra>'
    ))

    fig.add_trace(go.Scatter(
        x=time_slice,
        y=np.abs(s2[start_idx:end_idx]),
        mode='lines',
        name='Saturado',
        line=dict(color='blue', width=2),
        hovertemplate='<b>Saturado</b><br>Tempo: %{x:.3f} µs<br>Amplitude: %{y:.4f} V<extra></extra>'
    ))

    fig.add_trace(go.Scatter(
        x=time_slice,
        y=np.abs(s3[start_idx:end_idx]),
        mode='lines',
        name='Filtrado',
        line=dict(color='red', width=2),
        hovertemplate='<b>Filtrado</b><br>Tempo: %{x:.3f} µs<br>Amplitude: %{y:.4f} V<extra></extra>'
    ))

    fig.update_layout(
        title=f'{title} - Domínio do Tempo',
        xaxis_title='Tempo (µs)',
        yaxis_title='Amplitude (V)',
        hovermode='x unified',
        showlegend=True,
        template='plotly_white',
        width=800,
        height=500
    )

    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)

    return fig


def create_frequency_domain_plot(x1, y1, x2, y2, x3, y3, freq_mask, power_mask, title,
                                 ACPR_low1, ACPR_upper1, ACPR_mean1,
                                 ACPR_low2, ACPR_upper2, ACPR_mean2,
                                 ACPR_low3, ACPR_upper3, ACPR_mean3):
    """
    Cria gráfico interativo no domínio da frequência
    """
    fig = go.Figure()

    # Normalizar os sinais
    y1_norm = y1 - np.max(y1)
    y2_norm = y2 - np.max(y2)
    y3_norm = y3 - np.max(y3)

    fig.add_trace(go.Scatter(
        x=x1 / 1e6,
        y=y1_norm,
        mode='lines',
        name='Entrada',
        line=dict(color='black', width=2),
        hovertemplate='<b>Entrada</b><br>Freq: %{x:.2f} MHz<br>PSD: %{y:.2f} dBm/Hz<br>' +
                      f'ACPR Low: {ACPR_low1:.2f} dB<br>ACPR Upper: {ACPR_upper1:.2f} dB<br>' +
                      f'ACPR Mean: {ACPR_mean1:.2f} dB<extra></extra>'
    ))

    fig.add_trace(go.Scatter(
        x=x2 / 1e6,
        y=y2_norm,
        mode='lines',
        name='Saturado',
        line=dict(color='red', width=2),
        hovertemplate='<b>Saturado</b><br>Freq: %{x:.2f} MHz<br>PSD: %{y:.2f} dBm/Hz<br>' +
                      f'ACPR Low: {ACPR_low2:.2f} dB<br>ACPR Upper: {ACPR_upper2:.2f} dB<br>' +
                      f'ACPR Mean: {ACPR_mean2:.2f} dB<extra></extra>'
    ))

    fig.add_trace(go.Scatter(
        x=x3 / 1e6,
        y=y3_norm,
        mode='lines',
        name='Filtrado',
        line=dict(color='blue', width=2),
        hovertemplate='<b>Filtrado</b><br>Freq: %{x:.2f} MHz<br>PSD: %{y:.2f} dBm/Hz<br>' +
                      f'ACPR Low: {ACPR_low3:.2f} dB<br>ACPR Upper: {ACPR_upper3:.2f} dB<br>' +
                      f'ACPR Mean: {ACPR_mean3:.2f} dB<extra></extra>'
    ))

    # Adicionar máscara se disponível
    if freq_mask is not None and power_mask is not None:
        power_mask_norm = power_mask - np.max(power_mask)
        fig.add_trace(go.Scatter(
            x=freq_mask / 1e6,
            y=power_mask_norm,
            mode='lines',
            name='Mask',
            line=dict(color='orange', width=2, dash='dash'),
            hovertemplate='<b>Mask</b><br>Freq: %{x:.2f} MHz<br>PSD: %{y:.2f} dBm/Hz<extra></extra>'
        ))

    fig.update_layout(
        title=f'{title} - Domínio da Frequência (PSD)',
        xaxis_title='Frequência (MHz)',
        yaxis_title='PSD (dBm/Hz)',
        hovermode='x unified',
        showlegend=True,
        template='plotly_white',
        width=900,
        height=600,
        xaxis=dict(range=[-40, 40]),
        yaxis=dict(range=[-80, 0])
    )

    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)

    return fig


# ================= PROCESSAMENTO LTE =================
try:
    print("Processando dados LTE...")
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

    # Gráfico no domínio do tempo - LTE
    fig_time_lte = create_time_domain_plot(s1, s2, s3, tempo, 'LTE')
    fig_time_lte.show()

    # Parâmetros
    repetitions = 5
    pontos_para_media = 2
    Band = 20e6
    redBanda = 0.1
    janela = 1

    # Análise no domínio da frequência
    x1, y1, ACPR_low1, ACPR_upper1, ACPR_mean1 = temp_to_freq(s1, 120e6, repetitions, pontos_para_media, Band, redBanda,
                                                              janela)
    x2, y2, ACPR_low2, ACPR_upper2, ACPR_mean2 = temp_to_freq(s2, 120e6, repetitions, pontos_para_media, Band, redBanda,
                                                              janela)
    x3, y3, ACPR_low3, ACPR_upper3, ACPR_mean3 = temp_to_freq(s3, 120e6, repetitions, pontos_para_media, Band, redBanda,
                                                              janela)

    # Carregar máscara
    freq_mask, power_mask = load_and_validate_mask(os.path.join(path, "mask_lte_20M.csv"))

    # Gráfico no domínio da frequência - LTE
    fig_freq_lte = create_frequency_domain_plot(
        x1, y1, x2, y2, x3, y3, freq_mask, power_mask, 'LTE',
        ACPR_low1, ACPR_upper1, ACPR_mean1,
        ACPR_low2, ACPR_upper2, ACPR_mean2,
        ACPR_low3, ACPR_upper3, ACPR_mean3
    )
    fig_freq_lte.show()

    # Imprimir resultados ACPR - LTE
    print(f"\n=== RESULTADOS ACPR - LTE ===")
    print(f"Entrada - ACPR Low: {ACPR_low1:.2f} dB, Upper: {ACPR_upper1:.2f} dB, Mean: {ACPR_mean1:.2f} dB")
    print(f"Saturado - ACPR Low: {ACPR_low2:.2f} dB, Upper: {ACPR_upper2:.2f} dB, Mean: {ACPR_mean2:.2f} dB")
    print(f"Filtrado - ACPR Low: {ACPR_low3:.2f} dB, Upper: {ACPR_upper3:.2f} dB, Mean: {ACPR_mean3:.2f} dB")

except Exception as e:
    print(f"Error processing LTE data: {e}")

# ================= PROCESSAMENTO WIFI =================
try:
    print("\nProcessando dados WiFi...")
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

    # Gráfico no domínio do tempo - WiFi (usando slice diferente como no original)
    fig_time_wifi = create_time_domain_plot(s1, s2, s3, tempo, 'WiFi', start_idx=1000, end_idx=1300)
    fig_time_wifi.show()

    # Análise no domínio da frequência
    x1, y1, ACPR_low1, ACPR_upper1, ACPR_mean1 = temp_to_freq(s1, 120e6, repetitions, pontos_para_media, Band, redBanda,
                                                              janela)
    x2, y2, ACPR_low2, ACPR_upper2, ACPR_mean2 = temp_to_freq(s2, 120e6, repetitions, pontos_para_media, Band, redBanda,
                                                              janela)
    x3, y3, ACPR_low3, ACPR_upper3, ACPR_mean3 = temp_to_freq(s3, 120e6, repetitions, pontos_para_media, Band, redBanda,
                                                              janela)

    # Carregar máscara
    freq_mask, power_mask = load_and_validate_mask(os.path.join(path, "mask_wlan_20M.csv"))

    # Gráfico no domínio da frequência - WiFi
    fig_freq_wifi = create_frequency_domain_plot(
        x1, y1, x2, y2, x3, y3, freq_mask, power_mask, 'WiFi',
        ACPR_low1, ACPR_upper1, ACPR_mean1,
        ACPR_low2, ACPR_upper2, ACPR_mean2,
        ACPR_low3, ACPR_upper3, ACPR_mean3
    )
    fig_freq_wifi.show()

    # Imprimir resultados ACPR - WiFi
    print(f"\n=== RESULTADOS ACPR - WIFI ===")
    print(f"Entrada - ACPR Low: {ACPR_low1:.2f} dB, Upper: {ACPR_upper1:.2f} dB, Mean: {ACPR_mean1:.2f} dB")
    print(f"Saturado - ACPR Low: {ACPR_low2:.2f} dB, Upper: {ACPR_upper2:.2f} dB, Mean: {ACPR_mean2:.2f} dB")
    print(f"Filtrado - ACPR Low: {ACPR_low3:.2f} dB, Upper: {ACPR_upper3:.2f} dB, Mean: {ACPR_mean3:.2f} dB")

except Exception as e:
    print(f"Error processing WiFi data: {e}")

print("\nAnálise concluída! Os gráficos interativos foram gerados.")