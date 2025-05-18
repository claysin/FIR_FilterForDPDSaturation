
import numpy as np
import math
import matplotlib.pyplot as plt
import cmath
import pandas as pd


def Obter_X_MP(M, P, In):
    """
    :param M: valor inteiro
    :param P: valor inteiro
    :param In: Vetor linha
    :return: Uma matriz de 2 dimensões
    """
    vlinha = np.ndarray.flatten(In)
    vetor_x = []
    copia = vlinha.copy()

    for n in range(len(vlinha)):
        linhas = []
        for p in range(1, P + 1):
            for m in range(0, M + 1):
                if n - m < 0:
                    copia[n - m] = 0
                    numero = copia[n - m] * (abs(copia[n - m])) ** (p - 1)
                else:
                    numero = vlinha[n - m] * (abs(vlinha[n - m])) ** (p - 1)
                linhas.append(numero)
        vetor_x.append(linhas)

    return np.asarray(vetor_x)


def VetorCoeficiente(X, Y):
    """
    :param X: Matriz contendo N vetores x(n)
    :param Y: Vetor coluna contendo as N amostras y(n)
    :return: Vetor coluna
    """
    vcoluna = np.resize(Y, (len(Y), 1))
    X_ComplexoConjugado = np.transpose(X.conjugate())
    h = (np.linalg.inv(X_ComplexoConjugado @ X)) @ X_ComplexoConjugado @ vcoluna
    return h


def NMSE(Y_saida, Y_validado):
    """
    :param Y_saida: Sinal desejado
    :param Y_validado: Sinal estimado
    :return: NMSE
    """
    a = []
    b = []
    Y_saida = np.ndarray.flatten(Y_saida)
    Y_validado = np.ndarray.flatten(Y_validado)


    for n in range(1, len(Y_saida)):
        a.append(
            ((np.real(Y_saida[n]) - np.real(Y_validado[n])) ** 2) + ((np.imag(Y_saida[n]) - np.imag(Y_validado[n])) ** 2))
        b.append(((np.real(Y_saida[n])) ** 2) + ((np.imag(Y_saida[n])) ** 2))
    calculo = sum(a) / sum(b)

    return 10 * math.log(calculo, 10)

def MelhorPonto(vetor_entrada, vetor_saida):
    """
    :param vetor_entrada: Vetor linha
    :param vetor_saida: Vetor Coluna
    :return: Grafico contendo os valores de NMSE dados os M e P
    """

    M = int(input("Digite o valor máximo de M: "))
    P = int(input("Digite o valor máximo de P: "))
    vetor_entrada = np.ndarray.flatten(vetor_entrada)
    vetor_saida = np.resize(vetor_saida, (len(vetor_saida), 1))

    plt.subplots()
    plt.xlabel("Valores de P")
    for m in range(1, M + 1):
        lista_nmse = []
        for p in range(1, P+1):
            X = Obter_X_MP(m, p, vetor_entrada)
            H = VetorCoeficiente(X, vetor_saida)
            Y_valid = X @ H
            nmse = NMSE(vetor_saida, Y_valid)
            lista_nmse.append(nmse)
        plt.plot(range(1, P + 1), lista_nmse, label=f'M = {m}')
    plt.legend()
    plt.show()


def MelhorPonto_2D(vetor_entrada_1, vetor_entrada_2, vetor_saida_1, vetor_saida_2):
    """
    :param vetor_entrada: Vetor linha
    :param vetor_saida: Vetor Coluna
    :return: Grafico contendo os valores de NMSE dados os M e P
    """

    M = int(input("Digite o valor máximo de M: "))
    P = int(input("Digite o valor máximo de P: "))
    vetor_entrada_1 = np.ndarray.flatten(vetor_entrada_1)
    vetor_entrada_2 = np.ndarray.flatten(vetor_entrada_2)
    vetor_saida_1 = np.resize(vetor_saida_1, (len(vetor_saida_1), 1))
    vetor_saida_2 = np.resize(vetor_saida_2, (len(vetor_saida_2), 1))

    fig, (grafico1) = plt.subplots()
    fig, (grafico2) = plt.subplots()
    grafico1.set(xlabel="Valores de P", ylabel="NMSE")
    grafico1.set_title("ENTRADA 1")
    grafico2.set(xlabel="Valores de P", ylabel="NMSE")
    grafico2.set_title("ENTRADA 2")
    for m in range(M + 1):
        lista_nmse1 = []
        lista_nmse2 = []
        for p in range(P + 1):
            x1_MP2D, x2_MP2D = MP2D_1(m, p, vetor_entrada_1, vetor_entrada_2)
            h1_MP2D = VetorCoeficiente(x1_MP2D, vetor_saida_1)
            h2_MP2D = VetorCoeficiente(x2_MP2D, vetor_saida_2)
            y1 = x1_MP2D @ h1_MP2D
            y2 = x2_MP2D @ h2_MP2D
            #print(m, p, nmse)
            lista_nmse1.append(NMSE(vetor_saida_1, y1))
            lista_nmse2.append(NMSE(vetor_saida_2, y2))
        grafico1.plot(range(P + 1), lista_nmse1, label=f"M = {m}")
        grafico2.plot(range(P + 1), lista_nmse2, label=f"M = {m}")
    grafico1.legend()
    grafico2.legend()
    plt.show()


def AmAm(vetor_entrada, vetor_saida, vetor_saida_validado):

    tamanho = len(vetor_entrada)
    plt.subplots(figsize=(15, 8), layout="constrained")
    plt.plot(abs(np.squeeze(np.ndarray.flatten(vetor_entrada))), abs(np.ndarray.flatten(np.asarray(vetor_saida[:tamanho]))), '.', color="blue", label='Y')
    plt.plot(abs(np.squeeze(np.ndarray.flatten(vetor_entrada))), abs(np.ndarray.flatten(vetor_saida_validado[:tamanho])), '.', color='red', label='Y_validado')
    plt.title("AM - AM")
    plt.xlabel("|In|")
    plt.ylabel('|Out|')

    plt.grid()
    plt.legend()
    plt.show()


def AmFm(vetor_entrada, vetor_saida):

    lista_angulo = []
    vt = np.resize(vetor_saida, (len(vetor_saida), 1))

    for n in range(len(vetor_entrada)):
        anguloEntrada = cmath.phase(vetor_entrada[n])
        anguloSaida = cmath.phase(vt[n, 0])
        lista_angulo.append(anguloSaida - anguloEntrada)

    angulo = np.degrees(np.unwrap(np.asarray(lista_angulo)))

    plt.subplots(layout="constrained")
    plt.plot(abs(np.asarray(vetor_entrada)), angulo, '.')
    plt.xlabel("|In|")
    plt.ylabel("<Out - <In")
    plt.title("AM - FM")
    plt.show()


def MP2D_1(M, P, in1, in2):
    # Inicializa as matrizes de saída com a dimensão apropriada
    rows, cols = in1.shape[0], (M + 1) * (P + 1) * (P + 2) // 2
    x1 = np.zeros((rows, cols), dtype=complex)
    x2 = np.zeros((rows, cols), dtype=complex)

    col_idx = 0

    for m in range(M + 1):
        for k in range(P + 1):
            for l in range(k + 1):
                shifted_in1 = np.concatenate((np.zeros(m, dtype=complex), in1[:-m] if m != 0 else in1))
                shifted_in2 = np.concatenate((np.zeros(m, dtype=complex), in2[:-m] if m != 0 else in2))

                new_x1_term = shifted_in1 * np.abs(shifted_in1)**(k-l) * np.abs(shifted_in2)**l
                new_x2_term = shifted_in2 * np.abs(shifted_in1)**(k-l) * np.abs(shifted_in2)**l

                x1[:, col_idx] = new_x1_term
                x2[:, col_idx] = new_x2_term
                col_idx += 1

    return x1, x2

def MP2D(M, P, x1, x2):
    matriz1 = []
    matriz2 = []
    copia1 = x1.copy()
    copia2 = x2.copy()

    for n in range(len(x1)):
        lista1 = []
        lista2 = []
        for m in range(0, M+1):
            for k in range(0, P+1):
                for j in range(0, k+1):
                    if n - m < 0:
                        copia1[n - m] = 0
                        copia2[n - m] = 0
                        y1 = copia1[(n - m)] * np.abs(copia1[(n - m)]) ** (k - j) * np.abs(copia2[(n - m)]) ** j
                        y2 = copia2[(n - m)] * np.abs(copia2[(n - m)]) ** (k - j) * abs(copia1[(n - m)]) ** j
                    else:
                        y1 = x1[(n - m)] * np.abs(x1[(n - m)]) ** (k - j) * abs(x2[(n - m)]) ** j
                        y2 = x2[(n - m)] * np.abs(x2[(n - m)]) ** (k - j) * abs(x2[(n - m)]) ** j
                    lista1.append(y1)
                    lista2.append(y2)
        matriz1.append(lista1)
        matriz2.append(lista2)

    return np.asarray(matriz1), np.asarray(matriz2)


def IMP2D(P, M, Q, x1, x2):
    def parte_1(P, M, x1, x2):
        matriz1 = []
        matriz2 = []
        copia1 = x1.copy()
        copia2 = x2.copy()
        for n in range(len(x1)):
            lista1 = []
            lista2 = []

            for p in range(P+1):
                for m in range(0, M):
                    if (n - m) < 0:
                        copia1[n - m] = 0
                        copia2[n - m] = 0
                        y1 = copia1[(n - m)] * np.abs(copia1[(n - m)]) ** p
                        y2 = copia2[(n - m)] * np.abs(copia2[(n - m)]) ** p
                    else:
                        y1 = x1  [(n - m)] * np.abs(x1[(n - m)]) ** p
                        y2 = x2[(n - m)] * np.abs(x2[(n - m)]) ** p
                    lista1.append((np.round(y1, 4)))
                    lista2.append((np.round(y2, 4)))
            matriz1.append(lista1)
            matriz2.append(lista2)
        return np.asarray(matriz1), np.asarray(matriz2)

    def parte_2(P, M, Q, x1, x2):
        matriz1 = []
        matriz2 = []
        copia1 = x1.copy()
        copia2 = x2.copy()
        for n in range(len(x1)):
            lista1 = []
            lista2 = []
            for q in range(0, Q):
                for p in range(1, P+1):
                    for m in range(0, M):

                        if (n - m) < 0:
                            copia1[n - m] = 0
                            copia2[n - m] = 0
                            y1 = copia1[n - m] * np.abs(copia1[n - m]) ** q * np.abs(copia2[n - m]) ** p
                            y2 = copia2[n - m] * np.abs(copia2[n - m]) ** q * np.abs(copia1[n - m]) ** p
                        else:
                            y1 = x1[n - m] * np.abs(x1[n - m]) ** q * np.abs(x2[n - m]) ** p
                            y2 = x2[n - m] * np.abs(x2[n - m]) ** q * np.abs(x1[n - m]) ** p
                        lista1.append(np.round(y1, 4))
                        lista2.append(np.round(y2, 4))
            matriz1.append(lista1)
            matriz2.append(lista2)
        return np.asarray(matriz1), np.asarray(matriz2)

    def concatenar(A, B):
        matriz = []
        for n in range(len(A)):
            matriz.append(np.append(A[n], B[n]))
        return np.asarray(matriz)

    y1_1, y2_1 = parte_1(P, M, x1, x2)
    y1_2, y2_2 = parte_2(P, M, Q, x1, x2)
    y1 = concatenar(y1_1, y1_2)
    y2 = concatenar(y2_1, y2_2)

    return y1, y2

def IMP(P, M, Q, u1, u2):

    P = P - 1
    Q = Q - 1
    l = len(u1)  # Tamanho de u1 (vetor unidimensional)
    ind = 0

    # Inicializa x1 e x2 com o tamanho adequado, pronto para números complexos
    num_columns_x1 = (P + 1) * (M + 1) + (Q + 1) * (P) * (M + 1)
    num_columns_x2 = (P + 1) * (M + 1) + (Q + 1) * (P) * (M + 1)
    x1 = np.zeros((l, num_columns_x1), dtype=complex)
    x2 = np.zeros((l, num_columns_x2), dtype=complex)

    # Calcular x1
    for p in range(P + 1):
        for m in range(M + 1):
            shifted_u1 = np.concatenate([np.zeros(m, dtype=complex), u1[:-m]]) if m > 0 else u1
            x1[:, ind] = shifted_u1 * np.abs(shifted_u1) ** p
            ind += 1

    for q in range(Q + 1):
        for p in range(1, P + 1):
            for m in range(M + 1):
                shifted_u1 = np.concatenate([np.zeros(m, dtype=complex), u1[:-m]]) if m > 0 else u1
                shifted_u2 = np.concatenate([np.zeros(m, dtype=complex), u2[:-m]]) if m > 0 else u2
                x1[:, ind] = shifted_u1 * np.abs(shifted_u1) ** q * np.abs(shifted_u2) ** p
                ind += 1

    ind = 0  # Reset índice para x2

    # Calcular x2
    for p in range(P + 1):
        for m in range(M + 1):
            shifted_u2 = np.concatenate([np.zeros(m, dtype=complex), u2[:-m]]) if m > 0 else u2
            x2[:, ind] = shifted_u2 * np.abs(shifted_u2) ** p
            ind += 1

    for q in range(Q + 1):
        for p in range(1, P + 1):
            for m in range(M + 1):
                shifted_u2 = np.concatenate([np.zeros(m, dtype=complex), u2[:-m]]) if m > 0 else u2
                shifted_u1 = np.concatenate([np.zeros(m, dtype=complex), u1[:-m]]) if m > 0 else u1
                x2[:, ind] = shifted_u2 * np.abs(shifted_u2) ** q * np.abs(shifted_u1) ** p
                ind += 1

    return x1, x2


def AmFm_2(vetor_entrada, vetor_saida):
    vetor_entrada = np.ndarray.flatten(vetor_entrada)
    vetor_saida = np.ndarray.flatten(vetor_saida)
    lista_angulo = []
    vt = np.resize(vetor_saida, (len(vetor_saida), 1))

    for n in range(len(vetor_entrada)):
        anguloEntrada = cmath.phase(vetor_entrada[n])
        anguloSaida = cmath.phase(vetor_saida[n])
        lista_angulo.append(anguloSaida - anguloEntrada)

    angulo = np.degrees(np.unwrap(np.asarray(lista_angulo)))

    plt.subplots(layout="constrained")
    plt.plot(abs(np.asarray(vetor_entrada)), angulo, '.')
    plt.ylim(-50, 50)
    plt.xlabel("|In|")
    plt.ylabel("<Out - <In")
    plt.title("AM - FM")
    plt.grid()
    plt.show()

def separa_variaveis(real, imaginaria):
    lista = []

    for n in range(len(real)):
        lista.append(complex(real[n], imaginaria[n]))
    return np.asarray(lista)

def envoltoria(x1, x2, freq_Amostragem, delta):
    env = np.zeros(len(x1), dtype=complex)
    for n in range(len(x1)):
        env[n] = (x1[n] * np.exp((-1j * delta * (n)) / freq_Amostragem)) + (x2[n] * np.exp((1j * delta * (n)) / freq_Amostragem))
    return env

def saturacao_soma(xn, L, x1, x2):
    x1c = np.zeros_like(xn, dtype=np.complex128)
    x2c = np.zeros_like(xn, dtype=np.complex128)

    for n in range(len(xn)):
        mod_x1n = np.abs(x1[n])
        mod_x2n = np.abs(x2[n])
        mod_xn = np.abs(xn[n])

        if mod_xn <= L:
            x1c[n] = x1[n]
            x2c[n] = x2[n]

        else:
            z_n = mod_xn - L
            x1c[n] = (mod_x1n - (z_n / 2)) * np.exp(1j * cmath.phase(x1[n]))
            x2c[n] = (mod_x2n - (z_n / 2)) * np.exp(1j * cmath.phase(x2[n]))
    return x1c, x2c

def PSD(vetor, frequencia_de_amostragem, *args):
    in_tempo = vetor
    num_pts = 2 ** 9
    fs = frequencia_de_amostragem
    separar_variaveis = int(len(vetor) / num_pts)
    lista_separada = []

    for n in range(separar_variaveis):
        lista = []
        for numero in range((n * num_pts), ((n + 1) * num_pts)):
            lista.append(in_tempo[numero])
        lista_separada.append(lista)

    in_freq = abs(np.fft.fftshift(np.fft.fft(lista_separada)))

    y_data = []
    vetor_freq = []

    for numero in range(len(in_freq)):
        hold = []
        vetor_freq.append(np.linspace(-fs / 2, fs / 2, len(in_freq[numero])))
        for valor in range(len(in_freq[numero])):
            hold.append(20 * np.log(in_freq[numero][valor]))
        y_data.append(hold)
    vetor_freq = np.asarray(vetor_freq) / 1e6
    y_data = np.asarray(y_data)

    quantidades_de_valores_medios = 10
    med_pontos_x = []
    med_pontos_y = []

    for v in range(len(vetor_freq)):
        vetor_x = []
        for indice in range(len(vetor_freq[v]) - quantidades_de_valores_medios):
            vet_x = np.zeros(quantidades_de_valores_medios)
            for n in range(quantidades_de_valores_medios):
                vet_x[n] = vetor_freq[v, n + indice]
            vetor_x.append(sum(vet_x) / quantidades_de_valores_medios)
        med_pontos_x.append(vetor_x)

    for a in range(len(y_data)):
        vetor_y = []
        for indice in range(len(y_data[v]) - quantidades_de_valores_medios):
            vet_y = np.zeros(quantidades_de_valores_medios)
            for n in range(quantidades_de_valores_medios):
                vet_y[n] = y_data[v, n + indice]
            vetor_y.append(sum(vet_y) / quantidades_de_valores_medios)
        med_pontos_y.append(vetor_y)

    # plt.subplots()
    # plt.xlabel("Freq em MHz")
    # plt.ylabel("PSD em dBm/Hz")
    # for n in range(separar_variaveis):
    #     plt.plot(vetor_freq[n], y_data[n])

    plt.subplots()
    plt.xlabel("Freq em MHz")
    plt.ylabel("PSD em dBm/Hz")
    plt.title(*args)
    for n in range(separar_variaveis):
        plt.plot(med_pontos_x[n], med_pontos_y[n])
    plt.show()


def PSD3(vetor1, vetor2, vetor3, frequencia_de_amostragem, *args):
    url2 = '/home/clayson/Downloads/mask_wlan_20M.csv'
    padrao_2_4 = pd.read_csv(url2)
    X_2_4 = np.array(padrao_2_4.iloc[:, 0])
    Y_2_4 = np.array(padrao_2_4.iloc[:, 1])

    in_tempo1 = vetor1
    in_tempo2 = vetor2
    in_tempo3 = vetor3
    num_pts = 2 ** 9
    fs = frequencia_de_amostragem
    separar_variaveis1 = int(len(vetor1) / num_pts)
    separar_variaveis2 = int(len(vetor2) / num_pts)
    separar_variaveis3 = int(len(vetor3) / num_pts)

    lista_separada1 = []
    lista_separada2 = []
    lista_separada3 = []

    for n in range(separar_variaveis1):
        lista1 = []
        for numero in range((n * num_pts), ((n + 1) * num_pts)):
            lista1.append(in_tempo1[numero])
        lista_separada1.append(lista1)

    for n in range(separar_variaveis2):
        lista2 = []
        for numero in range((n * num_pts), ((n + 1) * num_pts)):
            lista2.append(in_tempo2[numero])
        lista_separada2.append(lista2)

    for n in range(separar_variaveis3):
        lista3 = []
        for numero in range((n * num_pts), ((n + 1) * num_pts)):
            lista3.append(in_tempo3[numero])
        lista_separada3.append(lista3)


    in_freq1 = abs(np.fft.fftshift(np.fft.fft(lista_separada1)))
    in_freq2 = abs(np.fft.fftshift(np.fft.fft(lista_separada2)))
    in_freq3 = abs(np.fft.fftshift(np.fft.fft(lista_separada3)))

    y_data1 = []
    y_data2 = []
    y_data3 = []
    vetor_freq1 = []
    vetor_freq2 = []
    vetor_freq3 = []

    for numero in range(len(in_freq1)):
        hold = []
        vetor_freq1.append(np.linspace(-fs / 2, fs / 2, len(in_freq1[numero])))
        for valor in range(len(in_freq1[numero])):
            hold.append(20 * np.log(in_freq1[numero][valor]))
        y_data1.append(hold)

    for numero in range(len(in_freq2)):
        hold = []
        vetor_freq2.append(np.linspace(-fs / 2, fs / 2, len(in_freq2[numero])))
        for valor in range(len(in_freq2[numero])):
            hold.append(20 * np.log(in_freq2[numero][valor]))
        y_data2.append(hold)

    for numero in range(len(in_freq3)):
        hold = []
        vetor_freq3.append(np.linspace(-fs / 2, fs / 2, len(in_freq3[numero])))
        for valor in range(len(in_freq3[numero])):
            hold.append(20 * np.log(in_freq3[numero][valor]))
        y_data3.append(hold)

    vetor_freq1 = np.asarray(vetor_freq1) / 1e6
    vetor_freq2 = np.asarray(vetor_freq2) / 1e6
    vetor_freq3 = np.asarray(vetor_freq3) / 1e6

    y_data1 = np.asarray(y_data1)
    y_data2 = np.asarray(y_data2)
    y_data3 = np.asarray(y_data3)

    quantidades_de_valores_medios = 10
    med_pontos_x1 = []
    med_pontos_x2 = []
    med_pontos_x3 = []
    med_pontos_y1 = []
    med_pontos_y2 = []
    med_pontos_y3 = []

    for v in range(len(vetor_freq1)):
        vetor_x = []
        for indice in range(len(vetor_freq1[v]) - quantidades_de_valores_medios):
            vet_x = np.zeros(quantidades_de_valores_medios)
            for n in range(quantidades_de_valores_medios):
                vet_x[n] = vetor_freq1[v, n + indice]
            vetor_x.append(sum(vet_x) / quantidades_de_valores_medios)
        med_pontos_x1.append(vetor_x)

    for v in range(len(vetor_freq2)):
        vetor_x = []
        for indice in range(len(vetor_freq2[v]) - quantidades_de_valores_medios):
            vet_x = np.zeros(quantidades_de_valores_medios)
            for n in range(quantidades_de_valores_medios):
                vet_x[n] = vetor_freq2[v, n + indice]
            vetor_x.append(sum(vet_x) / quantidades_de_valores_medios)
        med_pontos_x2.append(vetor_x)

    for v in range(len(vetor_freq3)):
        vetor_x = []
        for indice in range(len(vetor_freq3[v]) - quantidades_de_valores_medios):
            vet_x = np.zeros(quantidades_de_valores_medios)
            for n in range(quantidades_de_valores_medios):
                vet_x[n] = vetor_freq3[v, n + indice]
            vetor_x.append(sum(vet_x) / quantidades_de_valores_medios)
        med_pontos_x3.append(vetor_x)

    for a in range(len(y_data1)):
        vetor_y = []
        for indice in range(len(y_data1[a]) - quantidades_de_valores_medios):
            vet_y = np.zeros(quantidades_de_valores_medios)
            for n in range(quantidades_de_valores_medios):
                vet_y[n] = y_data1[a, n + indice]
            vetor_y.append(sum(vet_y) / quantidades_de_valores_medios)
        med_pontos_y1.append(vetor_y)

    for a in range(len(y_data2)):
        vetor_y = []
        for indice in range(len(y_data2[a]) - quantidades_de_valores_medios):
            vet_y = np.zeros(quantidades_de_valores_medios)
            for n in range(quantidades_de_valores_medios):
                vet_y[n] = y_data2[a, n + indice]
            vetor_y.append(sum(vet_y) / quantidades_de_valores_medios)
        med_pontos_y2.append(vetor_y)

    for a in range(len(y_data3)):
        vetor_y = []
        for indice in range(len(y_data3[a]) - quantidades_de_valores_medios):
            vet_y = np.zeros(quantidades_de_valores_medios)
            for n in range(quantidades_de_valores_medios):
                vet_y[n] = y_data3[a, n + indice]
            vetor_y.append(sum(vet_y) / quantidades_de_valores_medios)
        med_pontos_y3.append(vetor_y)

    # plt.subplots()
    # plt.xlabel("Freq em MHz")
    # plt.ylabel("PSD em dBm/Hz")
    # for n in range(separar_variaveis):
    #     plt.plot(vetor_freq[n], y_data[n])

    plt.subplots()
    plt.xlabel("Frequency [MHz]")
    plt.ylabel("PSD [dBm/Hz]")
    # plt.plot(X_2_4, Y_2_4)
    plt.plot(med_pontos_x1[n], med_pontos_y1[n], label="Input signal")
    plt.plot(med_pontos_x2[n], med_pontos_y2[n], label="Saturated Signal")
    plt.plot(med_pontos_x3[n], med_pontos_y3[n], label="Filtered Signal")

    # for n in range(separar_variaveis1):
    #     plt.plot(med_pontos_x1[n], med_pontos_y1[n], label="Entrada")
    # for n in range(separar_variaveis2):
    #     plt.plot(med_pontos_x2[n], med_pontos_y2[n], label="Sinal Saturado")
    # for n in range(separar_variaveis3):
    #     plt.plot(med_pontos_x3[n], med_pontos_y3[n], label="Sinal Filtrado")
    plt.legend()
    plt.show()


def PAPR(vetor_linha):
    absoluto = abs(vetor_linha)
    pico = max(absoluto ** 2)
    media = np.mean(absoluto ** 2)
    PAPR = 10 * np.log10(pico / media)
    return PAPR

