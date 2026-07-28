import numpy as np
from scipy.signal import butter, filtfilt, hilbert

def generate_distributed_sources(G, Nsrc, Ndistr, flanker, Ts, Fs):
    """
    Генерирует распределенные источники с блочной амплитудной модуляцией.

    Параметры:
    ----------
    G : numpy.ndarray
        Матрица прямой модели (Leadfield). Размер (Nsens, 3 * Nsites).
    Nsrc : int
        Общее количество источников.
    Ndistr : int
        Количество "целевых" распределенных источников (те, что нас интересуют).
    flanker : float
        Время "фланкеров" в секундах для фильтрации без краевых эффектов.
    Ts : float
        Общая длительность сигнала в секундах.
    Fs : float
        Частота дискретизации.

    Возвращает:
    -------
    X_s : numpy.ndarray
        Сенсорные данные целевых источников (Nsens, N).
    X_bg : numpy.ndarray
        Сенсорные данные фоновых источников (Nsens, N).
    X_n : numpy.ndarray
        Некоррелированный сенсорный белый шум (Nsens, N).
    z : numpy.ndarray
        Модуляция мощности целевых источников (Nsrc, N).
    GA : numpy.ndarray
        Проекция активных источников (Nsens, Nsrc).
    S : numpy.ndarray
        Сигналы в источниках до проецирования (Nsrc, N).
    labels : numpy.ndarray
        Временная маска состояний (N,), содержит метки 'Background', 'Target_1' и т.д.
    """
    N = int(Ts * Fs)
    flanker_samples = int(flanker * Fs)

    # Установка фильтров
    b, a = butter(5, [8 / (Fs / 2), 12 / (Fs / 2)], btype='bandpass')
    b_lp, a_lp = butter(5, 0.5 / (Fs / 2), btype='lowpass')

    Gx = G[:, 0::3]
    Gy = G[:, 1::3]
    Gz = G[:, 2::3]
    Nsens, Nsites = Gx.shape

    GA = np.zeros((Nsens, Nsrc))
    src_indsA = np.random.permutation(Nsites)
    for i in range(Nsrc):
        src_idx = src_indsA[i]
        r = np.random.rand(3)
        r = r / np.linalg.norm(r)
        GA[:, i] = Gx[:, src_idx]*r[0] + Gy[:, src_idx]*r[1] + Gz[:, src_idx]*r[2]

    raw_noise = np.random.randn(Nsrc, N + 2 * flanker_samples)
    S_full = filtfilt(b, a, raw_noise, axis=1)
    S = S_full[:, flanker_samples : -flanker_samples] if flanker_samples > 0 else S_full.copy()

    z = np.zeros((Nsrc, N))

    # --- НОВОЕ: Настройка блоков и массива меток ---
    num_blocks = Ndistr + 1
    samples_per_block = N // num_blocks
    
    # Создаем массив меток, изначально всё — фон
    labels = np.empty(N, dtype=object)
    labels[:] = 'Background'

    for k in range(Nsrc):
        analytic_signal = hilbert(S[k, :])
        carrier_env = np.abs(analytic_signal)
        S_norm = S[k, :] / carrier_env

        noise_mod = np.random.randn(N + 2 * flanker_samples)
        lp_noise_full = filtfilt(b_lp, a_lp, noise_mod)
        lp_noise = lp_noise_full[flanker_samples : -flanker_samples] if flanker_samples > 0 else lp_noise_full.copy()
        lp_noise = lp_noise / np.std(lp_noise)

        amp_mod = lp_noise - np.min(lp_noise) + 0.05

        if k < Ndistr:
            # Целевой источник: активен только в своем блоке (начиная со второго блока)
            mask = np.zeros(N)
            block_start = (k + 1) * samples_per_block
            block_end = (k + 2) * samples_per_block if k < Ndistr - 1 else N
            
            # Записываем метку для текущего блока
            labels[block_start:block_end] = f'Target_{k+1}'
            
            mask[block_start:block_end] = 1.0
            amp_mod_masked = amp_mod * mask
            
            S[k, :] = S_norm * amp_mod_masked
            
            sigma_s = np.std(S[k, block_start:block_end])
            if sigma_s > 0:
                S[k, :] = S[k, :] / sigma_s
                z[k, :] = (amp_mod_masked / sigma_s)**2
        else:
            # Фоновый источник: активен всегда
            S[k, :] = S_norm * amp_mod
            sigma_s = np.std(S[k, :])
            S[k, :] = S[k, :] / sigma_s
            z[k, :] = (amp_mod / sigma_s)**2

    X_s = GA[:, :Ndistr] @ S[:Ndistr, :]
    X_bg = GA[:, Ndistr:] @ S[Ndistr:, :]

    X_n = np.random.randn(Nsens, N)
    X_n = X_n - np.mean(X_n, axis=1, keepdims=True)
    X_n = X_n / np.std(X_n, axis=1, keepdims=True)

    # Не забываем вернуть labels
    return X_s, X_bg, X_n, z, GA, S, labels