# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 21:11:08 2026

@author: ansbel
"""

import numpy as np
from scipy.signal import butter, filtfilt, hilbert

def generate_bursty_sources(G, Nsrc, Ndistr, flanker, Ts, Fs, block_duration=10.0, burst_prob=0.15):
    """
    Генерирует сверхполную систему распределенных источников с блочной активацией.
    """
    N = int(Ts * Fs)
    flanker_samples = int(flanker * Fs)

    b, a = butter(5, [8 / (Fs / 2), 12 / (Fs / 2)], btype='bandpass')
    b_lp, a_lp = butter(5, 0.5 / (Fs / 2), btype='lowpass')

    Gx, Gy, Gz = G[:, 0::3], G[:, 1::3], G[:, 2::3]
    Nsens, Nsites = Gx.shape

    GA = np.zeros((Nsens, Nsrc))
    src_indsA = np.random.permutation(Nsites)
    
    # --- НОВОЕ: Массивы для сохранения положений (индексов) и направлений ---
    positions_idx = np.zeros(Nsrc, dtype=int)
    orientations = np.zeros((Nsrc, 3))

    for i in range(Nsrc):
        src_idx = src_indsA[i]
        r = np.random.rand(3)
        r = r / np.linalg.norm(r)
        
        # --- НОВОЕ: Сохраняем значения ---
        positions_idx[i] = src_idx
        orientations[i, :] = r
        
        GA[:, i] = Gx[:, src_idx]*r[0] + Gy[:, src_idx]*r[1] + Gz[:, src_idx]*r[2]

    raw_noise = np.random.randn(Nsrc, N + 2 * flanker_samples)
    S_full = filtfilt(b, a, raw_noise, axis=1)
    S = S_full[:, flanker_samples : -flanker_samples] if flanker_samples > 0 else S_full.copy()
    z = np.zeros((Nsrc, N))

    block_samples = int(block_duration * Fs)
    num_blocks = int(np.ceil(N / block_samples))

    for k in range(Nsrc):
        analytic_signal = hilbert(S[k, :])
        carrier_env = np.abs(analytic_signal)
        S_norm = S[k, :] / carrier_env

        noise_mod = np.random.randn(N + 2 * flanker_samples)
        lp_noise_full = filtfilt(b_lp, a_lp, noise_mod)
        lp_noise = lp_noise_full[flanker_samples : -flanker_samples] if flanker_samples > 0 else lp_noise_full.copy()
        lp_noise = lp_noise / np.std(lp_noise)
        
        # Базовая огибающая
        amp_mod = lp_noise - np.min(lp_noise) + 0.1

        # СПАРСИНГ (РАЗРЕЖЕННОСТЬ) ДЛЯ ЦЕЛЕВЫХ ИСТОЧНИКОВ
        if k < Ndistr:
            # --- ИЗМЕНЕНИЕ: Активируем источники строго по очереди ---
            active_blocks = np.zeros(num_blocks)
            blocks_per_source = max(1, num_blocks // Ndistr)
            
            start_block = k * blocks_per_source
            end_block = min(start_block + blocks_per_source, num_blocks)
            active_blocks[start_block:end_block] = 1.0
            
            mask = np.repeat(active_blocks, block_samples)[:N]
            
            smooth_mask = filtfilt(b_lp, a_lp, mask)
            smooth_mask = np.clip(smooth_mask, 0.0, 1.0)
            
            # Задаем физичный порог фоновой активности
            baseline_amp = 0.15 
            smooth_mask = smooth_mask * (1.0 - baseline_amp) + baseline_amp
            
            amp_mod = amp_mod * smooth_mask
        else:
            # Фоновые источники работают постоянно, но с подавленной мощностью
            amp_mod = amp_mod * 0.2

        amp_mod += 1e-5 # Защита от строгих нулей

        S[k, :] = S_norm * amp_mod

        # Нормализация
        sigma_s = np.std(S[k, :])
        if sigma_s > 1e-6:
            S[k, :] = S[k, :] / sigma_s
        else:
            sigma_s = 1.0

        z[k, :] = (amp_mod / sigma_s)**2

    X_s = GA[:, :Ndistr] @ S[:Ndistr, :]
    X_bg = GA[:, Ndistr:] @ S[Ndistr:, :]

    X_n = np.random.randn(Nsens, N)
    X_n = X_n - np.mean(X_n, axis=1, keepdims=True)
    X_n = X_n / np.std(X_n, axis=1, keepdims=True)

    # --- НОВОЕ: Возвращаем дополнительные параметры ---
    return X_s, X_bg, X_n, z, GA, S, positions_idx, orientations

def generate_distributed_sources(G, Nsrc, Ndistr, flanker, Ts, Fs):
    """
    !
    Параметры:
    ----------
    G : numpy.ndarray
        Матрица прямой модели (Leadfield). Размер (  , 3 * Nsites).
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
    """
    N = int(Ts * Fs)
    flanker_samples = int(flanker * Fs)

    # Установка фильтров
    # Для несущей (8-12 Гц)
    b, a = butter(5, [8 / (Fs / 2), 12 / (Fs / 2)], btype='bandpass')
    # ФНЧ < 0.5 Гц для амплитудной модуляции
    b_lp, a_lp = butter(5, 0.5 / (Fs / 2), btype='lowpass')

    # Инициализация forward model
    Gx = G[:, 0::3]
    Gy = G[:, 1::3]
    Gz = G[:, 2::3]
    Nsens, Nsites = Gx.shape

    # Создание случайных источников со случайным направлением
    GA = np.zeros((Nsens, Nsrc))
    src_indsA = np.random.permutation(Nsites)
    for i in range(Nsrc):
        src_idx = src_indsA[i]
        r = np.random.rand(3)
        r = r / np.linalg.norm(r)
        GA[:, i] = Gx[:, src_idx]*r[0] + Gy[:, src_idx]*r[1] + Gz[:, src_idx]*r[2]

    # Генерация временных рядов источников (несущие сигналы)
    raw_noise = np.random.randn(Nsrc, N + 2 * flanker_samples)
    S_full = filtfilt(b, a, raw_noise, axis=1)
    S = S_full[:, flanker_samples : -flanker_samples] if flanker_samples > 0 else S_full.copy()

    z = np.zeros((Nsrc, N))

    # Формирование огибающей и модуляция согласно Dähne et al. (2014)
    for k in range(Nsrc):
        # 1. Нормализация огибающей несущего сигнала к 1
        analytic_signal = hilbert(S[k, :])
        carrier_env = np.abs(analytic_signal)
        S_norm = S[k, :] / carrier_env

        # 2. Создание функции амплитудной модуляции (отфильтрованный белый шум < 0.5 Гц)
        noise_mod = np.random.randn(N + 2 * flanker_samples)
        lp_noise_full = filtfilt(b_lp, a_lp, noise_mod)
        lp_noise = lp_noise_full[flanker_samples : -flanker_samples] if flanker_samples > 0 else lp_noise_full.copy()

        # Восстанавливаем дисперсию модулирующего шума
        lp_noise = lp_noise / np.std(lp_noise)

        # 3. Добавление смещения
        amp_mod = lp_noise - np.min(lp_noise) + 0.05

        # 4. Применение амплитудной модуляции
        S[k, :] = S_norm * amp_mod

        # Приводим сигнал источника к единичной дисперсии
        sigma_s = np.std(S[k, :])
        S[k, :] = S[k, :] / sigma_s

        # 5. Целевая переменная z — это модуляция мощности.
        z[k, :] = (amp_mod / sigma_s)**2

    # Генерация чистых сенсорных данных (целевые источники)
    X_s = GA[:, :Ndistr] @ S[:Ndistr, :]

    # Генерация фоновой активности (остальные источники)
    X_bg = GA[:, Ndistr:] @ S[Ndistr:, :]

    # Генерация сенсорного шума (некоррелированный белый шум, нулевое среднее, единичная дисперсия)
    X_n = np.random.randn(Nsens, N)
    X_n = X_n - np.mean(X_n, axis=1, keepdims=True)
    X_n = X_n / np.std(X_n, axis=1, keepdims=True)

    return X_s, X_bg, X_n, z, GA, S

import itertools
import numpy as np
from scipy.signal import butter, filtfilt, hilbert


def _smooth_edges(sig, half_len):
    """Сглаживание 1-D сигнала raised-cosine ядром (без краевых эффектов)."""
    if half_len <= 0:
        return sig
    n = 2 * half_len + 1
    t = np.arange(n) - half_len
    k = np.cos(np.pi * t / (2 * half_len)) ** 2
    k = k / k.sum()
    padded = np.pad(sig, half_len, mode='edge')
    smoothed = np.convolve(padded, k, mode='same')
    return smoothed[half_len:-half_len]


def _rng_int(rng, low, high):
    """Случайное целое в [low, high) совместимо с np.random и Generator."""
    if hasattr(rng, "integers"):
        return int(rng.integers(low, high))
    return int(rng.randint(low, high))


def _rng_perm(rng, n):
    if hasattr(rng, "permutation"):
        return rng.permutation(n)
    return np.random.permutation(n)


def _count_transitions(labels):
    return sum(1 for i in range(1, len(labels)) if labels[i] != labels[i - 1])


def _reduce_transitions(labels, n_iter, rng):
    """
    Локальный поиск: перестановка двух случайных метк принимается,
    если она не увеличивает число переходов (баланс при этом сохраняется,
    так как мы только переставляем элементы).
    """
    n = len(labels)
    if n < 3:
        return labels
    best = _count_transitions(labels)
    for _ in range(n_iter):
        i = _rng_int(rng, 0, n)
        j = _rng_int(rng, 0, n)
        if i == j:
            continue
        labels[i], labels[j] = labels[j], labels[i]
        new = _count_transitions(labels)
        if new <= best:
            best = new
        else:
            labels[i], labels[j] = labels[j], labels[i]
    return labels


def _build_activity_mask(Ndistr, N, Fs,
                         min_seg_dur=1.0, max_seg_dur=3.0,
                         transition=0.1, rng=None,
                         balance='time'):
    """
    Маска активности целевых источников (Ndistr, N).

    Микросостояние = бинарный вектор активности Ndistr источников,
    всего M = 2**Ndistr штук (включая пустое и полное).

    balance : {'count', 'time'}
        'count' — одинаковое ЧИСЛО сегментов на микросостояние;
        'time'  — примерно равное СУММАРНОЕ ВРЕМЯ N/M на микросостояние.
    """
    if rng is None:
        rng = np.random

    subsets = []
    for k in range(Ndistr + 1):
        for comb in itertools.combinations(range(Ndistr), k):
            subsets.append(comb)
    M = len(subsets)

    min_len = max(1, int(min_seg_dur * Fs))
    max_len = max(min_len + 1, int(max_seg_dur * Fs))
    avg_len = (min_len + max_len) // 2

    K = max(M, int(np.ceil(N / max(1, avg_len))))

    # --- Сбалансированный по числу набор метк ---
    base, extra = divmod(K, M)
    labels = []
    for s in subsets:
        labels.extend([s] * base)
    if extra > 0:
        idx = _rng_perm(rng, M)
        for j in range(extra):
            labels.append(subsets[int(idx[j])])
    K = len(labels)

    perm = _rng_perm(rng, K)
    labels = [labels[int(i)] for i in perm]
    labels = _reduce_transitions(labels, n_iter=20 * K, rng=rng)

    # Группируем индексы сегментов по микросостояниям
    groups = {}
    for i, lab in enumerate(labels):
        groups.setdefault(lab, []).append(i)

    # --- Длительности сегментов ---
    if balance == 'time':
        target = N / M                       # целевое время на состояние
        seg_lens = [0] * K
        for lab, idxs in groups.items():
            n = len(idxs)
            raw = [_rng_int(rng, min_len, max_len + 1) for _ in range(n)]
            s = sum(raw)
            scaled = [x * target / s for x in raw]
            ints = [max(1, int(round(x))) for x in scaled]

            # Точечная подгонка суммы в группе под round(target)
            residual = int(round(target)) - sum(ints)
            k = 0
            while residual != 0 and k < 100 * n:
                i = k % n
                if residual > 0:
                    ints[i] += 1
                    residual -= 1
                elif ints[i] > 1:
                    ints[i] -= 1
                    residual += 1
                k += 1

            for i, L in zip(idxs, ints):
                seg_lens[i] = L

        # Финальная подгонка ровно под N
        diff = N - sum(seg_lens)
        step = 1 if diff > 0 else -1
        order = _rng_perm(rng, K)
        k = 0
        while diff != 0 and k < 100 * K:
            i = int(order[k % K])
            if seg_lens[i] + step >= 1:
                seg_lens[i] += step
                diff -= step
            k += 1
    else:
        seg_lens = [_rng_int(rng, min_len, max_len + 1) for _ in range(K)]
        total = sum(seg_lens)
        scale = N / total
        seg_lens = [max(1, int(round(l * scale))) for l in seg_lens]
        diff = N - sum(seg_lens)
        seg_lens[-1] += diff
        if seg_lens[-1] < 1:
            seg_lens[-1] = 1

    # --- Сборка маски ---
    mask = np.zeros((Ndistr, N))
    t = 0
    for label, L in zip(labels, seg_lens):
        if t >= N:
            break
        end = min(t + L, N)
        for s in label:
            mask[s, t:end] = 1.0
        t = end

    half = max(1, int(transition * Fs))
    for s in range(Ndistr):
        mask[s, :] = _smooth_edges(mask[s, :], half)

    return mask

def generate_microstates(G, Nsrc, Ndistr, flanker, Ts, Fs,
                         min_seg_dur=1.0, max_seg_dur=3.0,
                         transition=0.1,
                         snr_db_target=-10.0,
                         bg_to_noise=True,
                         rng=None):
    """
    Параметры
    ----------
    G : numpy.ndarray
        Матрица прямой модели (Leadfield). Размер (Nsens, 3 * Nsites).
    Nsrc : int
        Общее количество источников.
    Ndistr : int
        Количество "целевых" распределенных источников.
    flanker : float
        Время "фланкеров" в секундах для фильтрации без краевых эффектов.
    Ts : float
        Общая длительность сигнала в секундах.
    Fs : float
        Частота дискретизации.
    min_seg_dur, max_seg_dur : float
        Мин/макс длительность одного «режима» активности целевых источников (с).
    transition : float
        Длительность плавного перехода включения/выключения (с).
    snr_db_target : float
        Желаемое SNR целевых источников относительно сенсорного шума
        (10*log10(P_s / P_n), дБ). Отрицательные значения = источники
        «спрятаны» в шуме. По умолчанию -10 дБ.
    bg_to_noise : bool
        Если True — мощность фоновых источников приравнивается
        к мощности шума, чтобы фон тоже был неотличим по SNR.
    rng : numpy.random.Generator | None
        ГПСЧ (для воспроизводимости). Если None — используется np.random.

    Возвращает
    -------
    X_s : numpy.ndarray  (Nsens, N)  — сенсорные данные целевых источников
    X_bg : numpy.ndarray (Nsens, N)  — сенсорные данные фоновых источников
    X_n : numpy.ndarray  (Nsens, N)  — сенсорный белый шум
    z : numpy.ndarray    (Nsrc, N)   — модуляция мощности целевых источников
    GA : numpy.ndarray   (Nsens, Nsrc)
    S : numpy.ndarray    (Nsrc, N)
    activity : numpy.ndarray (Ndistr, N) — маска активности целевых источников
    """
    if rng is None:
        rng = np.random

    N = int(Ts * Fs)
    flanker_samples = int(flanker * Fs)

    # --- Фильтры ---
    b, a = butter(5, [8 / (Fs / 2), 12 / (Fs / 2)], btype='bandpass')
    b_lp, a_lp = butter(5, 0.5 / (Fs / 2), btype='lowpass')

    # --- Forward model ---
    Gx = G[:, 0::3]
    Gy = G[:, 1::3]
    Gz = G[:, 2::3]
    Nsens, Nsites = Gx.shape

    GA = np.zeros((Nsens, Nsrc))
    if hasattr(rng, "permutation"):
        src_indsA = rng.permutation(Nsites)
    else:
        src_indsA = np.random.permutation(Nsites)

    for i in range(Nsrc):
        src_idx = src_indsA[i]
        if hasattr(rng, "random"):
            r = rng.random(3)
        else:
            r = np.random.rand(3)
        r = r / np.linalg.norm(r)
        GA[:, i] = Gx[:, src_idx] * r[0] + Gy[:, src_idx] * r[1] + Gz[:, src_idx] * r[2]

    # --- Несущие сигналы источников ---
    if hasattr(rng, "standard_normal"):
        raw_noise = rng.standard_normal((Nsrc, N + 2 * flanker_samples))
    else:
        raw_noise = np.random.randn(Nsrc, N + 2 * flanker_samples)

    S_full = filtfilt(b, a, raw_noise, axis=1)
    S = S_full[:, flanker_samples:-flanker_samples] if flanker_samples > 0 else S_full.copy()

    z = np.zeros((Nsrc, N))

    # --- Амплитудная модуляция по Dähne et al. (2014) ---
    for k in range(Nsrc):
        analytic_signal = hilbert(S[k, :])
        carrier_env = np.abs(analytic_signal)
        S_norm = S[k, :] / carrier_env

        if hasattr(rng, "standard_normal"):
            noise_mod = rng.standard_normal(N + 2 * flanker_samples)
        else:
            noise_mod = np.random.randn(N + 2 * flanker_samples)

        lp_noise_full = filtfilt(b_lp, a_lp, noise_mod)
        lp_noise = (lp_noise_full[flanker_samples:-flanker_samples]
                    if flanker_samples > 0 else lp_noise_full.copy())
        lp_noise = lp_noise / np.std(lp_noise)

        amp_mod = lp_noise - np.min(lp_noise) + 0.05
        S[k, :] = S_norm * amp_mod
        sigma_s = np.std(S[k, :])
        S[k, :] = S[k, :] / sigma_s
        z[k, :] = (amp_mod / sigma_s) ** 2

    # --- Случайные режимы активности целевых источников ---
    activity = _build_activity_mask(
        Ndistr, N, Fs,
        min_seg_dur=min_seg_dur,
        max_seg_dur=max_seg_dur,
        transition=transition,
        rng=rng,
    )
    S[:Ndistr, :] *= activity
    z[:Ndistr, :] *= activity

    # --- Сенсорные данные ---
    X_s = GA[:, :Ndistr] @ S[:Ndistr, :]
    X_bg = GA[:, Ndistr:] @ S[Ndistr:, :]

    if hasattr(rng, "standard_normal"):
        X_n = rng.standard_normal((Nsens, N))
    else:
        X_n = np.random.randn(Nsens, N)
    X_n = X_n - np.mean(X_n, axis=1, keepdims=True)
    X_n = X_n / np.std(X_n, axis=1, keepdims=True)

    # === Нормировка по SNR ===
    # Мощность целевого сигнала (усреднение по всем отсчётам и каналам)
    P_s = np.mean(X_s ** 2)
    P_n = np.mean(X_n ** 2)

    # Желаемая мощность шума из snr_db_target = 10*log10(P_s / P_n)
    P_n_target = P_s / (10.0 ** (snr_db_target / 10.0))
    X_n = X_n * np.sqrt(P_n_target / (P_n + 1e-30))

    # Если нужно — приравниваем фон к шуму по мощности
    if bg_to_noise:
        P_bg = np.mean(X_bg ** 2)
        if P_bg > 0:
            X_bg = X_bg * np.sqrt(P_n_target / P_bg)

    return X_s, X_bg, X_n, z, GA, S, activity

def generate_sources(G, Nsrc, Ndistr, flanker, Ts, Fs, target_snr=1.0):
    """
    Генерирует распределенные источники.
    
    Параметры:
    ----------
    # ... (старые параметры) ...
    target_snr : float
        Желаемое отношение сигнал/шум (амплитудное) для целевых источников 
        относительно фоновых.
    """
    N = int(Ts * Fs)
    flanker_samples = int(flanker * Fs)

    b, a = butter(5, [8 / (Fs / 2), 12 / (Fs / 2)], btype='bandpass')
    b_lp, a_lp = butter(5, 0.5 / (Fs / 2), btype='lowpass')

    Gx, Gy, Gz = G[:, 0::3], G[:, 1::3], G[:, 2::3]
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

    for k in range(Nsrc):
        analytic_signal = hilbert(S[k, :])
        carrier_env = np.abs(analytic_signal)
        S_norm = S[k, :] / (carrier_env + 1e-12)

        noise_mod = np.random.randn(N + 2 * flanker_samples)
        lp_noise_full = filtfilt(b_lp, a_lp, noise_mod)
        lp_noise = lp_noise_full[flanker_samples : -flanker_samples] if flanker_samples > 0 else lp_noise_full.copy()
        
        # 1. Строго стандартизируем модулирующий шум (Mean=0, Var=1)
        lp_noise = (lp_noise - np.mean(lp_noise)) / np.std(lp_noise)

        if k < Ndistr:
            # ЦЕЛЕВЫЕ ИСТОЧНИКИ
            # Требование: Дисперсия огибающей == Среднему значению огибающей
            # То есть: Var[amp_mod] = target_snr
            # Mean[amp_mod] = target_snr
            
            # Так как Var[a * X + b] = a^2 * Var[X], а Var[lp_noise] = 1:
            # Нам нужно a^2 = target_snr => a = sqrt(target_snr)
            std_amp = np.sqrt(target_snr)
            
            # Формируем огибающую:
            amp_mod = std_amp * lp_noise + target_snr
            
        else:
            # ФОНОВЫЕ ИСТОЧНИКИ
            # Пусть их средняя мощность равна 1.0. 
            # Для них мы можем сделать дисперсию поменьше, чтобы они "гудели" ровнее (например, дисперсия = 0.25)
            bg_mean = 1.0
            bg_std = 0.5 # Корень из 0.25
            
            amp_mod = bg_std * lp_noise + bg_mean
            
        # 2. Жесткая защита от отрицательных значений (так как Гауссиана может зайти в минус)
        # Если дисперсия равна среднему, то 1 стандартное отклонение (68% данных) находится в плюсе.
        # Отрицательные значения (хвост) мы просто обрезаем до минимального порога.
        baseline_noise_level = 0.05
        amp_mod = np.maximum(amp_mod, baseline_noise_level)

        S[k, :] = S_norm * amp_mod

        # z - это квадрат амплитудной огибающей (мгновенная мощность)
        z[k, :] = amp_mod**2

    # Генерация сенсорных данных
    X_s = GA[:, :Ndistr] @ S[:Ndistr, :]
    X_bg = GA[:, Ndistr:] @ S[Ndistr:, :]

    X_n = np.random.randn(Nsens, N)
    X_n = X_n - np.mean(X_n, axis=1, keepdims=True)
    X_n = X_n / np.std(X_n, axis=1, keepdims=True)

    return X_s, X_bg, X_n, z, GA, S

import scipy.signal

def generate_complex_sources(G, Nsrc, Ndistr, flanker, Ts, Fs):
    """
    Генерирует распределенные источники со сложной динамикой огибающей:
    комбинация медленного дрейфа и транзиентных всплесков (полу-непрерывная активность).
    """
    N = int(Ts * Fs)
    flanker_samples = int(flanker * Fs)

    b, a = butter(5, [8 / (Fs / 2), 12 / (Fs / 2)], btype='bandpass')
    b_lp, a_lp = butter(5, 0.5 / (Fs / 2), btype='lowpass')

    Gx, Gy, Gz = G[:, 0::3], G[:, 1::3], G[:, 2::3]
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

    # Окно для сглаживания транзиентов (например, 1.5 секунды)
    window_len = int(Fs * 1.5)
    burst_window = scipy.signal.windows.hann(window_len)
    burst_window /= np.sum(burst_window) # Нормализация окна

    # --- параметры "молчания" ---
    silence_threshold = 0.8      # порог: чем выше, тем реже молчание
    silence_floor     = 0.1     # остаточная амплитуда в паузе (не ровно 0)
    silence_smooth    = int(Fs * 0.3)  # сглаживание фронтов переключения

    for k in range(Nsrc):
        analytic_signal = hilbert(S[k, :])
        carrier_env = np.abs(analytic_signal)
        S_norm = S[k, :] / (carrier_env + 1e-8)

        # 1. Медленный дрейф
        drift_noise = np.random.randn(N + 2 * flanker_samples)
        lp_drift = filtfilt(b_lp, a_lp, drift_noise)
        lp_drift = lp_drift[flanker_samples : -flanker_samples] if flanker_samples > 0 else lp_drift.copy()
        lp_drift = lp_drift / np.std(lp_drift)

        if k < Ndistr:
            # 2. Транзиентные всплески
            burst_events = (np.random.rand(N) > 0.995).astype(float)
            burst_amps = np.random.uniform(2.0, 8.0, size=N)
            burst_signal = burst_events * burst_amps
            smooth_bursts = scipy.signal.fftconvolve(burst_signal, burst_window, mode='same')

            # 3. Гейт "активен/молчит" из медленного шума
            gate_noise = np.random.randn(N + 2 * flanker_samples)
            lp_gate = filtfilt(b_lp, a_lp, gate_noise)
            lp_gate = lp_gate[flanker_samples : -flanker_samples] if flanker_samples > 0 else lp_gate.copy()
            lp_gate = lp_gate / np.std(lp_gate)

            gate = (lp_gate > silence_threshold).astype(float)
            # сглаживаем фронты, чтобы не было щелчков
            if silence_smooth > 1:
                gate = scipy.signal.fftconvolve(
                    gate, np.hanning(silence_smooth) / np.sum(np.hanning(silence_smooth)),
                    mode='same'
                )
            gate = silence_floor + (1 - silence_floor) * gate

            # 4. Нелинейное смешивание + гейт
            amp_mod = np.exp(0.4 * lp_drift + smooth_bursts) * gate
        else:
            amp_mod = np.exp(0.3 * lp_drift) * 0.1

        S[k, :] = S_norm * amp_mod
        sigma_s = np.std(S[k, :]) + 1e-8
        S[k, :] = S[k, :] / sigma_s
        z[k, :] = (amp_mod / sigma_s) ** 2
        
    X_s = GA[:, :Ndistr] @ S[:Ndistr, :]
    X_bg = GA[:, Ndistr:] @ S[Ndistr:, :]

    X_n = np.random.randn(Nsens, N)
    X_n = X_n - np.mean(X_n, axis=1, keepdims=True)
    X_n = X_n / np.std(X_n, axis=1, keepdims=True)

    return X_s, X_bg, X_n, z, GA, S

def generate_correlated_sources(G, Nsrc, Ndistr, flanker, Ts, Fs, rho=0.85):
    """
    Генерирует распределенные источники. Целевые источники (Ndistr) 
    имеют заданный уровень корреляции (rho) как по фазе, так и по мощности.
    """
    N = int(Ts * Fs)
    flanker_samples = int(flanker * Fs)

    b, a = butter(5, [8 / (Fs / 2), 12 / (Fs / 2)], btype='bandpass')
    b_lp, a_lp = butter(5, 0.5 / (Fs / 2), btype='lowpass')

    Gx, Gy, Gz = G[:, 0::3], G[:, 1::3], G[:, 2::3]
    Nsens, Nsites = Gx.shape

    GA = np.zeros((Nsens, Nsrc))
    src_indsA = np.random.permutation(Nsites)
    for i in range(Nsrc):
        src_idx = src_indsA[i]
        r = np.random.rand(3)
        r = r / np.linalg.norm(r)
        GA[:, i] = Gx[:, src_idx]*r[0] + Gy[:, src_idx]*r[1] + Gz[:, src_idx]*r[2]

    # Базовый независимый шум для несущих
    raw_noise = np.random.randn(Nsrc, N + 2 * flanker_samples)
    
    # Общий шум для связывания целевых источников
    shared_carrier = np.random.randn(N + 2 * flanker_samples)
    shared_envelope = np.random.randn(N + 2 * flanker_samples)

    # 1. СВЯЗЫВАЕМ ФАЗЫ (НЕСУЩИЕ) ДЛЯ ЦЕЛЕВЫХ ИСТОЧНИКОВ
    for k in range(Ndistr):
        raw_noise[k] = np.sqrt(1 - rho**2) * raw_noise[k] + rho * shared_carrier

    S_full = filtfilt(b, a, raw_noise, axis=1)
    S = S_full[:, flanker_samples : -flanker_samples] if flanker_samples > 0 else S_full.copy()
    z = np.zeros((Nsrc, N))

    for k in range(Nsrc):
        analytic_signal = hilbert(S[k, :])
        carrier_env = np.abs(analytic_signal)
        S_norm = S[k, :] / carrier_env

        noise_mod = np.random.randn(N + 2 * flanker_samples)
        
        # 2. СВЯЗЫВАЕМ ОГИБАЮЩИЕ ДЛЯ ЦЕЛЕВЫХ ИСТОЧНИКОВ
        if k < Ndistr:
            noise_mod = np.sqrt(1 - rho**2) * noise_mod + rho * shared_envelope

        lp_noise_full = filtfilt(b_lp, a_lp, noise_mod)
        lp_noise = lp_noise_full[flanker_samples : -flanker_samples] if flanker_samples > 0 else lp_noise_full.copy()
        lp_noise = lp_noise / np.std(lp_noise)
        amp_mod = lp_noise - np.min(lp_noise) + 0.05

        S[k, :] = S_norm * amp_mod

        sigma_s = np.std(S[k, :])
        S[k, :] = S[k, :] / sigma_s
        z[k, :] = (amp_mod / sigma_s)**2

    X_s = GA[:, :Ndistr] @ S[:Ndistr, :]
    X_bg = GA[:, Ndistr:] @ S[Ndistr:, :]

    X_n = np.random.randn(Nsens, N)
    X_n = X_n - np.mean(X_n, axis=1, keepdims=True)
    X_n = X_n / np.std(X_n, axis=1, keepdims=True)

    return X_s, X_bg, X_n, z, GA, S

def generate_manifold_sources(G, Nsrc, Ndistr, flanker, Ts, Fs,
                              manifold='spiral', noise_power=0.1, rho=0.85):
    """
    Генерирует источники, чьи мощности лежат на низкоразмерном многообразии.
    Добавлен параметр `rho` для фазовой синхронизации (слом ICA).
    """
    N = int(Ts * Fs)
    flanker_samples = int(flanker * Fs)

    b, a = butter(5, [8 / (Fs / 2), 12 / (Fs / 2)], btype='bandpass')
    b_lp, a_lp = butter(5, 0.5 / (Fs / 2), btype='lowpass')

    Gx, Gy, Gz = G[:, 0::3], G[:, 1::3], G[:, 2::3]
    Nsens, Nsites = Gx.shape
    GA = np.zeros((Nsens, Nsrc))
    src_inds = np.random.permutation(Nsites)
    for i in range(Nsrc):
        r = np.random.rand(3); r /= np.linalg.norm(r)
        GA[:, i] = Gx[:, src_inds[i]]*r[0] + Gy[:, src_inds[i]]*r[1] + Gz[:, src_inds[i]]*r[2]

    # 1. Создание скрытого 2D-многообразия
    t = np.linspace(0, 1, N)
    if manifold == 'spiral':
        theta = t * 4 * np.pi
        x = theta * np.cos(theta)
        y = theta * np.sin(theta)
    elif manifold == 's_curve':
        x = t * 2 - 1
        y = np.sin(2 * np.pi * t)
    elif manifold == 'torus':
        u = t * 2 * np.pi
        v = t * 4 * np.pi
        x = (2 + np.cos(v)) * np.cos(u)
        y = (2 + np.cos(v)) * np.sin(u)
    else:
        raise ValueError("Unknown manifold type")

    x = (x - x.mean()) / x.std()
    y = (y - y.mean()) / y.std()
    true_manifold = np.column_stack((x, y))

    # 2. Создание мощностей (z_target)
    z_target = np.zeros((Ndistr, N))
    for k in range(Ndistr):
        if k == 0: f = np.sin(x) * np.cos(y)
        elif k == 1: f = np.cos(x) * np.sin(y)
        elif k == 2: f = np.sin(x + y)
        elif k == 3: f = np.cos(x - y)
        else: f = np.sin(2*x) + np.cos(2*y)
        f -= f.min()
        f = f / f.std() + 0.5
        z_target[k] = f
    z_target += np.random.randn(Ndistr, N) * noise_power

    # 3. Генерация несущих (С ДОБАВЛЕНИЕМ СИНХРОНИЗАЦИИ RHO ДЛЯ СЛОМА ICA)
    raw_noise = np.random.randn(Nsrc, N + 2*flanker_samples)
    shared_carrier = np.random.randn(N + 2*flanker_samples) # Общая фаза
    
    for k in range(Ndistr):
        raw_noise[k] = np.sqrt(1 - rho**2) * raw_noise[k] + rho * shared_carrier

    S_full = filtfilt(b, a, raw_noise, axis=1)
    S = S_full[:, flanker_samples : -flanker_samples] if flanker_samples > 0 else S_full.copy()

    z = np.ones((Nsrc, N))
    for k in range(Nsrc):
        analytic = hilbert(S[k, :])
        env = np.abs(analytic)
        S_norm = S[k, :] / env

        if k < Ndistr:
            amp_mod = z_target[k, :]
        else:
            noise_mod = np.random.randn(N + 2*flanker_samples)
            lp_noise_full = filtfilt(b_lp, a_lp, noise_mod)
            lp_noise = lp_noise_full[flanker_samples : -flanker_samples] if flanker_samples > 0 else lp_noise_full.copy()
            lp_noise /= lp_noise.std()
            amp_mod = lp_noise - np.min(lp_noise) + 0.05

        S[k, :] = S_norm * amp_mod
        sigma_s = np.std(S[k, :])
        S[k, :] /= sigma_s
        z[k, :] = (amp_mod / sigma_s)**2 

    X_s = GA[:, :Ndistr] @ S[:Ndistr, :]
    X_bg = GA[:, Ndistr:] @ S[Ndistr:, :]

    X_n = np.random.randn(Nsens, N)
    X_n -= X_n.mean(axis=1, keepdims=True)
    X_n /= np.std(X_n, axis=1, keepdims=True)

    return X_s, X_bg, X_n, z, GA, S, true_manifold

def generate_switching_microstates(G, Nsrc, Ndistr=3, flanker=1.0, Ts=600, Fs=100, 
                                   min_dur=0.5, max_dur=2.0, snr_target=5.0):
    """
    Генерирует сигналы со строгим переключением дискретных микросостояний.
    Гарантирует формирование четких кластеров в пространстве ковариаций (UMAP).
    
    Параметры:
    ----------
    min_dur, max_dur : float
        Минимальная и максимальная длительность удержания одного состояния (в секундах).
    snr_target : float
        Уровень мощности активного состояния относительно фона.
    """
    N = int(Ts * Fs)
    flanker_samples = int(flanker * Fs)

    # Фильтры: несущая (8-12 Гц) и сглаживание переходов (1 Гц)
    b, a = butter(5, [8 / (Fs / 2), 12 / (Fs / 2)], btype='bandpass')
    b_lp, a_lp = butter(3, 1.0 / (Fs / 2), btype='lowpass')

    Gx, Gy, Gz = G[:, 0::3], G[:, 1::3], G[:, 2::3]
    Nsens, Nsites = Gx.shape
    GA = np.zeros((Nsens, Nsrc))
    src_inds = np.random.permutation(Nsites)
    
    for i in range(Nsrc):
        r = np.random.rand(3); r /= np.linalg.norm(r)
        GA[:, i] = Gx[:, src_inds[i]]*r[0] + Gy[:, src_inds[i]]*r[1] + Gz[:, src_inds[i]]*r[2]

    # Генерация несущих
    raw_noise = np.random.randn(Nsrc, N + 2 * flanker_samples)
    S_full = filtfilt(b, a, raw_noise, axis=1)
    S = S_full[:, flanker_samples : -flanker_samples] if flanker_samples > 0 else S_full.copy()

    z = np.zeros((Nsrc, N))
    
    # 1. ГЕНЕРАЦИЯ ПОСЛЕДОВАТЕЛЬНОСТИ СОСТОЯНИЙ (МАРКОВСКАЯ ЦЕПЬ)
    # Состояния: 0, 1, ..., Ndistr-1 (активен 1 источник), и Ndistr (пауза/тишина)
    current_time = 0
    gt_states = np.full(N, -1, dtype=int)
    Z_target = np.zeros((Ndistr, N))
    
    # Вероятности: 80% времени работает какой-то паттерн, 20% времени - тишина
    states_pool = list(range(Ndistr)) + [-1]
    probs = [0.8 / Ndistr] * Ndistr + [0.2]
    
    while current_time < N:
        dur_samples = np.random.randint(int(min_dur * Fs), int(max_dur * Fs))
        end_time = min(current_time + dur_samples, N)
        
        state = np.random.choice(states_pool, p=probs)
        gt_states[current_time:end_time] = state
        
        if state != -1:
            Z_target[state, current_time:end_time] = snr_target
            
        current_time = end_time

    # 2. ПРИМЕНЕНИЕ ОГИБАЮЩИХ К ИСТОЧНИКАМ
    for k in range(Nsrc):
        analytic_signal = hilbert(S[k, :])
        carrier_env = np.abs(analytic_signal)
        S_norm = S[k, :] / (carrier_env + 1e-12)

        if k < Ndistr:
            # Сглаживаем прямоугольные импульсы для плавных переходов
            smooth_env = filtfilt(b_lp, a_lp, Z_target[k])
            
            # Добавляем небольшой модуляционный шум (10% от амплитуды)
            noise_mod = np.random.randn(N)
            noise_mod = filtfilt(b_lp, a_lp, noise_mod)
            noise_mod = (noise_mod / np.std(noise_mod)) * (snr_target * 0.1)
            
            amp_mod = smooth_env + noise_mod + 0.5 # 0.5 - базовая остаточная мощность
            amp_mod = np.clip(amp_mod, 0.05, None) # Защита от провалов ниже нуля
        else:
            # Фоновые источники тихо шумят
            noise_mod = np.random.randn(N + 2 * flanker_samples)
            lp_noise = filtfilt(b_lp, a_lp, noise_mod)
            lp_noise = lp_noise[flanker_samples : -flanker_samples] if flanker_samples > 0 else lp_noise.copy()
            amp_mod = (lp_noise / np.std(lp_noise)) * 0.2 + 0.3
            amp_mod = np.clip(amp_mod, 0.05, None)

        S[k, :] = S_norm * amp_mod
        
        sigma_s = np.std(S[k, :])
        S[k, :] = S[k, :] / sigma_s
        z[k, :] = (amp_mod / sigma_s)**2

    X_s = GA[:, :Ndistr] @ S[:Ndistr, :]
    X_bg = GA[:, Ndistr:] @ S[Ndistr:, :]

    X_n = np.random.randn(Nsens, N)
    X_n = X_n - np.mean(X_n, axis=1, keepdims=True)
    X_n = X_n / np.std(X_n, axis=1, keepdims=True)

    return X_s, X_bg, X_n, z, GA, S, gt_states

import numpy as np
import tensorflow as tf

def build_unvec_matrix(M):
    """Создает матрицу для быстрого преобразования вектора обратно в симметричную матрицу"""
    D = M * (M + 1) // 2
    W = np.zeros((D, M * M), dtype=np.float32)
    idx_i, idx_j = np.triu_indices(M)
    for k, (i, j) in enumerate(zip(idx_i, idx_j)):
        if i == j:
            W[k, i * M + j] = 1.0
        else:
            W[k, i * M + j] = 1.0 / np.sqrt(2.0)
            W[k, j * M + i] = 1.0 / np.sqrt(2.0)
    return W


@tf.keras.utils.register_keras_serializable()
class UnflattenSymmetricLayer(tf.keras.layers.Layer):
    """Развёртка плоских векторов обратно в симметричные матрицы."""
    def __init__(self, M, **kwargs):
        super().__init__(**kwargs)
        self.M = int(M)

    def build(self, input_shape):
        # Создаем матрицу развертки и сохраняем как внутренний НЕобучаемый тензор слоя.
        # Это избавляет от необходимости использовать глобальные переменные.
        w_init = build_unvec_matrix(self.M)
        self.W_UNVEC = self.add_weight(
            shape=w_init.shape,
            initializer=tf.keras.initializers.Constant(w_init),
            trainable=False,  # Важно: этот вес не обучается!
            name='W_UNVEC'
        )
        super().build(input_shape)

    def call(self, inputs):
        # Дифференцируемое восстановление (batch, D) -> (batch, M, M)
        C_flat = tf.matmul(inputs, self.W_UNVEC)
        return tf.reshape(C_flat, [-1, self.M, self.M])

    def get_config(self):
        config = super().get_config()
        config.update({"M": self.M})
        return config


@tf.keras.utils.register_keras_serializable()
class BiMapLayer(tf.keras.layers.Layer):
    """
    Билинейное преобразование: X_out = W_norm · X · W_norm^T
    """
    def __init__(self, d_out, orth_weight=0.01, **kwargs):
        super().__init__(**kwargs)
        self.d_out = int(d_out)
        self.orth_weight = float(orth_weight)

    def build(self, input_shape):
        self.d_in = int(input_shape[-1])
        init_val = np.eye(self.d_out, self.d_in, dtype=np.float32)
        self.W = self.add_weight(
            shape=(self.d_out, self.d_in),
            initializer=tf.keras.initializers.Constant(init_val),
            trainable=True,
            name="W_bimap"
        )
        super().build(input_shape)

    def call(self, inputs):
        W_norm = tf.math.l2_normalize(self.W, axis=1)    
        X_out = tf.einsum('ij,bjk,lk->bil', W_norm, inputs, W_norm)
        X_out = 0.5 * (X_out + tf.transpose(X_out, perm=[0, 2, 1]))

        if self.orth_weight > 0.0:
            WWT = tf.matmul(W_norm, W_norm, transpose_b=True)
            I = tf.eye(self.d_out)
            orth_penalty = tf.reduce_sum(tf.square(WWT - I))
            self.add_loss(self.orth_weight * orth_penalty)

        return X_out

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_out": self.d_out,
            "orth_weight": self.orth_weight
        })
        return config


@tf.keras.utils.register_keras_serializable()
class LogDiagScaleLayer(tf.keras.layers.Layer):
    """
    z_i = softplus(s_i) · log( max( (W C W^T)_{ii}, eps ) ) + b_i
    """
    def __init__(self, n_filters, epsilon=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.n_filters = int(n_filters)
        self.epsilon = epsilon

    def build(self, input_shape):
        self.s_raw = self.add_weight(
            shape=(self.n_filters,),
            initializer='zeros',
            trainable=True,
            name='log_scale_raw'
        )
        self.bias = self.add_weight(
            shape=(self.n_filters,),
            initializer='zeros',
            trainable=True,
            name='log_bias'
        )
        super().build(input_shape)

    def call(self, inputs):
        diags = tf.linalg.diag_part(inputs)
        z = tf.math.log(tf.maximum(diags, self.epsilon))
        scale = tf.nn.softplus(self.s_raw)
        return scale * z + self.bias

    def get_config(self):
        config = super().get_config()
        config.update({
            "n_filters": self.n_filters,
            "epsilon": self.epsilon
        })
        return config