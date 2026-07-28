# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 21:11:08 2026

@author: ansbel
"""

import numpy as np
from scipy.signal import butter, filtfilt, hilbert

def generate_distributed_sources(G, Nsrc, Ndistr, flanker, Ts, Fs):
    """
    Генерирует распределенные источники с амплитудной модуляцией.

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