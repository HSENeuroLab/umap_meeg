# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 14:32:53 2026

@author: ansbel
"""
import os
import sys

lib_directory = os.path.abspath("C:/Users/ansbel/Documents/GitHub/pyRiemann") 
if lib_directory not in sys.path:
    sys.path.insert(0, lib_directory)
from pyriemann.estimation import Covariances
from pyriemann.geometry.distance import pairwise_distance

sim_dir = os.path.abspath("C:/Users/ansbel/Documents/GitHub/umap_meeg/examples/meeg_examples/simulations/")
if sim_dir not in sys.path:
    sys.path.insert(0, sim_dir)
from scipy.linalg import eigh

import mne

import numpy as np 
import matplotlib.pyplot as plt

import umap
from umap.parametric_umap import ParametricUMAP

import tensorflow as tf
from scipy.signal import butter, filtfilt, hilbert

def generate_Nstate_blocks(G, Nsrc=20, Ndistr=3, flanker=1.0, Ts=300.0, Fs=100):
    """
    Генерирует симуляцию с блоками, где ВСЕ источники (и целевые, и фоновые) 
    формируются по методу амплитудной модуляции (Dähne et al. 2014).
    
    В периоды "неактивности" целевые источники просто зануляются (умножаются на маску).
    Локальный SNR в моменты активности сохраняется корректным.
    """
    N = int(Ts * Fs)
    flanker_samples = int(flanker * Fs)

    # Фильтры (согласно Dähne et al.)
    b, a = butter(5, [8 / (Fs / 2), 12 / (Fs / 2)], btype='bandpass')  # Несущая
    b_lp, a_lp = butter(5, 0.5 / (Fs / 2), btype='lowpass')            # Огибающая / маска

    # Инициализация прямой модели
    Gx, Gy, Gz = G[:, 0::3], G[:, 1::3], G[:, 2::3]
    Nsens, Nsites = Gx.shape
    GA = np.zeros((Nsens, Nsrc))
    src_inds = np.random.permutation(Nsites)
    
    for i in range(Nsrc):
        r = np.random.rand(3)
        r /= np.linalg.norm(r)
        GA[:, i] = Gx[:, src_inds[i]]*r[0] + Gy[:, src_inds[i]]*r[1] + Gz[:, src_inds[i]]*r[2]

    # 1. ГЕНЕРАЦИЯ СИГНАЛОВ
    # Генерируем ровно Nsrc сигналов
    raw_noise = np.random.randn(Nsrc, N + 2 * flanker_samples)
    S_full = filtfilt(b, a, raw_noise, axis=1)
    S_raw = S_full[:, flanker_samples : -flanker_samples] if flanker_samples > 0 else S_full.copy()

    S_continuous = np.zeros_like(S_raw)
    z_continuous = np.zeros_like(S_raw)

    # Формируем все сигналы единообразно (по Dähne et al.)
    for k in range(Nsrc):
        analytic_signal = hilbert(S_raw[k, :])
        carrier_env = np.abs(analytic_signal)
        S_norm = S_raw[k, :] / (carrier_env)

        noise_mod = np.random.randn(N + 2 * flanker_samples)
        lp_noise_full = filtfilt(b_lp, a_lp, noise_mod)
        lp_noise = lp_noise_full[flanker_samples : -flanker_samples] if flanker_samples > 0 else lp_noise_full.copy()
        
        lp_noise = lp_noise / (np.std(lp_noise))
        amp_mod = lp_noise - np.min(lp_noise) + np.finfo(float).eps

        S_continuous[k, :] = S_norm * amp_mod
        
        sigma_s = np.std(S_continuous[k, :])
        S_continuous[k, :] = S_continuous[k, :] / sigma_s
        z_continuous[k, :] = (amp_mod / sigma_s)**2

    # Разделяем на целевые и фоновые
    S_target = S_continuous[:Ndistr, :]
    S_bg = S_continuous[Ndistr:, :]

    # 2. СОЗДАНИЕ МАСОК И БЛОКОВ
    num_blocks = Ndistr + 1
    chunk_len = N // num_blocks
    gt_states = np.zeros(N, dtype=int)
    Z_mask = np.zeros((Ndistr, N))
    
    for state in range(1, num_blocks):
        start_idx = state * chunk_len
        end_idx = (state + 1) * chunk_len if state < num_blocks - 1 else N
        gt_states[start_idx:end_idx] = state
        Z_mask[state - 1, start_idx:end_idx] = 1.0
        
    for k in range(Ndistr):
        # Сглаживаем фронты включения/выключения
        Z_mask[k] = filtfilt(b_lp, a_lp, Z_mask[k])
        Z_mask[k] = np.clip(Z_mask[k], 0.4, 1.0)

    # 3. СБОРКА И ПРОЕКЦИЯ НА СЕНСОРЫ
    
    # --- Целевая компонента ---
    # Считаем изначальный глобальный std до маскирования (чтобы масштаб в активных зонах был = 1)
    X_s_continuous = GA[:, :Ndistr] @ S_target
    std_Xs = np.std(X_s_continuous)
    
    # Зануляем неактивные участки (умножаем на маску)
    S_target_masked = S_target * Z_mask
    
    X_s = GA[:, :Ndistr] @ S_target_masked
    X_s = (X_s - np.mean(X_s, axis=1, keepdims=True)) / std_Xs

    # --- Фоновая компонента ---
    X_bg = GA[:, Ndistr:] @ S_bg
    X_bg = (X_bg - np.mean(X_bg, axis=1, keepdims=True)) / np.std(X_bg)

    # --- Сенсорный шум ---
    X_n = np.random.randn(Nsens, N)
    X_n = (X_n - np.mean(X_n, axis=1, keepdims=True)) / np.std(X_n, axis=1, keepdims=True)

    # 4. СОХРАНЕНИЕ ФИНАЛЬНЫХ S И Z
    S_final = np.zeros((Nsrc, N))
    z = np.zeros((Nsrc, N))
    
    for k in range(Ndistr):
        S_final[k, :] = S_target_masked[k, :]
        # Истинная мощность таргета падает в неактивные периоды пропорционально квадрату маски
        z[k, :] = (Z_mask[k]**2) * z_continuous[k, :]

    for k in range(Ndistr, Nsrc):
        S_final[k, :] = S_bg[k - Ndistr, :]
        z[k, :] = z_continuous[k, :] 

    return X_s, X_bg, X_n, gt_states, z, GA, S_final

# =============================================================================
# 1. ЗАГРУЗКА И ВЫБОР ЭЭГ КАНАЛОВ (МОНТАЖ 10-20)
# =============================================================================
fwd_fname = 'C:/Users/ansbel/Documents/GitHub/umap_meeg/examples/meeg_examples/simulations/fsaverage-fwd.fif'
info_fname = 'C:/Users/ansbel/Documents/GitHub/umap_meeg/examples/meeg_examples/simulations/fsaverage-info.fif'

fwd = mne.read_forward_solution(fwd_fname, verbose=False)
info = mne.io.read_info(info_fname, verbose=False)
Fs = info['sfreq']

# 25 каналов из вашей сетки, образующих симметричный монтаж 10-20:
montage_25 = [
    # Лобно-полюсные и сагиттальные вспомогательные
    'Fp1', 'Fpz', 'Fp2',
    # Лобные
    'F7', 'F3', 'Fz', 'F4', 'F8',
    # Лобно-центральные
    'FCz',
    # Височные и центральные
    'T7', 'C3', 'Cz', 'C4', 'T8',
    # Центрально-теменные
    'CPz',
    # Теменные
    'P7', 'P3', 'Pz', 'P4', 'P8',
    # Теменно-затылочные
    'POz',
    # Затылочные
    'O1', 'Oz', 'O2',
    # Нижний затылочный ориентир
    'Iz'
]

# Выбираем только те ЭЭГ каналы, которые есть в нашем списке
# Функция сама проигнорирует каналы из списка, которых нет в info
picks = mne.pick_types(info, eeg=True, meg=False, selection=montage_25)
info_sub = mne.pick_info(info, sel=picks)
G_sub = fwd['sol']['data'][picks, :]  # Форма: (M_channels, N_vertices * 3)

M_channels = len(picks)

# =============================================================================
# 2. СИМУЛЯЦИЯ СИГНАЛОВ
# =============================================================================
Ts = 300          
Nsrc = 100         
Ndistr = 2          
flanker = 1.0        
Wsize = 2
Ssize = 0.5
overlap = Wsize - Ssize

# Физиологичные параметры для реального сигнала
gamma = 0.3           # Аппаратный шум сенсоров
current_snr = 1     # Мощность индуцированного альфа-ритма (20% от общего фона)

X_s, X_bg, X_n, gt_states, z, GA, S = generate_realistic_blocks(
    G_sub, Nsrc=Nsrc, Ndistr=Ndistr, flanker=flanker, Ts=Ts, Fs=Fs
)

# Простое аддитивное смешивание
X = current_snr * X_s + X_bg + gamma * X_n
X = X / np.std(X)


# =============================================================================
# 3. УПАКОВКА В MNE И РАСЧЕТ КОВАРИАЦИЙ
# =============================================================================
print("Создание Raw и нарезка на эпохи...")
raw = mne.io.RawArray(X, info_sub)

epochs = mne.make_fixed_length_epochs(
    raw, 
    duration=Wsize, 
    overlap=overlap, 
    preload=True, 
    verbose=False
)
epochs_data = epochs.get_data(copy=False) 

covmats = Covariances(estimator='cov').fit_transform(epochs_data)
mean_cov = np.mean(covmats, axis=0)

# 2. Спектральное разложение
eigvals, eigvecs = eigh(mean_cov)

# 3. Сортировка по убыванию
idx_sorted = np.argsort(eigvals)[::-1]
eigvals = eigvals[idx_sorted]
eigvecs = eigvecs[:, idx_sorted]

# 4. Отсечение сингулярных / зашумленных компонент
valid_mask = eigvals > 1e-6
eigvals = eigvals[valid_mask]
eigvecs = eigvecs[:, valid_mask]

# 5. Вычисление матрицы отбеливания Ww = V * Lambda^(-1/2)
Ww = eigvecs / np.sqrt(eigvals)[np.newaxis, :]
n_components = Ww.shape[1]

# 6. Векторизованное преобразование ковариационных матриц
covmats_w = Ww.T @ covmats @ Ww

# 7. Проверка: средняя ковариация должна быть строго единичной
assert np.allclose(np.mean(covmats_w, axis=0), np.eye(n_components), atol=1e-5), "Отбеливание некорректно!"

# Извлекаем индексы начала каждой эпохи из MNE
epoch_starts = epochs.events[:, 0]
W_samples = int(Wsize * Fs)

epoch_labels = []
for start in epoch_starts:
    # Берем метку из середины временного окна
    mid_point = start + (W_samples // 2)
    epoch_labels.append(gt_states[mid_point])

y = np.array(epoch_labels, dtype=np.int32)

print(f"Количество эпох: {len(covmats)}")
print(f"Количество меток: {len(y)}")
print(f"Распределение классов: {np.unique(y, return_counts=True)}")

# %%
from mne.preprocessing import ICA
ica = ICA(method='fastica')

# 3. Обучение ICA на отфильтрованных данных
ica.fit(raw)

# 4. Визуализация компонент (для ручного поиска артефактов глаз/сердца)
# ica.plot_components()  # Карты топографии компонент
# ica.plot_sources(raw)  # Временные ряды компонент

# %%
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
import numpy as np
import mne
from scipy.signal import hilbert

# =============================================================================
# 1-5. ПОИСК НАИЛУЧШИХ СОВПАДЕНИЙ (ВРЕМЕННАЯ КОРРЕЛЯЦИЯ)
# =============================================================================
# 1. Извлекаем временные ряды ICA-компонент
ica_sources = ica.get_sources(raw).get_data()  

# 2. Берем временные ряды истинных целевых источников
true_target_sources = S[:Ndistr, :]  

# 3. Считаем матрицу корреляций (Пирсона) 
C_corr_ica = np.corrcoef(true_target_sources, ica_sources)[:Ndistr, Ndistr:]

# 4. Ищем наилучшие совпадения
row_true, col_ica = linear_sum_assignment(-np.abs(C_corr_ica))

# 5. Вывод численных результатов
print("\n=== Совпадение истинных источников и компонент ICA ===")
for t_idx, ica_idx in zip(row_true, col_ica):
    corr_val = C_corr_ica[t_idx, ica_idx]
    print(f"Истинный источник {t_idx + 1} -> Максимально похожая ICA компонента: {ica_idx} (Врем. корреляция: {corr_val:.3f})")

# =============================================================================
# ВИЗУАЛИЗАЦИЯ 1: ВРЕМЕННЫЕ РЯДЫ (ОГИБАЮЩИЕ)
# =============================================================================
time_axis = np.arange(ica_sources.shape[1]) / Fs

fig1, axes1 = plt.subplots(Ndistr, 1, figsize=(12, 4 * Ndistr), sharex=True)
if Ndistr == 1: axes1 = [axes1]
fig1.suptitle('Сравнение временных рядов (огибающих): Истинные источники vs ICA', fontsize=14, y=1.02)

for i, (t_idx, ica_idx) in enumerate(zip(row_true, col_ica)):
    corr_val = C_corr_ica[t_idx, ica_idx]
    sign = np.sign(corr_val) 
    
    true_ts = true_target_sources[t_idx, :]
    ica_ts = ica_sources[ica_idx, :] * sign
    
    true_env = np.abs(hilbert(true_ts))
    ica_env = np.abs(hilbert(ica_ts))
    
    true_env_n = (true_env - np.mean(true_env)) / (np.std(true_env) + 1e-12)
    ica_env_n = (ica_env - np.mean(ica_env)) / (np.std(ica_env) + 1e-12)
    
    ax = axes1[i]
    ax.plot(time_axis, true_env_n, 'b-', lw=2.5, alpha=0.6, label='Истинный источник (огибающая)')
    ax.plot(time_axis, ica_env_n, 'r-', lw=1.5, alpha=0.8, label='ICA компонента (огибающая)')
    
    ax.set_title(f'Источник {t_idx + 1} vs ICA Comp {ica_idx} (Корреляция рядов: {abs(corr_val):.3f})')
    ax.set_ylabel('Норм. амплитуда')
    ax.legend(loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.5)

axes1[-1].set_xlabel('Время (секунды)')
fig1.tight_layout()

# =============================================================================
# ВИЗУАЛИЗАЦИЯ 2: ПРОСТРАНСТВЕННЫЕ ПАТТЕРНЫ (ТОПОГРАФИИ)
# =============================================================================
# Извлекаем пространственные паттерны
# true_patterns - столбцы матрицы GA, ica_patterns - столбцы матрицы смешивания ICA
true_patterns = GA[:, :Ndistr] 
ica_patterns = ica.get_components() # Форма: (n_channels, n_ica_components)

fig2, axes2 = plt.subplots(Ndistr, 2, figsize=(8, 4 * Ndistr))
# Обработка случая, если Ndistr == 1 (axes2 будет 1D массивом, приводим к 2D)
if Ndistr == 1:
    axes2 = np.expand_dims(axes2, axis=0)

fig2.suptitle('Сравнение пространственных паттернов (Топографии)', fontsize=14, y=1.05)

for i, (t_idx, ica_idx) in enumerate(zip(row_true, col_ica)):
    # Извлекаем знак из временной корреляции для правильной ориентации полюсов паттерна
    corr_val = C_corr_ica[t_idx, ica_idx]
    sign = np.sign(corr_val)
    
    # Истинный паттерн и паттерн ICA (с учетом знака)
    true_pat = true_patterns[:, t_idx]
    ica_pat = ica_patterns[:, ica_idx] * sign
    
    # Считаем пространственную корреляцию (косинусное сходство / Пирсон)
    spat_corr = np.corrcoef(true_pat, ica_pat)[0, 1]
    
    # Отрисовка истинного паттерна
    ax_true = axes2[i, 0]
    mne.viz.plot_topomap(true_pat, info_sub, axes=ax_true, show=False)
    ax_true.set_title(f'Истинный паттерн {t_idx + 1}')
    
    # Отрисовка ICA паттерна
    ax_ica = axes2[i, 1]
    mne.viz.plot_topomap(ica_pat, info_sub, axes=ax_ica, show=False)
    ax_ica.set_title(f'ICA паттерн {ica_idx}\n(Простр. корр: {spat_corr:.3f})')

fig2.tight_layout()
plt.show()

# %%
# =============================================================================
# 3.5 ИЗВЛЕЧЕНИЕ ДИСПЕРСИИ ИСТИННЫХ ИСТОЧНИКОВ И ИХ МАСШТАБИРОВАНИЕ
# =============================================================================
print("Нарезка истинных источников S на эпохи и расчет дисперсий...")

# 1. Создаем фейковое info для источников (чтобы MNE мог их "проглотить")
info_S = mne.create_info(
    ch_names=[f"Src_{i+1}" for i in range(S.shape[0])], 
    sfreq=Fs, 
    ch_types='misc'
)

SS = S.copy()
SS[:Ndistr,:] *= SS[:Ndistr,:] * current_snr
raw_S = mne.io.RawArray(SS, info_S)

epochs_S = mne.make_fixed_length_epochs(
    raw_S, 
    duration=Wsize, 
    overlap=overlap, 
    preload=True, 
    verbose=False
)
epochs_S_data = epochs_S.get_data(copy=False)
true_source_variances = np.var(epochs_S_data, axis=2)  

plt.plot(true_source_variances[:,0])
plt.plot(true_source_variances[:,1])
plt.plot(true_source_variances[:,2])

# %%
dists_init = pairwise_distance(covmats_w, metric='riemann')

# %%
import matplotlib.pyplot as plt
import umap

reducer = umap.UMAP(n_components=2, n_neighbors=20, metric='precomputed')
coords = reducer.fit_transform(dists_init)

plt.figure(figsize=(8, 6))
plt.scatter(coords[:, 0], coords[:, 1], s=5, cmap='Spectral')

plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.colorbar() 
plt.show()

# %%
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
        
        # Базовая структура единичной матрицы
        init_val = np.eye(self.d_out, self.d_in, dtype=np.float32)
        
        # Добавляем небольшой гауссовский шум для нарушения симметрии
        noise = np.random.normal(loc=0.0, scale=0.01, size=(self.d_out, self.d_in)).astype(np.float32)
        init_val = init_val + noise
        
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
    z_i = scale_i * log( max( (W C W^T)_{ii}, eps ) ) + bias_i
    (По аналогии с аффинным преобразованием в BatchNorm / LayerNorm)
    """
    def __init__(self, n_filters, epsilon=1e-7, off_diag_penalty=1, **kwargs):
        super().__init__(**kwargs)
        self.n_filters = int(n_filters)
        self.epsilon = epsilon
        self.off_diag_penalty = off_diag_penalty

    def build(self, input_shape):
        self.scale = self.add_weight(
            shape=(self.n_filters,),
            initializer='ones',
            trainable=True,
            name='scale'
        )
        self.bias = self.add_weight(
            shape=(self.n_filters,),
            initializer='zeros',
            trainable=True,
            name='bias'
        )
        super().build(input_shape)

    def call(self, inputs):
        diags = tf.linalg.diag_part(inputs)
        
        # --- НАЛОЖЕНИЕ ШТРАФА НА ВНЕДИАГОНАЛЬНЫЕ ЭЛЕМЕНТЫ ---
        if self.off_diag_penalty > 0.0:
            zeros_diag = tf.zeros_like(diags)
            off_diagonals = tf.linalg.set_diag(inputs, zeros_diag)
            penalty_loss = self.off_diag_penalty * tf.reduce_mean(tf.square(off_diagonals))
            self.add_loss(penalty_loss)
        # ----------------------------------------------------

        z = tf.math.log(tf.maximum(diags, self.epsilon))
        
        # Применяем и масштаб (scale), и сдвиг (bias)
        return self.scale * z + self.bias

    def get_config(self):
        config = super().get_config()
        config.update({
            "n_filters": self.n_filters,
            "epsilon": self.epsilon,
           "off_diag_penalty": self.off_diag_penalty
        })
        return config
    
Npatt = 3

idx_i, idx_j = np.triu_indices(n_components)
multipliers = np.where(idx_i == idx_j, 1.0, np.sqrt(2.0)).astype(np.float32)
X_cov_flat = (covmats_w[:, idx_i, idx_j] * multipliers).astype(np.float32)
input_dim = int(X_cov_flat.shape[1])

inputs_enc = tf.keras.Input(shape=(input_dim,), dtype=tf.float32)
x = UnflattenSymmetricLayer(M=n_components)(inputs_enc)
x = BiMapLayer(d_out=Npatt, orth_weight=0.01, name="bimap_1")(x)
z_latent = LogDiagScaleLayer(n_filters=Npatt, epsilon=1e-4)(x)
encoder = tf.keras.Model(inputs=inputs_enc, outputs=z_latent)

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=50, min_delta=1e-4)
embedder = ParametricUMAP(
    encoder=encoder, n_components=Npatt, dims=(input_dim,), 
    metric="precomputed", n_neighbors=20, verbose=True,
    keras_fit_kwargs={"callbacks": [early_stopping], "verbose": 1}
)

embedder.n_training_epochs = 3 
embedder.loss_report_frequency = embedder.n_training_epochs * 100
embedder.fit(X_cov_flat, precomputed_distances=dists_init)

# %%
# =============================================================================
# ВИЗУАЛИЗАЦИЯ 
# =============================================================================
import matplotlib.pyplot as plt

history = embedder._history

# Создаем фигуру с двумя подграфиками (для Recon и для UMAP)
fig, ax1 = plt.subplots(1, 1)
# --- График 2: Истинный лосс графа UMAP ---
ax1.plot(history['umap_loss'], label='UMAP Graph Loss (Cross-Entropy)', color='purple', linewidth=2)
    
ax1.set_title('Качество кодера')
ax1.set_xlabel('Шаги оценки')
ax1.set_ylabel('Cross-Entropy')
ax1.legend()
ax1.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()


# %%
import numpy as np

# =============================================================================
# 1. ИЗВЛЕЧЕНИЕ ВЕСОВ ФИЛЬТРОВ ИЗ МОДЕЛИ
# =============================================================================
# Получаем веса обученного слоя BiMap. Размерность: (Npatt, n_components_ssd)
W_bimap = embedder.encoder.get_layer("bimap_1").get_weights()[0]

# Матрица фильтров в пространстве SSD (каждый столбец — отдельный фильтр)
W_filters_ssd = W_bimap.T 

# Фильтры в исходном пространстве сенсоров: X_latent = W_bimap @ W_ssd.T @ X_sensor
W_filters_sensor = Ww @ W_filters_ssd

# =============================================================================
# 2. РАСЧЕТ ИСТИННЫХ МОЩНОСТЕЙ ИСТОЧНИКОВ
# =============================================================================
# Считаем физические мощности напрямую для каждого окна: diag(W C W^T)
powers_true = np.zeros((len(covmats_w), Npatt))
for i, C in enumerate(covmats_w):
    C_filtered = W_bimap @ C @ W_bimap.T
    powers_true[i] = np.diag(C_filtered)

# =============================================================================
# 3. ОЦЕНКА ПАТТЕРНОВ ЧЕРЕЗ C @ W (В СЕНСОРНОМ ПРОСТРАНСТВЕ)
# =============================================================================
# Усредненная ковариационная матрица в пространстве SSD
C_avg_ssd = np.mean(covmats_w, axis=0)

# Паттерны в пространстве SSD
A_ssd = C_avg_ssd @ W_filters_ssd

# Проекция паттернов обратно в исходное пространство сенсоров
A_sensor_raw = np.linalg.pinv(Ww.T) @ A_ssd

# Нормировка физических паттернов для визуализации
scales = np.linalg.norm(A_sensor_raw, axis=0)
A_global_norm = A_sensor_raw / scales

# =============================================================================
# 4. СОРТИРОВКА ПО СРЕДНЕЙ МОЩНОСТИ
# =============================================================================
power_variances = np.mean(powers_true, axis=0)
sort_idx = np.argsort(power_variances)[::-1]

# Сортируем мощности, паттерны и фильтры по убыванию значимости
powers = powers_true[:, sort_idx]
A_sorted = A_global_norm[:, sort_idx]
W_sorted = W_filters_sensor[:, sort_idx]

print(f"Порядок компонент по убыванию средней мощности: {sort_idx}")

# =============================================================================
# 5. СПИСКИ ДЛЯ ОТРИСОВКИ MNE
# =============================================================================
found_filters = [W_sorted[:, i] for i in range(Npatt)]
found_patterns = [A_sorted[:, i] for i in range(Npatt)]

# %%
from scipy.optimize import linear_sum_assignment

# =============================================================================
# 6. СОГЛАСОВАНИЕ ВЫУЧЕННЫХ ПАТТЕРНОВ С ИСТИННЫМИ ИСТОЧНИКАМИ
# =============================================================================

# Истинные топографии целевых источников и их масштабы
A_true = GA[:, :Ndistr]                              # (Nsens, Ndistr)
ga_scales_sq = np.linalg.norm(A_true, axis=0) ** 2   # ||g_i||^2

# Истинная линейная мощность источника в сенсорном пространстве (по эпохам)
# z_true_epochs уже считается в блоке 3.5, но убедимся, что это именно линейная мощность
P_true = true_source_variances[:, :Ndistr] * ga_scales_sq   # (n_epochs, Ndistr)

# Выученная линейная мощность фильтров в whitened-пространстве
P_learned = powers_true.copy()                       # (n_epochs, Npatt)

# --- Согласование: какой learned-фильтр соответствует какому истинному источнику ---
# Считаем корреляционную матрицу между временными рядами мощностей
C_corr = np.corrcoef(P_learned.T, P_true.T)          # (Np+Nd, Np+Nd)
corr_block = np.abs(C_corr[:Npatt, Npatt:])  # (Npatt, Ndistr)

# Оптимальное назначение (максимизируем сумму |corr|)
row, col = linear_sum_assignment(-corr_block)
# row[k] -> индекс learned, col[k] -> индекс истинного источника

# Применяем перестановку: learned-массивы выстраиваются под истинные источники
learned_order = row[np.argsort(col)]                
P_learned_matched = P_learned[:, learned_order]
A_learned_matched = A_global_norm[:, learned_order]
W_learned_matched = W_filters_sensor[:, learned_order]

# --- НОВОЕ: Коррекция знака паттернов ---
for i in range(Ndistr):
    # Берем реальный знак корреляции (без модуля)
    real_corr = C_corr[learned_order[i], Npatt + i]
    sign = np.sign(real_corr)
    
    # Если корреляция отрицательная, разворачиваем паттерн
    A_learned_matched[:, i] *= sign
    W_learned_matched[:, i] *= sign

# =============================================================================
# 7. ТОПОГРАММЫ: истинные vs согласованные выученные
# =============================================================================
fig, axes = plt.subplots(2, Ndistr, figsize=(4 * Ndistr, 7))
for i in range(Ndistr):
    ax_t = axes[0, i]
    mne.viz.plot_topomap(A_true[:, i], info_sub, axes=ax_t, show=False, cmap='RdBu_r')
    ax_t.set_title(f'True source {i+1}')

    ax_l = axes[1, i]
    mne.viz.plot_topomap(A_learned_matched[:, i], info_sub, axes=ax_l,
                         show=False, cmap='RdBu_r')
    ax_l.set_title(f'Learned (matched) {i+1}')

plt.tight_layout()
plt.show()

# =============================================================================
# 8. КОРРЕЛЯЦИИ ВРЕМЕННЫХ РЯДОВ МОЩНОСТЕЙ
# =============================================================================
# По-столбцовая нормировка (каждую колонку — к единичной std)
P_true_n = P_true / P_true.std(axis=0, keepdims=True)
P_pred_n = P_learned_matched / P_learned_matched.std(axis=0, keepdims=True)

correlations = [
    np.corrcoef(P_true_n[:, i], P_pred_n[:, i])[0, 1]
    for i in range(Ndistr)
]

fig, axes = plt.subplots(Ndistr, 1, figsize=(12, 3 * Ndistr), sharex=True)
fig.suptitle('Мощность источника: истинная vs предсказанная (согласовано)', y=1.02)
for i in range(Ndistr):
    ax = axes[i] if Ndistr > 1 else axes
    ax.plot(P_true_n[:, i], 'b-', lw=2.5, alpha=0.6, label='Истинная')
    ax.plot(P_pred_n[:, i], 'r', lw=1.5, label='Предсказанная')
    ax.set_title(f'Источник {i+1} (corr = {correlations[i]:.3f})')
    ax.set_ylabel('Норм. мощность')
    ax.legend(loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.5)
plt.xlabel('Окна')
plt.tight_layout()
plt.show()

print("Корреляции по источникам:", correlations)

# %%
