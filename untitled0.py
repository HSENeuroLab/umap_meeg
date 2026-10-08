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

import scipy.signal
from scipy.signal import butter, filtfilt, hilbert

import numpy as np
from scipy.signal import butter, filtfilt, hilbert

def generate_distributed_sources(G, Nsrc, Ndistr, flanker, Ts, Fs, spiral_turns=2, target_env_std=10.0):

    """

    Параметры:

    ----------

    ...

    target_env_std : float

        Дисперсия (стандартное отклонение) огибающих таргетных источников.

        У фоновых источников она всегда равна 1.0. Увеличивая этот параметр,

        вы делаете размах витков спирали больше на фоне шума, не меняя среднее.

    """

    N = int(Ts * Fs)

    flanker_samples = int(flanker * Fs)



    # Установка фильтров

    # b, a = butter(5, [8 / (Fs / 2), 12 / (Fs / 2)], btype='bandpass')

    b, a = butter(5, [15 / (Fs / 2), 25 / (Fs / 2)], btype='bandpass')

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



    # =========================================================================

    # 1. ГЕНЕРАЦИЯ ФОРМ ОГИБАЮЩИХ (НИЖНЯЯ ГРАНИЦА ТАРГЕТОВ = СРЕДНЕЕ ШУМА)

    # =========================================================================

    amp_mods = np.zeros((Nsrc, N))

    

    t = np.linspace(0, Ts, N)

    f_rot = spiral_turns / Ts

    

    # Задаем среднее значение для шума (чтобы шум не падал ниже нуля при std=1)

    base_mean = 4.0

    

    # Амплитуда витков спирали (размах вверх от base_mean).

    # Начинаем почти от нуля (с 0.1) и постепенно раскручиваемся вверх до target_env_std

    r_spiral = np.linspace(0.1, target_env_std, N)



    for k in range(Nsrc):

        if k < Ndistr:

            # ТАРГЕТНЫЕ ИСТОЧНИКИ

            phase = k * (np.pi / 2) if Ndistr == 2 else k * (2 * np.pi / Ndistr)

            

            # (1 + np.cos) / 2 дает значения строго от 0 до 1.

            # Умножаем на r_spiral -> значения от 0 до r_spiral.

            # Прибавляем base_mean -> значения от base_mean до (base_mean + r_spiral).

            # Итог: нижняя граница спирали ВСЕГДА ровно касается base_mean!

            amp_mods[k, :] = base_mean + r_spiral * (1 + np.cos(2 * np.pi * f_rot * t + phase)) / 2

            

        else:

            # ФОНОВЫЕ ИСТОЧНИКИ

            noise_mod = np.random.randn(N + 2 * flanker_samples)

            lp_noise = filtfilt(b_lp, a_lp, noise_mod)

            lp_noise = lp_noise[flanker_samples : -flanker_samples] if flanker_samples > 0 else lp_noise.copy()

            

            # Стандартизация шума (mean=0, std=1)

            lp_noise = (lp_noise - np.mean(lp_noise)) / np.std(lp_noise)

            

            # Шум колеблется вокруг base_mean со стандартным отклонением 1.0

            amp_mods[k, :] = base_mean + lp_noise * 1.0



    amp_mods = np.clip(amp_mods, 0.01, None)

    # =========================================================================



    z = np.zeros((Nsrc, N))



    # Формирование огибающей и модуляция

    for k in range(Nsrc):

        analytic_signal = hilbert(S[k, :])

        carrier_env = np.abs(analytic_signal)

        

        # Получаем чистую несущую

        S_norm = S[k, :] / carrier_env

        

        # Нормализуем саму несущую к дисперсии 1.0, чтобы мощность сигнала 

        # зависела ИСКЛЮЧИТЕЛЬНО от нашей огибающей amp_mods

        S_norm = S_norm / np.std(S_norm)



        amp_mod = amp_mods[k, :]



        # Применение амплитудной модуляции

        S[k, :] = S_norm * amp_mod



        # ВАЖНО: Мы больше НЕ нормализуем S[k, :] к 1.0 (убрали деление на sigma_s). 

        # Если бы мы это сделали, мы бы сломали равенство средних значений огибающих!

        

        # Модуляция мощности — это просто квадрат нашей идеально выверенной огибающей

        z[k, :] = amp_mod**2



    # Генерация чистых сенсорных данных

    X_s = GA[:, :Ndistr] @ S[:Ndistr, :]

    X_bg = GA[:, Ndistr:] @ S[Ndistr:, :]



    # Генерация сенсорного шума

    X_n = np.random.randn(Nsens, N)

    X_n = X_n - np.mean(X_n, axis=1, keepdims=True)

    X_n = X_n / np.std(X_n, axis=1, keepdims=True)



    return X_s, X_bg, X_n, z, GA, S

# =============================================================================
# 1. ЗАГРУЗКА И ВЫБОР ЭЭГ КАНАЛОВ (МОНТАЖ 10-20)
# =============================================================================
fwd_fname = 'C:/Users/ansbel/Documents/GitHub/umap_meeg/examples/meeg_examples/simulations/fsaverage-fwd.fif'
info_fname = 'C:/Users/ansbel/Documents/GitHub/umap_meeg/examples/meeg_examples/simulations/fsaverage-info.fif'

fwd = mne.read_forward_solution(fwd_fname, verbose=False)
info = mne.io.read_info(info_fname, verbose=False)
Fs = info['sfreq']

picks = mne.pick_types(info, eeg=True, meg=False)
info_sub = mne.pick_info(info, sel=picks)
G_sub = fwd['sol']['data'][picks, :]  # Форма: (M_channels, N_vertices * 3)

M_channels = len(picks)

# =============================================================================
# 2. СИМУЛЯЦИЯ СИГНАЛОВ
# =============================================================================
Ts = 600          
Nsrc = 100      
Ndistr = 2

flanker = 25.0        
gamma = 0.1          

Wsize = 2
Ssize = 0.5
overlap = Wsize - Ssize

print("Генерация сетевых источников...")

X_s, X_bg, X_n, z, GA, S = generate_distributed_sources(
    G_sub, Nsrc, Ndistr=Ndistr,
    flanker=25.0, Ts=Ts, Fs=Fs
)

gamma = 0.1
X = X_s + X_bg + gamma * X_n / np.linalg.norm(X_s, 'fro')
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
covmats_w = np.array([Ww.T @ i @ Ww for i in covmats])

# 7. Проверка: средняя ковариация должна быть строго единичной
assert np.allclose(np.mean(covmats_w, axis=0), np.eye(n_components), atol=1e-5), "Отбеливание некорректно!"

# %%
plt.scatter(z[0,:],z[1,:])

# %%
plt.scatter(np.sqrt(z[0,:]), np.sqrt(z[1,:]), c=np.arange(z.shape[1]), cmap='viridis', s=5)
plt.axis('equal') # Чтобы оси были в одном масштабе
plt.show()

# %%
plt.plot(z[Ndistr:Ndistr+2,:].T)
plt.plot(z[:Ndistr,:].T)

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

raw_S = mne.io.RawArray(S, info_S)

# 2. Нарезаем на эпохи СТРОГО ТАК ЖЕ, как резали X
epochs_S = mne.make_fixed_length_epochs(
    raw_S, 
    duration=Wsize, 
    overlap=overlap, 
    preload=True, 
    verbose=False
)
epochs_S_data = epochs_S.get_data(copy=False)  # Форма: (n_epochs, Nsrc, n_samples)

# 3. Считаем дисперсию (мощность) сигнала по оси времени (axis=2) внутри каждой эпохи
true_source_variances = np.var(epochs_S_data, axis=2)  # Форма: (n_epochs, Nsrc)

plt.plot(true_source_variances[:,:10])

# %%
plt.scatter(np.sqrt(true_source_variances[:,0]),np.sqrt(true_source_variances[:,1]))

# %%
reducer = umap.UMAP(n_components=2, n_neighbors=20)
coords = reducer.fit_transform(true_source_variances)


# %%
dists_init = pairwise_distance(covmats_w, metric='riemann')

# %%
reducer = umap.UMAP(n_components=2, n_neighbors=20, metric='precomputed')
coords = reducer.fit_transform(dists_init)

# %%
plt.figure(figsize=(8, 6))
plt.scatter(coords[:, 0], coords[:, 1], s=5, cmap='Spectral')

plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.colorbar()
plt.show()

# %%
y_target = np.log(true_source_variances[:, :Ndistr]).astype(np.float32)

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
        w_init = build_unvec_matrix(self.M)
        self.W_UNVEC = self.add_weight(
            shape=w_init.shape,
            initializer=tf.keras.initializers.Constant(w_init),
            trainable=False,  
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
    def __init__(self, d_out, **kwargs):
        super().__init__(**kwargs)
        self.d_out = int(d_out)

    def build(self, input_shape):
        self.d_in = int(input_shape[-1])
        
        initializer = tf.keras.initializers.Orthogonal()
        init_val = initializer(shape=(self.d_out, self.d_in)).numpy()
        
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
        })
        return config

@tf.keras.utils.register_keras_serializable()
class LogDiagScaleLayer(tf.keras.layers.Layer):
    """
    z_i = log( scale_i * max( (W C W^T)_{ii}, eps ) )
    Штраф за внедиагональные элементы рассчитывается на основе матрицы корреляции.
    Масштаб строго больше нуля благодаря функции softplus, выступает как усилитель дисперсии.
    """
    def __init__(self, n_filters, epsilon=1e-9, off_diag_penalty=1.0, **kwargs):
        super().__init__(**kwargs)
        self.n_filters = int(n_filters)
        self.epsilon = epsilon
        self.off_diag_penalty = float(off_diag_penalty)

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs):
        diags = tf.linalg.diag_part(inputs)
        
        if self.off_diag_penalty > 0.0 and self.n_filters > 1:
            inv_std = tf.math.rsqrt(tf.maximum(diags, self.epsilon))
            inv_std_col = tf.expand_dims(inv_std, axis=-1)
            inv_std_row = tf.expand_dims(inv_std, axis=-2)
            R = inputs * inv_std_col * inv_std_row
            
            eps_eye = self.epsilon * tf.eye(self.n_filters, dtype=inputs.dtype)
            log_det_R = tf.linalg.logdet(R + eps_eye)
            
            N = tf.cast(self.n_filters, dtype=inputs.dtype)
            pairs_count = tf.maximum(N * (N - 1.0) / 2.0, 1.0) 
            
            pham_penalty = self.off_diag_penalty * tf.reduce_mean(-log_det_R) / pairs_count
            self.add_loss(pham_penalty)
                
        return tf.math.log(tf.maximum(diags, self.epsilon))

    def get_config(self):
        config = super().get_config()
        config.update({
            "n_filters": self.n_filters,
            "epsilon": self.epsilon,
            "off_diag_penalty": self.off_diag_penalty
        })
        return config

# %%
Npatt = 2

idx_i, idx_j = np.triu_indices(n_components)
multipliers = np.where(idx_i == idx_j, 1.0, np.sqrt(2.0)).astype(np.float32)
X_cov_flat = (covmats_w[:, idx_i, idx_j] * multipliers).astype(np.float32)
input_dim = int(X_cov_flat.shape[1])

inputs_enc = tf.keras.Input(shape=(input_dim,), dtype=tf.float32)
x = UnflattenSymmetricLayer(M=n_components)(inputs_enc)
x = BiMapLayer(d_out=Npatt, name="bimap_1")(x)
z_latent = LogDiagScaleLayer(n_filters=Npatt, epsilon=1e-9)(x)
encoder = tf.keras.Model(inputs=inputs_enc, outputs=z_latent)

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=50, min_delta=1e-4)

embedder = ParametricUMAP(
    encoder=encoder, 
    n_components=Npatt, 
    dims=(input_dim,), 
    metric="precomputed",
    
    # Метрика для учителя (так как y_target - это координаты, используем евклидову)
    target_metric="euclidean", 
    
    # КЛЮЧЕВОЙ ПАРАМЕТР: 1.0 означает 100% опору на граф учителя (y_target)
    target_weight=1.0, 
    
    n_neighbors=20, 
    verbose=True,
    keras_fit_kwargs={"callbacks": [early_stopping], "verbose": 1}
)

embedder.n_training_epochs = 3 
embedder.loss_report_frequency = embedder.n_training_epochs * 100

# 3. Запуск обучения
embedder.fit(
    X_cov_flat, 
    y=y_target, 
    precomputed_distances=dists_init
)

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
# Матрица фильтров в пространстве SSD (каждый столбец — отдельный фильтр)
W_bimap = embedder.encoder.get_layer("bimap_1").get_weights()[0]

# Фильтры в исходном пространстве сенсоров: X_latent = W_bimap @ W_ssd.T @ X_sensor
W_filters_sensor = Ww @ W_bimap.T

# =============================================================================
# 2. РАСЧЕТ ИСТИННЫХ МОЩНОСТЕЙ ИСТОЧНИКОВ
# =============================================================================
# Считаем физические мощности напрямую для каждого окна: diag(W C W^T)
powers_true = np.zeros((len(covmats_w), Npatt))
for i, C in enumerate(covmats_w):
    C_filtered = W_bimap @ C @ W_bimap.T
    powers_true[i] = np.diag(C_filtered)

A_sensor_raw =  np.mean(covmats, axis=0) @ W_filters_sensor

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
# 6. БЕЗОПАСНОЕ СОГЛАСОВАНИЕ (КОГДА Npatt != Ndistr)
# =============================================================================
# Убедитесь, что вы используете P_learned = powers.copy() (так как выше вы назвали переменную powers)
P_learned = powers.copy()  # (n_epochs, 12)
A_true = GA[:, :Ndistr]    # (Nsens, 6)
ga_scales_sq = np.linalg.norm(A_true, axis=0) ** 2            
P_true = true_source_variances[:, :Ndistr] * ga_scales_sq    # (n_epochs, 6)

n_learned = P_learned.shape[1] # В вашем случае 12
n_true = P_true.shape[1]       # В вашем случае 6

# Матрица корреляций (18x18)
C_corr = np.corrcoef(P_learned.T, P_true.T)                  

# ИЗМЕНЕНО: Правильно вырезаем блок корреляций между Learned и True (размер 12x6)
corr_block = np.abs(C_corr[:n_learned, n_learned:])                  

# Венгерский алгоритм
row, col = linear_sum_assignment(-corr_block)

# Сортируем пары по индексу истинного источника для красоты вывода
sort_idx = np.argsort(col)
matched_learned_idx = row[sort_idx]
matched_true_idx = col[sort_idx]
num_matched = len(matched_true_idx)

P_learned_matched = P_learned[:, matched_learned_idx]
A_learned_matched = A_global_norm[:, matched_learned_idx]
W_learned_matched = W_filters_sensor[:, matched_learned_idx]

print(f"Из {Ndistr} истинных источников алгоритм успешно сопоставил {num_matched} фильтров.")

# =============================================================================
# 7. ТОПОГРАММЫ (Только для совпавших пар)
# =============================================================================
fig, axes = plt.subplots(2, num_matched, figsize=(4 * num_matched, 7))
if num_matched == 1: axes = np.expand_dims(axes, axis=1) 

for i in range(num_matched):
    t_idx = matched_true_idx[i]
    
    ax_t = axes[0, i]
    mne.viz.plot_topomap(A_true[:, t_idx], info_sub, axes=ax_t, show=False, cmap='RdBu_r')
    ax_t.set_title(f'True Source {t_idx+1}')

    ax_l = axes[1, i]
    mne.viz.plot_topomap(A_learned_matched[:, i], info_sub, axes=ax_l, show=False, cmap='RdBu_r')
    ax_l.set_title(f'Learned {matched_learned_idx[i]+1}')

plt.tight_layout()
plt.show()

# =============================================================================
# 8. КОРРЕЛЯЦИИ (Только для совпавших пар)
# =============================================================================
P_true_matched = P_true[:, matched_true_idx]
P_true_n = P_true_matched / (P_true_matched.std(axis=0, keepdims=True) + 1e-9)
P_pred_n = P_learned_matched / (P_learned_matched.std(axis=0, keepdims=True) + 1e-9)

correlations = [np.corrcoef(P_true_n[:, i], P_pred_n[:, i])[0, 1] for i in range(num_matched)]

fig, axes = plt.subplots(num_matched, 1, figsize=(12, 3 * num_matched), sharex=True)
if num_matched == 1: axes = [axes]

fig.suptitle(f'Мощности: совпало {num_matched} из {Ndistr} источников', y=1.02)

# ИЗМЕНЕНО: Заменили жесткое range(6) на range(num_matched)
for i in range(num_matched):
    ax = axes[i]
    ax.plot(P_true_n[:, i], 'b-', lw=2.5, alpha=0.6, label='Истинная')
    ax.plot(P_pred_n[:, i], 'r', lw=1.5, label='Предсказанная')
    ax.set_title(f'True {matched_true_idx[i]+1} vs Learned {matched_learned_idx[i]+1} (corr = {correlations[i]:.3f})')
    ax.set_ylabel('Норм. мощность')
    ax.legend(loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.5)

plt.xlabel('Окна')
plt.tight_layout()
plt.show()

# %%

