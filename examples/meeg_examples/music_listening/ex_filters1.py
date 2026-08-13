# -*- coding: utf-8 -*-
"""
Created on Wed Oct 22 17:07:11 2025

@author: anton
"""

import mne
import numpy as np
import matplotlib.pyplot as plt
import umap

from scipy.signal import butter, filtfilt
from scipy.linalg import eigh
from scipy.linalg import inv, null_space

import sys
import os
import tensorflow as tf
from umap.parametric_umap import ParametricUMAP

lib_directory = os.path.abspath("C:/Users/ansbel/Documents/GitHub/pyRiemann") 
if lib_directory not in sys.path:
    sys.path.insert(0, lib_directory)
from pyriemann.estimation import Covariances
from pyriemann.utils.base import invsqrtm
from pyriemann.geometry.distance import pairwise_distance

# %%
# =============================================================================
# 1. ЗАГРУЗКА И ПРЕДОБРАБОТКА ДАННЫХ
# =============================================================================
fpath = "C:/Users/ansbel/Documents/GitHub/TriCo/data/external/music_listening/part1/eeg/10_07_g1_2223_raw.fif"
raw = mne.io.read_raw_fif(fpath, preload=True)
sfreq = raw.info['sfreq']

new_descriptions = [
    'RS_EC_1', 'RS_EO_1', '2Hz', '05Hz', '4Hz', '1Hz', '3Hz',
    'NoRy_1', 'Waltz_1', 'Waltz_2', 'NoRy_2', 'NoRy_3', 'Waltz_3',
    'NoRy_4', 'Waltz_4', 'NoRy_5', 'Waltz_5', 'RS_EC_2', 'RS_EO_2',
]

descriptions = raw.annotations.description
significant_mask = np.array([('BAD' not in d and 'EDGE' not in d) for d in descriptions])
significant_indices = np.where(significant_mask)[0]

new_desc = list(descriptions)
for idx, label in zip(significant_indices, new_descriptions):
    new_desc[idx] = label

old_annot = raw.annotations
new_annot = mne.Annotations(
    onset=old_annot.onset,
    duration=old_annot.duration,
    description=np.array(new_desc, dtype='U20'),
    orig_time=old_annot.orig_time
)
raw.set_annotations(new_annot)

raw_clean = raw.copy().pick_types(eeg=True)
data = raw.get_data()

# %%
# =============================================================================
# 2. ПОИСК SSD И ОТБЕЛИВАНИЕ
# =============================================================================
signal_pieces = []
noise_pieces = []

b_signal, a_signal = butter(3, np.array([15, 25]) / (int(sfreq) / 2), btype='band')
b_broad, a_broad = butter(3, np.array([13, 30]) / (int(sfreq) / 2), btype='band')
b_stop, a_stop = butter(3, np.array([14.5, 25.5]) / (int(sfreq) / 2), btype='stop')

crop_duration = 0.5
crop_samples = int(crop_duration * sfreq)

for annot in raw.annotations:
    desc = annot['description']
    if desc == 'BAD_' or annot['duration'] < 2.0:
        continue

    tmin = annot['onset']
    tmax = min(tmin + annot['duration'], raw.times[-1])

    start_idx = int(tmin * sfreq)
    end_idx = int(tmax * sfreq)
    seg_data = data[:, start_idx:end_idx]

    seg_signal = filtfilt(b_signal, a_signal, seg_data, axis=1)
    seg_noise_broad = filtfilt(b_broad, a_broad, seg_data, axis=1)
    seg_noise = filtfilt(b_stop, a_stop, seg_noise_broad, axis=1)

    if seg_signal.shape[1] > 2 * crop_samples:
        seg_signal_cropped = seg_signal[:, crop_samples:-crop_samples]
        seg_noise_cropped = seg_noise[:, crop_samples:-crop_samples]
    else:
        seg_signal_cropped = seg_signal
        seg_noise_cropped = seg_noise

    signal_pieces.append(seg_signal_cropped)
    noise_pieces.append(seg_noise_cropped)

signal_concatenated = np.concatenate(signal_pieces, axis=1)
noise_concatenated = np.concatenate(noise_pieces, axis=1)

C_signal = np.cov(signal_concatenated)
C_noise = np.cov(noise_concatenated)
C_noise_reg = C_noise + 1e-5 * np.trace(C_noise) * np.eye(C_noise.shape[0])

eigvals, eigvecs = eigh(C_signal, C_noise_reg)
idx_sorted = np.argsort(eigvals)[::-1]
W_ssd = eigvecs
component_variances = np.diag(W_ssd.T @ C_signal @ W_ssd)
valid_components = [v > 1e-6 for v in component_variances]
W_ssd = W_ssd[:, valid_components]
variances_filtered = component_variances[valid_components]
W_ssd = W_ssd / np.sqrt(variances_filtered)

A_ssd = C_signal @ W_ssd
n_components_ssd = W_ssd.shape[1]

# %%
# =============================================================================
# 3. НАРЕЗКА НА ЭПОХИ
# =============================================================================
Wsize = 2
Ssize = 0.5
overlap = Wsize - Ssize

X_windows_band = []
X_windows_unfilt = []
labels = []

raw_band = raw.copy().filter(l_freq=15, h_freq=25).pick_types(eeg=True)
raw_unfilt = raw.copy().pick_types(eeg=True)

for annot in raw.annotations:
    desc = annot['description']
    if annot['duration'] < Wsize or desc == 'BAD_':
        continue

    tmin = annot['onset']
    tmax = min(tmin + annot['duration'], raw.times[-1])

    raw_crop = raw_band.copy().crop(tmin=tmin, tmax=tmax)
    raw_crop_unfilt = raw_unfilt.copy().crop(tmin=tmin, tmax=tmax)

    epochs_band = mne.make_fixed_length_epochs(raw_crop, duration=Wsize, overlap=overlap, preload=True, verbose=False)
    epochs_unfilt = mne.make_fixed_length_epochs(raw_crop_unfilt, duration=Wsize, overlap=overlap, preload=True, verbose=False)

    if epochs_band and len(epochs_band) == len(epochs_unfilt):
        X_windows_band.append(epochs_band.get_data(copy=False))
        X_windows_unfilt.append(epochs_unfilt.get_data(copy=False))
        labels.extend([desc] * len(epochs_band))

X_windows_band = np.concatenate(X_windows_band, axis=0)
X_windows_unfilt = np.concatenate(X_windows_unfilt, axis=0)
labels = np.array(labels)

n_windows, n_ch, n_times = X_windows_band.shape
X_windows_ssd_proj = np.zeros((n_windows, n_components_ssd, n_times))
for i in range(n_windows):
    X_windows_ssd_proj[i] = W_ssd.T @ X_windows_band[i]

covmats_ssd = Covariances().fit_transform(X_windows_ssd_proj)
covmats = Covariances().fit_transform(X_windows_band / np.std(X_windows_band))

print("Отбеливание ковариационных матриц по среднему арифметическому...")
C_avg = np.mean(covmats_ssd, axis=0)                     
C_avg_invsqrt = invsqrtm(C_avg)                       
C_avg_sqrt = np.linalg.inv(C_avg_invsqrt)             
covmats_white = C_avg_invsqrt @ covmats_ssd @ C_avg_invsqrt

dist_matrix_init = pairwise_distance(covmats_white, metric='riemann')

# %%
from pyriemann.tangentspace import TangentSpace
import tensorflow as tf
from umap.parametric_umap import ParametricUMAP
import numpy as np

n_ch_white = covmats.shape[1]

# =============================================================================
# ПОДГОТОВКА ДАННЫХ
# =============================================================================
# Нормализация жизненно необходима, чтобы обучаемый шум стартовал в адекватном масштабе
X_cov_flat = covmats.reshape(covmats.shape[0], -1).astype(np.float32)
input_dim = X_cov_flat.shape[1]

N_dim = 2          
N_patterns = 20    

# =============================================================================
# ДЕКОДЕР С ОБУЧАЕМЫМ НЕСФЕРИЧНЫМ ШУМОМ
# =============================================================================
class SpatialPatternDecoder(tf.keras.layers.Layer):
    def __init__(self, M_channels, **kwargs):
        super().__init__(**kwargs)
        self.M_channels = M_channels

    def build(self, input_shape):
        n_filters = input_shape[-1]
        
        # 1. Обучаемые паттерны (матрица A)
        self.A = self.add_weight(
            shape=(self.M_channels, n_filters),
            initializer=tf.keras.initializers.RandomNormal(stddev=0.1),
            trainable=True,
            name="A_patterns"
        )
        
        # 2. ОБУЧАЕМЫЙ ГЛОБАЛЬНЫЙ ШУМ (индивидуальный для каждого сенсора)
        # Инициализируем отрицательным значением, чтобы после softplus шум стартовал с малого значения
        self.noise_log = self.add_weight(
            shape=(self.M_channels,),
            initializer=tf.keras.initializers.Constant(-3.0),
            trainable=True,
            name="sensor_noise"
        )

    def call(self, z):
        A_norm = tf.math.l2_normalize(self.A, axis=0)
        P = tf.exp(z)
        
        # Реконструкция полезного сигнала: A * P * A^T
        C_signal = tf.einsum("mf,bf,lf->bml", A_norm, P, A_norm)
        
        # Превращаем обучаемый вектор в строго положительную дисперсию (через softplus)
        noise_variance = tf.math.softplus(self.noise_log)
        
        # Создаем диагональную матрицу шума
        noise_diag = tf.linalg.diag(noise_variance)
        
        # Итоговая ковариация: Сигнал + Глобальный несферичный шум
        C_recon = C_signal + noise_diag
        return C_recon

def riemannian_distance_loss(y_true_flat, y_pred_flat):
    y_true_flat = tf.cast(y_true_flat, tf.float32)
    y_pred_flat = tf.cast(y_pred_flat, tf.float32)

    M = n_ch_white  
    
    C_true = tf.reshape(y_true_flat, [-1, M, M])
    C_pred = tf.reshape(y_pred_flat, [-1, M, M])
    
    # СГЛАЖИВАНИЕ МНОГООБРАЗИЯ (Riemannian Smoothing)
    # Защищает метрику от взрыва на направлениях с нулевой дисперсией (например, после Average Reference)
    smoothing_factor = 1e-2  
    eye_M = tf.eye(M, dtype=tf.float32)
    
    # Добавляем сглаживание к матрицам
    C_true_reg = C_true + smoothing_factor * eye_M
    C_pred_reg = C_pred + smoothing_factor * eye_M

    # Защитная константа для стабильности градиентов
    eps = 1e-7

    # Риманово расстояние
    eigvals_true, eigvecs_true = tf.linalg.eigh(C_true_reg)
    inv_sqrt_eigvals = 1.0 / tf.sqrt(tf.maximum(eigvals_true, eps))
    C_true_invsqrt = tf.einsum('bij,bj,bkj->bik', eigvecs_true, inv_sqrt_eigvals, eigvecs_true)

    C_mid = tf.einsum('bij,bjk,bkl->bil', C_true_invsqrt, C_pred_reg, C_true_invsqrt)
    C_mid = 0.5 * (C_mid + tf.transpose(C_mid, [0, 2, 1])) 

    eigvals_mid, eigvecs_mid = tf.linalg.eigh(C_mid)
    log_eigvals_mid = tf.math.log(tf.maximum(eigvals_mid, eps))
    logC_mid = tf.einsum('bij,bj,bkj->bik', eigvecs_mid, log_eigvals_mid, eigvecs_mid)

    dist_sq = tf.reduce_sum(tf.square(logC_mid), axis=(1, 2))
    return tf.reduce_mean(dist_sq)

# =============================================================================
# СБОРКА МОДЕЛЕЙ
# =============================================================================
encoder = tf.keras.Sequential([
    tf.keras.layers.InputLayer(shape=(input_dim,)),
    tf.keras.layers.Dense(100, activation="relu"),
    tf.keras.layers.Dense(100, activation="relu"),
    tf.keras.layers.Dense(100, activation="relu"),
    tf.keras.layers.Dense(N_dim, activation="linear", name="umap_embedding")
])

decoder = tf.keras.Sequential([
    tf.keras.layers.InputLayer(shape=(N_dim,)),
    tf.keras.layers.Dense(100, activation="relu"),
    tf.keras.layers.Dense(100, activation="relu"),
    tf.keras.layers.Dense(100, activation="relu"),
    tf.keras.layers.Dense(N_patterns, activation="linear", name="z_layer"),
    SpatialPatternDecoder(M_channels=n_ch_white, name="spatial_decoder"),
    tf.keras.layers.Flatten()
])

# =============================================================================
# ОБУЧЕНИЕ ParametricUMAP
# =============================================================================
print(f"Обучение с N_dim={N_dim}, N_patterns={N_patterns}")
embedder = ParametricUMAP(
    encoder=encoder,
    decoder=decoder,
    n_components=N_dim,
    dims=(input_dim,),
    metric="precomputed",
    parametric_reconstruction=True,
    autoencoder_loss=True,
    parametric_reconstruction_loss_fcn=riemannian_distance_loss,
    verbose=True
)

embedder.fit(X_cov_flat, precomputed_distances=dist_matrix_init)
umap_coords = embedder.embedding_

# %%
# =============================================================================
# 6. ИЗВЛЕЧЕНИЕ ПАРАМЕТРОВ И ОТРИСОВКА
# =============================================================================
import matplotlib as mpl

latent = embedder.encoder.predict(X_cov_flat)   # (n_windows, N_dim)

# 1. Извлекаем сырые значения z (логарифмы мощностей)
decoder_input = tf.keras.Input(shape=(N_dim,))
x = decoder_input
for layer in embedder.decoder.layers:
    x = layer(x)
    if layer.name == "z_layer":
        break

z_model = tf.keras.Model(inputs=decoder_input, outputs=x)
z_values = z_model.predict(latent)              # (n_windows, N_patterns)

# 2. Переводим в реальные мощности
powers = np.exp(z_values)

# 3. Извлекаем матрицу паттернов и нормируем 
spatial_decoder_layer = embedder.decoder.get_layer("spatial_decoder")
A_learned_raw = spatial_decoder_layer.get_weights()[0]
A_global = A_learned_raw / np.linalg.norm(A_learned_raw, axis=0) 

# ТЕПЕРЬ A_global - ЭТО И ЕСТЬ ГОТОВЫЕ ПАТТЕРНЫ В ПРОСТРАНСТВЕ СЕНСОРОВ!
found_patterns = [A_global[:, i] for i in range(N_patterns)]

# 5. Фильтры Хауфе (математика остается прежней)
C_global_mean = np.mean(covmats, axis=0)
C_global_inv = np.linalg.pinv(C_global_mean)
found_filters = [C_global_inv @ A_global[:, i] for i in range(N_patterns)]

print("Модели обучены end-to-end, сырые паттерны и мощности извлечены.")

# %%
# =============================================================================
# ДЕКОДЕР С ДИНАМИЧЕСКИМ ШУМОМ (ВНУТРЕННИЕ ПРОЕКЦИИ)
# =============================================================================
class SpatialPatternDecoder(tf.keras.layers.Layer):
    def __init__(self, M_channels, N_patterns, **kwargs):
        super().__init__(**kwargs)
        self.M_channels = M_channels
        self.N_patterns = N_patterns

    def build(self, input_shape):
        # input_shape здесь будет (None, N_dim), то есть 2D координаты UMAP
        
        # 1. Глобальные паттерны (Матрица A)
        self.A = self.add_weight(
            shape=(self.M_channels, self.N_patterns),
            initializer=tf.keras.initializers.RandomNormal(stddev=0.1),
            trainable=True,
            name="A_patterns"
        )
        
        # 2. Внутренний слой для генерации мощностей (z) из UMAP-координат
        self.z_dense = tf.keras.layers.Dense(
            units=self.N_patterns, 
            activation="linear", 
            name="z_projection"
        )
        
        # 3. Внутренний слой для генерации шума из UMAP-координат
        self.noise_dense = tf.keras.layers.Dense(
            units=self.M_channels, 
            activation="linear", 
            name="noise_projection"
        )

    def call(self, inputs):
        # inputs - это латентный вектор (batch_size, N_dim)
        
        # Генерируем параметры текущей эпохи
        z = self.z_dense(inputs)               # (batch_size, N_patterns)
        noise_log = self.noise_dense(inputs)   # (batch_size, M_channels)
        
        A_norm = tf.math.l2_normalize(self.A, axis=0)
        P = tf.exp(z)
        
        # Реконструкция полезного сигнала: A * P * A^T
        C_signal = tf.einsum("mf,bf,lf->bml", A_norm, P, A_norm)
        
        # Динамический вектор шума
        noise_variance = tf.math.softplus(noise_log)
        noise_diag = tf.linalg.diag(noise_variance)
        
        # Итоговая ковариация
        C_recon = C_signal + noise_diag
        return C_recon

# =============================================================================
# СБОРКА МОДЕЛЕЙ
# =============================================================================
encoder = tf.keras.Sequential([
    tf.keras.layers.InputLayer(shape=(input_dim,)),
    tf.keras.layers.Dense(100, activation="relu"),
    tf.keras.layers.Dense(100, activation="relu"),
    tf.keras.layers.Dense(100, activation="relu"),
    tf.keras.layers.Dense(N_dim, activation="linear", name="umap_embedding")
])

# Декодер теперь максимально простой снаружи, вся магия внутри нашего слоя
spatial_decoder = SpatialPatternDecoder(M_channels=n_ch_white, N_patterns=N_patterns, name="spatial_decoder")

decoder = tf.keras.Sequential([
    tf.keras.layers.InputLayer(shape=(N_dim,)),
    tf.keras.layers.Dense(100, activation="relu"),
    tf.keras.layers.Dense(100, activation="relu"),
    tf.keras.layers.Dense(100, activation="relu"),
    spatial_decoder,
    tf.keras.layers.Flatten()
])

# =============================================================================
# ОБУЧЕНИЕ ParametricUMAP
# =============================================================================
print(f"Обучение с N_dim={N_dim}, N_patterns={N_patterns}")
embedder = ParametricUMAP(
    encoder=encoder,
    decoder=decoder,
    n_components=N_dim,
    dims=(input_dim,),
    metric="precomputed",
    parametric_reconstruction=True,
    autoencoder_loss=True,
    parametric_reconstruction_loss_fcn=riemannian_distance_loss,
    verbose=True
)

embedder.fit(X_cov_flat, precomputed_distances=dist_matrix_init)
umap_coords = embedder.embedding_

# %%
# =============================================================================
# 6. ИЗВЛЕЧЕНИЕ ПАРАМЕТРОВ И ОТРИСОВКА
# =============================================================================
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import tensorflow as tf

latent = embedder.encoder.predict(X_cov_flat)   # (n_windows, N_dim)

# 1. Извлекаем признаки перед spatial_decoder
decoder_input = tf.keras.Input(shape=(N_dim,))
x = decoder_input
for layer in embedder.decoder.layers:
    if layer.name == "spatial_decoder":
        break
    x = layer(x)

# Создаем усеченную модель, которая выдает скрытые признаки
hidden_features_model = tf.keras.Model(inputs=decoder_input, outputs=x)
hidden_features = hidden_features_model.predict(latent)  # (n_windows, 100)

# 2. Вытаскиваем сам слой spatial_decoder
spatial_decoder_layer = embedder.decoder.get_layer("spatial_decoder")

# Прогоняем скрытые признаки через внутренние проекции слоя вручную
z_tensor = spatial_decoder_layer.z_dense(hidden_features)
noise_log_tensor = spatial_decoder_layer.noise_dense(hidden_features)

# 3. Переводим в numpy и в реальные физические величины
z_values = z_tensor.numpy()
noise_log_values = noise_log_tensor.numpy()

powers = np.exp(z_values)
dynamic_noise_variances = tf.math.softplus(noise_log_values).numpy()

# Проверка размерностей
print(f"Форма мощностей (z_values): {powers.shape} (ожидаем [n_windows, {N_patterns}])")
print(f"Форма шума (dynamic_noise_variances): {dynamic_noise_variances.shape} (ожидаем [n_windows, {n_ch_white}])")

# 4. Извлекаем матрицу паттернов и нормируем 
A_learned_raw = spatial_decoder_layer.A.numpy()
A_global = A_learned_raw / np.linalg.norm(A_learned_raw, axis=0) 

found_patterns = [A_global[:, i] for i in range(N_patterns)]

# 5. Фильтры Хауфе
C_global_mean = np.mean(covmats, axis=0)
C_global_inv = np.linalg.pinv(C_global_mean)
found_filters = [C_global_inv @ A_global[:, i] for i in range(N_patterns)]

print("Модели обучены, сырые паттерны, мощности и динамический шум извлечены.")

# =============================================================================
# ВИЗУАЛИЗАЦИЯ ДИНАМИКИ ШУМА
# =============================================================================
fig, ax = plt.subplots(figsize=(14, 6))

im = ax.imshow(dynamic_noise_variances.T, aspect='auto', cmap='magma', origin='lower')
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Дисперсия шума', rotation=270, labelpad=15)

ax.set_title('Динамика аппаратного (диагонального) шума по сенсорам', fontsize=14)
ax.set_xlabel('Номер окна (время)')
ax.set_ylabel('Индекс сенсора (канал ЭЭГ)')

plt.tight_layout()
plt.show()

# %%
from matplotlib.gridspec import GridSpec

def format_umap_axes(ax):
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(axis='both', which='both', length=0)
    ax.grid(True, linestyle='--', alpha=0.5, zorder=0)

# Выбираем индекс паттерна (0..N_patterns-1)
comp_idx = 2
W_sensor = found_filters[comp_idx]
A_pattern = found_patterns[comp_idx]

# 1. Мощность выбранного выученного источника
p_vals = powers[:, comp_idx]

# 2. Мощность через пространственный фильтр (w^T * C * w)
filtered_powers = np.array([W_sensor.T @ C @ W_sensor for C in covmats])

# 3. Нормализация (делим на std, без вычета среднего)
p_vals_norm = p_vals / np.std(p_vals)
filtered_powers_norm = filtered_powers / np.std(filtered_powers)

unique_labels_ordered = []
for lab in labels:
    if lab not in unique_labels_ordered:
        unique_labels_ordered.append(lab)

cmap = mpl.colormaps['viridis']
label_to_color = {lab: cmap(i / len(unique_labels_ordered)) for i, lab in enumerate(unique_labels_ordered)}

fig = plt.figure(figsize=(16, 10))
gs = GridSpec(2, 2, figure=fig, width_ratios=[1, 2], height_ratios=[1, 1])

ax_filt = fig.add_subplot(gs[0, 0])
mne.viz.plot_topomap(W_sensor, raw.info, axes=ax_filt, show=False)
ax_filt.set_title('Аппроксимация фильтра (Haufe)')

ax_patt = fig.add_subplot(gs[1, 0])
mne.viz.plot_topomap(A_pattern, raw.info, axes=ax_patt, show=False)
ax_patt.set_title('Истинный паттерн источника (A)')

# Отрисовка UMAP (используем выученную нормализованную мощность для цвета)
ax_umap = fig.add_subplot(gs[0, 1])
sc1 = ax_umap.scatter(umap_coords[:, 0], umap_coords[:, 1], c=p_vals_norm, cmap='plasma', s=15, zorder=2)
plt.colorbar(sc1, ax=ax_umap, label='Мощность источника (scaled)')
ax_umap.set_title(f'UMAP проекция\nЦвет: мощность компоненты {comp_idx+1}')
format_umap_axes(ax_umap)

ax_env = fig.add_subplot(gs[1, 1])
window_idx = np.arange(len(p_vals_norm))

# Отрисовка обеих мощностей для сравнения
ax_env.plot(window_idx, filtered_powers_norm, color='gray', lw=1.5, alpha=0.7, label='Пространственный фильтр', zorder=2)
ax_env.plot(window_idx, p_vals_norm, color='black', lw=1.2, label='Выученная сетью мощность', zorder=3)

xtick_positions = []
xtick_labels = []
for lab in unique_labels_ordered:
    mask = (labels == lab)
    if not np.any(mask):
        continue
    changes = np.diff(np.concatenate(([0], mask.astype(int), [0])))
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]
    for s, e in zip(starts, ends):
        ax_env.axvspan(s, e-1, facecolor=label_to_color[lab], alpha=0.2, zorder=0)
        ax_env.axvline(x=s, color='gray', linestyle='--', alpha=0.7, zorder=1)
        xtick_positions.append(s)
        xtick_labels.append(lab)

ax_env.set_xticks(xtick_positions)
ax_env.set_xticklabels(xtick_labels, rotation=45, ha='center', fontsize=8)
ax_env.set_xlabel('Номер окна')
ax_env.set_ylabel('Мощность (scaled by std)')
ax_env.set_title(f'Динамика мощности компоненты {comp_idx+1}')
ax_env.grid(True, axis='y', linestyle=':', alpha=0.6, zorder=0)
ax_env.legend(loc='upper right')

plt.suptitle(f'Анализ компоненты {comp_idx+1} из {N_patterns}', fontsize=16)
plt.tight_layout()
plt.show()
