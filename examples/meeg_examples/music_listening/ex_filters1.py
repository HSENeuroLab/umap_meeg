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
N_patterns = 40    

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
comp_idx = 7
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

# %%
# =============================================================================
# 7. СУПЕР-ИНТЕРАКТИВНЫЙ ДАШБОРД: АКТИВАЦИЯ ИСТОЧНИКОВ В UMAP
# =============================================================================
import matplotlib.patheffects as pe
from matplotlib.gridspec import GridSpec

print("Подготовка интерактивного дашборда...")

# 1. Отбираем самые "дисперсные" источники
power_variances = np.var(powers, axis=0)
top_n = min(15, N_patterns)
top_indices = np.argsort(power_variances)[::-1][:top_n]

print(f"Отображаем топ-{top_n} источников с наибольшей дисперсией: {top_indices}")

# 2. Настройка цветов и легенды с сохранением ИСХОДНОГО порядка!
# Используем ваш список new_descriptions из начала скрипта
unique_classes = []
for lab in new_descriptions:
    if lab in labels and lab not in unique_classes:
        unique_classes.append(lab)
        
# На всякий случай добавляем те, что есть в labels, но вдруг не попали в список
for lab in labels:
    if lab not in unique_classes:
        unique_classes.append(lab)

class_to_id = {lab: i + 1 for i, lab in enumerate(unique_classes)}
cmap_classes = plt.cm.tab20
color_map = {lab: cmap_classes(i / max(1, len(unique_classes) - 1)) for i, lab in enumerate(unique_classes)}
point_colors = [color_map[lab] for lab in labels]

# 3. Настройка сетки графика
N_COLS_TOPO = 3
N_ROWS_TOPO = int(np.ceil(top_n / N_COLS_TOPO))

fig = plt.figure(figsize=(18, max(8, 2 * N_ROWS_TOPO)))
gs = GridSpec(N_ROWS_TOPO + 1, N_COLS_TOPO + 2, width_ratios=[3.0, 0.5] + [1]*N_COLS_TOPO, height_ratios=[0.5] + [2]*N_ROWS_TOPO)

# =============================================================================
# ПОСТРОЕНИЕ UMAP (Левая панель)
# =============================================================================
ax_umap = fig.add_subplot(gs[:, 0])
ax_text = fig.add_subplot(gs[0, 2:])
ax_text.axis('off')

# Рисуем все эпохи
ax_umap.scatter(umap_coords[:, 0], umap_coords[:, 1], c=point_colors, 
                s=15, alpha=0.4, edgecolors='white', linewidths=0.2, zorder=1)

# Рисуем центроиды классов
for lab in unique_classes:
    mask = (labels == lab)
    if not np.any(mask):
        continue
    cx, cy = np.mean(umap_coords[mask], axis=0)
    ax_umap.scatter(cx, cy, marker='*', s=450, facecolor=color_map[lab], 
                    edgecolor='black', linewidths=1.0, zorder=3)
    # Цифра внутри звезды
    ax_umap.text(cx, cy, str(class_to_id[lab]), fontsize=11, fontweight='bold', color='white', 
                 ha='center', va='center', zorder=4, 
                 path_effects=[pe.withStroke(linewidth=2.5, foreground="black")])

ax_umap.set_title("Фазовое пространство UMAP", fontsize=14)
ax_umap.set_xlabel("UMAP 1")
ax_umap.set_ylabel("UMAP 2")
ax_umap.grid(True, linestyle='--', alpha=0.4, zorder=0)

# Легенда (увеличили ncol=4, уменьшили шрифт)
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map[lab], markersize=8) for lab in unique_classes]
legend_labels = [f"{class_to_id[lab]}: {lab}" for lab in unique_classes]
ax_umap.legend(handles, legend_labels, title="Условия", loc='upper center', 
               bbox_to_anchor=(0.5, -0.08), ncol=4, fontsize=8, title_fontsize=10)

# Информационный текст сверху
txt_info = ax_text.text(0.5, 0.5, "Наведите курсор на точки UMAP", 
                        ha='center', va='center', fontsize=16, color='gray', fontweight='bold')

# =============================================================================
# ПОСТРОЕНИЕ СТАТИЧНЫХ ТОПОМАПОВ (Правая панель)
# =============================================================================
print("Генерация топомапов (может занять несколько секунд)...")
ax_topos = []
overlays = []
titles = []

mne_info = raw.info 

for i, comp_idx in enumerate(top_indices):
    row = 1 + i // N_COLS_TOPO
    col = 2 + i % N_COLS_TOPO
    ax = fig.add_subplot(gs[row, col])
    
    mne.viz.plot_topomap(found_patterns[comp_idx], mne_info, axes=ax, show=False, contours=0)
    
    overlay = plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes, color='white', alpha=0.90, zorder=10)
    ax.add_patch(overlay)
    overlays.append(overlay)
    
    title = ax.set_title(f"Ист. {comp_idx+1}\nМощн: --", fontsize=11, color='gray')
    titles.append(title)
    
    ax.axis('off')
    ax_topos.append(ax)

highlighted_point = None

# =============================================================================
# ЛОГИКА ИНТЕРАКТИВНОСТИ
# =============================================================================
def update_dashboard(event):
    global highlighted_point
    
    if event.inaxes != ax_umap:
        return
    if event.name == 'motion_notify_event' and event.button is None:
        return
        
    x, y = event.xdata, event.ydata
    if x is None or y is None:
        return

    z_click = np.array([[x, y]], dtype=np.float32)
    hidden_feats_click = hidden_features_model.predict(z_click, verbose=0)
    z_values_click = spatial_decoder_layer.z_dense(hidden_feats_click).numpy()
    
    p_vals = np.exp(z_values_click)[0] 
    p_top = p_vals[top_indices]
    
    dists = np.hypot(umap_coords[:, 0] - x, umap_coords[:, 1] - y)
    min_dist_idx = np.argmin(dists)
    
    label = labels[min_dist_idx]
    txt_info.set_text(f"Зона: {class_to_id[label]} ({label}) | Координаты: ({x:.1f}, {y:.1f})")
    txt_info.set_color(color_map[label])

    p_local_max = np.max(p_top)
    p_local_min = np.min(p_top)
    denominator = p_local_max - p_local_min
    if denominator < 1e-6:
        denominator = 1e-6

    for i, comp_idx in enumerate(top_indices):
        power = p_top[i]
        norm_p = np.clip((power - p_local_min) / denominator, 0, 1)
        
        new_alpha = 0.95 * (1 - norm_p)
        overlays[i].set_alpha(new_alpha)
        
        titles[i].set_text(f"Ист. {comp_idx+1}\nМощн: {power:.2f}")
        
        if norm_p > 0.4:
            titles[i].set_color('black')
            titles[i].set_fontweight('bold')
        else:
            titles[i].set_color('gray')
            titles[i].set_fontweight('normal')

    if highlighted_point:
        highlighted_point.remove()
    highlighted_point = ax_umap.scatter(x, y, marker='+', color='black', s=150, lw=2.0, zorder=5)
    
    fig.canvas.draw_idle()

fig.canvas.mpl_connect('button_press_event', update_dashboard)
fig.canvas.mpl_connect('motion_notify_event', update_dashboard)

# Увеличили нижний отступ (bottom=0.25), чтобы влезла легенда из 4 колонок
plt.subplots_adjust(bottom=0.25, top=0.90, left=0.05, right=0.98, hspace=0.4, wspace=0.1)
print("Готово! Дашборд запущен. Кликните или ведите мышь по UMAP.")
plt.show()

# %%
# =============================================================================
# 8. ИНТЕРАКТИВНАЯ КАРТА ГРАДИЕНТОВ В ПРОСТРАНСТВЕ UMAP
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patheffects as pe

print("Расчет векторных полей градиентов для латентного пространства...")

# 1. Отбираем топ-N источников (как в прошлом скрипте)
power_variances = np.var(powers, axis=0)
top_n = min(15, N_patterns)
top_indices = np.argsort(power_variances)[::-1][:top_n]

# 2. Создаем регулярную сетку поверх пространства UMAP
grid_resolution = 25 # Количество стрелок по осям X и Y
margin = 1.0
x_min, x_max = umap_coords[:, 0].min() - margin, umap_coords[:, 0].max() + margin
y_min, y_max = umap_coords[:, 1].min() - margin, umap_coords[:, 1].max() + margin

X_grid, Y_grid = np.meshgrid(
    np.linspace(x_min, x_max, grid_resolution),
    np.linspace(y_min, y_max, grid_resolution)
)
grid_points = np.c_[X_grid.ravel(), Y_grid.ravel()].astype(np.float32)

# 3. ПРЕДСКАЗЫВАЕМ ПАРАМЕТРЫ ДЛЯ ВСЕЙ СЕТКИ
# Прогоняем сетку координат через декодер
hidden_grid = hidden_features_model.predict(grid_points, verbose=0)
z_grid_flat = spatial_decoder_layer.z_dense(hidden_grid).numpy()

# Восстанавливаем форму 3D: (Y_resolution, X_resolution, N_patterns)
Z_3D = z_grid_flat.reshape(grid_resolution, grid_resolution, N_patterns)

# 4. ВЫЧИСЛЯЕМ ГРАДИЕНТЫ ДЛЯ ВСЕХ ИСТОЧНИКОВ
# Мы берем градиент от z (логарифма мощности), так как градиент самой мощности exp(z) 
# будет слишком экстремальным (стрелки будут либо огромными, либо невидимыми).
# Градиент от z показывает ровное и понятное направление роста.
dZ_dY, dZ_dX = np.gradient(Z_3D, axis=(0, 1))

# =============================================================================
# ПОДГОТОВКА ИНТЕРФЕЙСА
# =============================================================================
active_sources = set() # Здесь храним индексы выбранных источников
source_colors = plt.cm.tab20.colors # Палитра для стрелок разных источников

# Настройка сетки графика
N_COLS_TOPO = 3
N_ROWS_TOPO = int(np.ceil(top_n / N_COLS_TOPO))

fig = plt.figure(figsize=(18, max(8, 2 * N_ROWS_TOPO)))
gs = GridSpec(N_ROWS_TOPO, N_COLS_TOPO + 2, width_ratios=[3.0, 0.2] + [1]*N_COLS_TOPO)

# Левая панель (UMAP с градиентами)
ax_umap = fig.add_subplot(gs[:, 0])

# Подготовка базовых цветов для фона
unique_classes = list(np.unique(labels))
class_to_id = {lab: i + 1 for i, lab in enumerate(unique_classes)}
cmap_bg = plt.cm.Pastel1
color_map_bg = {lab: cmap_bg(i % 9) for i, lab in enumerate(unique_classes)}
point_colors_bg = [color_map_bg[lab] for lab in labels]

# Правая панель (Кнопки-топомапы)
mne_info = raw.info 
ax_buttons = {}  # Связь: ось -> индекс источника
overlays = {}    # Связь: индекс источника -> белый прямоугольник (затемнение)
borders = {}     # Связь: индекс источника -> цветная рамка

for i, comp_idx in enumerate(top_indices):
    row = i // N_COLS_TOPO
    col = 2 + i % N_COLS_TOPO
    ax = fig.add_subplot(gs[row, col])
    
    mne.viz.plot_topomap(found_patterns[comp_idx], mne_info, axes=ax, show=False, contours=0)
    
    # Затемняющий слой (активен, когда источник НЕ выбран)
    overlay = plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes, color='white', alpha=0.75, zorder=10)
    ax.add_patch(overlay)
    overlays[comp_idx] = overlay
    
    # Цветная рамка (скрыта по умолчанию)
    border_color = source_colors[i % len(source_colors)]
    for spine in ax.spines.values():
        spine.set_edgecolor(border_color)
        spine.set_linewidth(4)
        spine.set_visible(False)
    borders[comp_idx] = ax.spines
    
    ax.set_title(f"Ист. {comp_idx+1}", fontsize=11, fontweight='bold')
    # Делаем оси кликабельными
    ax.set_xticks([])
    ax.set_yticks([])
    ax_buttons[ax] = (comp_idx, border_color)

# =============================================================================
# ЛОГИКА ОТРИСОВКИ UMAP
# =============================================================================
def draw_umap_gradients():
    ax_umap.clear()
    
    # 1. Рисуем тусклый фон из реальных эпох
    ax_umap.scatter(umap_coords[:, 0], umap_coords[:, 1], c=point_colors_bg, 
                    s=15, alpha=0.15, edgecolors='none', zorder=1)
    
    # 2. Рисуем центроиды
    for lab in unique_classes:
        mask = (labels == lab)
        if not np.any(mask): continue
        cx, cy = np.mean(umap_coords[mask], axis=0)
        ax_umap.scatter(cx, cy, marker='*', s=300, facecolor=color_map_bg[lab], 
                        edgecolor='gray', linewidths=0.5, zorder=2, alpha=0.5)
        ax_umap.text(cx, cy, str(class_to_id[lab]), fontsize=10, color='gray', 
                     ha='center', va='center', zorder=3)

    # 3. РИСУЕМ ГРАДИЕНТЫ ДЛЯ ВЫБРАННЫХ ИСТОЧНИКОВ
    for comp_idx, color in active_sources:
        # Извлекаем сетку dx и dy для конкретного источника
        U = dZ_dX[:, :, comp_idx]
        V = dZ_dY[:, :, comp_idx]
        
        # Отрисовываем векторное поле
        ax_umap.quiver(X_grid, Y_grid, U, V, color=color, 
                       alpha=0.9, scale_units='xy', angles='xy', 
                       headwidth=4, headlength=5, width=0.003, zorder=5)

    ax_umap.set_title("Векторные поля градиентов мощности (∇z)", fontsize=14, fontweight='bold')
    ax_umap.set_xlabel("UMAP 1")
    ax_umap.set_ylabel("UMAP 2")
    ax_umap.grid(True, linestyle='--', alpha=0.3)
    ax_umap.set_xlim(x_min, x_max)
    ax_umap.set_ylim(y_min, y_max)
    
    fig.canvas.draw_idle()

# Первичная отрисовка
draw_umap_gradients()

# =============================================================================
# ОБРАБОТЧИК КЛИКОВ ПО КНОПКАМ-ТОПОМАПАМ
# =============================================================================
def on_click(event):
    if event.inaxes not in ax_buttons:
        return
        
    comp_idx, color = ax_buttons[event.inaxes]
    source_tuple = (comp_idx, color)
    
    # Тоггл: добавляем или удаляем источник
    if source_tuple in active_sources:
        active_sources.remove(source_tuple)
        # Возвращаем затенение, скрываем рамку
        overlays[comp_idx].set_alpha(0.75)
        for spine in borders[comp_idx].values(): spine.set_visible(False)
    else:
        active_sources.add(source_tuple)
        # Убираем затенение, показываем цветную рамку
        overlays[comp_idx].set_alpha(0.0)
        for spine in borders[comp_idx].values(): spine.set_visible(True)
            
    # Перерисовываем основной график
    draw_umap_gradients()

fig.canvas.mpl_connect('button_press_event', on_click)

plt.subplots_adjust(left=0.05, right=0.98, bottom=0.1, top=0.92, wspace=0.1, hspace=0.3)
print("Готово! Кликайте по топомапам справа, чтобы включать/выключать их векторные поля на UMAP.")
plt.show()

# %%
# =============================================================================
# 8. ИНТЕРАКТИВНАЯ КАРТА ГРАДИЕНТОВ И СМЕШАННОЙ ТОПОГРАФИИ
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe
import matplotlib.colors as mcolors

print("Расчет векторных полей и подготовка топографии для латентного пространства...")

# 1. Отбираем топ-N источников
power_variances = np.var(powers, axis=0)
top_n = min(15, N_patterns)
top_indices = np.argsort(power_variances)[::-1][:top_n]

# 2. Создаем регулярную сетку
grid_resolution = 25
margin = 1.0
x_min, x_max = umap_coords[:, 0].min() - margin, umap_coords[:, 0].max() + margin
y_min, y_max = umap_coords[:, 1].min() - margin, umap_coords[:, 1].max() + margin

X_grid, Y_grid = np.meshgrid(
    np.linspace(x_min, x_max, grid_resolution),
    np.linspace(y_min, y_max, grid_resolution)
)
grid_points = np.c_[X_grid.ravel(), Y_grid.ravel()].astype(np.float32)

# 3. Предсказываем параметры
hidden_grid = hidden_features_model.predict(grid_points, verbose=0)
z_grid_flat = spatial_decoder_layer.z_dense(hidden_grid).numpy()
Z_3D = z_grid_flat.reshape(grid_resolution, grid_resolution, N_patterns)

# 4. Градиенты
dZ_dY, dZ_dX = np.gradient(Z_3D, axis=(0, 1))

# =============================================================================
# ПОДГОТОВКА ИНТЕРФЕЙСА И ЦВЕТОВ
# =============================================================================
active_sources = set()

# Максимально контрастные цвета для источников (палитра Келли)
distinct_colors = [
    '#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', 
    '#911eb4', '#42d4f4', '#f032e6', '#bfef45', '#fabed4', 
    '#469990', '#dcbeff', '#9A6324', '#fffac8', '#800000'
]
source_colors = distinct_colors[:top_n]

N_COLS_TOPO = 3
N_ROWS_TOPO = int(np.ceil(top_n / N_COLS_TOPO))

fig = plt.figure(figsize=(18, max(8, 2 * N_ROWS_TOPO)))
gs = GridSpec(N_ROWS_TOPO, N_COLS_TOPO + 2, width_ratios=[3.0, 0.2] + [1]*N_COLS_TOPO)

ax_umap = fig.add_subplot(gs[:, 0])

# --- СОХРАНЯЕМ ПОРЯДОК ИЗ ВАШЕГО СПИСКА new_descriptions ---
unique_classes = []
for lab in new_descriptions:
    if lab in labels and lab not in unique_classes:
        unique_classes.append(lab)
# На случай, если в labels есть что-то непредвиденное
for lab in labels:
    if lab not in unique_classes:
        unique_classes.append(lab)

class_to_id = {lab: i + 1 for i, lab in enumerate(unique_classes)}
cmap_bg = plt.cm.tab20
color_map_bg = {lab: cmap_bg(i / max(1, len(unique_classes) - 1)) for i, lab in enumerate(unique_classes)}
point_colors_bg = [color_map_bg[lab] for lab in labels]

mne_info = raw.info 
ax_buttons = {}  
overlays = {}    
borders = {}     

for i, comp_idx in enumerate(top_indices):
    row = i // N_COLS_TOPO
    col = 2 + i % N_COLS_TOPO
    ax = fig.add_subplot(gs[row, col])
    
    mne.viz.plot_topomap(found_patterns[comp_idx], mne_info, axes=ax, show=False, contours=0)
    
    overlay = plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes, color='white', alpha=0.75, zorder=10)
    ax.add_patch(overlay)
    overlays[comp_idx] = overlay
    
    border_color = source_colors[i % len(source_colors)]
    for spine in ax.spines.values():
        spine.set_edgecolor(border_color)
        spine.set_linewidth(4)
        spine.set_visible(False)
    borders[comp_idx] = ax.spines
    
    ax.set_title(f"Ист. {comp_idx+1}", fontsize=11, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])
    ax_buttons[ax] = (comp_idx, border_color)

# =============================================================================
# ЛОГИКА ОТРИСОВКИ UMAP СО СМЕШИВАНИЕМ ТОПОГРАФИЙ И Z-ORDER
# =============================================================================
def draw_umap_gradients():
    ax_umap.clear()
    
    # 1. РИСУЕМ ФОНОВУЮ ТОПОГРАФИЮ (САМЫЙ НИЖНИЙ СЛОЙ)
    for comp_idx, color in active_sources:
        Z_comp = Z_3D[:, :, comp_idx]
        
        # Создаем кастомную палитру: от прозрачного до цвета источника
        rgba = mcolors.to_rgba(color)
        color_transparent = (rgba[0], rgba[1], rgba[2], 0.0)
        color_solid = (rgba[0], rgba[1], rgba[2], 0.55)
        custom_cmap = mcolors.LinearSegmentedColormap.from_list(f'cmap_{comp_idx}', [color_transparent, color_solid])
        
        # zorder=1 (дно)
        ax_umap.contourf(X_grid, Y_grid, Z_comp, levels=15, cmap=custom_cmap, zorder=1)

    # 2. РИСУЕМ ТОЧКИ ЭПОХ ПОВЕРХ ТОПОГРАФИИ, НО ПОД СТРЕЛКАМИ
    # zorder=3 
    ax_umap.scatter(umap_coords[:, 0], umap_coords[:, 1], c=point_colors_bg, 
                    s=25, alpha=0.5, edgecolors='white', linewidths=0.3, zorder=3)

    # 3. РИСУЕМ СТРЕЛКИ ДЛЯ АКТИВНЫХ ИСТОЧНИКОВ ПОВЕРХ ТОЧЕК
    for comp_idx, color in active_sources:
        U = dZ_dX[:, :, comp_idx]
        V = dZ_dY[:, :, comp_idx]
        
        # color - цвет заливки, edgecolors - контур стрелки
        # zorder=4 (над точками)
        ax_umap.quiver(X_grid, Y_grid, U, V, 
                       color=color, edgecolors='black', linewidths=0.5,
                       alpha=0.95, scale_units='xy', angles='xy', 
                       headwidth=4, headlength=5, width=0.003, zorder=4)

    # 4. РИСУЕМ ЦЕНТРОИДЫ И ИХ ПОДПИСИ (САМЫЙ ВЕРХНИЙ СЛОЙ)
    for lab in unique_classes:
        mask = (labels == lab)
        if not np.any(mask): continue
        cx, cy = np.mean(umap_coords[mask], axis=0)
        
        # zorder=5 (над стрелками)
        ax_umap.scatter(cx, cy, marker='*', s=350, facecolor=color_map_bg[lab], 
                        edgecolor='black', linewidths=0.8, zorder=5, alpha=0.95)
        
        # zorder=6 (текст всегда на самом верху)
        ax_umap.text(cx, cy, str(class_to_id[lab]), fontsize=11, color='white', 
                     fontweight='bold', ha='center', va='center', zorder=6,
                     path_effects=[pe.withStroke(linewidth=2.5, foreground="black")])

    # 5. ДОБАВЛЕНИЕ ЛЕГЕНДЫ ДЛЯ КЛАССОВ (в правильном порядке)
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label=f"{class_to_id[lab]}: {lab}",
               markerfacecolor=color_map_bg[lab], markersize=8, 
               markeredgecolor='black', markeredgewidth=0.5) 
        for lab in unique_classes
    ]
    ax_umap.legend(handles=legend_elements, title="Условия / Классы", loc='upper center', 
                   bbox_to_anchor=(0.5, -0.08), ncol=6, fontsize=9, title_fontsize=10)

    # Оформление
    ax_umap.set_title("Смешанная топография мощностей и градиенты (∇z)", fontsize=14, fontweight='bold')
    ax_umap.set_xlabel("UMAP 1")
    ax_umap.set_ylabel("UMAP 2")
    ax_umap.grid(True, linestyle='--', alpha=0.3, zorder=0)
    ax_umap.set_xlim(x_min, x_max)
    ax_umap.set_ylim(y_min, y_max)
    
    fig.canvas.draw_idle()

# Первичная отрисовка
draw_umap_gradients()

# =============================================================================
# ОБРАБОТЧИК КЛИКОВ ПО КНОПКАМ-ТОПОМАПАМ
# =============================================================================
def on_click(event):
    if event.inaxes not in ax_buttons:
        return
        
    comp_idx, color = ax_buttons[event.inaxes]
    source_tuple = (comp_idx, color)
    
    if source_tuple in active_sources:
        active_sources.remove(source_tuple)
        overlays[comp_idx].set_alpha(0.75)
        for spine in borders[comp_idx].values(): spine.set_visible(False)
    else:
        active_sources.add(source_tuple)
        overlays[comp_idx].set_alpha(0.0)
        for spine in borders[comp_idx].values(): spine.set_visible(True)
            
    draw_umap_gradients()

fig.canvas.mpl_connect('button_press_event', on_click)

# Увеличили bottom до 0.18, чтобы вместить легенду
plt.subplots_adjust(left=0.05, right=0.98, bottom=0.18, top=0.92, wspace=0.1, hspace=0.3)
print("Готово! Выбирайте несколько источников: их топографии мощности будут смешиваться в пространстве UMAP.")
plt.show()