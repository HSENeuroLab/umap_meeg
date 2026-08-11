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
covmats = Covariances().fit_transform(X_windows_band)

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

# 1. Переводим отбеленные ковариации в касательное пространство
ts_projector = TangentSpace(metric='riemann')
X_ts = ts_projector.fit_transform(covmats_white) 

# %%
N_ts_features = X_ts.shape[1]
n_ch_white = covmats_white.shape[1]

# 2. ТРЮК: Конкатенируем X_ts и плоские матрицы ковариации в один датасет
X_cov_flat = covmats_white.reshape(covmats_white.shape[0], -1)
X_combined = np.hstack([X_ts, X_cov_flat])

# =============================================================================
# 2. ЭНКОДЕР (Берет только касательные векторы)
# =============================================================================
class TangentSliceEncoder(tf.keras.layers.Layer):
    def __init__(self, n_filters, N_ts_features, **kwargs):
        super().__init__(**kwargs)
        self.n_filters = n_filters
        self.N_ts_features = N_ts_features

    def build(self, input_shape):
        self.dense1 = tf.keras.layers.Dense(128, activation="relu")
        self.dense2 = tf.keras.layers.Dense(self.n_filters)
        self.log_scale = self.add_weight(
            shape=(self.n_filters,),
            initializer="zeros", 
            trainable=True,
            name="log_scale"
        )

    def call(self, X_combined):
        # ОТРЕЗАЕМ только касательные векторы (игнорируем ковариации)
        T = X_combined[:, :self.N_ts_features]
        
        y_raw = self.dense2(self.dense1(T))
        scale = tf.exp(self.log_scale)
        return y_raw * scale

# =============================================================================
# 3. ДЕКОДЕР (Реконструирует матрицы ковариации C = A * P * A^T)
# =============================================================================
class SpatialPatternDecoder(tf.keras.layers.Layer):
    def __init__(self, M_channels, **kwargs):
        super().__init__(**kwargs)
        self.M_channels = M_channels

    def build(self, input_shape):
        n_filters = input_shape[-1]
        self.A = self.add_weight(
            shape=(self.M_channels, n_filters),
            initializer=tf.keras.initializers.RandomNormal(stddev=0.1),
            trainable=True,
            name="A_patterns" # Это и есть наши искомые паттерны!
        )
        self.inv_log_scale = self.add_weight(
            shape=(n_filters,),
            initializer="zeros",
            trainable=True,
            name="inv_log_scale"
        )

    def call(self, z):
        inv_scale = tf.exp(self.inv_log_scale)
        P = tf.exp(z * inv_scale)
        # C_recon: (batch, M, M)
        C_recon = tf.einsum("mf,bf,lf->bml", self.A, P, self.A)
        return C_recon

# =============================================================================
# 4. КАСТОМНАЯ ФУНКЦИЯ ПОТЕРЬ ДЛЯ РЕКОНСТРУКЦИИ
# =============================================================================
def custom_cov_mse(y_true_combined, y_pred_cov_flat):
    """
    y_true_combined содержит и касательные векторы, и ковариации.
    Мы отрезаем ковариации, приводим типы к float32 и сравниваем с выходом декодера.
    """
    y_true_cov_flat = y_true_combined[:, N_ts_features:]
    y_true_cov_flat = tf.cast(y_true_cov_flat, tf.float32) # ИСПРАВЛЕНИЕ ОШИБКИ ТИПОВ
    return tf.reduce_mean(tf.square(y_true_cov_flat - y_pred_cov_flat))

# =============================================================================
# 5. СБОРКА И ОБУЧЕНИЕ UMAP-АВТОЭНКОДЕРА
# =============================================================================
N_dim = 21
n_ch_white = covmats_white.shape[1]

# ИСПРАВЛЕНИЕ ОШИБКИ ТИПОВ: сразу делаем float32
X_flat = covmats_white.reshape(covmats_white.shape[0], -1)
X_combined = np.hstack([X_ts, X_flat]).astype(np.float32) 

encoder = tf.keras.Sequential([
    tf.keras.layers.InputLayer(shape=(N_ts_features + n_ch_white**2,)),
    TangentSliceEncoder(n_filters=N_dim, N_ts_features=N_ts_features)
])

decoder = tf.keras.Sequential([
    tf.keras.layers.InputLayer(shape=(N_dim,)),
    SpatialPatternDecoder(M_channels=n_ch_white),
    tf.keras.layers.Flatten() 
])

print("Сборка и обучение Риманова UMAP-Автоэнкодера...")
embedder = ParametricUMAP(
    encoder=encoder,
    decoder=decoder,
    n_components=N_dim,
    dims=(N_ts_features + n_ch_white**2,),
    metric="precomputed", 
    parametric_reconstruction=True,
    autoencoder_loss=True,
    parametric_reconstruction_loss_fcn=custom_cov_mse,
    verbose=True
)

embedder.fit(X_combined, precomputed_distances=dist_matrix_init)
umap_coords = embedder.embedding_

# %%
class TangentEncoder(tf.keras.layers.Layer):
    def __init__(self, n_filters, **kwargs):
        super().__init__(**kwargs)
        self.n_filters = n_filters

    def build(self, input_shape):
        self.dense1 = tf.keras.layers.Dense(128, activation="relu")
        self.dense2 = tf.keras.layers.Dense(self.n_filters)
        self.log_scale = self.add_weight(
            shape=(self.n_filters,),
            initializer="zeros",
            trainable=True,
            name="log_scale"
        )

    def call(self, x):
        y_raw = self.dense2(self.dense1(x))
        scale = tf.exp(self.log_scale)
        return y_raw * scale
    
decoder = tf.keras.Sequential([
    tf.keras.layers.InputLayer(shape=(N_dim,)),
    SpatialPatternDecoder(M_channels=n_ch_white)   # выход (batch, M, M)
])

# Достаём из ts_projector (он уже обучен)
C_avg = ts_projector.reference_                  # среднее геометрическое
C_invsqrt = invsqrtm(C_avg)                      # C_avg^{-1/2}
C_sqrt = np.linalg.inv(C_invsqrt)                # C_avg^{1/2}

def tangent_space_mse(y_true, y_pred_cov):
    y_true = tf.cast(y_true, tf.float32)
    y_pred_cov = tf.cast(y_pred_cov, tf.float32)

    Cinv = tf.constant(C_invsqrt, dtype=tf.float32)
    Csqrt = tf.constant(C_sqrt, dtype=tf.float32)

    # 1. Перенос в точку отсчёта
    C_tilde = tf.einsum('ij,bjk,kl->bil', Cinv, y_pred_cov, Cinv)

    # 2. Регуляризация
    eps = 1e-6
    C_tilde += eps * tf.eye(tf.shape(C_tilde)[1], dtype=tf.float32)

    # 3. Матричный логарифм через собственные значения (градиент поддерживается)
    eigvals, eigvecs = tf.linalg.eigh(C_tilde)         # (batch, M) и (batch, M, M)
    eigvals = tf.maximum(eigvals, eps)                 # гарантируем положительность
    log_eigvals = tf.math.log(eigvals)
    logC_tilde = tf.einsum('bmi,bi,bni->bmn', eigvecs, log_eigvals, eigvecs)
    
    # 4. Возврат из whitened space
    logC = tf.einsum('ij,bjk,kl->bil', Csqrt, logC_tilde, Csqrt)

    # 5. Векторизация (M = n_ch_white — константа, известная заранее)
    M = n_ch_white
    triu_idx = np.triu_indices(M, k=0)
    rows, cols = triu_idx
    mult = np.where(rows == cols, 1.0, np.sqrt(2.0)).astype(np.float32)
    mult = tf.constant(mult, dtype=tf.float32)

    logC_flat = tf.stack([logC[:, r, c] for r, c in zip(rows, cols)], axis=1)
    logC_vector = logC_flat * mult

    return tf.reduce_mean(tf.square(y_true - logC_vector))

N_dim = 10

X_input = X_ts.astype(np.float32)   # (n_windows, N_ts_features)

encoder = tf.keras.Sequential([
    tf.keras.layers.InputLayer(shape=(N_ts_features,)),
    TangentEncoder(n_filters=N_dim)
])

decoder = tf.keras.Sequential([
    tf.keras.layers.InputLayer(shape=(N_dim,)),
    SpatialPatternDecoder(M_channels=n_ch_white)
])

embedder = ParametricUMAP(
    encoder=encoder,
    decoder=decoder,
    n_components=N_dim,
    dims=(N_ts_features,),                # только касательные векторы!
    metric="euclidean",                   # можно любое, т.к. граф задан жёстко
    parametric_reconstruction=True,
    autoencoder_loss=True,
    parametric_reconstruction_loss_fcn=tangent_space_mse,
    verbose=True
)

embedder.fit(X_input)    
umap_coords = embedder.embedding_

# %%
# =============================================================================
# 6. ИЗВЛЕЧЕНИЕ ПАРАМЕТРОВ И ОТРИСОВКА
# =============================================================================
from matplotlib.gridspec import GridSpec
import matplotlib as mpl
import numpy as np

# 1. Извлекаем слой паттернов из декодера
spatial_decoder_layer = [layer for layer in decoder.layers if isinstance(layer, SpatialPatternDecoder)][0]

# 2. Получаем ИСТИННЫЕ ПАТТЕРНЫ (A) и МАСШТАБЫ
A_white, inv_log_scale = spatial_decoder_layer.get_weights()
scale_dec = np.exp(inv_log_scale) # Форма: (N_dim,)

# 3. Восстанавливаем ИСТИННУЮ ДИНАМИКУ (Логарифм мощности источника)
# z_true = z_umap * scale
true_log_power = umap_coords * scale_dec

# 4. Проецируем паттерны из отбеленного пространства обратно на сенсоры
A_global = A_ssd @ C_avg_sqrt @ A_white
found_patterns = [A_global[:, d] for d in range(N_dim)]

# 5. Аппроксимируем фильтры через Хауфе для визуализации (W = C^{-1} A)
C_global_mean = np.mean(covmats, axis=0)
C_global_inv = np.linalg.pinv(C_global_mean)
found_filters = [C_global_inv @ A_global[:, d] for d in range(N_dim)]

# %%
def format_umap_axes(ax):
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(axis='both', which='both', length=0)
    ax.grid(True, linestyle='--', alpha=0.5, zorder=0)

comp_idx = 20
W_sensor = found_filters[comp_idx]
A_pattern = found_patterns[comp_idx]

# Используем истинную лог-мощность для графиков!
p_vals = true_log_power[:, comp_idx]

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

# Отрисовка скаттера (используем сырые координаты UMAP для 2D-проекции)
ax_umap = fig.add_subplot(gs[0, 1])
sc1 = ax_umap.scatter(umap_coords[:, 0], umap_coords[:, 1], c=p_vals, cmap='plasma', s=15, zorder=2)
plt.colorbar(sc1, ax=ax_umap, label='True Source Log-power')
ax_umap.set_title(f'UMAP проекция (оси 0 и 1)\nЦвет: истинная мощность компоненты {comp_idx+1}')
format_umap_axes(ax_umap)

ax_env = fig.add_subplot(gs[1, 1])
window_idx = np.arange(len(p_vals))
ax_env.plot(window_idx, p_vals, color='black', lw=0.8, zorder=2)

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
ax_env.set_ylabel('True Source log-power')
ax_env.set_title('Динамика истинной мощности источника')
ax_env.grid(True, axis='y', linestyle=':', alpha=0.6, zorder=0)

plt.suptitle(f'Анализ компоненты {comp_idx+1} из {N_dim}', fontsize=16)
plt.tight_layout()
plt.show()

# %%

