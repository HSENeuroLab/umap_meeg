# -*- coding: utf-8 -*-
"""
Created on Wed Oct 22 17:07:11 2025

@author: anton
"""
import os
import sys
import mne
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import umap

from scipy.signal import butter, filtfilt
from scipy.linalg import eigh
from scipy.linalg import inv, null_space
from umap.parametric_umap import ParametricUMAP

lib_directory = os.path.abspath("C:/Users/ansbel/Documents/GitHub/pyRiemann") 
if lib_directory not in sys.path:
    sys.path.insert(0, lib_directory)
from pyriemann.estimation import Covariances
from pyriemann.utils.base import invsqrtm
from pyriemann.geometry.distance import pairwise_distance

# %%
fpath = 'C:/Users/ansbel/Downloads/Telegram Desktop/CR_02.fif'
raw = mne.io.read_raw_fif(fpath, preload=True)

# %%
raw.plot()

# %%
cond_descriptions = ['rest/EC/1', 
                     'rest/EO/1', 
                     'rest/EC/2', 
                     'rest/EO/2', 
                     'ABA_1',
                     'ACA_1',
                     'ABA_2',
                     'ACA_2',
                     'rest/EO/3', 
                     'rest/EC/3']

# 4. Фильтрация индексов
bad_idx = [i for i, desc in enumerate(raw.annotations.description) if desc == "BAD_"]
good_idx = [i for i, desc in enumerate(raw.annotations.description) if desc in cond_descriptions]

# Теперь индексация сработает без ошибок
bad_ann = raw.annotations[bad_idx]
good_ann = raw.annotations[good_idx]

# Объединяем и сохраняем
new_ann = bad_ann + good_ann
raw.set_annotations(new_ann)

# %%
raw.plot()

# %%
raw.interpolate_bads()

# %%
from mne.preprocessing import ICA
ica = ICA(n_components=0.999, random_state=97, max_iter="auto")
ica.fit(raw)

# %%
ica.plot_components()  
ica.plot_sources(raw)  

# %%
ica.apply(raw)

# %%
raw.plot()

# %%
# =============================================================================
# 2. ПОИСК SSD И ОТБЕЛИВАНИЕ
# =============================================================================
data = raw.get_data(picks='eeg')
sfreq = raw.info['sfreq']

signal_pieces = []
noise_pieces = []

fmin = 15
fmax = 30 
b_signal, a_signal = butter(3, np.array([fmin, fmax]) / (int(sfreq) / 2), btype='band')
b_broad, a_broad = butter(3, np.array([fmin-2, fmax+2]) / (int(sfreq) / 2), btype='band')
b_stop, a_stop = butter(3, np.array([fmin-0.5, fmax+0.5]) / (int(sfreq) / 2), btype='stop')

crop_duration = 0.5
crop_samples = int(crop_duration * sfreq)

for annot in raw.annotations:
    desc = annot['description']
    if desc == 'BAD_' or annot['duration'] < 2.0:
        continue

    tmin = annot['onset'] - raw.first_time
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
n_components = W_ssd.shape[1]
n_components

# %%
# =============================================================================
# 3. НАРЕЗКА НА ЭПОХИ
# =============================================================================
Wsize = 2
Ssize = 0.5 
overlap = Wsize - Ssize

X_windows_band = []
X_windows_unfilt = []
window_labels = []

raw_band = raw.copy().filter(l_freq=fmin,h_freq=fmax)
raw_unfilt = raw.copy().pick_types(eeg=True)

for annot in raw.annotations:
    desc = annot['description']
    if annot['duration'] < Wsize or desc == 'BAD_':
        continue

    tmin = annot['onset'] - raw_band.first_time
    tmax = min(tmin + annot['duration'], raw.times[-1])

    raw_crop = raw_band.copy().crop(tmin=tmin, tmax=tmax)
    raw_crop_unfilt = raw_unfilt.copy().crop(tmin=tmin, tmax=tmax)

    epochs_band = mne.make_fixed_length_epochs(raw_crop, duration=Wsize, overlap=overlap, preload=True, verbose=False)
    epochs_unfilt = mne.make_fixed_length_epochs(raw_crop_unfilt, duration=Wsize, overlap=overlap, preload=True, verbose=False)

    if epochs_band and len(epochs_band) == len(epochs_unfilt):
        X_windows_band.append(epochs_band.get_data(copy=False,picks='eeg'))
        X_windows_unfilt.append(epochs_unfilt.get_data(copy=False,picks='eeg'))
        window_labels.extend([desc] * len(epochs_band))

X_windows_band = np.concatenate(X_windows_band, axis=0)
X_windows_unfilt = np.concatenate(X_windows_unfilt, axis=0)
window_labels = np.array(window_labels)

n_windows, n_ch, n_times = X_windows_band.shape
X_windows_ssd_proj = np.zeros((n_windows, n_components, n_times))
for i in range(n_windows):
    X_windows_ssd_proj[i] = W_ssd.T @ X_windows_band[i]

covmats_band = Covariances().fit_transform(X_windows_band / np.std(X_windows_band))
covmats_ssd = Covariances().fit_transform(X_windows_ssd_proj / np.std(X_windows_ssd_proj))

# %%
dists_init = pairwise_distance(covmats_ssd, metric='riemann')

# %%
reducer = umap.UMAP(n_components=2, n_neighbors=20, metric='precomputed')
coords = reducer.fit_transform(dists_init)

# %%
plt.figure(figsize=(11, 8))

# Уникальные метки (сохраняем исходный порядок из cond_descriptions, если они есть)
unique_labels = [c for c in cond_descriptions if c in np.unique(window_labels)]
for ul in np.unique(window_labels):
    if ul not in unique_labels:
        unique_labels.append(ul)

# Цветовая палитра под число условий (tab20 надежнее, если условий больше 10)
cmap = plt.colormaps.get_cmap('tab20' if len(unique_labels) > 10 else 'tab10')

for idx, label in enumerate(unique_labels, start=1):
    mask = (window_labels == label)
    if not np.any(mask):
        continue
    
    pts = coords[mask]
    color = cmap((idx - 1) % cmap.N)
    
    # Точки вложения с номером в легенде
    plt.scatter(
        pts[:, 0], pts[:, 1],
        color=color,
        s=25,
        alpha=0.6,
        label=f'[{idx}] {label}'
    )
    
    # Вычисление центроида
    centroid = np.mean(pts, axis=0)
    
    # Крупный маркер подложки под номер
    plt.scatter(
        centroid[0], centroid[1],
        color=color,
        s=320,
        marker='o',
        edgecolor='black',
        linewidth=1.4,
        zorder=6
    )
    
    # Номер условия прямо в центре маркера центроида
    plt.text(
        centroid[0], centroid[1],
        str(idx),
        color='white',
        fontsize=9,
        fontweight='bold',
        ha='center',
        va='center',
        zorder=7
    )

plt.title('UMAP (Riemannian Distance on SSD Covariances)', fontsize=14, pad=12)
plt.xlabel('UMAP 1', fontsize=12)
plt.ylabel('UMAP 2', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.3)

# Легенда с номерами условий
plt.legend(
    title='Conditions',
    bbox_to_anchor=(1.04, 1),
    loc='upper left',
    frameon=True,
    markerscale=1.2,
    fontsize=10
)

plt.tight_layout()
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
        })
        return config

@tf.keras.utils.register_keras_serializable()
class LogDiagScaleLayer(tf.keras.layers.Layer):
    """
    z_i = log( scale_i * max( (W C W^T)_{ii}, eps ) )
    Штраф за внедиагональные элементы рассчитывается на основе матрицы корреляции.
    Масштаб строго больше нуля благодаря функции softplus, выступает как усилитель дисперсии.
    """
    def __init__(self, n_filters, epsilon=1e-7, off_diag_penalty=1.0, **kwargs):
        super().__init__(**kwargs)
        self.n_filters = int(n_filters)
        self.epsilon = epsilon
        self.off_diag_penalty = float(off_diag_penalty)

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs):
        # 1. Извлекаем мощности (диагонали ковариационной матрицы)
        diags = tf.linalg.diag_part(inputs)
        
        # --- НАЛОЖЕНИЕ ШТРАФА НА ВНЕДИАГОНАЛЬНЫЕ ЭЛЕМЕНТЫ ---
        if self.off_diag_penalty > 0.0:
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
    
Npatt = 44 

idx_i, idx_j = np.triu_indices(n_components)
multipliers = np.where(idx_i == idx_j, 1.0, np.sqrt(2.0)).astype(np.float32)
X_cov_flat = (covmats_ssd[:, idx_i, idx_j] * multipliers).astype(np.float32)
input_dim = int(X_cov_flat.shape[1])

inputs_enc = tf.keras.Input(shape=(input_dim,), dtype=tf.float32)
x = UnflattenSymmetricLayer(M=n_components)(inputs_enc)
x = BiMapLayer(d_out=Npatt, name="bimap_1")(x)
z_latent = LogDiagScaleLayer(n_filters=Npatt, epsilon=1e-7)(x)
encoder = tf.keras.Model(inputs=inputs_enc, outputs=z_latent)

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=50, min_delta=1e-4)
embedder = ParametricUMAP(
    encoder=encoder, n_components=Npatt, dims=(input_dim,), 
    metric="precomputed", n_neighbors=50, verbose=True,
    keras_fit_kwargs={"callbacks": [early_stopping], "verbose": 1}
)

embedder.n_training_epochs = 3 
embedder.loss_report_frequency = embedder.n_training_epochs * 100
embedder.fit(X_cov_flat, precomputed_distances=dists_init)

# %%
# =============================================================================
# ВИЗУАЛИЗАЦИЯ (ТОЛЬКО ВАЛИДНЫЕ ОШИБКИ)
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
# =============================================================================
# ОБУЧЕНИЕ ОТОБРАЖЕНИЯ ДЛЯ КЛИКЕРА (20D Мощности <-> 2D Экран)
# =============================================================================
print("Обучение инверсной модели визуализации (20D -> 2D -> 20D)...")
from tensorflow.keras import regularizers

encoder_2d = tf.keras.Sequential([
    tf.keras.layers.InputLayer(shape=(Npatt,)),
    tf.keras.layers.Dense(100, activation="elu"),
    tf.keras.layers.Dense(100, activation="elu"),
    tf.keras.layers.Dense(100, activation="elu"),
    tf.keras.layers.Dense(2, activation="linear", name="2d_coords")
])

decoder_2d = tf.keras.Sequential([
    tf.keras.layers.InputLayer(shape=(2,)),
    tf.keras.layers.Dense(100, activation="elu", kernel_regularizer=regularizers.l2(1e-4)),
    tf.keras.layers.Dense(100, activation="elu", kernel_regularizer=regularizers.l2(1e-4)),
    tf.keras.layers.Dense(100, activation="elu", kernel_regularizer=regularizers.l2(1e-4)),
    tf.keras.layers.Dense(Npatt, activation="linear", name="z_reconstruction")
])

reducer_2d_nn = ParametricUMAP(
    encoder=encoder_2d,
    # decoder=decoder_2d,
    n_components=2,
    parametric_reconstruction=False, 
    parametric_reconstruction_loss_fcn=tf.keras.losses.MeanSquaredError(),
    verbose=True
)

pred = encoder.predict(X_cov_flat)
umap_coords = reducer_2d_nn.fit_transform(pred)
print("Визуальное пространство обучено!")

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
W_filters_sensor = W_ssd @ W_filters_ssd

# =============================================================================
# 2. РАСЧЕТ ИСТИННЫХ МОЩНОСТЕЙ ИСТОЧНИКОВ
# =============================================================================
# Считаем физические мощности напрямую для каждого окна: diag(W C W^T)
powers_true = np.zeros((len(covmats_ssd), Npatt))
for i, C in enumerate(covmats_ssd):
    C_filtered = W_bimap @ C @ W_bimap.T
    powers_true[i] = np.diag(C_filtered)

# =============================================================================
# 3. ОЦЕНКА ПАТТЕРНОВ ЧЕРЕЗ C @ W (В СЕНСОРНОМ ПРОСТРАНСТВЕ)
# =============================================================================
# Усредненная ковариационная матрица в пространстве SSD
C_avg_ssd = np.mean(covmats_ssd, axis=0)

# Паттерны в пространстве SSD
A_ssd = C_avg_ssd @ W_filters_ssd

# Проекция паттернов обратно в исходное пространство сенсоров
A_sensor_raw = np.linalg.pinv(W_ssd.T) @ A_ssd

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
import matplotlib as mpl
from matplotlib.gridspec import GridSpec

def format_umap_axes(ax):
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(axis='both', which='both', length=0)
    ax.grid(True, linestyle='--', alpha=0.5, zorder=0)

# =============================================================================
# ВЫБОР КОМПОНЕНТЫ
# =============================================================================
comp_idx = 43 

W_sensor = found_filters[comp_idx] 
A_pattern = found_patterns[comp_idx]

p_vals = powers[:, comp_idx]
filtered_powers = np.array([W_sensor.T @ C @ W_sensor for C in covmats_band])

p_vals_norm = p_vals / np.std(p_vals)
filtered_powers_norm = filtered_powers / np.std(filtered_powers)

# =============================================================================
# ПОДГОТОВКА ЦВЕТОВ, ЛЕЙБЛОВ И ПАЛИТРЫ УСЛОВИЙ
# =============================================================================
# Сохраняем исходный порядок из cond_descriptions, если переменная определена
if 'cond_descriptions' in locals():
    unique_labels_ordered = [c for c in cond_descriptions if c in np.unique(window_labels)]
    for ul in np.unique(window_labels):
        if ul not in unique_labels_ordered:
            unique_labels_ordered.append(ul)
else:
    unique_labels_ordered = []
    for lab in window_labels:
        if lab not in unique_labels_ordered:
            unique_labels_ordered.append(lab)

# Палитра для дискретных условий (подложка таймлайна и бейджи центроидов)
cmap_conds = mpl.colormaps['tab20' if len(unique_labels_ordered) > 10 else 'tab10']
label_to_color = {lab: cmap_conds(i % cmap_conds.N) for i, lab in enumerate(unique_labels_ordered)}

# =============================================================================
# ОТРИСОВКА
# =============================================================================
fig = plt.figure(figsize=(18, 10))
gs = GridSpec(2, 2, figure=fig, width_ratios=[1, 2.2], height_ratios=[1, 1])

# --- 1. ТОПОГРАФИЯ ФИЛЬТРА (W) ---
ax_filt = fig.add_subplot(gs[0, 0])
mne.viz.plot_topomap(W_sensor, raw.info, axes=ax_filt, show=False)
ax_filt.set_title('Пространственный фильтр')

# --- 2. ТОПОГРАФИЯ ПАТТЕРНА (A) ---
ax_patt = fig.add_subplot(gs[1, 0])
mne.viz.plot_topomap(A_pattern, raw.info, axes=ax_patt, show=False)
ax_patt.set_title('Пространственный паттерн')

# --- 3. UMAP ПРОЕКЦИЯ С ЦЕНТРОИДАМИ И ЛЕГЕНДОЙ ---
ax_umap = fig.add_subplot(gs[0, 1])

vmin_val = np.percentile(p_vals_norm, 5)
vmax_val = np.percentile(p_vals_norm, 95)

# 1. Точки UMAP с исходным цветом активации компоненты
sc1 = ax_umap.scatter(
    umap_coords[:, 0], 
    umap_coords[:, 1], 
    c=p_vals_norm, 
    cmap='plasma', 
    s=20, 
    alpha=0.65,
    zorder=2,
    vmin=vmin_val,
    vmax=vmax_val   
)
cbar = plt.colorbar(sc1, ax=ax_umap, label='Мощность источника (scaled)', extend='both', pad=0.02)

# Глобальный центр облака для адаптивного выноса сносок наружу
global_center = np.mean(umap_coords, axis=0)

legend_handles = []

for idx, label in enumerate(unique_labels_ordered, start=1):
    mask = (window_labels == label)
    if not np.any(mask):
        continue
    
    pts = umap_coords[mask]
    centroid = np.mean(pts, axis=0)
    cond_color = label_to_color[label]
    
    # Ненавязчивый маркер фактического центра (крестик без заливки)
    ax_umap.scatter(
        centroid[0], centroid[1],
        marker='+',
        color='black',
        s=70,
        linewidths=1.5,
        zorder=4
    )
    
    # Направление выноса сноски (от глобального центра наружу)
    vec = centroid - global_center
    norm = np.linalg.norm(vec)
    offset = (vec / norm * 35) if norm > 1e-3 else np.array([25, 25])
    
    # Сноска (callout) с номером условия
    ax_umap.annotate(
        f'{idx}',
        xy=(centroid[0], centroid[1]),
        xytext=(offset[0], offset[1]),
        textcoords='offset points',
        ha='center',
        va='center',
        fontsize=9,
        fontweight='bold',
        color='black',
        bbox=dict(
            boxstyle='circle,pad=0.25',
            facecolor='white',
            edgecolor=cond_color,
            linewidth=1.8,
            alpha=0.9
        ),
        arrowprops=dict(
            arrowstyle='->',
            connectionstyle='arc3,rad=0.1',
            color='black',
            lw=1.0,
            shrinkA=3,
            shrinkB=4
        ),
        zorder=6
    )
    
    # Прокси-элемент для легенды условий
    handle = mpl.lines.Line2D(
        [0], [0],
        marker='o',
        color='w',
        label=f'[{idx}] {label}',
        markerfacecolor=cond_color,
        markeredgecolor='black',
        markersize=9
    )
    legend_handles.append(handle)

# Дискретная легенда условий вынесена за colorbar
ax_umap.legend(
    handles=legend_handles,
    title='Conditions',
    bbox_to_anchor=(1.22, 1.0),
    loc='upper left',
    frameon=True,
    fontsize=9
)

ax_umap.set_title(f'UMAP проекция (цвет: мощность компоненты {comp_idx+1})', pad=10)
format_umap_axes(ax_umap)

# --- 4. ВРЕМЕННЫЕ РЯДЫ МОЩНОСТИ ---
ax_env = fig.add_subplot(gs[1, 1])
window_idx = np.arange(len(p_vals_norm))

ax_env.plot(window_idx, filtered_powers_norm, color='gray', lw=2.5, alpha=0.5, label='$w^T C w$', zorder=2)
ax_env.plot(window_idx, p_vals_norm, color='black', lw=1.2, label='Латентная мощность энкодера ($z$)', zorder=3)

# Отрисовка фона (событий) с синхронной нумерацией тиков
xtick_positions = []
xtick_labels = []
for idx, lab in enumerate(unique_labels_ordered, start=1):
    mask = (window_labels == lab)
    if not np.any(mask):
        continue
    changes = np.diff(np.concatenate(([0], mask.astype(int), [0])))
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]
    for s, e in zip(starts, ends):
        ax_env.axvspan(s, e-1, facecolor=label_to_color[lab], alpha=0.2, zorder=0)
        ax_env.axvline(x=s, color='gray', linestyle='--', alpha=0.7, zorder=1)
        xtick_positions.append(s)
        xtick_labels.append(f'[{idx}] {lab}')

ax_env.set_xticks(xtick_positions)
ax_env.set_xticklabels(xtick_labels, rotation=45, ha='right', fontsize=8)
ax_env.set_xlabel('Номер окна')
ax_env.set_ylabel('Мощность (scaled by std)')
ax_env.set_title(f'Динамика мощности компоненты {comp_idx+1}')
ax_env.grid(True, axis='y', linestyle=':', alpha=0.6, zorder=0)
ax_env.legend(loc='upper right')

plt.suptitle(f'Анализ компоненты {comp_idx+1} из {Npatt}', fontsize=16)
plt.tight_layout()
plt.show()

# %%





