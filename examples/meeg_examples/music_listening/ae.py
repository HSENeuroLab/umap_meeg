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
# 1. Путь к файлу
fpath = "C:/Users/ansbel/Documents/GitHub/TriCo/data/external/music_listening/part1/eeg/10_07_g1_2223_raw.fif"
raw = mne.io.read_raw_fif(fpath, preload=True)

# 2. Частота дискретизации
sfreq = raw.info["sfreq"]

# 3. Заменяем описания, используя numpy array вместо стандартного списка
raw.annotations.description = np.array(['EC1', 'EO1', 'BAD_', '2Hz', '05Hz', '4Hz', '1Hz', '3Hz', 'NoRy 1',
       'Waltz 1', 'Waltz 2', 'NoRy 2', 'NoRy 3', 'Waltz 3', 'NoRy 4',
       'Waltz 4', 'NoRy 5', 'Waltz 5', 'EC2', 'EO2'])
    
cond_descriptions = ['EC1', 'EO1', '2Hz', '05Hz', '4Hz', '1Hz', '3Hz', 'NoRy 1',
       'Waltz 1', 'Waltz 2', 'NoRy 2', 'NoRy 3', 'Waltz 3', 'NoRy 4',
       'Waltz 4', 'NoRy 5', 'Waltz 5', 'EC2', 'EO2']

# 4. Фильтрация индексов
bad_idx = [i for i, desc in enumerate(raw.annotations.description) if desc == "BAD_"]
good_idx = [i for i, desc in enumerate(raw.annotations.description) if desc != "BAD_"]

# Теперь индексация сработает без ошибок
bad_ann = raw.annotations[bad_idx]
good_ann = raw.annotations[good_idx]

# 5. Меняем длительность для хороших аннотаций
# Для надежности здесь тоже лучше использовать numpy array
good_ann.duration = np.array([110.0] * len(good_ann))

# Объединяем и сохраняем
new_ann = bad_ann + good_ann
raw.set_annotations(new_ann)

# %%
from mne.preprocessing import ICA
ica = ICA(method='fastica')

# 3. Обучение ICA на отфильтрованных данных
ica.fit(raw)

# 4. Визуализация компонент (для ручного поиска артефактов глаз/сердца)
ica.plot_components()  # Карты топографии компонент
ica.plot_sources(raw)  # Временные ряды компонент

# %%
ica.plot_properties(raw)

# %%
raw.plot()

# %%
data = raw.get_data()

# =============================================================================
# 2. ПОИСК SSD И ОТБЕЛИВАНИЕ
# =============================================================================
signal_pieces = []
noise_pieces = []

fmin = 3
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
from scipy.linalg import fractional_matrix_power
import numpy as np
import mne
from pyriemann.estimation import Covariances

# =============================================================================
# 3. НАРЕЗКА НА ЭПОХИ
# =============================================================================
Wsize = 2
Ssize = 0.5 
overlap = Wsize - Ssize

X_windows_band = []
X_windows_unfilt = []
window_labels = []

raw_band = raw.copy().filter(l_freq=15, h_freq=25)
raw_unfilt = raw.copy().pick_types(eeg=True)

for annot in raw.annotations:
    desc = annot['description']
    if annot['duration'] < Wsize or desc == 'BAD_':
        continue

    tmin = annot['onset'] - raw.first_time
    tmax = min(tmin + annot['duration'], raw.times[-1])

    raw_crop = raw_band.copy().crop(tmin=tmin, tmax=tmax)
    raw_crop_unfilt = raw_unfilt.copy().crop(tmin=tmin, tmax=tmax)

    epochs_band = mne.make_fixed_length_epochs(raw_crop, duration=Wsize, overlap=overlap, preload=True, verbose=False)
    epochs_unfilt = mne.make_fixed_length_epochs(raw_crop_unfilt, duration=Wsize, overlap=overlap, preload=True, verbose=False)

    if epochs_band and len(epochs_band) == len(epochs_unfilt):
        X_windows_band.append(epochs_band.get_data(copy=False))
        X_windows_unfilt.append(epochs_unfilt.get_data(copy=False))
        window_labels.extend([desc] * len(epochs_band))

X_windows_band = np.concatenate(X_windows_band, axis=0)
X_windows_unfilt = np.concatenate(X_windows_unfilt, axis=0)
window_labels = np.array(window_labels)

n_windows, n_ch, n_times = X_windows_band.shape

# =============================================================================
# ИНТЕГРАЦИЯ ОТБЕЛИВАНИЯ В SSD ФИЛЬТР
# =============================================================================
# 1. Считаем глобальную ковариацию по всем окнам (n_ch, n_windows * n_times)
X_band_flat = X_windows_band.transpose(1, 0, 2).reshape(n_ch, -1)
C_global = np.cov(X_band_flat)

# 2. Проецируем ковариацию в пространство SSD
C_ssd = W_ssd.T @ C_global @ W_ssd

# 3. Вычисляем матрицу отбеливания с легкой регуляризацией
alpha = 0
I_mat = np.eye(C_ssd.shape[0])
C_ssd_reg = (1.0 - alpha) * C_ssd + alpha * (np.trace(C_ssd) / C_ssd.shape[0]) * I_mat
W_white = fractional_matrix_power(C_ssd_reg, -0.5).real

# 4. Встраиваем отбеливание в исходный фильтр
# Теперь W_ssd_whitened снижает размерность И отбеливает за одну операцию
W_ssd_whitened = W_ssd @ W_white

# =============================================================================
# ПРОЕКЦИЯ И ВЫЧИСЛЕНИЕ КОВАРИАЦИЙ
# =============================================================================
X_windows_ssd_proj = np.zeros((n_windows, n_components_ssd, n_times))

for i in range(n_windows):
    X_windows_ssd_proj[i] = W_ssd_whitened.T @ X_windows_band[i]

covmats_band = Covariances(estimator='cov').fit_transform(X_windows_band / np.std(X_windows_band))
covmats_ssd = Covariances(estimator='cov').fit_transform(X_windows_ssd_proj / np.std(X_windows_ssd_proj))
covmats = Covariances(estimator='cov').fit_transform(X_windows_unfilt / np.std(X_windows_unfilt))

# %%
dists_init = pairwise_distance(Covariances(estimator='oas').fit_transform(X_windows_ssd_proj / np.std(X_windows_ssd_proj)), metric='riemann')
# dists_init = pairwise_distance(Covariances(estimator='oas').fit_transform(X_windows_unfilt / np.std(X_windows_unfilt)), metric='riemann')

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
import matplotlib.pyplot as plt
import numpy as np

# 1. Находим пороговое значение (например, 85-й или 95-й процентиль)
# Чем ниже процентиль, тем сильнее поднимется контраст для малых значений
percentile_value = 80
threshold = np.percentile(dists_init, percentile_value)

# 2. Отображаем матрицу
# Добавлена палитра (cmap), которая хорошо показывает контраст (например, 'viridis' или 'inferno')
plt.figure(figsize=(8, 6))
im = plt.imshow(dists_init, vmax=threshold, cmap='viridis')

# 3. Настраиваем цветовую шкалу
# Используем extend='max', чтобы показать, что на графике есть значения выше максимума шкалы
cbar = plt.colorbar(im, extend='max')
cbar.set_label(f"Расстояние (максимум ограничен {percentile_value}-м процентилем)")

plt.title("Матрица расстояний с повышенным контрастом")
plt.show()

# %%
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm, fractional_matrix_power
from sklearn.model_selection import train_test_split
from tensorflow.keras import regularizers

covmats = Covariances(estimator='cov').fit_transform(X_windows_ssd_proj / np.std(X_windows_ssd_proj))
M_channels = covmats.shape[1]

# Векторизация верхних треугольных частей матриц для подачи в MLP
idx_i, idx_j = np.triu_indices(M_channels)
multipliers = np.where(idx_i == idx_j, 1.0, np.sqrt(2.0)).astype(np.float32)

X_cov_flat = (covmats[:, idx_i, idx_j] * multipliers).astype(np.float32)
input_dim = int(X_cov_flat.shape[1])

print(f"Размерность векторизованной ковариации: {input_dim}")

# =============================================================================
# 2. РАЗДЕЛЕНИЕ НА ОБУЧАЮЩУЮ И ВАЛИДАЦИОННУЮ ВЫБОРКИ
# =============================================================================
indices = np.arange(len(X_cov_flat))
train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)

X_train = X_cov_flat[train_idx]
X_val = X_cov_flat[val_idx]

print(f"Обучающая выборка: {len(X_train)} окон. Валидационная: {len(X_val)} окон.")

# %%
# =============================================================================
# 3. АРХИТЕКТУРА АВТОЭНКОДЕРА (С НЕЗАВИСИМЫМ ФИЗИЧНЫМ ДЕКОДЕРОМ)
# =============================================================================
@tf.keras.utils.register_keras_serializable()
class PatternDecoder(tf.keras.layers.Layer):
    """
    Декодер с обучаемой матрицей паттернов A.
    Восстанавливает ковариацию: C_recon = A · diag(exp(z)) · A^T
    """
    def __init__(self, M_channels, N_patterns, **kwargs):
        super().__init__(**kwargs)
        self.M = int(M_channels)
        self.N = int(N_patterns)
        
    def build(self, input_shape):
        # 1. Базовая инициализация (единичная матрица для первых M паттернов)
        init_val = np.eye(self.M, self.N, dtype=np.float32)
        
        # 2. Если паттернов больше, чем каналов, заполняем остаток случайным шумом
        if self.N > self.M:
            random_tail = np.random.randn(self.M, self.N - self.M).astype(np.float32)
            # Нормируем каждый столбец на его L2-норму
            random_tail = random_tail / (np.linalg.norm(random_tail, axis=0, keepdims=True) + 1e-12)
            init_val[:, self.M:] = random_tail         
            
        self.A = self.add_weight(
            shape=(self.M, self.N),
            initializer=tf.keras.initializers.Constant(init_val),
            trainable=True,
            name="A_patterns"
        )
        
        # Подготовка индексов для векторизации 
        i, j = np.triu_indices(self.M)
        flat_indices_np = (i * self.M + j).astype(np.int32)
        mults_np = np.where(i == j, 1.0, np.sqrt(2.0)).astype(np.float32)

        self.flat_indices = self.add_weight(
            name="flat_indices",
            shape=flat_indices_np.shape,
            dtype=tf.int32,
            initializer=tf.keras.initializers.Constant(flat_indices_np),
            trainable=False,
        )
        self.multipliers = self.add_weight(
            name="multipliers",
            shape=mults_np.shape,
            dtype=tf.float32,
            initializer=tf.keras.initializers.Constant(mults_np),
            trainable=False,
        )
        super().build(input_shape)

    def call(self, z, training=None):
        # Ограничиваем z, чтобы избежать переполнения float32 при возведении в экспоненту.
        # Диапазон [-20.0, 10.0] безопасен для вычислений и достаточен для физики.
        z_clipped = tf.clip_by_value(z, -20.0, 20.0)
        P = tf.math.exp(z_clipped)  # (batch, N_patterns)
        
        C_recon = tf.einsum('mk,bk,nk->bmn', self.A, P, self.A)
        C_flat = tf.reshape(C_recon, [-1, self.M * self.M])
        vecs = tf.gather(C_flat, self.flat_indices, axis=1)
        return vecs * self.multipliers
    
import tensorflow as tf
import numpy as np

# =============================================================================
# 1. КАСТОМНЫЕ СЛОИ ДЛЯ SPD-NET
# =============================================================================

@tf.keras.utils.register_keras_serializable()
class UnflattenSymmetricLayer(tf.keras.layers.Layer):
    """Сворачивает векторизованную ковариацию обратно в симметричную матрицу M x M"""
    def __init__(self, M_channels, **kwargs):
        super().__init__(**kwargs)
        self.M = int(M_channels)
        
    def build(self, input_shape):
        D = input_shape[-1]
        P = np.zeros((D, self.M * self.M), dtype=np.float32)
        i, j = np.triu_indices(self.M)
        
        # Создаем матрицу проекции, которая отменяет умножение на sqrt(2) вне диагонали
        for idx, (r, c) in enumerate(zip(i, j)):
            if r == c:
                P[idx, r * self.M + c] = 1.0
            else:
                val = 1.0 / np.sqrt(2.0)
                P[idx, r * self.M + c] = val
                P[idx, c * self.M + r] = val
                
        self.P = tf.constant(P)
        super().build(input_shape)

    def call(self, inputs):
        # Восстанавливаем плоскую матрицу M*M и решейпим в (batch, M, M)
        flat_matrices = tf.matmul(inputs, self.P)
        return tf.reshape(flat_matrices, [-1, self.M, self.M])


import tensorflow as tf
import tensorflow_riemopt as ro
from tensorflow_riemopt.manifolds.stiefel import StiefelCanonical
from tensorflow_riemopt.variable import assign_to_manifold   # <-- import this

@tf.keras.utils.register_keras_serializable()
class BiMapLayer(tf.keras.layers.Layer):
    """
    BiMap Layer: X_out = W * X_in * W^T
    Weights W are strictly constrained to the Stiefel manifold (orthogonal).
    """
    def __init__(self, d_out, **kwargs):
        super().__init__(**kwargs)
        self.d_out = int(d_out)

    def build(self, input_shape):
        self.d_in = int(input_shape[-1])

        # 1. Create a normal Keras variable (orthogonal initializer)
        init_val = tf.keras.initializers.Orthogonal()(
            shape=(self.d_out, self.d_in)
        )
        self.W = self.add_weight(
            name="W_bimap",
            shape=(self.d_out, self.d_in),
            initializer=tf.keras.initializers.Constant(init_val),
            trainable=True,
        )

        # 2. Attach the Stiefel manifold to the variable
        assign_to_manifold(self.W, StiefelCanonical())   # <-- key step

        super().build(input_shape)

    def call(self, X):
        # W stays orthogonal because the Riemannian optimizer projects it
        # back to the manifold after each update.
        X_out = tf.einsum('oi,bij,kj->bok', self.W, X, self.W)
        return 0.5 * (X_out + tf.transpose(X_out, perm=[0, 2, 1]))


@tf.keras.utils.register_keras_serializable()
class ReEigLayer(tf.keras.layers.Layer):
    """
    ReEig Layer: Нелинейность, ограничивающая минимальные собственные числа.
    """
    def __init__(self, epsilon=1e-4, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def call(self, X):
        # На всякий случай симметризуем для стабильности градиентов
        X = 0.5 * (X + tf.transpose(X, perm=[0, 2, 1]))
        eigvals, eigvecs = tf.linalg.eigh(X)
        
        # Rectification (ReLU-подобная функция для собственных чисел)
        eigvals_rect = tf.maximum(eigvals, self.epsilon)
        
        # Сборка матрицы обратно: U * Sigma_rect * U^T
        X_rect = tf.einsum('bij,bj,bkj->bik', eigvecs, eigvals_rect, eigvecs)
        return 0.5 * (X_rect + tf.transpose(X_rect, perm=[0, 2, 1]))


@tf.keras.utils.register_keras_serializable()
class LogEigAndFlattenLayer(tf.keras.layers.Layer):
    """
    LogEig Layer: Проецирует риманово пространство в плоское евклидово.
    Сразу векторизует верхнюю треугольную часть, чтобы подать её в Dense.
    """
    def __init__(self, epsilon=1e-4, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        self.M = int(input_shape[-1])
        i, j = np.triu_indices(self.M)
        self.flat_indices = tf.constant((i * self.M + j).astype(np.int32))
        
        # Снова умножаем на sqrt(2) вне диагонали для корректного касательного пространства
        mults = np.where(i == j, 1.0, np.sqrt(2.0)).astype(np.float32)
        self.multipliers = tf.constant(mults)
        super().build(input_shape)

    def call(self, X):
        X = 0.5 * (X + tf.transpose(X, perm=[0, 2, 1]))
        eigvals, eigvecs = tf.linalg.eigh(X)
        
        # Взятие матричного логарифма
        log_eigvals = tf.math.log(tf.maximum(eigvals, self.epsilon))
        log_X = tf.einsum('bij,bj,bkj->bik', eigvecs, log_eigvals, eigvecs)
        
        # Извлечение признаков (векторизация)
        log_X_flat = tf.reshape(log_X, [-1, self.M * self.M])
        vecs = tf.gather(log_X_flat, self.flat_indices, axis=1)
        return vecs * self.multipliers

@tf.keras.utils.register_keras_serializable()
class LogEuclideanLoss(tf.keras.losses.Loss):
    """
    Вычисляет Лог-Евклидово расстояние между векторизованными ковариационными матрицами.
    """
    def __init__(self, M_channels, epsilon=1e-7, **kwargs):
        super().__init__(**kwargs)
        self.M = int(M_channels)
        self.epsilon = epsilon
        
        # Предвычисляем матрицу восстановления (Unflatten)
        D = (self.M * (self.M + 1)) // 2
        P = np.zeros((D, self.M * self.M), dtype=np.float32)
        i, j = np.triu_indices(self.M)
        
        for idx, (r, c) in enumerate(zip(i, j)):
            if r == c:
                P[idx, r * self.M + c] = 1.0
            else:
                val = 1.0 / np.sqrt(2.0)
                P[idx, r * self.M + c] = val
                P[idx, c * self.M + r] = val
                
        self.P = tf.constant(P)

    def matrix_log(self, C):
        # Симметризация для стабильности вычисления градиентов
        C = 0.5 * (C + tf.transpose(C, perm=[0, 2, 1]))
        eigvals, eigvecs = tf.linalg.eigh(C)
        
        # Ограничиваем снизу для избежания log(0) и NaN в градиентах
        log_eigvals = tf.math.log(tf.maximum(eigvals, self.epsilon))
        
        # Собираем матрицу обратно
        return tf.einsum('bij,bj,bkj->bik', eigvecs, log_eigvals, eigvecs)

    def call(self, y_true, y_pred):
        # 1. Восстанавливаем квадратные матрицы (batch, M, M) из векторов (batch, D)
        C_true = tf.reshape(tf.matmul(y_true, self.P), [-1, self.M, self.M])
        C_pred = tf.reshape(tf.matmul(y_pred, self.P), [-1, self.M, self.M])
        
        # 2. Проецируем в касательное пространство (Log-Euclidean)
        log_C_true = self.matrix_log(C_true)
        log_C_pred = self.matrix_log(C_pred)
        
        # 3. Квадрат нормы Фробениуса разности
        diff = log_C_true - log_C_pred
        loss = tf.reduce_sum(tf.square(diff), axis=[1, 2])
        
        return tf.reduce_mean(loss)
    
# =============================================================================
# 2. СБОРКА АВТОЭНКОДЕРА С SPD-NET
# =============================================================================

N_patterns = 21 
N_dim = N_patterns

print(f"Архитектура: SPD-Net ({input_dim}) -> Latent ({N_patterns}) -> Physical Decoder")

# ----------------- ЭНКОДЕР (SPD-Net) -----------------
inputs_enc = tf.keras.Input(shape=(input_dim,), dtype=tf.float32, name="encoder_input")

# 1. Восстанавливаем матрицу из вектора
x = UnflattenSymmetricLayer(M_channels, name="unflatten_cov")(inputs_enc)

# 2. Блок 1: BiMap -> ReEig
# Размерность d_out можно сделать равной M_channels или слегка сжать (например, M_channels - 2)
# Оставим M_channels, чтобы не терять информацию до перехода к паттернам
x = BiMapLayer(d_out=M_channels, name="bimap_1")(x)
x = ReEigLayer(epsilon=1e-7, name="reeig_1")(x)

# 3. Блок 2: LogEig -> Векторизация
x = LogEigAndFlattenLayer(epsilon=1e-7, name="logeig_flatten")(x)

# 4. Латентное пространство 
z_latent = tf.keras.layers.Dense(
    N_patterns, 
    activation="linear", 
    name="latent_z",
)(x)

encoder = tf.keras.Model(inputs=inputs_enc, outputs=z_latent, name="spdnet_encoder")

inputs_dec = tf.keras.Input(shape=(N_patterns,), dtype=tf.float32, name="decoder_input")
recon_out = PatternDecoder(
    M_channels=M_channels, 
    N_patterns=N_patterns, 
    name="spatial_decoder"
)(inputs_dec)
decoder = tf.keras.Model(inputs=inputs_dec, outputs=recon_out, name="physical_decoder")

# %%
# =============================================================================
# 2. СБОРКА АВТОЭНКОДЕРА С MLP (БЕЙЗЛАЙН ДЛЯ СРАВНЕНИЯ)
# =============================================================================

N_patterns = 21 
N_dim = N_patterns

print(f"Архитектура: Standard MLP ({input_dim}) -> Latent ({N_patterns}) -> Physical Decoder")

# ----------------- ЭНКОДЕР (Standard MLP) -----------------
inputs_enc = tf.keras.Input(shape=(input_dim,), dtype=tf.float32, name="encoder_input")

# 1. Входное расширение
x = tf.keras.layers.Dense(512, activation="elu", name="enc_dense_1")(inputs_enc)
x = tf.keras.layers.BatchNormalization(name="enc_bn_1")(x)
x = tf.keras.layers.Dropout(0.3, name="enc_drop_1")(x)

# 2. Residual Block (помогает прокидывать градиенты)
res = tf.keras.layers.Dense(512, activation="elu", name="enc_res_dense")(x)
res = tf.keras.layers.BatchNormalization(name="enc_res_bn")(res)
res = tf.keras.layers.Dropout(0.3, name="enc_res_drop")(res)
x = tf.keras.layers.Add(name="enc_res_add")([x, res])

# 3. Постепенное сужение (Bottleneck)
x = tf.keras.layers.Dense(256, activation="elu", name="enc_dense_2")(x)
x = tf.keras.layers.BatchNormalization(name="enc_bn_2")(x)
x = tf.keras.layers.Dropout(0.2, name="enc_drop_2")(x)

x = tf.keras.layers.Dense(128, activation="elu", name="enc_dense_3")(x)
x = tf.keras.layers.BatchNormalization(name="enc_bn_3")(x)

# 4. Латентное пространство
# (Для сверхполного словаря N > M добавьте activity_regularizer=tf.keras.regularizers.l1(1e-7))
z_latent = tf.keras.layers.Dense(
    N_patterns, 
    activation="linear", 
    name="latent_z",
    activity_regularizer=tf.keras.regularizers.l1(1e-7)
)(x)

encoder = tf.keras.Model(inputs=inputs_enc, outputs=z_latent, name="mlp_encoder")

# ----------------- ДЕКОДЕР (Физическая модель A) -----------------
inputs_dec = tf.keras.Input(shape=(N_patterns,), dtype=tf.float32, name="decoder_input")
recon_out = PatternDecoder(
    M_channels=M_channels, 
    N_patterns=N_patterns, 
    name="spatial_decoder"
)(inputs_dec)
decoder = tf.keras.Model(inputs=inputs_dec, outputs=recon_out, name="physical_decoder")

# %%
# Сквозной Автоэнкодер
autoencoder_output = decoder(encoder(inputs_enc))
autoencoder = tf.keras.Model(inputs=inputs_enc, outputs=autoencoder_output, name="mlp_phys_autoencoder")

# Компиляция
autoencoder.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='mse' # Замените на riemannian_distance_loss при необходимости
)
autoencoder.summary()

# =============================================================================
# 4. ОБУЧЕНИЕ АВТОЭНКОДЕРА
# =============================================================================
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', 
    patience=100, 
    restore_best_weights=True
)

history = autoencoder.fit(
    x=X_train, 
    y=X_train,  # Пытаемся восстановить вход
    validation_data=(X_val, X_val),
    epochs=300,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=1
)

plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Train Loss (MSE)', color='blue', linewidth=2)
plt.plot(history.history['val_loss'], label='Val Loss (MSE)', color='green', linewidth=2)
plt.title('Качество декодера: Ошибка ковариаций')
plt.xlabel('Эпохи')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# %%
import tensorflow as tf
from tensorflow_riemopt.optimizers import RiemannianAdam

import tensorflow as tf
import keras
from keras import ops


@keras.utils.register_keras_serializable()
class KerasRiemannianAdam(keras.optimizers.Optimizer):
    """
    Keras 3 compatible Riemannian Adam.

    For variables that have a `manifold` attribute (attached via
    tensorflow_riemopt.variable.assign_to_manifold), the gradient is
    projected onto the tangent space, Adam moments are updated there,
    and the new point is obtained via the manifold's exponential map.
    Variables without a manifold receive a standard Euclidean Adam update.
    """

    def __init__(
        self,
        learning_rate=1e-3,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7,
        name="KerasRiemannianAdam",
        **kwargs,
    ):
        super().__init__(learning_rate=learning_rate, name=name, **kwargs)
        self.beta_1 = float(beta_1)
        self.beta_2 = float(beta_2)
        self.epsilon = float(epsilon)
        self._built_slots = False
        self._m = {}
        self._v = {}

    # ------------------------------------------------------------------
    # Slot creation
    # ------------------------------------------------------------------
    def build(self, var_list):
        super().build(var_list)
        if self._built_slots:
            return
        for var in var_list:
            self._m[id(var)] = self.add_variable_from_reference(var, "m")
            self._v[id(var)] = self.add_variable_from_reference(var, "v")
        self._built_slots = True

    # ------------------------------------------------------------------
    # Per-variable update (Keras 3 entry point)
    # ------------------------------------------------------------------
    def update_step(self, gradient, variable, learning_rate):
        key = id(variable)

        # Lazily create slots if build() did not see this variable.
        if key not in self._m:
            self._m[key] = self.add_variable_from_reference(variable, "m")
            self._v[key] = self.add_variable_from_reference(variable, "v")

        m = self._m[key]
        v = self._v[key]

        # Cast LR to the variable's dtype.
        lr = ops.cast(learning_rate, variable.dtype)

        # Does this variable live on a manifold?
        manifold = getattr(variable, "manifold", None)

        if manifold is not None:
            # 1. Project the Euclidean gradient onto the tangent space.
            grad_proj = manifold.proju(variable, gradient)
        else:
            grad_proj = gradient

        # 2. Adam moments in the tangent space (or Euclidean space).
        m_t = self.beta_1 * m + (1.0 - self.beta_1) * grad_proj
        v_t = self.beta_2 * v + (1.0 - self.beta_2) * ops.square(grad_proj)

        step = ops.cast(self.iterations + 1, variable.dtype)
        m_hat = m_t / (1.0 - ops.power(self.beta_1, step))
        v_hat = v_t / (1.0 - ops.power(self.beta_2, step))

        update = m_hat / (ops.sqrt(v_hat) + self.epsilon)

        # 3. Apply the update.
        if manifold is not None:
            # Retract back to the manifold.
            new_val = manifold.expmap(variable, -lr * update)
            variable.assign(new_val)
        else:
            variable.assign_sub(lr * update)

        # 4. Store updated moments.
        m.assign(m_t)
        v.assign(v_t)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "beta_1": self.beta_1,
                "beta_2": self.beta_2,
                "epsilon": self.epsilon,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
from umap.parametric_umap import ParametricUMAP

dists_train = dists_init[np.ix_(train_idx, train_idx)]

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='loss', # В UMAPModel Keras отслеживает композитный loss (UMAP + Recon)
    patience=20, 
    restore_best_weights=True
)

keras_fit_args = {
    "callbacks": [early_stopping],
    "batch_size": 64
}

# 1. Build the ParametricUMAP model WITHOUT the optimizer kwarg.
embedder = ParametricUMAP(
    encoder=encoder,
    decoder=decoder,
    dims=(input_dim,),
    n_components=N_patterns,
    metric="precomputed",
    n_neighbors=20,
    parametric_reconstruction=True, # Включаем реконструкционный лосс[cite: 1]
    autoencoder_loss=True, # Разрешаем градиентам от декодера протекать в энкодер[cite: 1]
    parametric_reconstruction_loss_fcn=LogEuclideanLoss(M_channels=M_channels),
    parametric_reconstruction_loss_weight=1.0, # Баланс между графом и MSE[cite: 1]
    reconstruction_validation=X_val, # Передаем валидационную выборку для оценки[cite: 1]
    keras_fit_kwargs=keras_fit_args,
    verbose=True
    )

# 2. Replace the internal Adam with the Riemannian Adam.
embedder.optimizer = KerasRiemannianAdam(learning_rate=1e-3)

embedder.loss_report_frequency = 100
embedder.n_training_epochs = 1

# 3. Fit as usual.
embedder.fit(X_train, precomputed_distances=dists_train)

# =============================================================================
# 5. ВИЗУАЛИЗАЦИЯ ИСТОРИИ ОБУЧЕНИЯ PARAMETRIC UMAP
# =============================================================================
import matplotlib.pyplot as plt

history = embedder._history

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# График 1: Ошибка реконструкции ковариаций (Физика)
if 'recon_loss' in history and 'val_recon_loss' in history:
    ax1.plot(history['recon_loss'], label='Train Recon Loss', color='orange', linewidth=2)
    ax1.plot(history['val_recon_loss'], label='Val Recon Loss', color='red', linewidth=2)
ax1.set_title('Качество декодера: MSE Реконструкции')
ax1.set_xlabel('Шаги оценки')
ax1.set_ylabel('MSE')
ax1.legend()
ax1.grid(True, linestyle='--', alpha=0.7)

# График 2: Ошибка графа UMAP (Топология)
if 'umap_loss' in history and 'val_umap_loss' in history:
    ax2.plot(history['umap_loss'], label='Train UMAP Loss', color='purple', linewidth=2)
ax2.set_title('Качество проекции: UMAP Граф (Кросс-Энтропия)')
ax2.set_xlabel('Шаги оценки')
ax2.set_ylabel('Loss (Cross-Entropy)')
ax2.legend()
ax2.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

# %%
# =============================================================================
# 6. ОБУЧЕНИЕ ОТОБРАЖЕНИЯ ДЛЯ КЛИКЕРА (N_patterns -> 2D Экран)
# =============================================================================
print("Обучение инверсной модели визуализации (Latent -> 2D -> Latent)...")

encoder_2d = tf.keras.Sequential([
    tf.keras.layers.InputLayer(shape=(N_dim,)),
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
    tf.keras.layers.Dense(N_dim, activation="linear", name="z_reconstruction")
])

# Внимание: Убедитесь, что ParametricUMAP импортирован корректно
try:
    reducer_2d_nn = ParametricUMAP(
        encoder=encoder_2d,
        decoder=decoder_2d,
        n_components=2,
        parametric_reconstruction=True, 
        parametric_reconstruction_loss_fcn=tf.keras.losses.MeanSquaredError(),
        verbose=True
    )

    # Перезапустите этот кусок кода перед отрисовкой графиков:
    pred = encoder.predict(X_cov_flat) # Здесь должно получиться (5214, 38)
    umap_coords = reducer_2d_nn.fit_transform(pred) # Здесь должно получиться (5214, 2)
    
    print(f"Размер umap_coords: {umap_coords.shape}")
    print(f"Размер p_vals_norm: {p_vals_norm.shape}")

    # Визуализация 2D пространства
    plt.figure(figsize=(8, 6))
    plt.scatter(umap_coords[:, 0], umap_coords[:, 1], alpha=0.6, edgecolors='w', s=30)
    plt.title('Визуализация 2D проекции (ParametricUMAP)')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()
    
except NameError:
    print("ParametricUMAP не найден. Убедитесь, что umap-learn установлен и импортирован.")

# %%
import numpy as np
from scipy.linalg import inv

# 1. Извлекаем матрицу A напрямую из слоя декодера
decoder_layer = decoder.get_layer("spatial_decoder")
# Формат: (n_components_ssd, N_patterns)
A_learned_raw = decoder_layer.get_weights()[0]  

# 2. Нормируем столбцы топографий в латентном пространстве
scales = np.linalg.norm(A_learned_raw, axis=0)
A_norm = A_learned_raw / (scales + 1e-12)

# 3. Получаем предсказанные мощности (P)
z_amplitudes = encoder.predict(X_cov_flat)
P_raw = np.exp(z_amplitudes)
powers_true = P_raw * (scales ** 2)

# 4. Вычисляем СКВОЗНОЙ пространственный фильтр (Сенсоры -> Нейросеть)
# W_ssd_whitened: (n_ch, n_ssd), A_norm: (n_ssd, N_patterns)
# Результат W_total: (n_ch, N_patterns)
W_total = W_ssd_whitened @ A_norm  

# 5. Вычисляем истинные физические паттерны (по формуле Хауфе: A = C @ W)
C_global_mean = np.mean([c for c in covmats_band], 0)

# C_global_mean: (n_ch, n_ch), W_total: (n_ch, N_patterns)
# Результат A_global_raw: (n_ch, N_patterns)
A_global_raw = C_global_mean @ W_total  

# Масштабируем паттерны и корректируем физические мощности
scales_global = np.linalg.norm(A_global_raw, axis=0)
A_global = A_global_raw / (scales_global)
powers_true = powers_true * (scales_global ** 2)

# =============================================================================
# ДОПОЛНИТЕЛЬНЫЙ ШАГ: ВЫЧИСЛЕНИЕ ФИЛЬТРОВ ИЗ ПАТТЕРНОВ (W = C_{reg}^{-1} @ A)
# =============================================================================
# Регуляризация средней ковариационной матрицы для стабильного обращения
alpha = 1e-4
I_mat = np.eye(C_global_mean.shape[0])
C_global_mean_reg = C_global_mean + alpha * (np.trace(C_global_mean) / C_global_mean.shape[0]) * I_mat
C_global_inv = inv(C_global_mean_reg)

# Формат: (n_ch, N_patterns)
W_from_A = C_global_inv @ A_global

# =============================================================================
# СОРТИРОВКА И ПОДГОТОВКА К ОТРИСОВКЕ
# =============================================================================
power_variances = np.mean(powers_true, axis=0)
sort_idx = np.argsort(power_variances)[::-1]

A_sorted = A_global[:, sort_idx]
W_sorted = W_total[:, sort_idx]  
W_from_A_sorted = W_from_A[:, sort_idx]  # Сортируем и альтернативные фильтры
powers = powers_true[:, sort_idx]  

print(f"Топ-5 компонент по средней мощности: {sort_idx[:5]}")

# 7. Списки для отрисовки в MNE (массивы [N_patterns, n_ch])
found_patterns = np.array([A_sorted[:, i] for i in range(N_patterns)])
found_filters_haufe = np.array([W_sorted[:, i] for i in range(N_patterns)])

# Дополнительный список фильтров, полученных из паттернов
found_filters_invC = np.array([W_from_A_sorted[:, i] for i in range(N_patterns)])

# %%
# =============================================================================
# 5. ГРАФИК: СРЕДНЯЯ МОЩНОСТЬ ПО КОМПОНЕНТАМ С ОТКЛОНЕНИЯМИ
# =============================================================================
import matplotlib.pyplot as plt

# powers уже отсортированы по убыванию среднего (см. блок выше)
mean_power = np.mean(powers, axis=0)
std_power  = np.std(powers, axis=0)

component_idx = np.arange(1, N_patterns + 1)

fig, ax = plt.subplots(figsize=(10, 5))

# errorbar: точка — среднее, усы — ±std
ax.errorbar(
    component_idx, mean_power,
    yerr=std_power,
    fmt='o', 
    capsize=4, 
    elinewidth=1.2,
    color='tab:blue',
    ecolor='tab:gray',
    markersize=6,
    label='mean ± std'
)

# Дополнительно — "лента" ±std для наглядности
ax.fill_between(
    component_idx,
    mean_power - std_power,
    mean_power + std_power,
    color='tab:blue',
    alpha=0.15
)

ax.set_xlabel('Номер компоненты (по убыванию средней мощности)')
ax.set_ylabel('Мощность')
ax.set_title('Средняя мощность компонент с отклонениями (±1 std)')
ax.grid(True, alpha=0.3)
ax.legend()
plt.tight_layout()
plt.show()

# %%
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
# Предположу, что mne и raw у тебя уже импортированы/загружены
# import mne

def format_umap_axes(ax): 
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(axis='both', which='both', length=0)
    ax.grid(True, linestyle='--', alpha=0.5, zorder=0)

print("Toha, the best coder in the world")  # AI slope  

# =============================================================================
# ВЫБОР КОМПОНЕНТЫ
# =============================================================================
# Выбираем индекс паттерна (0..N_patterns-1)
comp_idx = 20+

# Берем СТРОКУ из массива (это и есть нужный вектор длины n_ch)
W_sensor = found_filters_haufe[comp_idx] 
A_pattern = found_patterns[comp_idx]

# 1. Мощность выбранного выученного источника (латентная переменная сети)
p_vals = powers[:, comp_idx]

# 2. Мощность через ручное применение пространственного фильтра (w^T * C * w)
# ВАЖНО: Фильтр W_sensor УЖЕ сквозной, поэтому применяем его прямо к исходной С!
filtered_powers = np.array([W_sensor.T @ C @ W_sensor for C in covmats_band])

# 3. Нормализация
p_vals_norm = p_vals / np.std(p_vals)
filtered_powers_norm = filtered_powers / np.std(filtered_powers)

# =============================================================================
# ПОДГОТОВКА ЦВЕТОВ И ЛЕЙБЛОВ
# =============================================================================
unique_labels_ordered = []
for lab in window_labels:
    if lab not in unique_labels_ordered:
        unique_labels_ordered.append(lab)

cmap = mpl.colormaps['viridis']
label_to_color = {lab: cmap(i / len(unique_labels_ordered)) for i, lab in enumerate(unique_labels_ordered)}

# =============================================================================
# ОТРИСОВКА
# =============================================================================
fig = plt.figure(figsize=(16, 10))
gs = GridSpec(2, 2, figure=fig, width_ratios=[1, 2], height_ratios=[1, 1])

# --- 1. ТОПОГРАФИЯ ФИЛЬТРА (W) ---
ax_filt = fig.add_subplot(gs[0, 0])
mne.viz.plot_topomap(W_sensor, raw.info, axes=ax_filt, show=False)
ax_filt.set_title('Пространственный фильтр (Нейросеть W)')

# --- 2. ТОПОГРАФИЯ ПАТТЕРНА (A) ---
ax_patt = fig.add_subplot(gs[1, 0])
mne.viz.plot_topomap(A_pattern, raw.info, axes=ax_patt, show=False)
ax_patt.set_title('Истинный паттерн источника (A = W^+)')

# --- 3. UMAP ПРОЕКЦИЯ ---
ax_umap = fig.add_subplot(gs[0, 1])

# Вычисляем робастные границы (5-й и 95-й перцентили)
vmin_val = np.percentile(p_vals_norm, 5)
vmax_val = np.percentile(p_vals_norm, 95)

sc1 = ax_umap.scatter(
    umap_coords[:, 0], 
    umap_coords[:, 1], 
    c=p_vals_norm, 
    cmap='plasma', 
    s=15, 
    zorder=2,
    vmin=vmin_val,
    vmax=vmax_val   
)
plt.colorbar(sc1, ax=ax_umap, label='Мощность источника (scaled)', extend='both')
ax_umap.set_title(f'UMAP проекция\nЦвет: мощность компоненты {comp_idx+1}')
format_umap_axes(ax_umap)

# --- 4. ВРЕМЕННЫЕ РЯДЫ МОЩНОСТИ ---
ax_env = fig.add_subplot(gs[1, 1])
window_idx = np.arange(len(p_vals_norm))

# Отрисовка обеих мощностей для сравнения
# Если они идут "ноздря в ноздрю", значит модель выучила честный фильтр!
ax_env.plot(window_idx, filtered_powers_norm, color='gray', lw=2.5, alpha=0.5, label='Математический W-фильтр ($w^T C w$)', zorder=2)
ax_env.plot(window_idx, p_vals_norm, color='black', lw=1.2, label='Латентная мощность от сети ($z$)', zorder=3)

# Отрисовка фона (событий)
xtick_positions = []
xtick_labels = []
for lab in unique_labels_ordered:
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