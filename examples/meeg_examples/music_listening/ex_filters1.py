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
# fpath = "C:/Users/ansbel/Documents/GitHub/TriCo/data/external/music_listening/part2/eeg/TumAle_raw.fif"
fpath = "D:/OS(CURRENT)/data/music/exp2/20.03_g1/Tumyalis_clear.fif"
raw = mne.io.read_raw_fif(fpath, preload=True)
sfreq = raw.info['sfreq']

new_descriptions = [
    'RS_EC_1', 'RS_EO_1', '2Hz', '05Hz', '4Hz', '1Hz', '3Hz',
    'NoRy_1', 'Waltz_1', 'Waltz_2', 'NoRy_2', 'NoRy_3', 'Waltz_3',
    'NoRy_4', 'Waltz_4', 'NoRy_5', 'Waltz_5', 'RS_EC_2', 'RS_EO_2',
    'Waltz_6', 'Waltz_7', 'Waltz_8'
]

descriptions = raw.annotations.description
significant_mask = np.array([('BAD' not in d and 'EDGE' not in d) for d in descriptions])
significant_indices = np.where(significant_mask)[0]

new_desc = list(descriptions)
for idx, label in zip(significant_indices, new_descriptions):
    new_desc[idx] = label

old_annot = raw.annotations
new_durations = np.array(old_annot.duration, copy=True)
new_durations[significant_indices] = 120

new_annot = mne.Annotations(
    onset=old_annot.onset,
    duration=new_durations,
    description=np.array(new_desc, dtype='U20'),
    orig_time=old_annot.orig_time
)
raw.set_annotations(new_annot)

raw_clean = raw.copy().pick_types(eeg=True)
data = raw.get_data()

# %%
raw.plot()

# %%
# =============================================================================
# 2. ПОИСК SSD И ОТБЕЛИВАНИЕ
# =============================================================================
signal_pieces = []
noise_pieces = []

b_signal, a_signal = butter(3, np.array([15, 25]) / (int(sfreq) / 2), btype='band')
b_broad, a_broad = butter(3, np.array([13, 30]) / (int(sfreq) / 2), btype='band')
b_stop, a_stop = butter(3, np.array([14.5, 25.5]) / (int(sfreq) / 2), btype='stop')
# b_signal, a_signal = butter(3, np.array([8, 12]) / (int(sfreq) / 2), btype='band')
# b_broad, a_broad = butter(3, np.array([6, 14]) / (int(sfreq) / 2), btype='band')
# b_stop, a_stop = butter(3, np.array([7.5, 12.5]) / (int(sfreq) / 2), btype='stop')

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
window_labels = []

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
        window_labels.extend([desc] * len(epochs_band))

X_windows_band = np.concatenate(X_windows_band, axis=0)
X_windows_unfilt = np.concatenate(X_windows_unfilt, axis=0)
window_labels = np.array(window_labels)

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
import matplotlib.pyplot as plt
import numpy as np

# 1. Находим пороговое значение для 95-го процентиля
threshold = np.percentile(dist_matrix_init, 95)

# 2. Отображаем всю матрицу целиком, ограничив верхний диапазон цвета
plt.imshow(dist_matrix_init, vmax=threshold)

# 3. Добавляем цветовую шкалу
plt.colorbar(label="Расстояние (максимум ограничен 95-м процентилем)")
plt.show()

# %%
n_ch_white = covmats.shape[1]

# =============================================================================
# ПОДГОТОВКА ДАННЫХ
# =============================================================================
# Нормализация жизненно необходима, чтобы обучаемый шум стартовал в адекватном масштабе
X_cov_flat = covmats.reshape(covmats.shape[0], -1).astype(np.float32)
input_dim = X_cov_flat.shape[1]

# =============================================================================
# ДЕКОДЕР С ОБУЧАЕМЫМ НЕСФЕРИЧНЫМ ШУМОМ
# =============================================================================
class SpatialPatternDecoder(tf.keras.layers.Layer):
    def __init__(self, M_channels, N_patterns, **kwargs):
        super().__init__(**kwargs)
        self.M_channels = M_channels
        self.N_patterns = N_patterns

    def build(self, input_shape):
        # 1. Обучаемые паттерны (матрица A)
        self.A = self.add_weight(
            shape=(self.M_channels, self.N_patterns),
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
        # Нормализуем столбцы A
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

def log_euclidean_loss(y_true_flat, y_pred_flat):
    y_true_flat = tf.cast(y_true_flat, tf.float32)
    y_pred_flat = tf.cast(y_pred_flat, tf.float32)

    M = n_ch_white  
    C_true = tf.reshape(y_true_flat, [-1, M, M])
    C_pred = tf.reshape(y_pred_flat, [-1, M, M])
    
    # 1. Жесткая симметризация (защита от ошибок float32)
    C_true = 0.5 * (C_true + tf.transpose(C_true, perm=[0, 2, 1]))
    C_pred = 0.5 * (C_pred + tf.transpose(C_pred, perm=[0, 2, 1]))
    
    # 2. Регуляризация (чтобы собственные значения не были нулями)
    eps = 1e-4
    eye_M = tf.eye(M, dtype=tf.float32)
    C_true_reg = C_true + eps * eye_M
    C_pred_reg = C_pred + eps * eye_M

    # 3. Вычисляем Матричный Логарифм для ТАРГЕТА (C_true)
    eigvals_true, eigvecs_true = tf.linalg.eigh(C_true_reg)
    log_eigvals_true = tf.math.log(tf.maximum(eigvals_true, 1e-9))
    log_C_true = tf.einsum('bij,bj,bkj->bik', eigvecs_true, log_eigvals_true, eigvecs_true)

    # 4. Вычисляем Матричный Логарифм для ПРЕДСКАЗАНИЯ (C_pred)
    eigvals_pred, eigvecs_pred = tf.linalg.eigh(C_pred_reg)
    log_eigvals_pred = tf.math.log(tf.maximum(eigvals_pred, 1e-9))
    log_C_pred = tf.einsum('bij,bj,bkj->bik', eigvecs_pred, log_eigvals_pred, eigvecs_pred)

    # 5. Считаем Фробениусово расстояние между логарифмами (это и есть LEM!)
    # Разница матриц -> возводим каждый элемент в квадрат -> суммируем
    diff = log_C_true - log_C_pred
    dist_sq = tf.reduce_sum(tf.square(diff), axis=(1, 2))
    
    return tf.reduce_mean(dist_sq)

# =============================================================================
# ПОДГОТОВКА ДАННЫХ
# =============================================================================
N_patterns = 20  # Количество источников
N_dim = N_patterns  

# =============================================================================
# ЭНКОДЕР И ДЕКОДЕР (С ОРТОГОНАЛЬНОСТЬЮ)
# =============================================================================
# Энкодер переводит матрицу напрямую в 5 мощностей (z)
encoder = tf.keras.Sequential([
    tf.keras.layers.InputLayer(shape=(input_dim,)),
    tf.keras.layers.Dense(100, activation="relu"),
    tf.keras.layers.Dense(100, activation="relu"),
    tf.keras.layers.Dense(100, activation="relu"),
    tf.keras.layers.Dense(N_dim, activation="linear", name="z_powers")
])

decoder = tf.keras.Sequential([
    tf.keras.layers.InputLayer(shape=(N_dim,)),
    SpatialPatternDecoder(M_channels=n_ch_white, N_patterns=N_patterns, name="spatial_decoder"),
    tf.keras.layers.Flatten()
])

# %%
import tensorflow as tf

# =============================================================================
# НАСТРОЙКА CUDA / GPU
# =============================================================================
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Включаем динамическое выделение памяти для каждой видеокарты
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ CUDA АКТИВНА! Доступно GPU: {len(gpus)}. Используется: {gpus[0]}")
    except RuntimeError as e:
        print(e)
else:
    print("❌ ВНИМАНИЕ: CUDA не найдена или не настроена. TensorFlow будет использовать CPU!")

# %%
# =============================================================================
# НАСТРОЙКА ПАРАМЕТРОВ ОБУЧЕНИЯ (RECOMENDATIONS)
# =============================================================================

# 1. Параметры Keras (Нейросети)
batch_size = 64
keras_epochs = 10  # Сколько раз нейросеть пройдет по всему сгенерированному графу
loss_weight = 1.0 # Баланс. Если Риманово расстояние падает плохо, увеличьте до 5.0 - 10.0

# 2. Параметры UMAP (Топологии)
n_neighbors = 20  # Размер локальной окрестности (10-15 оптимально для ЭЭГ)
umap_n_epochs = 500 # Количество итераций оптимизации графа (для датасетов <10000 точек можно 500)

print(f"Обучение ParametricUMAP: N_dim={N_dim}, N_patterns={N_patterns}")

# =============================================================================
# КОЛЛБЕКИ KERAS (ДЛЯ ПОЛНОГО КОНТРОЛЯ)
# =============================================================================
# ParametricUMAP позволяет передавать любые аргументы напрямую в Keras Model.fit()
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='loss', 
    patience=3, 
    restore_best_weights=True,
    verbose=1
)

keras_fit_args = {
    "callbacks": [early_stopping],
    # "validation_split": 0.1 # Можно добавить, если используете валидационную выборку
}

# =============================================================================
# ИНИЦИАЛИЗАЦИЯ PARAMETRIC UMAP
# =============================================================================
embedder = ParametricUMAP(
    encoder=encoder,
    decoder=decoder,
    n_components=N_dim, 
    dims=(input_dim,),
    metric="precomputed", # Обязательно, так как мы передаем матрицу dist_matrix_init
    n_neighbors=n_neighbors,
    n_epochs=umap_n_epochs, # Параметр алгоритма UMAP
    batch_size=batch_size,
    parametric_reconstruction=True,
    autoencoder_loss=True, # Градиенты от декодера текут в энкодер
    parametric_reconstruction_loss_fcn=riemannian_distance_loss,
    parametric_reconstruction_loss_weight=loss_weight,
    keras_fit_kwargs=keras_fit_args, # Проброс аргументов в Keras
    verbose=True
)

# =============================================================================
# ТОНКАЯ НАСТРОЙКА ВНУТРЕННИХ ПАРАМЕТРОВ КЛАССА
# =============================================================================
# По умолчанию ParametricUMAP делит 1 эпоху на 10 частей (loss_report_frequency = 10) 
# для более частого вывода логов. 
# Мы устанавливаем это значение в 1, чтобы 1 Keras-эпоха строго равнялась 1 проходу по графу.
embedder.loss_report_frequency = 1 

# Устанавливаем количество реальных проходов нейросети по датасету[cite: 4]
embedder.n_training_epochs = keras_epochs 

# =============================================================================
# ЗАПУСК ОБУЧЕНИЯ
# =============================================================================
# Библиотека сама:
# 1. Построит нечеткие симплициальные множества на основе dist_matrix_init[cite: 4]
# 2. Сгенерирует tf.data.Dataset из пар точек (to_x, from_x)[cite: 4]
# 3. Вызовет Keras .fit()[cite: 4]
embedder.fit(X_cov_flat, precomputed_distances=dist_matrix_init)

# =============================================================================
# ВИЗУАЛИЗАЦИЯ И ИЗВЛЕЧЕНИЕ ПАТТЕРНОВ
# =============================================================================
import matplotlib.pyplot as plt

# Вытягиваем историю напрямую из скрытого атрибута объекта
history = embedder._history

plt.figure(figsize=(10, 5))
plt.plot(history['loss'], label='Total Loss (UMAP CE + Riemannian)', color='purple', linewidth=2)
plt.title('График обучения Parametric UMAP')
plt.xlabel('Эпохи (Keras)')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# %%
# =============================================================================
# ОБУЧЕНИЕ ОТОБРАЖЕНИЯ ДЛЯ КЛИКЕРА (20D Мощности <-> 2D Экран)
# =============================================================================
print("Обучение инверсной модели визуализации (20D -> 2D -> 20D)...")
from tensorflow.keras import regularizers

# Энкодер: сжимает 20D мощности в 2D для отрисовки на экране
encoder_2d = tf.keras.Sequential([
    tf.keras.layers.InputLayer(shape=(N_dim,)),
    tf.keras.layers.Dense(100, activation="elu"),
    tf.keras.layers.Dense(100, activation="elu"),
    tf.keras.layers.Dense(100, activation="elu"),
    tf.keras.layers.Dense(2, activation="linear", name="2d_coords")
])

# Декодер: с мягкой L2-регуляризацией для страховки от выбросов
decoder_2d = tf.keras.Sequential([
    tf.keras.layers.InputLayer(shape=(2,)),
    tf.keras.layers.Dense(100, activation="elu", kernel_regularizer=regularizers.l2(1e-4)),
    tf.keras.layers.Dense(100, activation="elu", kernel_regularizer=regularizers.l2(1e-4)),
    tf.keras.layers.Dense(100, activation="elu", kernel_regularizer=regularizers.l2(1e-4)),
    tf.keras.layers.Dense(N_dim, activation="linear", name="z_reconstruction")
])

reducer_2d_nn = ParametricUMAP(
    encoder=encoder_2d,
    decoder=decoder_2d,
    n_components=2,
    parametric_reconstruction=True, 
    parametric_reconstruction_loss_fcn=tf.keras.losses.MeanSquaredError(),
    # =========================
    verbose=True
)

# Обучаем визуальное пространство
pred = encoder.predict(X_cov_flat)
umap_coords = reducer_2d_nn.fit_transform(pred)
print("Визуальное пространство обучено!")

# %%
# 2. Визуализация
plt.figure(figsize=(8, 6))
plt.scatter(umap_coords[:, 0], umap_coords[:, 1], alpha=0.6, edgecolors='w', s=30)
plt.title('Визуализация данных через PCA (2D)')
plt.xlabel('Главная компонента 1')
plt.ylabel('Главная компонента 2')
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

# %%
# =============================================================================
# 6. ИЗВЛЕЧЕНИЕ ПАРАМЕТРОВ И ОТРИСОВКА
# =============================================================================
import matplotlib as mpl

# 1. Извлекаем сырые значения z (логарифмы мощностей) 
# Теперь выход энкодера — это и есть наше пространство источников!
z_values = embedder.encoder.predict(X_cov_flat)   # Размерность: (n_windows, N_patterns)

# 2. Переводим в реальные мощности
powers = np.exp(z_values)

# 3. Извлекаем матрицу паттернов и нормируем 
spatial_decoder_layer = embedder.decoder.get_layer("spatial_decoder")
# Нулевой индекс весов — это матрица A. Первый индекс был бы шумом (noise_log).
A_learned_raw = spatial_decoder_layer.get_weights()[0]
A_global = A_learned_raw / np.linalg.norm(A_learned_raw, axis=0) 

# ТЕПЕРЬ A_global - ЭТО И ЕСТЬ ГОТОВЫЕ ПАТТЕРНЫ В ПРОСТРАНСТВЕ СЕНСОРОВ!
found_patterns = [A_global[:, i] for i in range(N_patterns)]

# 4. Фильтры Хауфе (математика остается прежней)
C_global_mean = np.mean(covmats, axis=0)
C_global_inv = np.linalg.pinv(C_global_mean)
found_filters = [C_global_inv @ A_global[:, i] for i in range(N_patterns)]

print(f"Модели обучены end-to-end. Размерность powers: {powers.shape}")

# %%
import numpy as np
import matplotlib.pyplot as plt

# 1. Расчет матрицы корреляций (20 x 20)
# rowvar=False указывает, что переменные (компоненты) находятся в столбцах
corr_matrix = np.corrcoef(powers, rowvar=False)

# 2. Настройка графика
fig, ax = plt.subplots(figsize=(12, 10))

# Отображаем матрицу в виде тепловой карты
# cmap='coolwarm' центрирует цвета (синий = -1, белый = 0, красный = 1)
im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)

# Добавляем цветовую шкалу справа
cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8)
cbar.ax.set_ylabel("Коэффициент корреляции", rotation=-90, va="bottom")

# 3. Настройка осей (индексы от 1 до 20)
num_components = corr_matrix.shape[1]
ticks = np.arange(num_components)
labels = [f"C{i+1}" for i in ticks]

ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(labels, rotation=45, ha="right")
ax.set_yticklabels(labels)

# 4. Отображение числовых значений внутри ячеек (по желанию)
# Из-за плотности 20x20 используем мелкий шрифт
for i in range(num_components):
    for j in range(num_components):
        text = ax.text(j, i, f"{corr_matrix[i, j]:.2f}",
                       ha="center", va="center", color="black", fontsize=8)

# Финальное оформление
ax.set_title("Матрица корреляций компонент (NumPy & Matplotlib)", fontsize=14, pad=20)
fig.tight_layout()
plt.show()

# %%
from matplotlib.gridspec import GridSpec

def format_umap_axes(ax):
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(axis='both', which='both', length=0)
    ax.grid(True, linestyle='--', alpha=0.5, zorder=0)

print("Toha, the best coder in the world")  # AI slope  
# Выбираем индекс паттерна (0..N_patterns-1)
comp_idx = 0
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

# =============================================================================
ax_umap = fig.add_subplot(gs[0, 1])

# Вычисляем робастные границы (5-й и 95-й перцентили)
# Все значения ниже 5% будут окрашены в цвет минимума, а выше 95% - в цвет максимума
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

# Аргумент extend='both' добавит красивые треугольники на концах colorbar, 
# показывая зрителю, что есть значения, выходящие за пределы шкалы
plt.colorbar(sc1, ax=ax_umap, label='Мощность источника (scaled)', extend='both')
ax_umap.set_title(f'UMAP проекция\nЦвет: мощность компоненты {comp_idx+1}')
format_umap_axes(ax_umap)
# =============================================================================

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
# 7. СУПЕР-ИНТЕРАКТИВНЫЙ ДАШБОРД:
#    АКТИВАЦИЯ ИСТОЧНИКОВ В 2D-ПРОСТРАНСТВЕ
# =============================================================================

import matplotlib.patheffects as pe
from matplotlib.gridspec import GridSpec
import numpy as np
import matplotlib.pyplot as plt

print("Подготовка интерактивного дашборда...")

# =============================================================================
# 1. ПРОВЕРКА ДАННЫХ
# =============================================================================

original_labels = window_labels.copy()

print("powers:", powers.shape)
print("umap_coords:", umap_coords.shape)
print("window_labels:", window_labels.shape)

assert len(window_labels) == len(powers), (
    f"Количество labels ({len(window_labels)}) "
    f"не совпадает с количеством окон ({len(powers)})"
)

assert len(umap_coords) == len(powers), (
    f"Количество UMAP-точек ({len(umap_coords)}) "
    f"не совпадает с количеством окон ({len(powers)})"
)

# =============================================================================
# 2. ВЫБИРАЕМ НАИБОЛЕЕ ДИСПЕРСНЫЕ ИСТОЧНИКИ
# =============================================================================

power_variances = np.var(powers, axis=0)

top_n = min(15, N_patterns)

top_indices = np.argsort(power_variances)[::-1][:top_n]

print(
    f"Отображаем топ-{top_n} источников "
    f"с наибольшей дисперсией:"
)
print(top_indices + 1)


# =============================================================================
# 3. ЦВЕТА КЛАССОВ
# =============================================================================

# Сначала сохраняем порядок, заданный new_descriptions.
unique_classes = []

for lab in new_descriptions:
    if lab in original_labels and lab not in unique_classes:
        unique_classes.append(lab)

# Добавляем всё, что вдруг отсутствует в new_descriptions.
for lab in original_labels:
    if lab not in unique_classes:
        unique_classes.append(lab)

class_to_id = {
    lab: i + 1
    for i, lab in enumerate(unique_classes)
}

cmap_classes = plt.cm.tab20

if len(unique_classes) == 1:
    color_map = {
        unique_classes[0]: cmap_classes(0)
    }
else:
    color_map = {
        lab: cmap_classes(i / (len(unique_classes) - 1))
        for i, lab in enumerate(unique_classes)
    }

point_colors = [
    color_map[lab]
    for lab in original_labels
]


# =============================================================================
# 4. FIGURE
# =============================================================================

N_COLS_TOPO = 3
N_ROWS_TOPO = int(np.ceil(top_n / N_COLS_TOPO))

fig = plt.figure(
    figsize=(18, max(8, 2.5 * N_ROWS_TOPO))
)

gs = GridSpec(
    N_ROWS_TOPO + 1,
    N_COLS_TOPO + 2,
    figure=fig,
    width_ratios=[3.0, 0.5] + [1] * N_COLS_TOPO,
    height_ratios=[0.5] + [2] * N_ROWS_TOPO
)


# =============================================================================
# 5. UMAP
# =============================================================================

ax_umap = fig.add_subplot(gs[:, 0])

ax_text = fig.add_subplot(gs[0, 2:])
ax_text.axis('off')


# Все точки
ax_umap.scatter(
    umap_coords[:, 0],
    umap_coords[:, 1],
    c=point_colors,
    s=15,
    alpha=0.4,
    edgecolors='white',
    linewidths=0.2,
    zorder=1
)


# =============================================================================
# 6. ЦЕНТРОИДЫ КЛАССОВ
# =============================================================================

for lab in unique_classes:

    mask = original_labels == lab

    if not np.any(mask):
        continue

    cx = np.mean(umap_coords[mask, 0])
    cy = np.mean(umap_coords[mask, 1])

    ax_umap.scatter(
        cx,
        cy,
        marker='*',
        s=450,
        facecolor=color_map[lab],
        edgecolor='black',
        linewidths=1.0,
        zorder=3
    )

    ax_umap.text(
        cx,
        cy,
        str(class_to_id[lab]),
        fontsize=11,
        fontweight='bold',
        color='white',
        ha='center',
        va='center',
        zorder=4,
        path_effects=[
            pe.withStroke(
                linewidth=2.5,
                foreground="black"
            )
        ]
    )


ax_umap.set_title(
    "Фазовое пространство UMAP",
    fontsize=14
)

ax_umap.set_xlabel("UMAP 1")
ax_umap.set_ylabel("UMAP 2")

ax_umap.grid(
    True,
    linestyle='--',
    alpha=0.4,
    zorder=0
)


# =============================================================================
# 7. ЛЕГЕНДА
# =============================================================================

handles = [
    plt.Line2D(
        [0],
        [0],
        marker='o',
        color='w',
        markerfacecolor=color_map[lab],
        markersize=8
    )
    for lab in unique_classes
]

legend_labels = [
    f"{class_to_id[lab]}: {lab}"
    for lab in unique_classes
]

ax_umap.legend(
    handles,
    legend_labels,
    title="Условия",
    loc='upper center',
    bbox_to_anchor=(0.5, -0.08),
    ncol=4,
    fontsize=8,
    title_fontsize=10
)


# =============================================================================
# 8. ИНФОРМАЦИОННАЯ ПАНЕЛЬ
# =============================================================================

txt_info = ax_text.text(
    0.5,
    0.5,
    "Наведите курсор на точки UMAP",
    ha='center',
    va='center',
    fontsize=16,
    color='gray',
    fontweight='bold'
)


# =============================================================================
# 9. ТОПОМАПЫ ИСТОЧНИКОВ
# =============================================================================

print(
    "Генерация топомапов "
    "(может занять несколько секунд)..."
)

ax_topos = []
overlays = []
titles = []

mne_info = raw.info

for i, comp_idx in enumerate(top_indices):

    row = 1 + i // N_COLS_TOPO
    col = 2 + i % N_COLS_TOPO

    ax = fig.add_subplot(gs[row, col])

    mne.viz.plot_topomap(
        found_patterns[comp_idx],
        mne_info,
        axes=ax,
        show=False,
        contours=0
    )

    # Белый overlay скрывает топомап.
    # При наведении мы уменьшаем alpha,
    # и соответствующий паттерн становится видимым.
    overlay = plt.Rectangle(
        (0, 0),
        1,
        1,
        transform=ax.transAxes,
        color='white',
        alpha=0.90,
        zorder=10
    )

    ax.add_patch(overlay)

    overlays.append(overlay)

    title = ax.set_title(
        f"Ист. {comp_idx + 1}\nМощн: --",
        fontsize=11,
        color='gray'
    )

    titles.append(title)

    ax.axis('off')

    ax_topos.append(ax)


# =============================================================================
# 10. МАРКЕР ТЕКУЩЕЙ ПОЗИЦИИ
# =============================================================================

highlighted_point = None


# =============================================================================
# 11. ФУНКЦИЯ ОБНОВЛЕНИЯ DASHBOARD
# =============================================================================

def update_dashboard(event):

    global highlighted_point

    # Реагируем только на события внутри UMAP
    if event.inaxes != ax_umap:
        return

    # Только движение мыши / клик
    if event.name == 'motion_notify_event':
        if event.button is not None:
            return

    x = event.xdata
    y = event.ydata

    if x is None or y is None:
        return


    # -------------------------------------------------------------------------
    # 11.1. ДЕКОДИРУЕМ 2D -> 20D
    # -------------------------------------------------------------------------

    z_click = np.array([[x, y]], dtype=np.float32)
    
    z_20d = decoder_2d.predict(
        z_click,
        verbose=0
    )[0]
    
    p_vals = z_20d

    print(
        "z_20d:",
        "min =", np.min(z_20d),
        "max =", np.max(z_20d),
        "mean =", np.mean(z_20d),
        "std =", np.std(z_20d),
        "finite =", np.all(np.isfinite(z_20d))
    )

    p_top = p_vals[top_indices]


    # -------------------------------------------------------------------------
    # 11.2. ИЩЕМ БЛИЖАЙШУЮ РЕАЛЬНУЮ ТОЧКУ
    # -------------------------------------------------------------------------

    dists = np.hypot(
        umap_coords[:, 0] - x,
        umap_coords[:, 1] - y
    )

    min_dist_idx = np.argmin(dists)

    nearest_label = original_labels[min_dist_idx]


    # -------------------------------------------------------------------------
    # 11.3. ОБНОВЛЯЕМ ТЕКСТ
    # -------------------------------------------------------------------------

    txt_info.set_text(
        f"Зона: {class_to_id[nearest_label]} "
        f"({nearest_label})\n"
        f"Координаты: ({x:.2f}, {y:.2f})"
    )

    txt_info.set_color(
        color_map[nearest_label]
    )


    # -------------------------------------------------------------------------
    # 11.4. НОРМИРОВКА МОЩНОСТЕЙ
    # -------------------------------------------------------------------------

    p_local_max = np.max(p_top)
    p_local_min = np.min(p_top)

    denominator = p_local_max - p_local_min

    if denominator < 1e-6:
        denominator = 1e-6


    # -------------------------------------------------------------------------
    # 11.5. ОБНОВЛЯЕМ ТОПОМАПЫ
    # -------------------------------------------------------------------------

    for i, comp_idx in enumerate(top_indices):

        power = p_top[i]

        norm_p = np.clip(
            (power - p_local_min) / denominator,
            0,
            1
        )

        # Большая мощность -> прозрачность overlay меньше
        new_alpha = 0.95 * (1.0 - norm_p)

        overlays[i].set_alpha(new_alpha)

        titles[i].set_text(
            f"Ист. {comp_idx + 1}\n"
            f"Мощн: {power:.2f}"
        )

        if norm_p > 0.4:
            titles[i].set_color('black')
            titles[i].set_fontweight('bold')
        else:
            titles[i].set_color('gray')
            titles[i].set_fontweight('normal')


    # -------------------------------------------------------------------------
    # 11.6. МАРКЕР ПОЗИЦИИ
    # -------------------------------------------------------------------------

    if highlighted_point is not None:
        highlighted_point.remove()

    highlighted_point = ax_umap.scatter(
        x,
        y,
        marker='+',
        color='black',
        s=150,
        linewidths=2.0,
        zorder=5
    )


    # Перерисовываем figure
    fig.canvas.draw_idle()


# =============================================================================
# 12. ПОДКЛЮЧАЕМ INTERACTION
# =============================================================================

fig.canvas.mpl_connect(
    'button_press_event',
    update_dashboard
)

fig.canvas.mpl_connect(
    'motion_notify_event',
    update_dashboard
)


# =============================================================================
# 13. LAYOUT
# =============================================================================

plt.subplots_adjust(
    bottom=0.25,
    top=0.90,
    left=0.05,
    right=0.98,
    hspace=0.4,
    wspace=0.1
)

print(
    "Готово! Дашборд запущен. "
    "Кликните или ведите мышь по UMAP."
)

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

