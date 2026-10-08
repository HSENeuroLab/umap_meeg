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
from umap.parametric_umap import ParametricUMAP

lib_directory = os.path.abspath("C:/Users/ansbel/Documents/GitHub/pyRiemann") 
if lib_directory not in sys.path:
    sys.path.insert(0, lib_directory)
from pyriemann.estimation import Covariances
from pyriemann.geometry.distance import pairwise_distance
from sklearn.model_selection import train_test_split

from matplotlib.gridspec import GridSpec
import matplotlib as mpl

# %%
# 1. Путь к файлу
fpath = "C:/Users/ansbel/Downloads/10_07_g1_2223_raw.fif"
raw = mne.io.read_raw_fif(fpath, preload=True)

# 2. Частота дискретизации
sfreq = raw.info["sfreq"]

# 3. Заменяем описания, используя numpy array вместо стандартного списка
# raw.annotations.description = np.array([
#     "RS_EC_1", "BAD_", "RS_EO_1", "BAD_", "2Hz", "05Hz", "4Hz", "1Hz", "BAD_", "BAD_", 
#     "BAD_", "3Hz", "NoRy_1", "Waltz_1", "Waltz_2", "BAD_", "NoRy_2", "BAD_", "NoRy_3", 
#     "BAD_", "BAD_", "BAD_", "Waltz_3", "BAD_", "BAD_", "BAD_", "NoRy_4", "BAD_", 
#     "BAD_", "Waltz_4", "BAD_", "NoRy_5", "Waltz_5", "RS_EC_2", "BAD_", "RS_EO_2", 
#     "BAD_", "BAD_", "Waltz_6", "BAD_", "BAD_", "Waltz_7", "Waltz_8"
# ])
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

data = raw.get_data()

raw.plot()

# %%
from mne.preprocessing import ICA
ica = ICA(n_components=0.999, method='fastica')

# 3. Обучение ICA на отфильтрованных данных
ica.fit(raw)

# 4. Визуализация компонент (для ручного поиска артефактов глаз/сердца)
ica.plot_components()  # Карты топографии компонент
# ica.plot_sources(raw)  # Временные ряды компонент

# %%
from mne.preprocessing import ICA

# Переводит данные от сенсоров в пространство главных компонент
W_pca = ica.pca_components_  # Форма: (n_pca_components, n_channels)

# 2. Вектор средних значений каналов 
# MNE вычитает его из сырых данных перед умножением на W_pca
pca_mean = ica.pca_mean_     # Форма: (n_channels,)

# %%
ica.plot_properties(raw)

# %%
# =============================================================================
# 2. ПОИСК SSD И ОТБЕЛИВАНИЕ
# =============================================================================
signal_pieces = []
noise_pieces = []

fmin = 3
fmax = 27
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
n_components = W_ssd.shape[1]

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
        X_windows_band.append(epochs_band.get_data(copy=False))
        X_windows_unfilt.append(epochs_unfilt.get_data(copy=False))
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
label_map = {
    'EC1': 0, 'EC2': 0,
    'EO1': 0, 'EO2': 0,
    '05Hz': 2, '1Hz': 2, '2Hz': 2, '3Hz': 2, '4Hz': 2,
    'Waltz 1': 0, 'Waltz 2': 0, 'Waltz 3': 0, 'Waltz 4': 0, 'Waltz 5': 0,
    'NoRy 1': 0, 'NoRy 2': 0, 'NoRy 3': 0, 'NoRy 4': 0, 'NoRy 5': 0
}

class_names = {
    0: 'Закрытые глаза',
    1: 'Открытые глаза',
    2: 'Стимуляция',
    3: 'Вальс',
    4: 'Неритмичные'
}

y = np.array([label_map.get(lbl, -1) for lbl in window_labels], dtype=np.int32)

# %%
dists_init = pairwise_distance(covmats_ssd, metric='riemann')

# %%
reducer = umap.UMAP(n_components=2, n_neighbors=20, metric='precomputed')
coords = reducer.fit_transform(dists_init)

plt.figure(figsize=(10, 8))

# 1. Отрисовываем точки. 
# Обязательно передаем y в параметр c=y для раскраски
scatter = plt.scatter(coords[:, 0], coords[:, 1], s=15, cmap='Spectral', alpha=0.7)

# 2. Вычисляем и рисуем центроиды
unique_classes = np.unique(y)

for cls in unique_classes:
    # Если вы не фильтровали метку -1 (неизвестный класс) и хотите её пропустить:
    # if cls == -1: continue 
    
    # Находим индексы точек, принадлежащих текущему классу
    idx = (y == cls)
    
    # Считаем среднее по осям X и Y для этих точек
    centroid = coords[idx].mean(axis=0)
    
    # Рисуем центроид (большой маркер с черной окантовкой для контраста)
    # plt.scatter(centroid[0], centroid[1], marker='X', s=250, edgecolor='black', linewidth=1.5, label=f'Класс {cls}')
    
    # Опционально: добавляем текстовую подпись прямо над крестиком
    plt.text(centroid[0], centroid[1] + 0.3, str(cls), fontsize=12, fontweight='bold', ha='center')

plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.title('UMAP-эмбеддинги ковариационных матриц с центроидами')

# Добавляем легенду для крестиков и colorbar для точек
plt.legend()
plt.colorbar(scatter, label='Метки классов')
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
    def __init__(self, n_filters, epsilon=1e-9, off_diag_penalty=0.0, **kwargs):
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
Npatt = 21

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
    metric="precomputed", n_neighbors=20, verbose=True,
    keras_fit_kwargs={"callbacks": [early_stopping], "verbose": 1}
)

embedder.n_training_epochs = 3 
embedder.loss_report_frequency = embedder.n_training_epochs * 100
embedder.fit(X_cov_flat, precomputed_distances=dists_init)

# %%
from scipy.linalg import null_space

n_iterations = 7
Npatt = 3

W_bimap_list = [] 
history_list = []

# Списки для сохранения результатов каждой итерации
covmats_reduced_list = []  
# dists_list = []            

for it in range(n_iterations):
    print(f"\n{'='*50}\n ИТЕРАЦИЯ {it + 1} ИЗ {n_iterations}\n{'='*50}")
    
    # 1. ПОДГОТОВКА ПОДПРОСТРАНСТВА И РАССТОЯНИЙ
    if it == 0:
        # Пропускаем лишние вычисления: берем уже готовые исходные данные
        V = np.eye(n_components)
        covmats_reduced = covmats_ssd.copy()
        dists_current = dists_init.copy()
    else:
        # Строим базис нуль-пространства из всех ранее найденных фильтров
        W_stacked = np.vstack(W_bimap_list)
        V = null_space(W_stacked).T
        M_current = V.shape[0]
        
        # Проецируем ковариации в меньшую размерность
        covmats_reduced = np.zeros((len(covmats_ssd), M_current, M_current))
        for i in range(len(covmats_ssd)):
            covmats_reduced[i] = V @ covmats_ssd[i] @ V.T

        # Идеальный перерасчет топологии без регуляризации
        # dists_current = pairwise_distance(covmats_reduced, metric='riemann')
        # dists_current = pairwise_distance(covmats_reduced, metric='riemann')
    
    # Сохраняем спроецированные матрицы и их дистанции в списки
    covmats_reduced_list.append(covmats_reduced.copy())
    # dists_list.append(dists_current.copy())

    M_current = V.shape[0]
    
    # 2. ВЕКТОРИЗАЦИЯ УМЕНЬШЕННЫХ КОВАРИАЦИЙ
    idx_i, idx_j = np.triu_indices(M_current)
    multipliers = np.where(idx_i == idx_j, 1.0, np.sqrt(2.0)).astype(np.float32)
    X_cov_flat = (covmats_reduced[:, idx_i, idx_j] * multipliers).astype(np.float32)
    input_dim = int(X_cov_flat.shape[1])

    # 3. ИНИЦИАЛИЗАЦИЯ И ОБУЧЕНИЕ МОДЕЛИ
    tf.keras.backend.clear_session() # Очищаем память
    
    inputs_enc = tf.keras.Input(shape=(input_dim,), dtype=tf.float32)
    x = UnflattenSymmetricLayer(M=M_current)(inputs_enc)
    layer_name = f"bimap_{it}"
    x = BiMapLayer(d_out=Npatt, name=layer_name)(x)
    z_latent = LogDiagScaleLayer(n_filters=Npatt, epsilon=1e-9)(x)
    encoder = tf.keras.Model(inputs=inputs_enc, outputs=z_latent)

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=50, min_delta=1e-4)
    embedder = ParametricUMAP(
        encoder=encoder, n_components=Npatt, dims=(input_dim,), 
        metric="precomputed", n_neighbors=20, verbose=True,
        keras_fit_kwargs={"callbacks": [early_stopping], "verbose": 1}
    )

    embedder.n_training_epochs = 3
    embedder.loss_report_frequency = embedder.n_training_epochs * 100

    embedder.fit(X_cov_flat, precomputed_distances=dists_current)
    history_list.append(embedder._history)
    
    # 4. ИЗВЛЕЧЕНИЕ ФИЛЬТРА И ВОЗВРАТ В ИСХОДНОЕ ПРОСТРАНСТВО
    # w_reduced имеет размер (1, M_current)
    w_reduced = encoder.get_layer(layer_name).get_weights()[0]
    
    # Проецируем обратно в исходное отбеленное пространство: (1, M_current) @ (M_current, M) -> (1, M)
    w_orig = w_reduced @ V
    
    # Нормализуем
    w_orig = w_orig / np.linalg.norm(w_orig, axis=1, keepdims=True)
    W_bimap_list.append(w_orig)

# %%
# =============================================================================
# ПОСТПРОЦЕССИНГ 
# =============================================================================
W_bimap_stacked = np.vstack(W_bimap_list)
W_sorted = W_ssd @ W_bimap_stacked.T

# Дальнейший код для мощностей, сортировки и отрисовки...

# 2. РАСЧЕТ ИСТИННЫХ МОЩНОСТЕЙ ИСТОЧНИКОВ
powers = np.zeros((len(covmats_ssd), n_iterations * Npatt))
for i, C in enumerate(covmats_ssd):
    # Применяем матрицу ко всем ИСХОДНЫМ отбеленным ковариациям
    C_filtered = W_bimap_stacked @ C @ W_bimap_stacked.T
    powers[i] = np.diag(C_filtered)

# Расчет паттернов: A = Sigma_mean * W
A_sensor_raw = np.mean(covmats_band, axis=0) @ W_sorted

# Нормировка физических паттернов для визуализации
scales = np.linalg.norm(A_sensor_raw, axis=0)
A_sorted = A_sensor_raw / scales

found_filters = [W_sorted[:, i] for i in range(n_iterations * Npatt)]
found_patterns = [A_sorted[:, i] for i in range(n_iterations * Npatt)]

# %%
# =============================================================================
# ВИЗУАЛИЗАЦИЯ (ТОЛЬКО ВАЛИДНЫЕ ОШИБКИ)
# =============================================================================
import matplotlib.pyplot as plt

# Создаем фигуру с двумя подграфиками (для Recon и для UMAP)
fig, ax1 = plt.subplots(1, 1)
# --- График 2: Истинный лосс графа UMAP ---
for idx, hist in enumerate(history_list):
    ax1.plot(hist['umap_loss'], label=idx+1, linewidth=2)
    
ax1.set_title('Сохранение топологии')
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
    decoder=decoder_2d,
    n_components=2,
    parametric_reconstruction=True, 
    parametric_reconstruction_loss_fcn=tf.keras.losses.MeanSquaredError(),
    verbose=True
)

pred = encoder.predict(X_cov_flat)
umap_coords = reducer_2d_nn.fit_transform(pred)
print("Визуальное пространство обучено!")

# %%
plt.scatter(umap_coords[:,0], umap_coords[:,1])

# %%
umap_coords = coords 

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
import matplotlib.pyplot as plt
import numpy as np

# Вычисляем матрицу ковариации один раз, чтобы использовать для графиков и текста
cov_matrix = np.cov(found_filters @ signal_concatenated)

fig, ax = plt.subplots(figsize=(6, 6)) # Увеличим размер, чтобы цифры поместились
im = ax.imshow(cov_matrix, cmap='viridis')

# Добавляем colorbar
plt.colorbar(im)

# Добавляем числа в каждую ячейку

rows, cols = cov_matrix.shape
for i in range(rows):
    for j in range(cols):
        value = cov_matrix[i, j]
        
        # Форматирование: '.2f' для обычных чисел (0.12), '.2e' если порядок очень мал (1.23e-05)
        text_label = f"{value:.2f}" 
        
        # Меняем цвет текста в зависимости от яркости ячейки, чтобы его было видно
        # (выбираем белый для темных ячеек, черный для светлых)
        text_color = "white" if im.norm(value) < 0.5 else "black"
        
        ax.text(j, i, text_label,
                ha="center", va="center", 
                color=text_color, fontsize=9)

plt.title("Матрица ковариации с числовыми значениями")
plt.show()

# %%
def format_umap_axes(ax):
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(axis='both', which='both', length=0)
    ax.grid(True, linestyle='--', alpha=0.5, zorder=0)

# =============================================================================
# ВЫБОР КОМПОНЕНТЫ
# =============================================================================
# Выбираем индекс паттерна (0..Npatt-1)
comp_idx = 20 

# Берём фильтр, который ВЫУЧИЛА НЕЙРОСЕТЬ (из матрицы W)
W_sensor = found_filters[comp_idx] 
# Берём соответствующий ему паттерн (из матрицы A = W^+)
A_pattern = found_patterns[comp_idx]

# 1. Мощность выбранного выученного источника (латентная переменная сети)
p_vals = powers[:, comp_idx]

# 2. Мощность через ручное применение пространственного фильтра (w^T * C * w)
# ВАЖНО: используем исходные ковариации (covmats), чтобы проверить честность сети
filtered_powers = np.array([W_sensor.T @ C @ W_sensor for C in covmats_band])

# 3. Нормализация (делим на std, без вычета среднего, чтобы сохранить масштаб)
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
ax_filt.set_title('Пространственный фильтр')

# --- 2. ТОПОГРАФИЯ ПАТТЕРНА (A) ---
ax_patt = fig.add_subplot(gs[1, 0])
mne.viz.plot_topomap(A_pattern, raw.info, axes=ax_patt, show=False)
ax_patt.set_title('Пространственный паттерн')

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
ax_env.plot(window_idx, filtered_powers_norm, color='gray', lw=2.5, alpha=0.5, label='$w^T C w$', zorder=2)
ax_env.plot(window_idx, p_vals_norm, color='black', lw=1.2, label='Латентная мощность энкодера ($z$)', zorder=3)

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

plt.suptitle(f'Анализ компоненты {comp_idx+1} из {Npatt}', fontsize=16)
plt.tight_layout()
plt.show()

# %%
import matplotlib.pyplot as plt
import numpy as np
import mne

# =============================================================================
# ВИЗУАЛИЗАЦИЯ: ПАТТЕРНЫ -> ИСХОДНЫЙ UMAP КОВАРИАЦИЙ
# =============================================================================
# Используем классический UMAP (coords), полученный из pairwise_distance
# Npatt берем из обученной модели

fig, axes = plt.subplots(2, Npatt, figsize=(5 * Npatt, 8))

# Защита размерности, если Npatt == 1
if Npatt == 1:
    axes = np.expand_dims(axes, axis=1)

for i in range(Npatt):
    # 1. Извлекаем паттерн (для отрисовки) и фильтр (для расчета мощности)
    w = W_sorted[:, i] # Фильтр в пространстве сенсоров
    a = A_sorted[:, i] # Паттерн в пространстве сенсоров
    
    # 2. Вычисляем честную мощность для исходных ковариационных матриц
    # Применяем выученный фильтр напрямую к сырым ковариациям эпох
    p_real = np.array([w.T @ C @ w for C in covmats_band])
    
    # 3. ВЕРХНИЙ РЯД: Топомапа паттернов
    ax_topo = axes[0, i]
    mne.viz.plot_topomap(a, raw.info, axes=ax_topo, show=False)
    ax_topo.set_title(f'Паттерн {i+1}', fontweight='bold', fontsize=12)
    
    # 4. НИЖНИЙ РЯД: Оригинальный UMAP (coords), раскрашенный по мощности
    ax_scatter = axes[1, i]
    
    # Срезаем 5% экстремальных значений, чтобы выбросы не портили цветовую шкалу
    vmin = np.percentile(p_real, 5)
    vmax = np.percentile(p_real, 95)
    
    # Отрисовываем первые две оси оригинального графа (coords[:, 0], coords[:, 1])
    sc = ax_scatter.scatter(
        coords[:, 0], coords[:, 1], 
        c=p_real, cmap='plasma', s=15, alpha=0.8,
        vmin=vmin, vmax=vmax, edgecolors='none'
    )
    
    ax_scatter.set_title(f'UMAP исходных матриц\nМощность $w^T C w$', fontsize=11)
    ax_scatter.set_xlabel('UMAP 1')
    ax_scatter.set_ylabel('UMAP 2')
    ax_scatter.grid(True, linestyle='--', alpha=0.3)
    
    # Добавляем цветовые шкалы к каждому графику рассеяния
    cbar = fig.colorbar(sc, ax=ax_scatter, orientation='horizontal', pad=0.15)
    cbar.set_label('Абсолютная мощность', fontsize=9)

plt.suptitle('Связь исходного риманова многообразия с выделенными источниками', 
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

# %%
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

# =============================================================================
# РУЧНОЙ ВЫБОР КОМПОНЕНТ (индексы от 0 до Npatt - 1)
# =============================================================================
# Укажите 3 компоненты для визуализации
selected_comps = [0, 1, 2] 

if len(selected_comps) != 3:
    raise ValueError("Для этого графика нужно выбрать ровно 3 компоненты!")

c1, c2, c3 = selected_comps

# =============================================================================
# ПОДГОТОВКА ЦВЕТОВ
# =============================================================================
unique_classes = []
for lab in np.unique(window_labels):
    if lab not in unique_classes:
        unique_classes.append(lab)

cmap_classes = plt.cm.tab20
color_map = {lab: cmap_classes(i / max(1, len(unique_classes) - 1)) 
             for i, lab in enumerate(unique_classes)}

# =============================================================================
# НАСТРОЙКА СЕТКИ ГРАФИКОВ (6 строк, 3 столбца)
# =============================================================================
fig = plt.figure(figsize=(16, 10))
# Левая колонка уже, центральная и правая - шире
gs = GridSpec(6, 3, figure=fig, width_ratios=[1, 2.5, 2.5], wspace=0.3, hspace=1.0)

# --- 1. ТОПОМАПЫ (ЛЕВЫЙ СТОЛБЕЦ) ---
# Паттерн 1 (строки 0-1)
ax_topo1 = fig.add_subplot(gs[0:2, 0])
mne.viz.plot_topomap(found_patterns[c1], raw.info, axes=ax_topo1, show=False)
ax_topo1.set_title(f'Pattern {c1+1}', fontweight='bold')

# Паттерн 2 (строки 2-3)
ax_topo2 = fig.add_subplot(gs[2:4, 0])
mne.viz.plot_topomap(found_patterns[c2], raw.info, axes=ax_topo2, show=False)
ax_topo2.set_title(f'Pattern {c2+1}', fontweight='bold')

# Паттерн 3 (строки 4-5)
ax_topo3 = fig.add_subplot(gs[4:6, 0])
mne.viz.plot_topomap(found_patterns[c3], raw.info, axes=ax_topo3, show=False)
ax_topo3.set_title(f'Pattern {c3+1}', fontweight='bold')

# --- 2. 3D ГРАФИК (ЦЕНТР-ВЕРХ) ---
ax3d = fig.add_subplot(gs[0:3, 1], projection='3d')
for lab in unique_classes:
    mask = window_labels == lab
    if np.any(mask):
        pts = powers[mask][:, selected_comps]
        ax3d.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                     c=[color_map[lab]], s=15, alpha=0.5, label=lab)

ax3d.set_xlabel(f'Pattern {c1+1}')
ax3d.set_ylabel(f'Pattern {c2+1}')
ax3d.set_zlabel(f'Pattern {c3+1}')
# Подгоняем угол обзора примерно как на картинке
ax3d.view_init(elev=20, azim=-45)

# --- 3. 2D ПРОЕКЦИИ ---
# Паттерн 1 vs Паттерн 2 (ПРАВЫЙ-ВЕРХ)
ax2d_12 = fig.add_subplot(gs[0:3, 2])
# Паттерн 1 vs Паттерн 3 (ЦЕНТР-НИЗ)
ax2d_13 = fig.add_subplot(gs[3:6, 1])
# Паттерн 2 vs Паттерн 3 (ПРАВЫЙ-НИЗ)
ax2d_23 = fig.add_subplot(gs[3:6, 2])

projections = [
    (0, 1, ax2d_12, f'Pattern {c1+1}', f'Pattern {c2+1}'),
    (0, 2, ax2d_13, f'Pattern {c1+1}', f'Pattern {c3+1}'),
    (1, 2, ax2d_23, f'Pattern {c2+1}', f'Pattern {c3+1}')
]

for idx_x, idx_y, ax, label_x, label_y in projections:
    for lab in unique_classes:
        mask = window_labels == lab
        if np.any(mask):
            pts = powers[mask][:, selected_comps]
            ax.scatter(pts[:, idx_x], pts[:, idx_y],
                       c=[color_map[lab]], s=10, alpha=0.5)
    
    ax.set_xlabel(label_x)
    ax.set_ylabel(label_y)
    ax.grid(True, linestyle='-', alpha=0.3)

plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
plt.show()

# %%
import matplotlib.pyplot as plt
import numpy as np
import itertools

# =============================================================================
# ПОДГОТОВКА ЦВЕТОВ
# =============================================================================
unique_classes = []
for lab in np.unique(window_labels):
    if lab not in unique_classes:
        unique_classes.append(lab)

cmap_classes = plt.cm.tab20
color_map = {lab: cmap_classes(i / max(1, len(unique_classes) - 1)) 
             for i, lab in enumerate(unique_classes)}

# =============================================================================
# ГЕНЕРАЦИЯ КОМБИНАЦИЙ И НАСТРОЙКА СЕТОК
# =============================================================================
# Получаем все уникальные пары осей для 2D проекций
combinations = list(itertools.combinations(range(Npatt), 2))
n_combs = len(combinations)

# Определяем размер сетки для графиков рассеяния (например, 2 строки х 3 столбца для 6 графиков)
n_cols = 3 if n_combs > 3 else n_combs
n_rows = int(np.ceil(n_combs / n_cols))

# Размер фигуры подстраивается под количество паттернов и графиков
fig = plt.figure(figsize=(18, max(8, Npatt * 2.5)))

# Две независимые зоны: слева (топомапы) и справа (2D проекции)
gs_left = fig.add_gridspec(nrows=Npatt, ncols=1, left=0.02, right=0.18, hspace=0.3)
gs_right = fig.add_gridspec(nrows=n_rows, ncols=n_cols, left=0.25, right=0.98, wspace=0.25, hspace=0.3)

# --- 1. ТОПОМАПЫ (СЛЕВА) ---
for i in range(Npatt):
    ax_topo = fig.add_subplot(gs_left[i, 0])
    mne.viz.plot_topomap(found_patterns[i], raw.info, axes=ax_topo, show=False)
    ax_topo.set_title(f'Pattern {i+1}', fontweight='bold')

# --- 2. 2D ПРОЕКЦИИ (СПРАВА) ---
for idx, (comp_x, comp_y) in enumerate(combinations):
    row = idx // n_cols
    col = idx % n_cols
    
    ax2d = fig.add_subplot(gs_right[row, col])
    
    for lab in unique_classes:
        mask = window_labels == lab
        if np.any(mask):
            pts = powers[mask]
            # Добавляем label только на первом графике, чтобы не дублировать легенду
            ax2d.scatter(
                pts[:, comp_x], pts[:, comp_y],
                c=[color_map[lab]], s=15, alpha=0.5, 
                label=lab if idx == 0 else ""
            )
    
    ax2d.set_xlabel(f'Pattern {comp_x+1}')
    ax2d.set_ylabel(f'Pattern {comp_y+1}')
    ax2d.grid(True, linestyle='--', alpha=0.3)

# =============================================================================
# ОБЩАЯ ЛЕГЕНДА
# =============================================================================
# Вытаскиваем легенду из первого графика и располагаем её в самом низу
handles, labels = ax2d.get_legend_handles_labels()
if handles:
    fig.legend(handles, labels, loc='lower center', ncol=len(unique_classes)//2, 
               bbox_to_anchor=(0.5, 0.01), fontsize=10, markerscale=2)

# Добавляем отступ снизу, чтобы легенда не перекрывала графики
plt.subplots_adjust(bottom=0.12)
plt.suptitle(f'Матрица 2D-проекций {Npatt}-мерного латентного пространства', fontsize=16, y=0.98, fontweight='bold')
plt.show()

# %%
import matplotlib.pyplot as plt
import numpy as np
import itertools

# =============================================================================
# ПОДГОТОВКА ЦВЕТОВ И НУМЕРАЦИЯ КЛАССОВ
# =============================================================================
unique_classes = []
for lab in np.unique(window_labels):
    if lab not in unique_classes:
        unique_classes.append(lab)

cmap_classes = plt.cm.tab20
color_map = {lab: cmap_classes(i / max(1, len(unique_classes) - 1)) 
             for i, lab in enumerate(unique_classes)}

# Создаем словарь для маппинга "Имя класса" -> "Номер"
class_to_num = {lab: str(i + 1) for i, lab in enumerate(unique_classes)}

# =============================================================================
# ГЕНЕРАЦИЯ КОМБИНАЦИЙ И НАСТРОЙКА СЕТОК
# =============================================================================
# Получаем все уникальные пары осей для 2D проекций
combinations = list(itertools.combinations(range(Npatt), 2))
n_combs = len(combinations)

# Определяем размер сетки для графиков рассеяния
n_cols = 3 if n_combs > 3 else n_combs
n_rows = int(np.ceil(n_combs / n_cols))

# Размер фигуры
fig = plt.figure(figsize=(18, max(8, Npatt * 2.5)))

# Две независимые зоны: слева (топомапы) и справа (2D проекции)
gs_left = fig.add_gridspec(nrows=Npatt, ncols=1, left=0.02, right=0.18, hspace=0.3)
gs_right = fig.add_gridspec(nrows=n_rows, ncols=n_cols, left=0.25, right=0.98, wspace=0.25, hspace=0.3)

# --- 1. ТОПОМАПЫ (СЛЕВА) ---
for i in range(Npatt):
    ax_topo = fig.add_subplot(gs_left[i, 0])
    mne.viz.plot_topomap(found_patterns[i], raw.info, axes=ax_topo, show=False)
    ax_topo.set_title(f'Pattern {i+1}', fontweight='bold')

# --- 2. 2D ПРОЕКЦИИ И ЦЕНТРОИДЫ (СПРАВА) ---
legend_handles = []
legend_labels = []

for idx, (comp_x, comp_y) in enumerate(combinations):
    row = idx // n_cols
    col = idx % n_cols
    
    ax2d = fig.add_subplot(gs_right[row, col])
    
    for lab in unique_classes:
        mask = window_labels == lab
        if np.any(mask):
            pts = powers[mask]
            
            # Формируем подпись с номером для легенды
            lab_with_num = f"{class_to_num[lab]}. {lab}"
            
            # Отрисовка облака точек
            sc = ax2d.scatter(
                pts[:, comp_x], pts[:, comp_y],
                c=[color_map[lab]], s=15, alpha=0.5, 
                label=lab_with_num if idx == 0 else ""
            )
            
            # Собираем данные для легенды только один раз
            if idx == 0:
                legend_handles.append(sc)
                legend_labels.append(lab_with_num)
            
            # --- Расчет и отрисовка центроидов ---
            cx = np.mean(pts[:, comp_x])
            cy = np.mean(pts[:, comp_y])
            
            # Размещаем цифру в центре кластера (с белым фоном для читаемости)
            ax2d.text(
                cx, cy, class_to_num[lab],
                color='black', fontsize=10, fontweight='bold',
                ha='center', va='center',
                bbox=dict(boxstyle='circle,pad=0.2', facecolor='white', edgecolor=color_map[lab], alpha=0.85)
            )
    
    ax2d.set_xlabel(f'Pattern {comp_x+1}')
    ax2d.set_ylabel(f'Pattern {comp_y+1}')
    ax2d.grid(True, linestyle='--', alpha=0.3)

# =============================================================================
# ОБЩАЯ ЛЕГЕНДА
# =============================================================================
if legend_handles:
    # Динамический расчет количества колонок для легенды
    n_legend_cols = max(1, len(unique_classes) // 2)
    fig.legend(
        legend_handles, legend_labels, 
        loc='lower center', ncol=n_legend_cols, 
        bbox_to_anchor=(0.5, 0.01), fontsize=11, markerscale=2
    )

# Увеличиваем нижний отступ, чтобы легенда поместилась
plt.subplots_adjust(bottom=0.15)
plt.suptitle(f'Матрица 2D-проекций {Npatt}-мерного латентного пространства', fontsize=16, y=0.98, fontweight='bold')
plt.show()

# %%
# =============================================================================
# 5. ГРАФИК: СРЕДНЯЯ МОЩНОСТЬ ПО КОМПОНЕНТАМ С ОТКЛОНЕНИЯМИ
# =============================================================================
import matplotlib.pyplot as plt

# powers уже отсортированы по убыванию среднего (см. блок выше)
mean_power = np.mean(powers, axis=0)
std_power  = np.std(powers, axis=0)

component_idx = np.arange(1, Npatt + 1)

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
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patheffects as pe

# 1. Собираем уникальные классы в правильном порядке
unique_classes = []
for lab in cond_descriptions:
    if lab in window_labels and lab not in unique_classes:
        unique_classes.append(lab)
# На случай, если в window_labels есть что-то, чего нет в cond_descriptions
for lab in np.unique(window_labels):
    if lab not in unique_classes:
        unique_classes.append(lab)

# 2. Настройка цветов (используем tab20 для большого количества классов)
cmap_classes = plt.cm.tab20
if len(unique_classes) == 1:
    color_map = {unique_classes[0]: cmap_classes(0)}
else:
    color_map = {
        lab: cmap_classes(i / (len(unique_classes) - 1))
        for i, lab in enumerate(unique_classes)
    }

# 3. Визуализация 3D
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(projection='3d')

for lab in unique_classes:
    # Фильтруем точки по реальному массиву меток
    mask = window_labels == lab
    
    # Если окон с таким лейблом нет, пропускаем
    if not np.any(mask):
        continue
        
    cluster_points = powers[mask,:3]
    color = color_map[lab]
    
    # Отрисовка облака точек (первые 3 паттерна)
    ax.scatter(
        cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2],
        c=[color], marker='o', s=35, alpha=0.5,
        edgecolors='white', linewidths=0.3, label=lab
    )
    
    # Расчет координат центроида
    cx, cy, cz = np.mean(cluster_points, axis=0)
    
ax.set_xlabel('Pattern 1')
ax.set_ylabel('Pattern 2')
ax.set_zlabel('Pattern 3')
ax.set_title('3D Фазовое пространство мощностей (Топ-3 паттерна)')

# Легенда в 2 колонки, чтобы поместились все 19 условий
# ax.legend(
#     loc='center left', bbox_to_anchor=(1.05, 0.5),
#     fontsize=10, title="Условия", title_fontsize=12,
#     ncol=2, markerscale=1.5
# )

ax.legend()
plt.tight_layout()
plt.show()

# %%

