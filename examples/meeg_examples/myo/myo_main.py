import numpy as np
import matplotlib.pyplot as plt
import mne
import umap

import os
import sys
lib_directory = os.path.abspath("C:/Users/ansbel/Documents/GitHub/pyRiemann") 
if lib_directory not in sys.path:
    sys.path.insert(0, lib_directory)
from pyriemann.estimation import Covariances
from scipy.interpolate import griddata
from scipy.stats import pearsonr
import tensorflow as tf
from umap.parametric_umap import ParametricUMAP

# %%
# =====================================================================
# 1. ИНИЦИАЛИЗАЦИЯ И ЗАГРУЗКА ДАННЫХ (MNE Pipeline)
# =====================================================================
Fs = 500
Wsize = 0.5
Ssize = 0.1
overlap = Wsize - Ssize  # Шаг окна определяется через перекрытие
base_path = r'C:\Users\ansbel\Documents\GitHub\myo-tools\data\\'

epochs_list = []
file_labels_list = []

raws = []
for file_idx in range(10): # В MATLAB было for file_idx = 0
    filename = f"{base_path}{file_idx:04d}.npz"
    print(f"Чтение файла: {file_idx:04d}.npz...")
    
    # Загрузка
    npz_file = np.load(filename)
    data_myo = npz_file['data_myo'].astype(float)
    
    # MNE ожидает формат [Channels x Samples]
    if data_myo.shape[0] > data_myo.shape[1]:
        data_myo = data_myo.T
        
    num_channels = data_myo.shape[0]
    
    # Создание объекта Raw
    info = mne.create_info(ch_names=[str(i) for i in range(num_channels)], 
                           sfreq=Fs, ch_types='eeg') # Можно использовать 'emg'
    raw = mne.io.RawArray(data_myo, info, verbose=False)
    raws.append(raw)
    
    # Нарезка на эпохи фиксированной длины
    epochs = mne.make_fixed_length_epochs(raw, duration=Wsize, overlap=overlap, 
                                          preload=True, verbose=False)
    
    # Получаем массив данных [n_epochs, n_channels, n_times]
    ep_data = epochs.get_data(copy=False)
    
    epochs_list.append(ep_data)
    file_labels_list.extend([file_idx] * len(epochs))

# Объединяем данные со всех файлов
X_epochs = np.concatenate(epochs_list, axis=0)
File_Labels = np.array(file_labels_list)

# %%
# =====================================================================
# 1.5. ВИЗУАЛИЗАЦИЯ СПЕКТРОВ (PSD)
# =====================================================================
# Создаем сетку 2x5 для 10 файлов
fig, axes = plt.subplots(2, 5, figsize=(20, 8), facecolor='w')
axes = axes.flatten()

for i, raw in enumerate(raws):
    # Вычисляем PSD. n_fft=int(Fs) дает разрешение в 1 Гц.
    # Ограничим fmax=100 (или Fs/2), чтобы детальнее рассмотреть рабочий диапазон
    spectrum = raw.compute_psd(fmax=150, n_fft=int(Fs), verbose=False)
    
    # Строим график. average=True усреднит спектр по всем каналам для наглядности
    spectrum.plot(axes=axes[i], average=True, show=False)
    
    axes[i].set_title(f'Файл: {i:04d}.npz')
    axes[i].set_xlabel('Частота (Гц)')
    if i % 5 != 0:
        axes[i].set_ylabel('') # Убираем лишние подписи Y для чистоты

plt.tight_layout()
plt.show()

# %%
raw.plot(scalings='auto')

# %%
# =====================================================================
# 2. РАСЧЕТ КОВАРИАЦИОННЫХ МАТРИЦ И КАСАТЕЛЬНОГО ПРОСТРАНСТВА
# =====================================================================
from pyriemann.geometry.distance import pairwise_distance

print(f"Собрано эпох: {X_epochs.shape[0]}")

# PyRiemann для расчета ковариаций и проекции в Tangent Space
Covs = Covariances(estimator='scm').fit_transform(X_epochs / np.std(X_epochs))
dist_matrix_init = pairwise_distance(Covs, metric='riemann')

# %%
import matplotlib.pyplot as plt
import umap

reducer = umap.UMAP(n_components=2, n_neighbors=20, metric='precomputed')
coords = reducer.fit_transform(dist_matrix_init)

# 2. Строим график рассеяния (Scatter Plot)
plt.figure(figsize=(8, 6))
plt.scatter(coords[:, 0], coords[:, 1], s=5, cmap='Spectral')

plt.title('UMAP проекция матрицы расстояний')
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.colorbar() 
plt.show()

# %%
import matplotlib.pyplot as plt
import umap
from umap.parametric_umap import ParametricUMAP
import tensorflow as tf

# =============================================================================
# 1. КЛАССИЧЕСКИЙ UMAP
# =============================================================================
print("Обучение классического UMAP...")
reducer_standard = umap.UMAP(n_components=2, n_neighbors=20, metric='precomputed')
coords_standard = reducer_standard.fit_transform(dist_matrix_init)

# =============================================================================
# 2. ПАРАМЕТРИЧЕСКИЙ UMAP (ЧИСТЫЙ ЭНКОДЕР)
# =============================================================================
print("Обучение параметрического UMAP (без декодера)...")

Covs_flat_full = Covs.reshape(Covs.shape[0], -1).astype(np.float32)
input_dim = int(Covs_flat_full.shape[1])

# Создаем легкий энкодер для проекции в 2D
encoder_only = tf.keras.Sequential([
    tf.keras.Input(shape=(int(input_dim),), dtype=tf.float32),
    tf.keras.layers.Dense(256, activation="relu"), 
    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.Dense(2, activation="linear", name="z_2d")
])

embedder_parametric = ParametricUMAP(
    encoder=encoder_only,
    dims=(input_dim,),
    n_components=2,
    n_neighbors=20,
    metric="precomputed",
    parametric_reconstruction=False,  
    autoencoder_loss=False,
    verbose=True
)

# Обучаем только на UMAP-графе (без кастомных лоссов и ранней остановки по декодеру)
embedder_parametric.fit(Covs_flat_full, precomputed_distances=dist_matrix_init)

# Получаем координаты для всех данных
coords_parametric = embedder_parametric.transform(Covs_flat_full)

# =============================================================================
# 3. ВИЗУАЛЬНОЕ СРАВНЕНИЕ
# =============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), facecolor='w')

# График 1: Стандартный UMAP
sc1 = ax1.scatter(coords_standard[:, 0], coords_standard[:, 1], 
                  c=File_Labels, cmap='jet', s=10, alpha=0.7, edgecolors='none')
ax1.set_title('Классический UMAP (Непараметрический)', fontsize=14)
ax1.set_xlabel('UMAP 1')
ax1.set_ylabel('UMAP 2')
ax1.grid(True, linestyle='--', alpha=0.5)

# График 2: Параметрический UMAP
sc2 = ax2.scatter(coords_parametric[:, 0], coords_parametric[:, 1], 
                  c=File_Labels, cmap='jet', s=10, alpha=0.7, edgecolors='none')
ax2.set_title('Parametric UMAP (Только нейросеть-энкодер)', fontsize=14)
ax2.set_xlabel('UMAP 1')
ax2.set_ylabel('UMAP 2')
ax2.grid(True, linestyle='--', alpha=0.5)

# Добавляем общий Colorbar
cbar = fig.colorbar(sc2, ax=[ax1, ax2], fraction=0.02, pad=0.04)
cbar.set_label('Номер класса (движения)')

plt.suptitle('Сравнение формирования латентного пространства', fontsize=16)
plt.show()

# %%
import numpy as np
import tensorflow as tf

n_ch_white = Covs.shape[1]
M = n_ch_white
K = M * (M + 1) // 2  # Количество уникальных элементов

# =============================================================================
# МАТРИЦЫ ПРЕОБРАЗОВАНИЯ (ПОЛНАЯ <-> ТРЕУГОЛЬНАЯ)
# =============================================================================
# Получаем индексы верхнего треугольника
i_idx, j_idx = np.triu_indices(M)

# T_to_tri: проецирует [M*M] -> [K] (извлекает верхний треугольник)
T_to_tri_np = np.zeros((M * M, K), dtype=np.float32)
# T_to_sym: проецирует [K] -> [M*M] (восстанавливает симметричную матрицу)
T_to_sym_np = np.zeros((K, M * M), dtype=np.float32)

for k, (r, c) in enumerate(zip(i_idx, j_idx)):
    T_to_tri_np[r * M + c, k] = 1.0
    T_to_sym_np[k, r * M + c] = 1.0
    if r != c:  # Симметричное отражение для внедиагональных элементов
        T_to_sym_np[k, c * M + r] = 1.0

# Переводим в константы TF для использования внутри графа
T_to_tri_tf = tf.constant(T_to_tri_np)
T_to_sym_tf = tf.constant(T_to_sym_np)

# =============================================================================
# ПОДГОТОВКА ДАННЫХ
# =============================================================================
# Разворачиваем полностью, а затем матричным умножением оставляем только 21 признак
Covs_flat_full = Covs.reshape(Covs.shape[0], -1).astype(np.float32)
X_cov_flat = Covs_flat_full @ T_to_tri_np  # Размерность станет [n_epochs, K]
input_dim = int(X_cov_flat.shape[1])

print(f"Размерность входа снижена с {M*M} до {input_dim} признаков.")

# =============================================================================
# ДЕКОДЕР С ОБУЧАЕМЫМ КОРРЕЛИРОВАННЫМ ШУМОМ
# =============================================================================
class SpatialPatternDecoder(tf.keras.layers.Layer):
    def __init__(self, M_channels, N_patterns, **kwargs):
        super().__init__(**kwargs)
        self.M_channels = int(M_channels)
        self.N_patterns = int(N_patterns)
        
    def build(self, input_shape):
        # 1. Обучаемые паттерны (матрица A)
        self.A = self.add_weight(
            shape=(self.M_channels, self.N_patterns),
            initializer=tf.keras.initializers.RandomNormal(stddev=0.1),
            trainable=True,
            name="A_patterns"
        )
        
        # 2. Обучаемая матрица для выявления кросс-корреляций в шуме (W)
        self.noise_corr = self.add_weight(
            shape=(self.M_channels, self.M_channels),
            initializer=tf.keras.initializers.RandomNormal(stddev=0.05),
            trainable=True,
            name="sensor_noise_corr"
        )
        
        # 3. Диагональный шум для математической стабильности (D)
        self.noise_log = self.add_weight(
            shape=(self.M_channels,),
            initializer=tf.keras.initializers.Constant(-3.0),
            trainable=True,
            name="sensor_noise_diag"
        )

    def call(self, z):
        # Нормализуем столбцы A
        A_norm = tf.math.l2_normalize(self.A, axis=0)
        
        P = tf.exp(z)
        
        # Реконструкция полезного сигнала: A * P * A^T
        C_signal = tf.einsum("mf,bf,lf->bml", A_norm, P, A_norm)
        
        # Формируем матрицу коррелированного шума: W * W^T
        # Это гарантирует симметричность и неотрицательную определенность
        C_noise_corr = tf.matmul(self.noise_corr, self.noise_corr, transpose_b=True)
        
        # Формируем диагональный шум (предотвращает вырождение матрицы)
        noise_variance = tf.exp(self.noise_log)
        C_noise_diag = tf.linalg.diag(noise_variance)
        
        # Итоговая матрица шума: одинаковая для всего батча
        C_noise_total = C_noise_corr + C_noise_diag
        
        # Итоговая ковариация: Сигнал + Коррелированный Шум
        C_recon = C_signal + C_noise_total
        
        # Сплющиваем и извлекаем только треугольную часть для лосса/UMAP
        C_recon_flat = tf.reshape(C_recon, [-1, self.M_channels * self.M_channels])
        C_recon_tri = tf.matmul(C_recon_flat, T_to_tri_tf)
        
        return C_recon_tri

def riemannian_distance_loss(y_true_tri, y_pred_tri):
    y_true_tri = tf.cast(y_true_tri, tf.float32)
    y_pred_tri = tf.cast(y_pred_tri, tf.float32)

    # НОВОВВЕДЕНИЕ: Восстанавливаем полные симметричные матрицы из усеченных векторов
    C_true_flat = tf.matmul(y_true_tri, T_to_sym_tf)
    C_pred_flat = tf.matmul(y_pred_tri, T_to_sym_tf)

    C_true = tf.reshape(C_true_flat, [-1, M, M])
    C_pred = tf.reshape(C_pred_flat, [-1, M, M])
    
    smoothing_factor = 1e-4  
    eye_M = tf.eye(M, dtype=tf.float32)
    
    C_true_reg = C_true + smoothing_factor * eye_M
    C_pred_reg = C_pred + smoothing_factor * eye_M

    eps = 1e-7

    # Шаг 1: C_true^{-1/2}
    eigvals_true, eigvecs_true = tf.linalg.eigh(C_true_reg)
    inv_sqrt_eigvals = 1.0 / tf.sqrt(tf.maximum(eigvals_true, eps))
    C_true_invsqrt = tf.einsum('bij,bj,bkj->bik', eigvecs_true, inv_sqrt_eigvals, eigvecs_true)

    # Шаг 2: C_true^{-1/2} * C_pred * C_true^{-1/2}
    C_mid = tf.einsum('bij,bjk,bkl->bil', C_true_invsqrt, C_pred_reg, C_true_invsqrt)
    C_mid = 0.5 * (C_mid + tf.transpose(C_mid, perm=[0, 2, 1])) 

    # Шаг 3: Собственные значения полученной матрицы
    eigvals_mid, _ = tf.linalg.eigh(C_mid) 
    log_eigvals_mid = tf.math.log(tf.maximum(eigvals_mid, eps))

    # Шаг 4: Расчет дистанции
    dist_sq = tf.reduce_sum(tf.square(log_eigvals_mid), axis=1) / tf.cast(M * M, tf.float32)

    return tf.reduce_mean(dist_sq)

# =============================================================================
# ПОДГОТОВКА ДАННЫХ
# =============================================================================
N_patterns = int(9)  
N_dim = int(N_patterns)

print(f"Обучение ParametricUMAP: N_dim={N_dim}, N_patterns={N_patterns}")

# # =============================================================================
# # ЭНКОДЕР И ДЕКОДЕР
# # =============================================================================
# encoder = tf.keras.Sequential([
#     tf.keras.Input(shape=(int(input_dim),), dtype=tf.float32),
#     tf.keras.layers.Dense(512, activation="relu"), 
#     tf.keras.layers.Dense(512, activation="relu"),
#     tf.keras.layers.Dense(512, activation="relu"),
#     tf.keras.layers.Dense(int(N_dim), activation="linear", name="z_powers")
# ])

# decoder = tf.keras.Sequential([
#     tf.keras.Input(shape=(int(N_dim),), dtype=tf.float32),
#     SpatialPatternDecoder(
#         M_channels=int(n_ch_white),
#         N_patterns=int(N_patterns),
#         name="spatial_decoder"
#     ),
#     tf.keras.layers.Flatten()
# ])
# =============================================================================
# ЭНКОДЕР И ДЕКОДЕР
# =============================================================================
encoder = tf.keras.Sequential([
    tf.keras.Input(shape=(int(input_dim),), dtype=tf.float32),
    tf.keras.layers.Dense(512, activation="relu"), 
    tf.keras.layers.Dense(512, activation="relu"),
    tf.keras.layers.Dense(512, activation="relu"),
    # Выход энкодера - это "сырые" топологические координаты
    tf.keras.layers.Dense(int(N_dim), activation="linear", name="umap_topology_coords")
])

decoder = tf.keras.Sequential([
    tf.keras.Input(shape=(int(N_dim),), dtype=tf.float32),
    
    # НОВОВВЕДЕНИЕ: Слой аффинного выравнивания (Rotation, Scaling & Translation)
    # Позволяет декодеру повернуть и сдвинуть UMAP-пространство так, 
    # чтобы координаты идеально ложились на функцию softplus(z)
    tf.keras.layers.Dense(int(N_dim), activation="linear", use_bias=True, name="latent_alignment"),
    
    SpatialPatternDecoder(
        M_channels=int(n_ch_white),
        N_patterns=int(N_patterns),
        name="spatial_decoder"
    ),
    tf.keras.layers.Flatten()
])

from sklearn.model_selection import train_test_split

# =============================================================================
# РАЗДЕЛЕНИЕ НА ОБУЧАЮЩУЮ И ВАЛИДАЦИОННУЮ ВЫБОРКИ
# =============================================================================
# Индексы для сплита (откладываем 10% данных на валидацию)
indices = np.arange(len(X_cov_flat))
train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)

X_train = X_cov_flat[train_idx]
X_val = X_cov_flat[val_idx]

# Важно: для графа UMAP нужна матрица расстояний ТОЛЬКО между обучающими окнами
dist_matrix_train = dist_matrix_init[train_idx][:, train_idx]

print(f"Обучающая выборка: {len(X_train)} окон. Валидационная: {len(X_val)} окон.")

# =============================================================================
# ПОДГОТОВКА ДАННЫХ И КОЛЛБЕКОВ
# =============================================================================
n_neighbors = 20 
loss_weight = 1

class HonestEarlyStoppingCallback(tf.keras.callbacks.Callback):
    def __init__(self, X_train, X_val, loss_weight, patience=5):
        super().__init__()
        self.X_train = X_train
        self.X_val = X_val
        self.loss_weight = loss_weight  # Передаем вес реконструкции сюда
        self.patience = patience
        self.best_loss = np.inf
        self.wait = 0
        self.best_weights = None

    def _compute_honest_recon_loss(self, X_data):
        batch_sz = 256
        total_loss = 0.0
        n_batches = int(np.ceil(len(X_data) / batch_sz))
        
        for i in range(n_batches):
            X_batch = X_data[i * batch_sz : (i + 1) * batch_sz]
            z = self.model.encoder(X_batch, training=False)
            X_recon = self.model.decoder(z, training=False)
            
            batch_loss = riemannian_distance_loss(X_batch, X_recon).numpy()
            total_loss += batch_loss * len(X_batch)
            
        return total_loss / len(X_data)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        
        # 1. Считаем честную Риманову ошибку
        val_recon_loss = self._compute_honest_recon_loss(self.X_val)
        
        idx = np.random.choice(len(self.X_train), len(self.X_val), replace=False)
        train_recon_loss = self._compute_honest_recon_loss(self.X_train[idx])
        
        # 2. Вычисляем чистый UMAP loss из общего лосса Keras
        total_loss = logs.get('loss', 0.0)
        approx_umap_loss = total_loss - (self.loss_weight * train_recon_loss)
        
        # 3. Сохраняем в логи для графиков
        logs['honest_train_recon'] = train_recon_loss
        logs['honest_val_recon'] = val_recon_loss
        logs['umap_graph_loss'] = approx_umap_loss
        
        print(f"\n[Отчет {epoch+1}] Total Loss: {total_loss:.4f} | "
              f"UMAP Loss: {approx_umap_loss:.4f} | "
              f"Recon (Train: {train_recon_loss:.4f}, Val: {val_recon_loss:.4f})")
        
        # 4. Логика ранней остановки
        if val_recon_loss < self.best_loss:
            self.best_loss = val_recon_loss
            self.wait = 0
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                print(f"\n[EarlyStopping] Валидационный лосс реконструкции не улучшается {self.patience} шагов.")
                self.model.stop_training = True
                self.model.set_weights(self.best_weights)

# При инициализации не забудь передать loss_weight
keras_fit_args = {
    "callbacks": [HonestEarlyStoppingCallback(X_train, X_val, loss_weight=loss_weight, patience=15)]
}

# =============================================================================
# ИНИЦИАЛИЗАЦИЯ И ЗАПУСК PARAMETRIC UMAP
# =============================================================================
embedder = ParametricUMAP(
    encoder=encoder,
    decoder=decoder,
    n_components=N_dim, 
    dims=(input_dim,),
    metric="precomputed", 
    n_neighbors=n_neighbors,
    parametric_reconstruction=True,
    autoencoder_loss=False, 
    parametric_reconstruction_loss_fcn=riemannian_distance_loss, 
    parametric_reconstruction_loss_weight=loss_weight,
    reconstruction_validation=X_val, 
    keras_fit_kwargs=keras_fit_args, 
    verbose=True
)

embedder.n_training_epochs = 1      
embedder.loss_report_frequency = 200 * embedder.n_training_epochs

embedder.fit(X_train, precomputed_distances=dist_matrix_train)

# %%
# =============================================================================
# ВИЗУАЛИЗАЦИЯ (ТОЛЬКО ВАЛИДНЫЕ ОШИБКИ)
# =============================================================================
import matplotlib.pyplot as plt

history = embedder._history

# Создаем фигуру с двумя подграфиками (для Recon и для UMAP)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# --- График 1: Честные ошибки реконструкции (Риманово расстояние) ---
if 'honest_train_recon' in history and 'honest_val_recon' in history:
    ax1.plot(history['honest_train_recon'], label='Honest Train Recon Loss', color='blue', linewidth=2)
    ax1.plot(history['honest_val_recon'], label='Honest Val Recon Loss', color='green', linewidth=2)
    
ax1.set_title('Качество декодера: Ошибка ковариаций')
ax1.set_xlabel('Шаги оценки')
ax1.set_ylabel('Квадрат Риманова расстояния')
ax1.legend()
ax1.grid(True, linestyle='--', alpha=0.7)

# --- График 2: Истинный лосс графа UMAP ---
if 'umap_graph_loss' in history:
    ax2.plot(history['umap_graph_loss'], label='UMAP Graph Loss (Cross-Entropy)', color='purple', linewidth=2)
    
ax2.set_title('Качество проекции: Сходимость графа UMAP')
ax2.set_xlabel('Шаги оценки')
ax2.set_ylabel('Loss (Cross-Entropy)')
ax2.legend()
ax2.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
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
    tf.keras.layers.Dense(128, activation="elu"),
    tf.keras.layers.Dense(128, activation="elu"),
    tf.keras.layers.Dense(128, activation="elu"),
    tf.keras.layers.Dense(2, activation="linear", name="2d_coords")
])

# Декодер: с мягкой L2-регуляризацией для страховки от выбросов
decoder_2d = tf.keras.Sequential([
    tf.keras.layers.InputLayer(shape=(2,)),
    tf.keras.layers.Dense(128, activation="elu"),
    tf.keras.layers.Dense(128, activation="elu"),
    tf.keras.layers.Dense(128, activation="elu"),
    tf.keras.layers.Dense(N_dim, activation="linear", name="z_reconstruction")
])

reducer_2d_nn = ParametricUMAP(
    encoder=encoder_2d,
    decoder=decoder_2d,
    n_components=2,
    parametric_reconstruction=False, 
    autoencoder_loss=False, 
    parametric_reconstruction_loss_fcn=tf.keras.losses.MeanSquaredError(),
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
plt.xlabel('Главная компонента 1')
plt.ylabel('Главная компонента 2')
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

# %%
# =============================================================================
# 6. ИЗВЛЕЧЕНИЕ ПАРАМЕТРОВ И ОТРИСОВКА
# =============================================================================
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
from scipy.interpolate import griddata

# 1. Извлекаем сырые значения z (логарифмы мощностей) 
z_values = embedder.encoder.predict(X_cov_flat)   # Размерность: (n_windows, N_patterns)

# 2. Переводим в реальные мощности
# powers = np.exp(z_values)
z_raw = embedder.encoder.predict(X_cov_flat)

# 2. Прогоняем их через слои декодера ДО SpatialPatternDecoder, 
# чтобы учесть выравнивание (latent_alignment) и получить корректные z для мощностей
alignment_layer = embedder.decoder.get_layer("latent_alignment")
z_aligned = alignment_layer(z_raw).numpy() # Переводим в numpy массив

# 3. Переводим в реальные мощности (учитывая softplus, который зашит в spatial_decoder)
# Проще всего вычислить softplus явно, так как именно он переводит латентные переменные в положительные мощности P:
powers = np.exp(z_aligned) # Эквивалент tf.math.softplus(z_aligned) в numpy

# 3. Извлекаем матрицу паттернов и нормируем 
spatial_decoder_layer = embedder.decoder.get_layer("spatial_decoder")
A_learned_raw = spatial_decoder_layer.get_weights()[0]
A_global = A_learned_raw / np.linalg.norm(A_learned_raw, axis=0) 

# Готовые паттерны в пространстве сенсоров
found_patterns = [A_global[:, i] for i in range(N_patterns)]

# 4. Фильтры Хауфе (с Тихоновской регуляризацией)
C_global_mean = np.mean(Covs, axis=0)
alpha = 1e-4  
I = np.eye(C_global_mean.shape[0])
C_global_reg = C_global_mean + alpha * np.trace(C_global_mean) * I
C_global_inv = np.linalg.inv(C_global_reg)

found_filters = [C_global_inv @ A_global[:, i] for i in range(N_patterns)]

# %%
# =============================================================================
# ФУНКЦИЯ ДЛЯ ОТРИСОВКИ БРАСЛЕТА (ВМЕСТО MNE ТOПОМАП)
# =============================================================================
def plot_wrist_emg(ax, weights, title_str):
    num_ch = len(weights)
    angles = np.linspace(0, 2 * np.pi, num_ch, endpoint=False)
    x_pos, y_pos = np.cos(angles), np.sin(angles)
    
    xi, yi = np.linspace(-1.5, 1.5, 100), np.linspace(-1.5, 1.5, 100)
    Xg, Yg = np.meshgrid(xi, yi)
    
    # Добавляем фиктивный центр (среднее значение), чтобы интерполяция не падала
    points = np.column_stack((x_pos, y_pos))
    points = np.vstack((points, [0, 0])) 
    vals = np.append(weights, np.mean(weights))
    
    Vq = griddata(points, vals, (Xg, Yg), method='cubic')
    
    # Обрезаем радиус
    Rq = np.sqrt(Xg**2 + Yg**2)
    Vq[Rq > 1.3] = np.nan
    
    ax.pcolormesh(Xg, Yg, Vq, shading='gouraud', cmap='viridis')
    ax.contour(Xg, Yg, Vq, levels=5, colors='k', linewidths=0.5, alpha=0.5)
    ax.scatter(x_pos, y_pos, s=150, c=weights, cmap='viridis', edgecolors='k', linewidths=1.5)
    
    for e in range(len(x_pos)):
        ax.text(x_pos[e]*1.25, y_pos[e]*1.25, str(e+1), ha='center', va='center', fontweight='bold', fontsize=10)
                
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title_str, fontsize=12)

# =============================================================================
def format_umap_axes(ax):
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(axis='both', which='both', length=0)
    ax.grid(True, linestyle='--', alpha=0.5, zorder=0)

comp_idx = 0
W_sensor = found_filters[comp_idx]
A_pattern = found_patterns[comp_idx]

# 1. Мощность выбранного выученного источника
p_vals = powers[:, comp_idx]

# 2. Мощность через пространственный фильтр (w^T * C * w)
filtered_powers = np.array([W_sensor.T @ C @ W_sensor for C in Covs])

# 3. Нормализация (делим на std, без вычета среднего)
p_vals_norm = p_vals / np.std(p_vals)
filtered_powers_norm = filtered_powers / np.std(filtered_powers)

unique_labels_ordered = []
for lab in File_Labels:
    if lab not in unique_labels_ordered:
        unique_labels_ordered.append(lab)

cmap = mpl.colormaps['viridis']
label_to_color = {lab: cmap(i / len(unique_labels_ordered)) for i, lab in enumerate(unique_labels_ordered)}

fig = plt.figure(figsize=(16, 10))
gs = GridSpec(2, 2, figure=fig, width_ratios=[1, 2], height_ratios=[1, 1])

# Заменяем mne.viz.plot_topomap на plot_wrist_emg
ax_filt = fig.add_subplot(gs[0, 0])
plot_wrist_emg(ax_filt, W_sensor, 'Аппроксимация фильтра')

ax_patt = fig.add_subplot(gs[1, 0])
plot_wrist_emg(ax_patt, A_pattern, 'Паттерн')

# =============================================================================
ax_umap = fig.add_subplot(gs[0, 1])

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
# =============================================================================

ax_env = fig.add_subplot(gs[1, 1])
window_idx = np.arange(len(p_vals_norm))

ax_env.plot(window_idx, filtered_powers_norm, color='gray', lw=1.5, alpha=0.7, label='Пространственный фильтр', zorder=2)
ax_env.plot(window_idx, p_vals_norm, color='black', lw=1.2, label='Выученная сетью мощность', zorder=3)

xtick_positions = []
xtick_labels = []
for lab in unique_labels_ordered:
    mask = (File_Labels == lab)
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