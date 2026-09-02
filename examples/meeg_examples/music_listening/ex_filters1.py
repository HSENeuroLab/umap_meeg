# -*- coding: utf-8 -*-
"""
Created on Wed Oct 22 17:07:11 2025

@author: anton
"""
import tensorflow as tf

import os

import mne
import numpy as np
import matplotlib.pyplot as plt
import umap

from scipy.signal import butter, filtfilt
from scipy.linalg import eigh
from scipy.linalg import inv, null_space

import sys
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
fpath = "C:/Users/ansbel/Documents/GitHub/TriCo/data/external/music_listening/part2/eeg/DmiAna_raw.fif"
# fpath = "C:/Users/ansbel/Documents/GitHub/TriCo/data/external/music_listening/part1/eeg/10_07_g1_2223_raw.fif"
raw = mne.io.read_raw_fif(fpath, preload=True)
sfreq = raw.info['sfreq']

new_descriptions = [
    'RS_EC_1', 'RS_EO_1', '2Hz', '05Hz', '4Hz', '1Hz', '3Hz',
    'NoRy_1', 'Waltz_1', 'Waltz_2', 'NoRy_2', 'NoRy_3', 'Waltz_3',
    'NoRy_4', 'Waltz_4', 'NoRy_5', 'Waltz_5', 'RS_EC_2', 'RS_EO_2',
    # 'Waltz_6', 'Waltz_7', 'Waltz_8'
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

raw_band = raw.copy().filter(l_freq=15,h_freq=25)
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

covmats_ssd = Covariances().fit_transform(X_windows_ssd_proj / np.std(X_windows_ssd_proj))
# covmats = Covariances().fit_transform(X_windows_band / np.std(X_windows_band))
covmats = Covariances().fit_transform(X_windows_band / np.std(X_windows_band))

# print("Отбеливание ковариационных матриц по среднему арифметическому...")
# C_avg = np.mean(covmats_ssd, axis=0)                     
# C_avg_invsqrt = invsqrtm(C_avg)                       
# C_avg_sqrt = np.linalg.inv(C_avg_invsqrt)             
# covmats_white = C_avg_invsqrt @ covmats_ssd @ C_avg_invsqrt

# %%
# 1. Вычисляем чистые, сырые ковариации
covmats = Covariances(estimator='oas').fit_transform(X_windows_band / np.std(X_windows_band))

# 2. ИЩЕМ МИНИМАЛЬНЫЙ GLOBAL_EPS ЧЕРЕЗ АНАЛИЗ СПЕКТРА
# Вычисляем собственные значения для всех матриц: форма (N_windows, M_channels)
eigvals = np.linalg.eigvalsh(covmats)

max_eig = np.max(eigvals, axis=1)  # Максимальные собственные значения каждой эпохи
min_eig = np.min(eigvals, axis=1)  # Минимальные собственные значения каждой эпохи

target_kappa = 1e6  # Максимально допустимое число обусловленности (безопасно для логарифма)

# Математика: нам нужно, чтобы (max_eig + eps) / (min_eig + eps) <= target_kappa
# Поскольку eps мало по сравнению с max_eig, упрощаем: max_eig / (min_eig + eps) <= target_kappa
# Отсюда: eps >= (max_eig / target_kappa) - min_eig
required_eps = (max_eig / target_kappa) - min_eig

# Также eps обязан вытягивать любые отрицательные собственные значения (ошибки округления) в плюс
strict_positivity_eps = -min_eig

# Для каждой матрицы находим нужный ей eps, и берем худший случай по всему датасету
eps_candidates = np.maximum(required_eps, strict_positivity_eps)
global_eps = np.max(eps_candidates) + np.finfo(float).eps

global_eps = max(0.0, global_eps) # Защита, если матрицы уже идеальны

print(f"Найденный минимальный global_eps: {global_eps:.4e}")

# 3. Применяем единую регуляризацию ко всем окнам
M = covmats.shape[1]
covmats_reg = covmats + global_eps * np.eye(M)

# 4. ОЦЕНИВАЕМ ИСКРИВЛЕНИЕ ПРОСТРАНСТВА (Distortion)
# Возьмем среднюю ковариационную матрицу (как центр нашего пространства)
C_mean = np.mean(covmats, axis=0)
eigvals_mean = np.linalg.eigvalsh(C_mean)

# Искривление Риманова пространства из-за добавления eps * I вычисляется аналитически.
# Поскольку I не меняет базис, расстояние d(C, C + eps*I) = sqrt( sum( ln^2(1 + eps / lambda_i) ) )
# Мы игнорируем lambda <= 0, так как исходно расстояние там бесконечно.
valid_eigvals = eigvals_mean[eigvals_mean > 0]
distortion = np.sqrt(np.sum(np.log(1 + global_eps / valid_eigvals)**2))

print(f"Римановское искажение (дистанция) центра пространства: {distortion:.4f}")

# Для контекста: посмотрим, как eps повлиял на самую слабую и самую сильную ось
print(f"Искажение главной оси (lambda={np.max(valid_eigvals):.4e}): {np.log(1 + global_eps/np.max(valid_eigvals)):.4e}")
print(f"Искажение слабой оси (lambda={np.min(valid_eigvals):.4e}): {np.log(1 + global_eps/np.min(valid_eigvals)):.4f}")

# 5. Считаем классическую Риманову метрику (AIRM)
print("Расчет матрицы попарных расстояний (AIRM)")
dists_init = pairwise_distance(covmats, metric='riemann')

# %%
covmats_reg = covmats.copy()
dist_matrix_init = dists_init.copy()

# %%
import matplotlib.pyplot as plt
import numpy as np

plt.hist(eigvals[:,2], bins=1000, rwidth=0.85)
plt.show()

# %%
from pyriemann.estimation import Covariances
from pyriemann.geometry.distance import pairwise_distance
import numpy as np

# 1. Вычисляем чистые, сырые ковариации (Sample Covariance Matrix)
covmats = Covariances(estimator='cov').fit_transform(X_windows_band)

# 2. Вычисляем среднюю энергию (след) по всему датасету
mean_trace = np.mean(np.trace(covmats, axis1=1, axis2=2))

# 3. Задаем глобальный порог шума 
# (например, 1e-4 от средней энергии, что отлично совпадает с твоим smoothing_factor в слое)
global_eps = 1e-4 * mean_trace 
# global_eps = 1e-6

# 4. Применяем единую регуляризацию ко всем окнам
M = covmats.shape[1]
covmats_reg = covmats + global_eps * np.eye(M)

# 5. Считаем классическую Риманову метрику (AIRM)
# Теперь pyriemann отработает без ошибок, а геометрия будет идеальной для UMAP
print("Расчет матрицы попарных расстояний (AIRM)...")
dist_matrix_init = pairwise_distance(covmats_reg, metric='riemann')

# %%
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed

def process_chunk_kl(start_idx, end_idx, covmats, covmats_inv, M):
    """Функция-воркер для обработки куска строк (Kullback-Sym)"""
    chunk_results = []
    for i in range(start_idx, end_idx):
        C_j_subset = covmats[i+1:]
        if len(C_j_subset) == 0:
            chunk_results.append((i, []))
            continue
            
        C_inv_i = covmats_inv[i]
        C_i = covmats[i]
        C_inv_j_subset = covmats_inv[i+1:]
        
        # МАГИЯ ЛИНЕЙНОЙ АЛГЕБРЫ: 
        # Tr(A^-1 * B) для симметричных матриц равен сумме поэлементного умножения!
        # Это избавляет от тяжелого матричного умножения @
        term1 = np.sum(C_inv_i * C_j_subset, axis=(1, 2))
        term2 = np.sum(C_inv_j_subset * C_i, axis=(1, 2))
        
        # Формула Джеффриса: 0.5 * (Tr(A^-1 B) + Tr(B^-1 A)) - M
        d2 = 0.5 * (term1 + term2) - M
        
        # Защита от микроскопических минусов из-за float
        chunk_results.append((i, np.maximum(d2, 0.0)))
        
    return chunk_results

def compute_super_fast_kullback_sym(covmats, n_jobs=-1, chunk_size=100):
    print("Предвычисление обратных матриц...")
    N, M, _ = covmats.shape
    
    # Считаем инверсию всего один раз для каждой матрицы
    covmats_inv = np.linalg.inv(covmats)
    
    dist_matrix = np.zeros((N, N))
    
    # Разбиваем индексы на чанки (размер чанка можно сделать больше, так как операции легкие)
    chunks = [(i, min(i + chunk_size, N)) for i in range(0, N, chunk_size)]
    
    print("Расчет матрицы попарных расстояний Kullback-Sym (Мультипроцессинг)...")
    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(process_chunk_kl)(start, end, covmats, covmats_inv, M) 
        for start, end in tqdm(chunks, desc="Chunks")
    )
    
    # Собираем всё в матрицу
    for chunk_res in results:
        for i, row_data in chunk_res:
            if len(row_data) > 0:
                dist_matrix[i, i+1:] = row_data
                
    dist_matrix = dist_matrix + dist_matrix.T
    return dist_matrix

# ЗАПУСК (Подаем матрицы, которые УЖЕ регуляризованы вашим global_eps)
dist_matrix_chi2 = compute_super_fast_kullback_sym(covmats_reg)

# %%
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed

def process_chunk(start_idx, end_idx, covmats_reg, covmats_sqrt, traces, eps):
    """Функция-воркер для обработки куска строк в отдельном процессе"""
    chunk_results = []
    for i in range(start_idx, end_idx):
        C_j_subset = covmats_reg[i+1:]
        if len(C_j_subset) == 0:
            chunk_results.append((i, []))
            continue
            
        C_sqrt_i = covmats_sqrt[i]
        
        # Перемножаем батчом
        C_mid = C_sqrt_i @ C_j_subset @ C_sqrt_i
        C_mid = 0.5 * (C_mid + np.transpose(C_mid, axes=(0, 2, 1)))
        
        # Находим собственные значения
        eigvals_mid = np.linalg.eigvalsh(C_mid)
        trace_mid_sqrt = np.sum(np.sqrt(np.maximum(eigvals_mid, eps)), axis=1)
        
        # Считаем Вассерштейна
        d2 = traces[i] + traces[i+1:] - 2.0 * trace_mid_sqrt
        chunk_results.append((i, np.maximum(d2, 0.0)))
        
    return chunk_results

def compute_super_fast_wasserstein(covmats, eps=0, n_jobs=-1, chunk_size=50):
    print("Предвычисление матричных корней...")
    N, M, _ = covmats.shape
    
    eye_M = np.eye(M)
    covmats_reg = covmats + eps * eye_M
    traces = np.trace(covmats_reg, axis1=1, axis2=2)
    
    eigvals, eigvecs = np.linalg.eigh(covmats_reg)
    eigvals_sqrt = np.sqrt(np.maximum(eigvals, eps))
    covmats_sqrt = (eigvecs * eigvals_sqrt[:, np.newaxis, :]) @ np.transpose(eigvecs, axes=(0, 2, 1))
    
    dist_matrix = np.zeros((N, N))
    
    # Разбиваем индексы на чанки
    chunks = [(i, min(i + chunk_size, N)) for i in range(0, N, chunk_size)]
    
    print("Расчет матрицы попарных расстояний (Мультипроцессинг)...")
    # Используем backend "loky" (процессы), чтобы обойти все ограничения Python
    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(process_chunk)(start, end, covmats_reg, covmats_sqrt, traces, eps) 
        for start, end in tqdm(chunks, desc="Chunks")
    )
    
    # Собираем всё в матрицу
    for chunk_res in results:
        for i, row_data in chunk_res:
            if len(row_data) > 0:
                dist_matrix[i, i+1:] = row_data
                
    dist_matrix = dist_matrix + dist_matrix.T
    return dist_matrix

eigvals, eigvecs = np.linalg.eigh(covmats)

eigvals = np.maximum(eigvals, 0.0)

# U @ diag(lambda) @ U.T для каждой матрицы
covmats_psd = np.einsum(
    'nij,nj,nkj->nik',
    eigvecs,
    eigvals,
    eigvecs
)

dist_matrix_init = compute_super_fast_wasserstein(covmats_psd, eps=0)

# %%
import matplotlib.pyplot as plt
import umap

reducer = umap.UMAP(n_components=2, n_neighbors=20, metric='precomputed')
coords = reducer.fit_transform(dist_matrix_init)

plt.figure(figsize=(8, 6))
plt.scatter(coords[:, 0], coords[:, 1], s=5, cmap='Spectral')

plt.title('UMAP проекция матрицы расстояний')
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
threshold = np.percentile(dist_matrix_init, percentile_value)

# 2. Отображаем матрицу
# Добавлена палитра (cmap), которая хорошо показывает контраст (например, 'viridis' или 'inferno')
plt.figure(figsize=(8, 6))
im = plt.imshow(dist_matrix_init, vmax=threshold, cmap='viridis')

# 3. Настраиваем цветовую шкалу
# Используем extend='max', чтобы показать, что на графике есть значения выше максимума шкалы
cbar = plt.colorbar(im, extend='max')
cbar.set_label(f"Расстояние (максимум ограничен {percentile_value}-м процентилем)")

plt.title("Матрица расстояний с повышенным контрастом")
plt.show()

# %%
import numpy as np

n_ch_white = covmats_reg.shape[1]

# =============================================================================
# ПОДГОТОВКА ДАННЫХ (УНИКАЛЬНЫЕ ЭЛЕМЕНТЫ)
# =============================================================================
M_channels = covmats_reg.shape[1]

# Индексы верхней треугольной матрицы
idx_i, idx_j = np.triu_indices(M_channels)

# Множители: 1.0 для диагонали, sqrt(2) для внедиагональных
multipliers = np.where(idx_i == idx_j, 1.0, np.sqrt(2.0)).astype(np.float32)

# Вытаскиваем уникальные элементы и сразу масштабируем
X_cov_flat = (covmats_reg[:, idx_i, idx_j] * multipliers).astype(np.float32)

input_dim = int(X_cov_flat.shape[1]) # Теперь размерность M*(M+1)/2
print(f"Новая размерность входа: {input_dim}")

class SpatialPatternDecoder(tf.keras.layers.Layer):
    def __init__(self, M_channels, N_patterns, **kwargs):
        super().__init__(**kwargs)
        self.M_channels = int(M_channels)
        self.N_patterns = int(N_patterns)
        
    def build(self, input_shape):
        self.A = self.add_weight(
            shape=(self.M_channels, self.N_patterns),
            initializer=tf.keras.initializers.RandomNormal(stddev=0.1),
            trainable=True,
            name="A_patterns"
        )
        
        self.noise_log = self.add_weight(
            shape=(self.M_channels,),
            initializer=tf.keras.initializers.Constant(-3.0),
            trainable=True,
            name="sensor_noise"
        )
        
        # Подготовка констант для векторизации
        i, j = np.triu_indices(self.M_channels)
        self.flat_indices = tf.constant(i * self.M_channels + j, dtype=tf.int32)
        self.multipliers = tf.constant(np.where(i == j, 1.0, np.sqrt(2.0)), dtype=tf.float32)

    def call(self, z):
        A_norm = tf.math.l2_normalize(self.A, axis=0)
        P = tf.exp(z)
        
        C_signal = tf.einsum("mf,bf,lf->bml", A_norm, P, A_norm)
        noise_variance = tf.math.softplus(self.noise_log)
        noise_diag = tf.linalg.diag(noise_variance)
        
        # Полная матрица (batch, M, M)
        C_recon = C_signal + noise_diag
        
        # Выборка уникальных элементов и масштабирование
        C_recon_flat = tf.reshape(C_recon, [-1, self.M_channels * self.M_channels])
        vecs = tf.gather(C_recon_flat, self.flat_indices, axis=1)
        
        return vecs * self.multipliers # Выход: (batch, M*(M+1)/2)

def build_unvec_matrix(M):
    """Создает матрицу для быстрого преобразования вектора обратно в симметричную матрицу"""
    D = M * (M + 1) // 2
    W = np.zeros((D, M * M), dtype=np.float32)
    idx_i, idx_j = np.triu_indices(M)
    for k, (i, j) in enumerate(zip(idx_i, idx_j)):
        if i == j:
            W[k, i * M + j] = 1.0
        else:
            # Делим на sqrt(2), чтобы снять примененный ранее масштаб
            W[k, i * M + j] = 1.0 / np.sqrt(2.0)
            W[k, j * M + i] = 1.0 / np.sqrt(2.0)
    return W

# Инициализируем константу один раз
W_UNVEC = tf.constant(build_unvec_matrix(n_ch_white), dtype=tf.float32)

def reconstruct_sym_matrix(vecs, M):
    """Дифференцируемое восстановление (batch, D) -> (batch, M, M)"""
    C_flat = tf.matmul(vecs, W_UNVEC)
    return tf.reshape(C_flat, [-1, M, M])

def riemannian_distance_loss(y_true_flat, y_pred_flat):
    y_true_flat = tf.cast(y_true_flat, tf.float32)
    y_pred_flat = tf.cast(y_pred_flat, tf.float32)

    M = n_ch_white  
    
    C_true = reconstruct_sym_matrix(y_true_flat, M)
    C_pred = reconstruct_sym_matrix(y_pred_flat, M)
    
    C_true = 0.5 * (C_true + tf.transpose(C_true, [0, 2, 1]))
    C_pred = 0.5 * (C_pred + tf.transpose(C_pred, [0, 2, 1]))

    # 1. Отсекаем построение градиентного графа для истинных данных
    C_true = tf.stop_gradient(C_true)
        
    # 2. Безопасный порог для float32 (машинный эпсилон ~1.19e-7)
    eps = 1e-4

    # Шаг 1: C_true^{-1/2}
    eigvals_true, eigvecs_true = tf.linalg.eigh(C_true)
    inv_sqrt_eigvals = 1.0 / tf.sqrt(tf.maximum(eigvals_true, eps))
    C_true_invsqrt = tf.einsum('bij,bj,bkj->bik', eigvecs_true, inv_sqrt_eigvals, eigvecs_true)

    # Шаг 2: C_true^{-1/2} * C_pred * C_true^{-1/2}
    C_mid = tf.einsum('bij,bjk,bkl->bil', C_true_invsqrt, C_pred, C_true_invsqrt)
    C_mid = 0.5 * (C_mid + tf.transpose(C_mid, perm=[0, 2, 1])) 

    # Шаг 3: Собственные значения полученной матрицы
    eigvals_mid, _ = tf.linalg.eigh(C_mid) 
    log_eigvals_mid = tf.math.log(tf.maximum(eigvals_mid, eps))

    # Шаг 4: Считаем дистанцию напрямую по собственным значениям    
    # dist_sq = tf.reduce_sum(tf.square(log_eigvals_mid), axis=1) / tf.cast(M * M, tf.float32)
    dist_sq = tf.sqrt(tf.reduce_sum(tf.square(log_eigvals_mid), axis=1))

    return tf.reduce_mean(dist_sq)

def log_euclidean_loss(y_true_flat, y_pred_flat):
    y_true_flat = tf.cast(y_true_flat, tf.float32)
    y_pred_flat = tf.cast(y_pred_flat, tf.float32)

    M = n_ch_white  
    C_true = reconstruct_sym_matrix(y_true_flat, M)
    C_pred = reconstruct_sym_matrix(y_pred_flat, M)
    
    C_true = 0.5 * (C_true + tf.transpose(C_true, perm=[0, 2, 1]))
    C_pred = 0.5 * (C_pred + tf.transpose(C_pred, perm=[0, 2, 1]))
        
    C_true = tf.stop_gradient(C_true)

    eigvals_true, eigvecs_true = tf.linalg.eigh(C_true)
    log_eigvals_true = tf.math.log(tf.maximum(eigvals_true, 1e-9))
    log_C_true = tf.stop_gradient(tf.einsum('bij,bj,bkj->bik', eigvecs_true, log_eigvals_true, eigvecs_true))

    eigvals_pred, eigvecs_pred = tf.linalg.eigh(C_pred)
    log_eigvals_pred = tf.math.log(tf.maximum(eigvals_pred, 1e-9))
    log_C_pred = tf.einsum('bij,bj,bkj->bik', eigvecs_pred, log_eigvals_pred, eigvecs_pred)

    diff = log_C_true - log_C_pred
    dist_sq = tf.reduce_sum(tf.square(diff), axis=(1, 2))
    
    return tf.reduce_mean(dist_sq)

def wasserstein_distance_loss(y_true_flat, y_pred_flat):
    y_true_flat = tf.cast(y_true_flat, tf.float32)
    y_pred_flat = tf.cast(y_pred_flat, tf.float32)

    M = n_ch_white  
    C_true = reconstruct_sym_matrix(y_true_flat, M)
    C_pred = reconstruct_sym_matrix(y_pred_flat, M)
    
    eps = 0 

    # 1. Вычисляем следы исходных матриц
    tr_true = tf.linalg.trace(C_true)
    tr_pred = tf.linalg.trace(C_pred)

    # 2. Вычисляем корень из C_true через собственные значения
    eigvals_true, eigvecs_true = tf.linalg.eigh(C_true)
    sqrt_eigvals_true = tf.sqrt(tf.maximum(eigvals_true, eps))
    C_true_sqrt = tf.einsum('bij,bj,bkj->bik', eigvecs_true, sqrt_eigvals_true, eigvecs_true)

    # 3. Центральная матрица C_mid = C_true^{1/2} * C_pred * C_true^{1/2}
    C_mid = tf.einsum('bij,bjk,bkl->bil', C_true_sqrt, C_pred, C_true_sqrt)
    C_mid = 0.5 * (C_mid + tf.transpose(C_mid, perm=[0, 2, 1])) 

    # 4. След корня из C_mid (сумма корней её собственных значений)
    eigvals_mid, _ = tf.linalg.eigh(C_mid) 
    tr_mid_sqrt = tf.reduce_sum(tf.sqrt(tf.maximum(eigvals_mid, eps)), axis=1)

    # 5. Итоговое расстояние Вассерштейна
    d2 = tr_true + tr_pred - 2.0 * tr_mid_sqrt
    
    # Защита от отрицательных значений из-за погрешности float32
    d2 = tf.maximum(d2, 0.0)
    
    # Усредняем и масштабируем (делим на M для совместимости градиентов)
    return tf.reduce_mean(d2 / tf.cast(M, tf.float32))

# =============================================================================
# ПОДГОТОВКА ДАННЫХ
# =============================================================================
N_patterns = int(40)
N_dim = int(N_patterns)

print(f"Обучение ParametricUMAP: N_dim={N_dim}, N_patterns={N_patterns}")

# =============================================================================
# ЭНКОДЕР И ДЕКОДЕР
# =============================================================================
# Легкий L2-штраф удержит логарифмы мощностей от бесконечности
reg = tf.keras.regularizers.l2(1e-4)

encoder = tf.keras.Sequential([
    tf.keras.Input(shape=(int(input_dim),), dtype=tf.float32),
    
    # Блок 1
    tf.keras.layers.Dense(512, use_bias=False),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation("relu"),
    tf.keras.layers.Dropout(0.2),
    
    # Блок 2
    tf.keras.layers.Dense(512, use_bias=False),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation("relu"),
    tf.keras.layers.Dropout(0.2),
    
    # Блок 3
    tf.keras.layers.Dense(512, use_bias=False),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation("relu"),
    tf.keras.layers.Dropout(0.1),
    
    # Латентное пространство (z_powers)
    tf.keras.layers.Dense(
        int(N_dim), 
        activation="linear", 
        use_bias=True, 
        activity_regularizer=reg, 
        name="z_powers"
    )
])

decoder = tf.keras.Sequential([
    tf.keras.Input(shape=(int(N_dim),), dtype=tf.float32),    
    tf.keras.layers.Dense(int(N_dim), activation="linear", name="latent_alignment"),
    SpatialPatternDecoder(
        M_channels=int(n_ch_white),
        N_patterns=int(N_patterns),
        name="spatial_decoder"
    )
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
loss_weight = 1.0

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
            # batch_loss = log_euclidean_loss(X_batch, X_recon).numpy()
            # batch_loss = wasserstein_distance_loss(X_batch, X_recon).numpy()

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
    autoencoder_loss=True, 
    parametric_reconstruction_loss_fcn=riemannian_distance_loss, 
    parametric_reconstruction_loss_weight=loss_weight,
    reconstruction_validation=X_val, 
    keras_fit_kwargs=keras_fit_args, 
    verbose=True
)

embedder.loss_report_frequency = 200
embedder.n_training_epochs = 1      

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
    parametric_reconstruction=False, 
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
plt.title('Визуализация данных')
plt.xlabel('Главная компонента 1')
plt.ylabel('Главная компонента 2')
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

# %%
# =============================================================================
# 6. ИЗВЛЕЧЕНИЕ ПАРАМЕТРОВ И ОТРИСОВКА (С АВТОСОРТИРОВКОЙ)
# =============================================================================
import matplotlib as mpl
import numpy as np # На всякий случай убедимся, что np импортирован

z_raw = embedder.encoder.predict(X_cov_flat)

# 1. Прогоняем их через слои декодера ДО SpatialPatternDecoder, 
# чтобы учесть выравнивание (latent_alignment) и получить корректные z для мощностей
alignment_layer = embedder.decoder.get_layer("latent_alignment")
z_aligned = alignment_layer(z_raw).numpy() 

# 2. Переводим в реальные мощности
powers = np.exp(z_aligned) 

# 3. Извлекаем матрицу паттернов и нормируем 
spatial_decoder_layer = embedder.decoder.get_layer("spatial_decoder")
A_learned_raw = spatial_decoder_layer.get_weights()[0]
A_global = A_learned_raw / np.linalg.norm(A_learned_raw, axis=0) 

# =============================================================================
# НОВОЕ: СОРТИРОВКА ПО ДИСПЕРСИИ МОЩНОСТИ (ПО УБЫВАНИЮ)
# =============================================================================
# Считаем дисперсию (разброс) каждого из N_patterns вдоль всех эпох (axis=0)
power_variances = np.var(powers, axis=0)

# Получаем индексы сортировки по убыванию [::-1]
sort_idx = np.argsort(power_variances)[::-1]

# Сортируем массив мощностей и столбцы матрицы прямой модели A_global
powers = powers[:, sort_idx]
A_global = A_global[:, sort_idx]

print(f"Топ-5 компонент по дисперсии мощности (оригинальные индексы): {sort_idx[:5]}")
# =============================================================================

# 4. ТЕПЕРЬ A_global - ЭТО ОТСОРТИРОВАННЫЕ ПАТТЕРНЫ В ПРОСТРАНСТВЕ СЕНСОРОВ!
found_patterns = [A_global[:, i] for i in range(N_patterns)]

# 5. Фильтры Хауфе (с Тихоновской регуляризацией)
C_global_mean = np.mean(covmats, axis=0)

# Коэффициент регуляризации 
alpha = 1e-4  

# Добавляем единичную матрицу, масштабированную на след C_global_mean
I = np.eye(C_global_mean.shape[0])
C_global_reg = C_global_mean + alpha * np.trace(C_global_mean) * I

# Обращаем регуляризованную матрицу
C_global_inv = np.linalg.inv(C_global_reg)

# Фильтры тоже автоматически получаются отсортированными, так как мы берем столбцы из отсортированной A_global
found_filters = [C_global_inv @ A_global[:, i] for i in range(N_patterns)]

# %%
import matplotlib.pyplot as plt
import numpy as np

# Пример создания матрицы (100 точек, у каждой по 3 координаты)
# powers = np.random.rand(100, 3) 

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

# Передаем столбцы 0, 1 и 2 в качестве координат X, Y, Z
ax.scatter(powers[:, 5], powers[:, 3], powers[:, 7], c='blue', marker='o')

ax.set_xlabel('Ось X')
ax.set_ylabel('Ось Y')
ax.set_xlabel('Ось Z')

plt.show()


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
# Выбираем индекс паттерна (0..N_patterns-1
comp_idx = 0
W_sensor = found_filters[comp_idx]
A_pattern = found_patterns[comp_idx]

# 1. Мощность выбранного выученного источника
p_vals = powers[:, comp_idx]

# 2. Мощность через пространственный фильтр (w^T * C * w)
filtered_powers = np.array([W_sensor.T @ C @ W_sensor for C in covmats_reg])

# 3. Нормализация (делим на std, без вычета среднего)
p_vals_norm = p_vals / np.std(p_vals)
filtered_powers_norm = filtered_powers / np.std(filtered_powers)

unique_labels_ordered = []
for lab in window_labels:
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
    
    p_vals = np.exp(z_20d)

    # print(
    #     "z_20d:",
    #     "min =", np.min(z_20d),
    #     "max =", np.max(z_20d),
    #     "mean =", np.mean(z_20d),
    #     "std =", np.std(z_20d),
    #     "finite =", np.all(np.isfinite(z_20d))
    # )

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

        if norm_p > 0.1:
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

