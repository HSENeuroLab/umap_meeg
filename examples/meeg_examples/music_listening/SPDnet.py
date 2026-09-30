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
fpath = "C:/Users/ansbel/Documents/GitHub/TriCo/data/external/music_listening/part1/eeg/10_07_g1_2223_raw.fif"
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
ica.plot_sources(raw)  # Временные ряды компонент

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
fmax = 28
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
X_windows_ssd_proj = np.zeros((n_windows, n_components_ssd, n_times))
for i in range(n_windows):
    X_windows_ssd_proj[i] = W_ssd.T @ X_windows_band[i]

covmats_band = Covariances().fit_transform(X_windows_band / np.std(X_windows_band))
covmats_ssd = Covariances().fit_transform(X_windows_ssd_proj / np.std(X_windows_ssd_proj))

# %%
dists_init = pairwise_distance(covmats_ssd, metric='riemann')

# %%
reducer = umap.UMAP(n_components=3, n_neighbors=20, metric='precomputed')
coords = reducer.fit_transform(dists_init)

plt.figure(figsize=(8, 6))
plt.scatter(coords[:, 0], coords[:, 1], s=5, cmap='Spectral')

plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.colorbar() 
plt.show()

# %%
sim_dir = os.path.abspath("C:/Users/ansbel/Documents/GitHub/umap_meeg/examples/meeg_examples/music_listening/")
if sim_dir not in sys.path:
    sys.path.insert(0, sim_dir)

import matplotlib.pyplot as plt
from mspoc import mspoc

# =============================================================================
# 4. ПРИМЕНЕНИЕ mSPoC
# =============================================================================

# 1. Приводим размерности X к нужному виду: (n_times, n_channels, n_epochs)
X_input = np.transpose(X_windows_band, (2, 1, 0))

# 2. Приводим размерности Y (UMAP координаты) к виду: (n_channels_y, n_epochs)
# В нашем случае Y - это 3-мерное вложение UMAP
Y_input = coords.T 

# 3. Запускаем алгоритм
# Просим найти столько же компонент, сколько осей в UMAP (максимум 3)
Wx, Wy, Wtau, Ax, Ay, out = mspoc(X_input, Y_input, n_component_sets=10, verbose=1)

# %%
# 4. Проецируем исходные ЭЭГ данные на найденные оси mSPoC
# Получаем временные ряды новых компонент формы (n_windows, n_components, n_times)
# Wx имеет форму (n_channels, n_components)
X_windows_mspoc_proj = np.array([Wx.T @ epoch for epoch in X_windows_band])

# =============================================================================
# 5. ВИЗУАЛИЗАЦИЯ ИСТОЧНИКОВ
# =============================================================================
# Пространственные паттерны (Ax) показывают, как найденные источники
# проецируются на скальп (аналог топомапов в ICA/SSD).

# Убедитесь, что epochs_band содержит правильный info (с координатами каналов)
info = epochs_band.info
n_components_extracted = Wx.shape[1]

fig, axes = plt.subplots(1, n_components_extracted, figsize=(4 * n_components_extracted, 4))
if n_components_extracted == 1:
    axes = [axes]

for i, ax in enumerate(axes):
    # Отрисовка топомапа для i-й компоненты mSPoC
    mne.viz.plot_topomap(Ax[:, i], info, axes=ax, show=False)
    
    # В out['corr_values'] лежит коэффициент корреляции мощности 
    # этой компоненты с соответствующей проекцией Y
    corr_val = out['corr_values'][i]
    ax.set_title(f'mSPoC Source {i+1}\nCorr: {corr_val:.3f}')

plt.suptitle('Пространственные паттерны (Ax) компонент mSPoC', y=1.05)
plt.tight_layout()
plt.show()

# =============================================================================
# 6. АНАЛИЗ ВЕСОВ UMAP
# =============================================================================
# Wy (размерность 3 x n_components) показывает, как именно оси UMAP 
# комбинируются для формирования таргета для каждой mSPoC-компоненты.
print("Веса осей UMAP для каждой mSPoC-компоненты (Wy):")
print(Wy)

# %%
from scipy import signal as sp_signal

def recompute_corr(X_windows, Y, Wx, Wy, Wtau, tau_vector, use_log=True):
    """
    Пересчитывает знаковые корреляции компонент mSPoC, согласованные
    с финальными (пост-флиповыми) векторами Wx, Wy.

    X_windows : (n_epochs, n_channels, n_times)
    Y         : (n_y, n_epochs)     — тот же Y, что подавался в mspoc
    Wx        : (n_channels, K)
    Wy        : (n_y, K)
    Wtau      : (n_tau, K)
    """
    X_windows = np.asarray(X_windows)
    Y         = np.asarray(Y)
    K         = Wx.shape[1]

    # 1) Мощность компоненты в каждой эпохе
    comp = np.einsum('ck,ect->ekt', Wx, X_windows)   # (n_epochs, K, n_times)
    Px   = np.var(comp, axis=2)                      # (n_epochs, K)

    # 2) Логарифм — как в optimize_filters при use_log=True
    if use_log:
        Px = np.log(Px + 1e-12)

    # 3) Фильтр Wtau и корреляция с sy = Wy^T Y
    corrs = np.zeros(K)
    for k in range(K):
        px  = Px[:, k] - Px[:, k].mean()
        pxf = sp_signal.lfilter(Wtau[:, k], [1.0], px)

        sy  = Wy[:, k] @ Y
        sy  = sy - sy.mean()

        corrs[k] = np.corrcoef(pxf, sy)[0, 1]

    return corrs
corr_values = recompute_corr(
    X_windows = X_windows_band,   # (n_epochs, n_channels, n_times)
    Y         = Y_input,          # (n_y, n_epochs), тот же что шёл в mspoc
    Wx        = Wx[:, selected_comps],
    Wy        = W_sel,
    Wtau      = Wtau,
    tau_vector= [0],              # если не передавал в mspoc — дефолт [0]
    use_log   = True,
)

# %%
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# =============================================================================
# 5. РУЧНОЙ ВЫБОР КОМПОНЕНТ И ТОПОМАПЫ
# =============================================================================
# Укажите индексы 3 компонентов, которые вам показались наиболее интересными (с нуля)
# Например, выберем 1-ю, 3-ю и 5-ю компоненты:
selected_comps = [2, 3, 7] 

info = epochs_band.info

fig, axes = plt.subplots(1, len(selected_comps), figsize=(4 * len(selected_comps), 4))
for ax, idx in zip(axes, selected_comps):
    mne.viz.plot_topomap(Ax[:, idx], info, axes=ax, show=False)
    corr_val = out['corr_values'][idx]
    ax.set_title(f'mSPoC Source {idx+1}\nCorr: {corr_val:.3f}')

plt.suptitle('Выбранные пространственные паттерны (Ax)', y=1.05)
plt.tight_layout()
plt.show()

# =============================================================================
# 6. ВИЗУАЛИЗАЦИЯ НАПРАВЛЕНИЙ В ИСХОДНОМ ПРОСТРАНСТВЕ UMAP (3D)
# =============================================================================
# Центрируем координаты UMAP для красивой отрисовки векторов из начала координат
coords_centered = coords - coords.mean(axis=0)

# 1. Выделяем матрицу базиса и считаем матрицу перехода
W_sel = Wy[:, selected_comps]
W_trans = np.linalg.inv(W_sel.T)

# 2. Переводим координаты и все векторы в новый базис
Y_proj = coords_centered @ W_trans
Wy_proj = Wy.T @ W_trans 

base_scale = np.max(np.abs(Y_proj)) * 0.8 

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.patheffects as pe

# =============================================================================
# ПОДГОТОВКА ЦВЕТОВ И КЛАССОВ (ДЛЯ ОБОИХ ГРАФИКОВ)
# =============================================================================
unique_classes = []
try:
    for lab in cond_descriptions:
        if lab in window_labels and lab not in unique_classes:
            unique_classes.append(lab)
except NameError:
    pass

for lab in np.unique(window_labels):
    if lab not in unique_classes:
        unique_classes.append(lab)

cmap_classes = plt.cm.tab20
color_map = {
    lab: cmap_classes(i / max(1, len(unique_classes) - 1))
    for i, lab in enumerate(unique_classes)
}

# =============================================================================
# РАСЧЕТЫ ДЛЯ СМЕНЫ БАЗИСА И ПРОЕКЦИЙ
# =============================================================================
coords_centered = coords - coords.mean(axis=0)
coords_centered /= np.max(np.abs(coords_centered))

W_sel = Wy[:, selected_comps]
W_trans = np.linalg.inv(W_sel.T)

# Новые точки и векторы
Y_proj = coords_centered @ W_trans
Wy_proj = Wy.T @ W_trans 
Y_proj /= np.max(np.abs(Y_proj))

# Масштабы для графиков (чтобы векторы длиной 1.0 вписывались в облако)
scale_orig = np.max(np.abs(coords_centered)) * 0.8 
scale_proj = np.max(np.abs(Y_proj)) * 0.8 

# =============================================================================
# ПОСТРОЕНИЕ 2-Х ГРАФИКОВ РЯДОМ
# =============================================================================
# Масштабы больше не нужны как множители — облака уже нормированы в [-1, 1].
# Но оставим переменные для подписей центроидов (они теперь равны 1.0)
scale_orig = np.max(np.abs(coords_centered))   # = 1.0
scale_proj = np.max(np.abs(Y_proj))            # = 1.0

fig = plt.figure(figsize=(20, 9))

# -----------------------------------------------------------------------------
# ГРАФИК 1: ИСХОДНОЕ ПРОСТРАНСТВО UMAP (СЛЕВА)
# -----------------------------------------------------------------------------
ax1 = fig.add_subplot(121, projection='3d')

for lab in unique_classes:
    mask = window_labels == lab
    if not np.any(mask):
        continue
    cluster_points = coords_centered[mask, :]
    ax1.scatter(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2],
                c=[color_map[lab]], marker='o', s=10, alpha=0.2, edgecolors='none')

for i in range(n_components_extracted):
    vec = Wy[:, i]
    norm = np.linalg.norm(vec)
    if norm == 0:
        continue

    corr = out['corr_values'][i]
    # Длина = |corr|, знак = sign(corr), максимум 1 — ровно в границах куба
    vec_norm = (vec / norm) * corr

    if i in selected_comps:
        ax1.quiver(0, 0, 0, vec_norm[0], vec_norm[1], vec_norm[2],
                   color='red', linewidth=3, arrow_length_ratio=0.1)
        ax1.text(vec_norm[0]*1.1, vec_norm[1]*1.1, vec_norm[2]*1.1,
                 f'S{i+1}\n(r={corr:.2f})', color='darkred',
                 fontsize=10, weight='bold')
    else:
        ax1.quiver(0, 0, 0, vec_norm[0], vec_norm[1], vec_norm[2],
                   color='gray', linewidth=1, arrow_length_ratio=0.1, alpha=0.5)

ax1.set_xlabel('UMAP 1')
ax1.set_ylabel('UMAP 2')
ax1.set_zlabel('UMAP 3')
ax1.set_title('Направления mSPoC в исходном пространстве UMAP')
ax1.set_xlim(-1, 1); ax1.set_ylim(-1, 1); ax1.set_zlim(-1, 1)
ax1.set_box_aspect((1, 1, 1))   # куб, а не «кирпич»

# -----------------------------------------------------------------------------
# ГРАФИК 2: СПРОЕЦИРОВАННОЕ ПРОСТРАНСТВО С ЦЕНТРОИДАМИ (СПРАВА)
# -----------------------------------------------------------------------------
ax2 = fig.add_subplot(122, projection='3d')

for lab in unique_classes:
    mask = window_labels == lab
    if not np.any(mask):
        continue

    cluster_points = Y_proj[mask, :]
    color = color_map[lab]

    ax2.scatter(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2],
                c=[color], marker='o', s=15, alpha=0.2, edgecolors='none', label=lab)

    cx, cy, cz = np.mean(cluster_points, axis=0)
    ax2.scatter(cx, cy, cz, c=[color], marker='o', s=150, alpha=1.0,
                edgecolors='black', linewidths=1.5)

    ax2.text(cx, cy, cz + 0.05, lab[:5],
             color='black', fontsize=9, weight='bold', ha='center',
             path_effects=[pe.withStroke(linewidth=2, foreground="white")])

for i in range(n_components_extracted):
    vec_proj = Wy_proj[i, :]
    norm = np.linalg.norm(vec_proj)
    if norm == 0:
        continue

    corr = out['corr_values'][i]
    # Длина = |corr| в тех же единицах, что и нормированное облако
    vec_norm = (vec_proj / norm) * corr

    if i in selected_comps:
        ax2.quiver(0, 0, 0, vec_norm[0], vec_norm[1], vec_norm[2],
                   color='red', linewidth=3, arrow_length_ratio=0.15)
        ax2.text(vec_norm[0]*1.15, vec_norm[1]*1.15, vec_norm[2]*1.15,
                 f'S{i+1}\n(r={corr:.2f})', color='darkred',
                 fontsize=11, weight='bold')
    else:
        ax2.quiver(0, 0, 0, vec_norm[0], vec_norm[1], vec_norm[2],
                   color='gray', linewidth=1, arrow_length_ratio=0.15, alpha=0.5)

ax2.set_xlabel(f'Ось S{selected_comps[0]+1}')
ax2.set_ylabel(f'Ось S{selected_comps[1]+1}')
ax2.set_zlabel(f'Ось S{selected_comps[2]+1}')
ax2.set_title('Новое фазовое пространство 3-х паттернов\n')
ax2.set_xlim(-1, 1); ax2.set_ylim(-1, 1); ax2.set_zlim(-1, 1)
ax2.set_box_aspect((1, 1, 1))   

ax2.legend(loc='center left', bbox_to_anchor=(1.05, 0.5),
           fontsize=9, title="Условия", title_fontsize=11, ncol=2, markerscale=2.5)

plt.tight_layout()
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
def init_identity_with_noise(shape, dtype=None):
    M, N = shape
    init = np.zeros((M, N), dtype=np.float32)
    # Первые M столбцов — единичные
    for i in range(min(M, N)):
        init[i, i] = 1.0
    # Остальные — случайные малые, чтобы после нормализации не были нулевыми
    if N > M:
        init[:, M:] = np.random.normal(0, 0.1, (M, N - M)).astype(np.float32)
    return tf.constant(init, dtype=dtype)
n_ch_white = covmats_ssd.shape[1]

# =============================================================================
# ПОДГОТОВКА ДАННЫХ (УНИКАЛЬНЫЕ ЭЛЕМЕНТЫ)
# =============================================================================
M_channels = covmats_ssd.shape[1]

# Индексы верхней треугольной матрицы
idx_i, idx_j = np.triu_indices(M_channels)

# Множители: 1.0 для диагонали, sqrt(2) для внедиагональных
multipliers = np.where(idx_i == idx_j, 1.0, np.sqrt(2.0)).astype(np.float32)

# Вытаскиваем уникальные элементы и сразу масштабируем
X_cov_flat = (covmats_ssd[:, idx_i, idx_j] * multipliers).astype(np.float32)

input_dim = int(X_cov_flat.shape[1]) # Теперь размерность M*(M+1)/2
print(f"Новая размерность входа: {input_dim}")

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
    dist_sq = tf.sqrt(tf.reduce_sum(tf.square(log_eigvals_mid), axis=1))

    return tf.reduce_mean(dist_sq)

@tf.keras.utils.register_keras_serializable()
class BiMapLayer(tf.keras.layers.Layer):
    """
    Билинейное преобразование: X_k = W_k · X_{k-1} · W_k^T
    Вход:  (batch, d_in, d_in)
    Выход: (batch, d_out, d_out)
    
    W_k — свободно обучаемая матрица (без ограничений на ортогональность/Штифеля).
    """
    def __init__(self, d_out, **kwargs):
        super().__init__(**kwargs)
        self.d_out = int(d_out)

    def build(self, input_shape):
        # Размерность входа берется автоматически из формы тензора
        self.d_in = int(input_shape[-1])
        
        # Инициализируем через единичную матрицу для хорошего старта (identity init)
        init_val = np.eye(self.d_out, self.d_in, dtype=np.float32)
        
        self.W = self.add_weight(
            shape=(self.d_out, self.d_in),
            initializer=tf.keras.initializers.Constant(init_val),
            trainable=True,
            name="W_bimap"
        )
        super().build(input_shape)

    def call(self, inputs):
        # Билинейное умножение: X_out = W · X · W^T
        X_out = tf.einsum('ij,bjk,lk->bil', self.W, inputs, self.W)
        
        # Принудительная симметризация для сохранения свойств SPD-матриц
        return 0.5 * (X_out + tf.transpose(X_out, perm=[0, 2, 1]))
    
@tf.keras.utils.register_keras_serializable()
class ReEigLayer(tf.keras.layers.Layer):
    """
    Eigenvalue Rectification Layer:
        X_k = U · max(eps · I, Sigma) · U^T
    Вводит нелинейность в сеть, отсекая малые/отрицательные собственные числа.
    """
    def __init__(self, epsilon=1e-4, jitter=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon
        self.jitter = jitter

    def call(self, inputs):
        X = 0.5 * (inputs + tf.transpose(inputs, perm=[0, 2, 1]))
        if self.jitter > 0:
            d = tf.shape(X)[-1]
            X = X + tf.eye(d, dtype=X.dtype) * self.jitter

        eigvals, eigvecs = tf.linalg.eigh(X)
        rectified_eigvals = tf.maximum(eigvals, self.epsilon)
        # Обратная сборка: U · Sigma_rect · U^T
        X_rect = tf.einsum('bij,bj,bkj->bik', eigvecs, rectified_eigvals, eigvecs)
        return 0.5 * (X_rect + tf.transpose(X_rect, perm=[0, 2, 1]))


@tf.keras.utils.register_keras_serializable()
class LogEigAndFlattenLayer(tf.keras.layers.Layer):
    """
    LogEig Layer + Tangent Space Vectorization:
        log(X) = U · log(Sigma) · U^T
    После взятия матричного логарифма переводит матрицу в плоский изометричный вектор.
    """
    def __init__(self, epsilon=1e-4, jitter=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon
        self.jitter = jitter

    def build(self, input_shape):
        self.M = int(input_shape[-1])
        idx_i, idx_j = np.triu_indices(self.M)
        flat_indices_np = (idx_i * self.M + idx_j).astype(np.int32)
        mults_np = np.where(idx_i == idx_j, 1.0, np.sqrt(2.0)).astype(np.float32)

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

    def call(self, inputs):
        X = 0.5 * (inputs + tf.transpose(inputs, perm=[0, 2, 1]))
        if self.jitter > 0:
            X = X + tf.eye(self.M, dtype=X.dtype) * self.jitter

        eigvals, eigvecs = tf.linalg.eigh(X)
        log_eigvals = tf.math.log(tf.maximum(eigvals, self.epsilon))
        log_X = tf.einsum('bij,bj,bkj->bik', eigvecs, log_eigvals, eigvecs)

        log_X_flat = tf.reshape(log_X, [-1, self.M * self.M])
        vecs = tf.gather(log_X_flat, self.flat_indices, axis=1)
        return vecs * self.multipliers

@tf.keras.utils.register_keras_serializable()
class LogDiagLayer(tf.keras.layers.Layer):
    """
    Извлекает диагональ матрицы и берет логарифм:
        z_i = log( (W C W^T)_{ii} )
    Выполняет роль расчета логарифмической дисперсии независимых компонент.
    """
    def __init__(self, epsilon=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def call(self, inputs):
        # inputs имеет форму (batch, M, M). 
        # tf.linalg.diag_part извлекает элементы [:, i, i]
        diags = tf.linalg.diag_part(inputs)
        
        # Защита от нуля и логарифмирование
        return tf.math.log(tf.maximum(diags, self.epsilon))

@tf.keras.utils.register_keras_serializable()
class UnflattenSymmetricLayer(tf.keras.layers.Layer):
    """Развёртка плоских векторов обратно в симметричные матрицы."""
    def __init__(self, M, **kwargs):
        super().__init__(**kwargs)
        self.M = int(M)

    def call(self, inputs):
        return reconstruct_sym_matrix(inputs, self.M)

@tf.keras.utils.register_keras_serializable()
class PatternDecoder(tf.keras.layers.Layer):
    """
    Восстанавливает ковариацию на сенсорах через свободно обучаемую матрицу A 
    с нормализованными столбцами:
        C_recon = A_norm · diag(exp(z)) · A_norm^T
    """
    def __init__(self, M_channels, N_patterns, **kwargs):
        super().__init__(**kwargs)
        self.M = int(M_channels)
        self.N = int(N_patterns)
        
    def build(self, input_shape):
        # 1. Свободно обучаемая прямая модель (A)
        # Инициализируем единичной матрицей
        init_val = np.eye(self.M, self.N, dtype=np.float32) 
        self.A = self.add_weight(
            name="A_pattern",
            shape=(self.M, self.N),
            initializer=tf.keras.initializers.Constant(init_val),
            trainable=True,
        )
        
        # 2. Индексы для векторизации
        M = self.M
        i, j = np.triu_indices(M)
        flat_indices_np = (i * M + j).astype(np.int32)
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
        # 1. Нормализация столбцов A (паттернов) до единичной длины (L2-норма)
        # axis=0 означает, что мы нормируем независимо каждый из N паттернов
        norms = tf.norm(self.A, axis=0, keepdims=True)
        A_norm = self.A / (norms)  

        # 2. Физические мощности источников
        P = tf.math.exp(z)  # (batch, N_patterns)

        # 3. Восстановление ковариации C = A_norm · P · A_norm^T
        C_recon = tf.einsum('mk,bk,nk->bmn', A_norm, P, A_norm)

        # 4. Векторизация для функции потерь
        C_flat = tf.reshape(C_recon, [-1, self.M * self.M])
        vecs = tf.gather(C_flat, self.flat_indices, axis=1)
        
        return vecs * self.multipliers

@tf.keras.utils.register_keras_serializable()
class LatentBalanceLoss(tf.keras.layers.Layer):
    """
    Штрафует различие дисперсий между латентными размерностями:
        L_bal = Var_k( Var_batch(z_k) )
    """
    def __init__(self, weight=0.1, **kwargs):
        super().__init__(**kwargs)
        self.weight = weight

    def call(self, z):
        z_var = tf.math.reduce_variance(z, axis=0)      # (N,)
        mean_var = tf.reduce_mean(z_var)
        balance = tf.reduce_mean(tf.square(z_var - mean_var))
        self.add_loss(self.weight * balance)
        return z
    
# # # =============================================================================
# # 3. СБОРКА АРХИТЕКТУРЫ ЭНКОДЕРА (Свободный BiMap)
# # =============================================================================

# M_channels = covmats_ssd.shape[1]
# N_patterns = M_channels
# d0 = N_patterns
# d1 = N_patterns

# inputs_enc = tf.keras.Input(shape=(input_dim,), dtype=tf.float32, name="encoder_input")

# # Развертка в симметричные матрицы
# x = UnflattenSymmetricLayer(M=d0, name="unflatten_to_sym")(inputs_enc)

# # --- Свободный слой BiMap + ReEig ---
# x = BiMapLayer(d_out=d1, name="bimap_1")(x)
# x = ReEigLayer(epsilon=1e-4, name="reeig_1")(x)

# # --- LogEig проекция ---
# x_log = LogEigAndFlattenLayer(epsilon=1e-4, name="logeig")(x)

# # Линейный слой до N_patterns латентных переменных (z)
# z_latent = tf.keras.layers.Dense(
#     N_patterns, activation="linear", use_bias=True, name="latent_alignment"
# )(x_log)

# encoder = tf.keras.Model(inputs=inputs_enc, outputs=z_latent, name="spdnet_1bire_encoder")

# =============================================================================
# 3. СБОРКА АРХИТЕКТУРЫ ЭНКОДЕРА (Диагонализирующий фильтр)
# =============================================================================

M_channels = covmats_ssd.shape[1]
N_patterns = M_channels

inputs_enc = tf.keras.Input(shape=(input_dim,), dtype=tf.float32, name="encoder_input")

# 1. Развертка в симметричные матрицы
x = UnflattenSymmetricLayer(M=M_channels, name="unflatten_to_sym")(inputs_enc)

# 2. Свободный слой BiMap (Применение пространственных фильтров W)
x = BiMapLayer(d_out=N_patterns, name="bimap_1")(x)

# 3. Берем только диагональ (мощность фильтров) и логарифмируем
# Это сразу дает нам латентный вектор z размера N_patterns!
z_latent = LogDiagLayer(epsilon=1e-4, name="log_diag")(x)

z_latent = tf.keras.layers.Dense(
    N_patterns, activation="linear", use_bias=True, name="latent_alignment"
)(z_latent)

encoder = tf.keras.Model(inputs=inputs_enc, outputs=z_latent, name="spdnet_diag_encoder")

# --------------------------------- ДЕКОДЕР ----------------------------------
inputs_dec = tf.keras.Input(shape=(N_patterns,), dtype=tf.float32, name="decoder_input")

# Линейный слой: аффинное преобразование латентного вектора.
# Инициализация identity: W = I, b = 0 — стартуем ровно как без слоя,
# чтобы обучение не деградировало на первых шагах.
z_dec = tf.keras.layers.Dense(
    N_patterns,
    activation="linear",
    use_bias=True,
    kernel_initializer=tf.keras.initializers.Identity(gain=1.0),
    bias_initializer="zeros",
    name="decoder_affine",
)(inputs_dec)

recon_out = PatternDecoder(
    M_channels=M_channels,
    N_patterns=N_patterns,
    name="spatial_decoder"
)(z_dec)

decoder = tf.keras.Model(inputs=inputs_dec, outputs=recon_out, name="spdnet_decoder")

# =============================================================================
# 5. ПОДГОТОВКА ДАННЫХ
# =============================================================================
M_channels = covmats_ssd.shape[1]
idx_i, idx_j = np.triu_indices(M_channels)
multipliers = np.where(idx_i == idx_j, 1.0, np.sqrt(2.0)).astype(np.float32)

X_cov_flat = (covmats_ssd[:, idx_i, idx_j] * multipliers).astype(np.float32)
input_dim = int(X_cov_flat.shape[1])
print(f"Обучение ParametricUMAP: N_dim={N_patterns}, N_patterns={N_patterns}")

# =============================================================================
# 7. РАЗДЕЛЕНИЕ НА ОБУЧАЮЩУЮ / ВАЛИДАЦИОННУЮ ВЫБОРКИ
# =============================================================================

indices = np.arange(len(X_cov_flat))
train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)

X_train = X_cov_flat[train_idx]
X_val = X_cov_flat[val_idx]
dist_matrix_train = dists_init[train_idx][:, train_idx]

print(f"Обучающая выборка: {len(X_train)} окон. Валидационная: {len(X_val)} окон.")

# =============================================================================
# 8. КОЛЛБЕКИ И ЗАПУСК
# =============================================================================

n_neighbors = 20
loss_weight = 1

N_dim = N_patterns

# 1. Создаем коллбек EarlyStopping с параметром patience
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='loss',        # Можно изменить на 'val_loss', если валидация полностью настроена
    patience=15,           # Количество шагов/эпох без улучшений до остановки
    min_delta=1e-5,        # Минимальное изменение для признания улучшения
    restore_best_weights=True
)

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
    # parametric_reconstruction_loss_fcn=tf.keras.losses.MeanSquaredError(),
    parametric_reconstruction_loss_weight=loss_weight,
    reconstruction_validation=X_val,
    verbose=True,
    keras_fit_kwargs={"callbacks": [early_stopping]} 
)

embedder.loss_report_frequency = 100
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
ax2.plot(history['recon_loss'], label='Train Recon Loss', color='blue', linewidth=2)
ax2.plot(history['val_recon_loss'], label='Val Recon Loss', color='green', linewidth=2)
    
ax2.set_title('Качество декодера')
ax2.set_xlabel('Шаги оценки')
ax2.set_ylabel('Riemann Loss')
ax2.legend()
ax2.grid(True, linestyle='--', alpha=0.7)

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
# 1. Извлекаем z и прогоняем через аффинный слой декодера
# =============================================================================
import numpy as np

z_encoder = embedder.encoder.predict(X_cov_flat)          # (batch, N_patterns)

affine_layer = embedder.decoder.get_layer("decoder_affine")
W_aff = affine_layer.get_weights()[0]                     # (N_patterns, N_patterns)
b_aff = affine_layer.get_weights()[1]                     # (N_patterns,)

z_dec = z_encoder @ W_aff + b_aff                         # (batch, N_patterns)

# Физические мощности паттернов, которые реально уходят в пространственный декодер
powers_true = np.exp(z_dec)                               # (batch, N_patterns)

# =============================================================================
# 2. Извлекаем паттерны (Пространственные фильтры сети больше не линейны!)
# =============================================================================

# Паттерны декодера (в SSD-пространстве)
A_ssd_learned = embedder.decoder.get_layer("spatial_decoder").get_weights()[0]

# Возврат паттернов в пространство сенсоров остается абсолютно корректным!
A_sensor_raw = np.linalg.pinv(W_ssd.T) @ A_ssd_learned

# Нормировка физических паттернов
scales = np.linalg.norm(A_sensor_raw, axis=0)
A_global_norm = A_sensor_raw / scales

# =============================================================================
# 3. Сортировка по средней мощности (по убыванию)
# =============================================================================
power_variances = np.mean(powers_true, axis=0)
sort_idx = np.argsort(power_variances)[::-1]

powers = powers_true[:, sort_idx]
A_sorted = A_global_norm[:, sort_idx]

print(f"Топ-5 компонент по средней мощности (индексы): {sort_idx[:5]}")

# =============================================================================
# 4. Списки для отрисовки
# =============================================================================
found_patterns = [A_sorted[:, i] for i in range(N_patterns)]

# Выученные фильтры нейросети (W) извлечь нельзя из-за нелинейности логарифма.
# Поэтому полагаемся исключительно на теоретические фильтры Хауфе, 
# выведенные из выученных паттернов A.

C_global_mean = np.mean(covmats_band, axis=0)
alpha = 1e-4
I = np.eye(C_global_mean.shape[0])
C_global_reg = C_global_mean + alpha * np.trace(C_global_mean) * I
C_global_inv = np.linalg.inv(C_global_reg)

found_filters_haufe = [C_global_inv @ A_sorted[:, i] for i in range(N_patterns)]

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
def format_umap_axes(ax):
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(axis='both', which='both', length=0)
    ax.grid(True, linestyle='--', alpha=0.5, zorder=0)

# =============================================================================
# ВЫБОР КОМПОНЕНТЫ
# =============================================================================
# Выбираем индекс паттерна (0..N_patterns-1)
comp_idx = 15 

# Берём фильтр, который ВЫУЧИЛА НЕЙРОСЕТЬ (из матрицы W)
W_sensor = found_filters_haufe[comp_idx] 
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

plt.suptitle(f'Анализ компоненты {comp_idx+1} из {N_patterns}', fontsize=16)
plt.tight_layout()
plt.show()

# %%
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
import mne

# =============================================================================
# 1. ПОДГОТОВКА ЦВЕТОВОЙ ПАЛИТРЫ И КЛАССОВ
# =============================================================================
# Собираем уникальные классы с сохранением порядка
unique_classes = []
for lab in cond_descriptions:
    if lab in window_labels and lab not in unique_classes:
        unique_classes.append(lab)
for lab in np.unique(window_labels):
    if lab not in unique_classes:
        unique_classes.append(lab)

# Назначаем цвета (tab20)
cmap_classes = plt.cm.tab20
color_map = {lab: cmap_classes(i / max(1, len(unique_classes) - 1)) 
             for i, lab in enumerate(unique_classes)}

# =============================================================================
# 2. ИНИЦИАЛИЗАЦИЯ ФИГУРЫ И СЕТКИ
# =============================================================================
# Уменьшаем высоту фигуры, так как оставили только 3 строки
fig = plt.figure(figsize=(18, 10))
# 3 строки, 2 колонки (соотношение ширины 1:4)
gs = GridSpec(3, 2, figure=fig, width_ratios=[1, 4], hspace=0.4)

# =============================================================================
# 3. ДЕТАЛИЗАЦИЯ ТОП-3 КОМПОНЕНТ (ТОПОГРАФИЯ + ДИНАМИКА)
# =============================================================================
mne_info = raw.info
window_idx = np.arange(len(window_labels))

for i in range(3):
    # Извлечение данных компоненты
    W_sensor = found_filters_haufe[i] 
    A_pattern = found_patterns[i]
    
    p_vals = powers[:, i]
    filtered_powers = np.array([W_sensor.T @ C @ W_sensor for C in covmats_band])
    
    p_vals_norm = p_vals / np.std(p_vals)
    filtered_powers_norm = filtered_powers / np.std(filtered_powers)

    # --- Подграфик: Пространственный паттерн ---
    ax_topo = fig.add_subplot(gs[i, 0])
    mne.viz.plot_topomap(A_pattern, mne_info, axes=ax_topo, show=False)
    ax_topo.set_title(f'Паттерн {i+1}', fontsize=12, fontweight='bold')
    
    # --- Подграфик: Временная динамика ---
    ax_ts = fig.add_subplot(gs[i, 1])
    
    # Линии мощности
    # ax_ts.plot(window_idx, filtered_powers_norm, color='gray', lw=3.0, alpha=0.4, zorder=2, label='Хауфе (Linear)')
    ax_ts.plot(window_idx, p_vals_norm, color='black', lw=1.5, zorder=3)
    
    # Цветовая заливка (события)
    xtick_positions = []
    xtick_labels = []
    
    for lab in unique_classes:
        mask = (window_labels == lab)
        if not np.any(mask): continue
        
        changes = np.diff(np.concatenate(([0], mask.astype(int), [0])))
        starts = np.where(changes == 1)[0]
        ends = np.where(changes == -1)[0]
        
        for s, e in zip(starts, ends):
            ax_ts.axvspan(s, e-1, facecolor=color_map[lab], alpha=0.25, zorder=0)
            ax_ts.axvline(x=s, color='gray', linestyle='--', alpha=0.5, zorder=1)
            # Избегаем дублирования подписей осей при частых переключениях
            if len(xtick_positions) == 0 or s - xtick_positions[-1] > 5:
                xtick_positions.append(s)
                xtick_labels.append(lab)

    ax_ts.set_xticks(xtick_positions)
    ax_ts.set_xticklabels(xtick_labels, rotation=45, ha='right', fontsize=9)
    ax_ts.set_ylabel('Мощность (норм.)', fontsize=10)
    ax_ts.grid(True, linestyle=':', alpha=0.6)
    
    # Легенда для линий (только на первом графике, чтобы не дублировать)
    if i == 0:
        ax_ts.legend(loc='upper right', ncol=2)
    
    # Подпись оси X (только на нижнем графике)
    if i == 2:
        ax_ts.set_xlabel('Номер окна (время)', fontsize=12, fontweight='bold')

# =============================================================================
# 4. ГЛОБАЛЬНАЯ ЛЕГЕНДА ДЛЯ УСЛОВИЙ
# =============================================================================
legend_handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map[lab], 
                         markersize=10, markeredgecolor='black', markeredgewidth=0.5) 
                  for lab in unique_classes]

# fig.legend(
#     handles=legend_handles, 
#     labels=unique_classes, 
#     title="Условия (События)", 
    # loc='lower center', 
    # bbox_to_anchor=(0.5, 0.02),
    # ncol=min(10, len(unique_classes)), 
    # fontsize=10, 
    # title_fontsize=12
# )

# plt.subplots_adjust(bottom=0.15, top=0.92, left=0.05, right=0.98, hspace=0.4)
plt.show()

# %%
# =============================================================================
# ДОПОЛНИТЕЛЬНАЯ ВИЗУАЛИЗАЦИЯ: ТОПОГРАФИИ, 3D ПРОСТРАНСТВО И ПРОЕКЦИИ
# =============================================================================
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import mne

# Инициализация фигуры
fig_proj = plt.figure(figsize=(18, 12))

# Создаем сетку 6 строк x 3 колонки.
# Колонка 0 (для топографий) чуть уже, колонки 1 и 2 (для скаттеров) шире.
gs_proj = GridSpec(6, 3, figure=fig_proj, width_ratios=[1, 1.5, 1.5], wspace=0.3, hspace=0.8)

# Извлекаем топ-3 компоненты
p1 = powers[:, 0]
p2 = powers[:, 1]
p3 = powers[:, 2]

# -----------------------------------------------------------------------------
# 1. ЛЕВЫЙ СТОЛБЕЦ: Топографии паттернов (3x1)
# -----------------------------------------------------------------------------
mne_info = raw.info  # Информация о сенсорах из вашего raw-файла

# Паттерн 1 (занимает строки 0 и 1 в 0-й колонке)
ax_topo1 = fig_proj.add_subplot(gs_proj[0:2, 0])
mne.viz.plot_topomap(found_patterns[0], mne_info, axes=ax_topo1, show=False)
ax_topo1.set_title('Pattern 1', fontsize=14, fontweight='bold')

# Паттерн 2 (занимает строки 2 и 3 в 0-й колонке)
ax_topo2 = fig_proj.add_subplot(gs_proj[2:4, 0])
mne.viz.plot_topomap(found_patterns[1], mne_info, axes=ax_topo2, show=False)
ax_topo2.set_title('Pattern 2', fontsize=14, fontweight='bold')

# Паттерн 3 (занимает строки 4 и 5 в 0-й колонке)
ax_topo3 = fig_proj.add_subplot(gs_proj[4:6, 0])
mne.viz.plot_topomap(found_patterns[2], mne_info, axes=ax_topo3, show=False)
ax_topo3.set_title('Pattern 3', fontsize=14, fontweight='bold')


# -----------------------------------------------------------------------------
# 2. ПРАВАЯ ЧАСТЬ: Графики рассеяния (2x2)
# -----------------------------------------------------------------------------

# --- 3D-График (Изометрический вид) --- 
# Занимает верхнюю левую часть правой зоны (строки 0-2, колонка 1)
ax_3d = fig_proj.add_subplot(gs_proj[0:3, 1], projection='3d')
ax_3d.scatter(p1, p2, p3, c=point_colors, marker='o', s=30, alpha=0.6, edgecolors='white', linewidths=0.3)

ax_3d.set_xlabel('Pattern 1')
ax_3d.set_ylabel('Pattern 2')
ax_3d.set_zlabel('Pattern 3')
ax_3d.view_init(elev=25, azim=45)

# --- Проекция XY (Pattern 1 vs Pattern 2) ---
# Занимает верхнюю правую часть (строки 0-2, колонка 2)
ax_xy = fig_proj.add_subplot(gs_proj[0:3, 2])
ax_xy.scatter(p1, p2, c=point_colors, s=25, alpha=0.6, edgecolors='white', linewidths=0.3)
ax_xy.set_xlabel('Pattern 1')
ax_xy.set_ylabel('Pattern 2')
ax_xy.grid(True, linestyle='--', alpha=0.5)

# --- Проекция XZ (Pattern 1 vs Pattern 3) ---
# Занимает нижнюю левую часть (строки 3-5, колонка 1)
ax_xz = fig_proj.add_subplot(gs_proj[3:6, 1])
ax_xz.scatter(p1, p3, c=point_colors, s=25, alpha=0.6, edgecolors='white', linewidths=0.3)
ax_xz.set_xlabel('Pattern 1')
ax_xz.set_ylabel('Pattern 3')
ax_xz.grid(True, linestyle='--', alpha=0.5)

# --- Проекция YZ (Pattern 2 vs Pattern 3) ---
# Занимает нижнюю правую часть (строки 3-5, колонка 2)
ax_yz = fig_proj.add_subplot(gs_proj[3:6, 2])
ax_yz.scatter(p2, p3, c=point_colors, s=25, alpha=0.6, edgecolors='white', linewidths=0.3)
ax_yz.set_xlabel('Pattern 2')
ax_yz.set_ylabel('Pattern 3')
ax_yz.grid(True, linestyle='--', alpha=0.5)


# -----------------------------------------------------------------------------
# 3. Глобальная легенда и форматирование
# -----------------------------------------------------------------------------
# Размещаем легенду сбоку или снизу. В данном случае удобно поместить ее внизу по центру
fig_proj.legend(
    handles=legend_handles, 
    labels=unique_classes, 
    # title="Условия",
    # loc='lower center',
    # bbox_to_anchor=(0.5, 0.02),
    # ncol=min(8, len(unique_classes))
)

# Выравниваем отступы, чтобы легенда влезла
# plt.subplots_adjust(bottom=0.1, top=0.95, left=0.05, right=0.98)
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
import mne

print("Подготовка интерактивного дашборда...")

# =============================================================================
# 1. ПРОВЕРКА ДАННЫХ
# =============================================================================

original_labels = window_labels.copy()

print("powers:", powers.shape)
print("umap_coords:", umap_coords.shape)
print("window_labels:", window_labels.shape)

assert len(window_labels) == len(powers), (
    f"Количество labels ({len(window_labels)}) не совпадает с количеством окон ({len(powers)})"
)

assert len(umap_coords) == len(powers), (
    f"Количество UMAP-точек ({len(umap_coords)}) не совпадает с количеством окон ({len(powers)})"
)

# =============================================================================
# 2. ВЫБИРАЕМ НАИБОЛЕЕ МОЩНЫЕ ИСТОЧНИКИ
# =============================================================================

power_variances = np.mean(powers, axis=0)
top_n = min(15, N_patterns)
top_indices = np.argsort(power_variances)[::-1][:top_n]

print(f"Отображаем топ-{top_n} источников с наибольшей дисперсией:")
print(top_indices + 1)

# =============================================================================
# 3. ЦВЕТА КЛАССОВ
# =============================================================================

unique_classes = []
for lab in cond_descriptions:
    if lab in original_labels and lab not in unique_classes:
        unique_classes.append(lab)

for lab in original_labels:
    if lab not in unique_classes:
        unique_classes.append(lab)

class_to_id = {lab: i + 1 for i, lab in enumerate(unique_classes)}
cmap_classes = plt.cm.tab20

if len(unique_classes) == 1:
    color_map = {unique_classes[0]: cmap_classes(0)}
else:
    color_map = {
        lab: cmap_classes(i / (len(unique_classes) - 1))
        for i, lab in enumerate(unique_classes)
    }

point_colors = [color_map[lab] for lab in original_labels]

# =============================================================================
# 4. FIGURE
# =============================================================================

N_COLS_TOPO = 3
N_ROWS_TOPO = int(np.ceil(top_n / N_COLS_TOPO))

fig = plt.figure(figsize=(18, max(8, 2.5 * N_ROWS_TOPO)))
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
    umap_coords[:, 0], umap_coords[:, 1],
    c=point_colors, s=15, alpha=0.4,
    edgecolors='white', linewidths=0.2, zorder=1
)

# =============================================================================
# 6. ЦЕНТРОИДЫ КЛАССОВ
# =============================================================================

for lab in unique_classes:
    mask = original_labels == lab
    if not np.any(mask):
        continue

    cx, cy = np.mean(umap_coords[mask, 0]), np.mean(umap_coords[mask, 1])

    ax_umap.scatter(
        cx, cy, marker='*', s=450, facecolor=color_map[lab],
        edgecolor='black', linewidths=1.0, zorder=3
    )
    ax_umap.text(
        cx, cy, str(class_to_id[lab]),
        fontsize=11, fontweight='bold', color='white',
        ha='center', va='center', zorder=4,
        path_effects=[pe.withStroke(linewidth=2.5, foreground="black")]
    )

ax_umap.set_title("Фазовое пространство UMAP", fontsize=14)
ax_umap.set_xlabel("UMAP 1")
ax_umap.set_ylabel("UMAP 2")
ax_umap.grid(True, linestyle='--', alpha=0.4, zorder=0)

# =============================================================================
# 7. ЛЕГЕНДА
# =============================================================================

handles = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map[lab], markersize=8)
    for lab in unique_classes
]
legend_labels = [f"{class_to_id[lab]}: {lab}" for lab in unique_classes]

ax_umap.legend(
    handles, legend_labels, title="Условия",
    loc='upper center', bbox_to_anchor=(0.5, -0.08),
    ncol=4, fontsize=8, title_fontsize=10
)

# =============================================================================
# 8. ИНФОРМАЦИОННАЯ ПАНЕЛЬ
# =============================================================================

txt_info = ax_text.text(
    0.5, 0.5, "Наведите курсор на точки UMAP",
    ha='center', va='center', fontsize=16,
    color='gray', fontweight='bold'
)

# =============================================================================
# 9. ТОПОМАПЫ ИСТОЧНИКОВ
# =============================================================================

print("Генерация топомапов (может занять несколько секунд)...")

ax_topos = []
overlays = []
titles = []
mne_info = raw.info

for i, comp_idx in enumerate(top_indices):
    row, col = 1 + i // N_COLS_TOPO, 2 + i % N_COLS_TOPO
    ax = fig.add_subplot(gs[row, col])

    mne.viz.plot_topomap(found_patterns[comp_idx], mne_info, axes=ax, show=False, contours=0)

    overlay = plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes, color='white', alpha=0.90, zorder=10)
    ax.add_patch(overlay)
    overlays.append(overlay)

    title = ax.set_title(f"Ист. {comp_idx + 1}\nМощн: --", fontsize=11, color='gray')
    titles.append(title)
    ax.axis('off')
    ax_topos.append(ax)

# =============================================================================
# 10. МАРКЕР ТЕКУЩЕЙ ПОЗИЦИИ
# =============================================================================

highlighted_point = ax_umap.scatter([], [], marker='+', color='black', s=150, linewidths=2.0, zorder=5)

# =============================================================================
# 11. ФУНКЦИЯ ОБНОВЛЕНИЯ DASHBOARD
# =============================================================================

# Компилируем TF-граф для инференса 2D -> 20D (z)
@tf.function
def fast_decode_to_z(z_coords):
    affine_layer = embedder.decoder.get_layer("decoder_affine")
    W_aff = affine_layer.get_weights()[0]                     # (N_patterns, N_patterns)
    b_aff = affine_layer.get_weights()[1]                     # (N_patterns,)

    return  decoder_2d(z_coords, training=False) @ W_aff + b_aff 

last_x, last_y = None, None

def update_dashboard(event):
    global last_x, last_y

    if event.inaxes != ax_umap:
        return
    if event.name == 'motion_notify_event' and event.button is not None:
        return

    x, y = event.xdata, event.ydata
    if x is None or y is None:
        return

    if last_x is not None and np.hypot(x - last_x, y - last_y) < 0.05:
        return

    last_x, last_y = x, y

    # 11.1. ДЕКОДИРОВАНИЕ 2D -> Z
    z_click = tf.constant([[x, y]], dtype=tf.float32)
    z_decoded_tf = fast_decode_to_z(z_click)
    
    # Мощность = exp(z)
    p_vals_unsorted = np.exp(z_decoded_tf.numpy()[0])
    
    # Синхронизируем сортировку с топомапами
    p_vals_sorted = p_vals_unsorted[sort_idx]
    p_top = p_vals_sorted[top_indices]

    # 11.2. ИЩЕМ БЛИЖАЙШУЮ ТОЧКУ
    dist_sq = (umap_coords[:, 0] - x)**2 + (umap_coords[:, 1] - y)**2
    nearest_label = original_labels[np.argmin(dist_sq)]

    # 11.3. ОБНОВЛЯЕМ ИНФО-ПАНЕЛЬ
    txt_info.set_text(f"Зона: {class_to_id[nearest_label]} ({nearest_label})\nКоординаты: ({x:.2f}, {y:.2f})")
    txt_info.set_color(color_map[nearest_label])

    # 11.4. ЛОКАЛЬНАЯ НОРМИРОВКА ПРОЗРАЧНОСТИ
    p_local_max, p_local_min = np.max(p_top), np.min(p_top)
    denominator = max(p_local_max - p_local_min, 1e-12)

    # 11.5. ОБНОВЛЯЕМ ТОПОМАПЫ
    for i, comp_idx in enumerate(top_indices):
        power = p_top[i]
        norm_p = np.clip((power - p_local_min) / denominator, 0, 1)
        overlays[i].set_alpha(0.95 * (1.0 - norm_p))
        titles[i].set_text(f"Ист. {comp_idx + 1}\nМощн: {power:.5f}")

        if norm_p > 0.1:
            titles[i].set_color('black')
            titles[i].set_fontweight('bold')
        else:
            titles[i].set_color('gray')
            titles[i].set_fontweight('normal')

    # 11.6. ДВИГАЕМ МАРКЕР
    highlighted_point.set_offsets([[x, y]])
    fig.canvas.draw_idle()

# =============================================================================
# 12. ПОДКЛЮЧАЕМ ИНТЕРАКТИВНОСТЬ
# =============================================================================
fig.canvas.mpl_connect('button_press_event', update_dashboard)
fig.canvas.mpl_connect('motion_notify_event', update_dashboard)

plt.subplots_adjust(bottom=0.25, top=0.90, left=0.05, right=0.98, hspace=0.4, wspace=0.1)
print("Готово! Дашборд запущен. Кликните или ведите мышь по UMAP.")
plt.show()

# %%
# =============================================================================
# ОБРАБОТЧИК КЛИКОВ (ОПТИМИЗИРОВАННЫЙ, БЕЗ AX.CLEAR())
# =============================================================================
def on_click(event):
    if event.inaxes not in ax_buttons:
        return

    comp_idx, color = ax_buttons[event.inaxes]

    # ЕСЛИ ИСТОЧНИК УЖЕ ВКЛЮЧЕН -> ВЫКЛЮЧАЕМ
    if comp_idx in active_quivers:
        # 1. Безопасное удаление стрелок
        quiver_obj = active_quivers.pop(comp_idx, None)
        if quiver_obj is not None:
            try: quiver_obj.remove()
            except Exception: pass

        # 2. Безопасное удаление топографии (заливки)
        contour_obj = active_contours.pop(comp_idx, None)
        if contour_obj is not None:
            if hasattr(contour_obj, 'remove'):
                try: contour_obj.remove()
                except Exception: pass
            elif hasattr(contour_obj, 'collections'):
                for c in contour_obj.collections:
                    try: c.remove()
                    except Exception: pass

        # 2.5. Гарантированное удаление изолиний и числовых меток
        contour_data = active_contour_lines.pop(comp_idx, None)
        if contour_data is not None:
            c_lines, c_labels = contour_data
            
            if hasattr(c_lines, 'remove'):
                try: c_lines.remove()
                except: pass
            elif hasattr(c_lines, 'collections'):
                for c in c_lines.collections:
                    try: c.remove()
                    except: pass
            
            for txt in c_labels:
                try: txt.remove()
                except: pass

        # 3. Обновляем кнопку (визуально "выключаем")
        overlays[comp_idx].set_alpha(0.85)
        for spine in borders[comp_idx].values(): 
            spine.set_visible(False)
        event.inaxes.title.set_color('gray')

    # ЕСЛИ ИСТОЧНИК ВЫКЛЮЧЕН -> ВКЛЮЧАЕМ
    else:
        # 1. Рисуем топографию (contourf)
        P_comp = P_3D[:, :, comp_idx]

        # --- НАСТРОЙКА КОНТРАСТА ПО 100% РЕАЛЬНЫХ ТОЧЕК ---
        real_p = powers[:, comp_idx]
        
        # Берем абсолютный максимум и минимум ИМЕННО ЭТОЙ компоненты
        vmax = np.max(real_p)
        vmin = np.min(real_p)

        if vmax <= vmin:
            vmax = vmin + 1e-6

        levels = np.linspace(vmin, vmax, 50)
        # ----------------------------------------------

        rgba = mcolors.to_rgba(color)
        color_transparent = (rgba[0], rgba[1], rgba[2], 0.0)
        color_solid = (rgba[0], rgba[1], rgba[2], 0.55)
        custom_cmap = mcolors.LinearSegmentedColormap.from_list(f'cmap_{comp_idx}', [color_transparent, color_solid])

        # Заливка (contourf)
        contour = ax_umap.contourf(X_grid, Y_grid, P_comp, levels=levels, cmap=custom_cmap, extend='max', zorder=1)
        active_contours[comp_idx] = contour

        # Отрисовка изолиний
        contour_lines = ax_umap.contour(X_grid, Y_grid, P_comp, levels=levels, colors=[color], alpha=1, linewidths=0.8, zorder=2)
        clabels = ax_umap.clabel(contour_lines, inline=True, fontsize=10, colors='black', fmt='%.2g')
        active_contour_lines[comp_idx] = (contour_lines, clabels) 

        # 2. Рисуем градиенты (quiver)
        U = dP_dX[:, :, comp_idx]
        V = dP_dY[:, :, comp_idx]
        
        quiver = ax_umap.quiver(X_grid, Y_grid, U, V, 
            color=color, 
            edgecolors='black', 
            linewidths=0.5,     
            alpha=0.4, 
            scale_units='xy', 
            angles='xy', 
            width=0.0015,        
            headwidth=4,        
            headlength=5,       
            headaxislength=4.5, 
            zorder=1,
            scale=0.5)          

        active_quivers[comp_idx] = quiver

        # 3. Обновляем кнопку
        overlays[comp_idx].set_alpha(0.0)
        for spine in borders[comp_idx].values(): 
            spine.set_visible(True)
        event.inaxes.title.set_color('black')

    fig.canvas.draw_idle()

# =============================================================================
# 8. ИНТЕРАКТИВНАЯ КАРТА ГРАДИЕНТОВ И СМЕШАННОЙ ТОПОГРАФИИ (ОПТИМИЗИРОВАННО)
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe
import matplotlib.colors as mcolors
import mne

print("Расчет векторных полей и подготовка топографии для латентного пространства...")

# 1. Отбираем топ-N источников 
top_n = min(15, N_patterns)
top_indices = np.arange(top_n)

# 2. Создаем регулярную сетку
grid_resolution = 50 
margin = 1.0
x_min, x_max = umap_coords[:, 0].min() - margin, umap_coords[:, 0].max() + margin
y_min, y_max = umap_coords[:, 1].min() - margin, umap_coords[:, 1].max() + margin

X_grid, Y_grid = np.meshgrid(
    np.linspace(x_min, x_max, grid_resolution),
    np.linspace(y_min, y_max, grid_resolution)
)
grid_points = np.c_[X_grid.ravel(), Y_grid.ravel()].astype(np.float32)

# =============================================================================
# 3. Предсказываем параметры
# =============================================================================
z_grid = decoder_2d.predict(grid_points, verbose=0)
affine_layer = embedder.decoder.get_layer("decoder_affine")
W_aff = affine_layer.get_weights()[0]                     
b_aff = affine_layer.get_weights()[1]                     
z_grid = z_grid @ W_aff + b_aff                         

p_grid = np.exp(z_grid)
p_sorted_grid = p_grid[:, sort_idx]
P_3D = p_sorted_grid.reshape(grid_resolution, grid_resolution, N_patterns)

# --- БЛОКИРОВКА ЭКСТРАПОЛЯЦИОННЫХ ВЗРЫВОВ ---
# Находим 99.5-й перцентиль для каждой компоненты из РЕАЛЬНЫХ данных
real_max_bounds = np.percentile(powers, 99.5, axis=0)

# Обрезаем сетку P_3D. Все значения, превышающие реальный максимум, станут плато.
for i in range(N_patterns):
    P_3D[:, :, i] = np.clip(P_3D[:, :, i], a_min=None, a_max=real_max_bounds[i])
# ---------------------------------------------

# 4. Градиенты от обрезанной абсолютной мощности
# На образовавшихся плато градиент будет строго 0, поэтому стрелки там исчезнут
dy = (y_max - y_min) / (grid_resolution - 1)
dx = (x_max - x_min) / (grid_resolution - 1)

dP_dY, dP_dX = np.gradient(P_3D, dy, dx, axis=(0, 1))

# =============================================================================
# ПОДГОТОВКА ИНТЕРФЕЙСА И ЦВЕТОВ
# =============================================================================
active_contours = {} 
active_quivers = {}
active_contour_lines = {} 

distinct_colors = [
    '#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', 
    '#911eb4', '#42d4f4', '#f032e6', '#bfef45', '#fabed4', 
    '#469990', '#dcbeff', '#9A6324', '#fffac8', '#800000'
]
source_colors = distinct_colors[:top_n]

N_COLS_TOPO = 3
N_ROWS_TOPO = int(np.ceil(top_n / N_COLS_TOPO))

fig = plt.figure(figsize=(18, max(8, 2.5 * N_ROWS_TOPO)))
gs = GridSpec(N_ROWS_TOPO, N_COLS_TOPO + 2, width_ratios=[3.0, 0.2] + [1]*N_COLS_TOPO)

ax_umap = fig.add_subplot(gs[:, 0])

original_labels = window_labels.copy()
unique_classes = []
for lab in cond_descriptions:
    if lab in original_labels and lab not in unique_classes:
        unique_classes.append(lab)
for lab in original_labels:
    if lab not in unique_classes:
        unique_classes.append(lab)

class_to_id = {lab: i + 1 for i, lab in enumerate(unique_classes)}
cmap_bg = plt.cm.tab20
color_map_bg = {lab: cmap_bg(i / max(1, len(unique_classes) - 1)) for i, lab in enumerate(unique_classes)}
point_colors_bg = [color_map_bg[lab] for lab in original_labels]

mne_info = raw.info 
ax_buttons = {}  
overlays = {}    
borders = {}     

for i, comp_idx in enumerate(top_indices):
    row = i // N_COLS_TOPO
    col = 2 + i % N_COLS_TOPO
    ax = fig.add_subplot(gs[row, col])

    mne.viz.plot_topomap(found_patterns[comp_idx], mne_info, axes=ax, show=False, contours=0)

    overlay = plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes, color='white', alpha=0.85, zorder=10)
    ax.add_patch(overlay)
    overlays[comp_idx] = overlay

    border_color = source_colors[i % len(source_colors)]
    for spine in ax.spines.values():
        spine.set_edgecolor(border_color)
        spine.set_linewidth(5)
        spine.set_visible(False)
    borders[comp_idx] = ax.spines

    ax.set_title(f"Ист. {comp_idx+1}", fontsize=11, fontweight='bold', color='gray')
    ax.set_xticks([])
    ax.set_yticks([])
    ax_buttons[ax] = (comp_idx, border_color)

# =============================================================================
# ПЕРВИЧНАЯ ОТРИСОВКА БАЗОВОГО СЛОЯ UMAP
# =============================================================================
ax_umap.scatter(umap_coords[:, 0], umap_coords[:, 1], c=point_colors_bg, 
                s=25, alpha=0.5, edgecolors='white', linewidths=0.3, zorder=3)

for lab in unique_classes:
    mask = (original_labels == lab)
    if not np.any(mask): continue
    cx, cy = np.mean(umap_coords[mask], axis=0)

    ax_umap.scatter(cx, cy, marker='*', s=350, facecolor=color_map_bg[lab], 
                    edgecolor='black', linewidths=0.8, zorder=5, alpha=0.95)

    ax_umap.text(cx, cy, str(class_to_id[lab]), fontsize=11, color='white', 
                 fontweight='bold', ha='center', va='center', zorder=6,
                 path_effects=[pe.withStroke(linewidth=2.5, foreground="black")])

legend_elements = [
    Line2D([0], [0], marker='o', color='w', label=f"{class_to_id[lab]}: {lab}",
           markerfacecolor=color_map_bg[lab], markersize=8, 
           markeredgecolor='black', markeredgewidth=0.5) 
    for lab in unique_classes
]
ax_umap.legend(handles=legend_elements, title="Условия / Классы", loc='upper center', 
               bbox_to_anchor=(0.5, -0.08), ncol=6, fontsize=9, title_fontsize=10)

ax_umap.set_xlabel("UMAP 1")
ax_umap.set_ylabel("UMAP 2")
ax_umap.grid(True, linestyle='--', alpha=0.3, zorder=0)
ax_umap.set_xlim(x_min, x_max)
ax_umap.set_ylim(y_min, y_max)

fig.canvas.mpl_connect('button_press_event', on_click)

plt.subplots_adjust(left=0.05, right=0.98, bottom=0.18, top=0.92, wspace=0.1, hspace=0.3)
print("Готово! Выбирайте источники справа: их топографии и векторные поля будут мгновенно появляться на графике.")
plt.show()

# %%
# =============================================================================
# СРАВНЕНИЕ ПРОСТРАНСТВЕННЫХ ПАТТЕРНОВ: ICA vs AUTOENCODER
# =============================================================================
from scipy.optimize import linear_sum_assignment
import mne
import numpy as np
import matplotlib.pyplot as plt

print("Извлечение и нормализация паттернов...")

# 1. Извлекаем паттерны ICA (матрица смешивания в пространстве сенсоров)
# ica.get_components() возвращает массив формы (n_channels, n_components)
ica_patterns = ica.get_components()

# 2. Извлекаем паттерны автоэнкодера
# found_patterns - это список из N_patterns векторов. Собираем их в матрицу (n_channels, N_patterns)
ae_patterns = np.column_stack(found_patterns)

# 3. Нормализуем столбцы (L2-норма) для корректного расчета косинусного сходства
ica_patterns_norm = ica_patterns / (np.linalg.norm(ica_patterns, axis=0, keepdims=True) + 1e-12)
ae_patterns_norm = ae_patterns / (np.linalg.norm(ae_patterns, axis=0, keepdims=True) + 1e-12)

print("Сопоставление компонент (Венгерский алгоритм)...")

# 4. Вычисляем матрицу косинусного сходства
# Берем модуль, так как полярность (знак) паттерна не влияет на его физический смысл
similarity_matrix = np.abs(ica_patterns_norm.T @ ae_patterns_norm)

# 5. Ищем оптимальное соответствие (максимизируем сходство)
row_ind, col_ind = linear_sum_assignment(-similarity_matrix)

# =============================================================================
# ВИЗУАЛИЗАЦИЯ НАИБОЛЕЕ ПОХОЖИХ ПАТТЕРНОВ
# =============================================================================
# Собираем совпадения в список и сортируем по убыванию сходства
matches = [(row_ind[i], col_ind[i], similarity_matrix[row_ind[i], col_ind[i]]) for i in range(len(row_ind))]
matches.sort(key=lambda x: x[2], reverse=True)

# Отрисуем топ-5 совпадений
top_n = min(10, len(matches))
top_n = min(10, len(matches))
top_matches = matches[:top_n]

fig, axes = plt.subplots(2, top_n, figsize=(3 * top_n, 5.5))
fig.suptitle('Сравнение пространственных паттернов: ICA (верх) vs Autoencoder (низ)', fontsize=15, y=1.05)

mne_info = raw.info

for i, (ica_idx, ae_idx, score) in enumerate(top_matches):
    # Корректируем знак (полярность) AE паттерна для визуального совпадения с ICA
    sign = np.sign(np.dot(ica_patterns_norm[:, ica_idx], ae_patterns_norm[:, ae_idx]))
    matched_ae_pattern = ae_patterns[:, ae_idx] * sign

    # Верхний ряд: ICA
    ax_ica = axes[0, i] if top_n > 1 else axes[0]
    mne.viz.plot_topomap(ica_patterns[:, ica_idx], mne_info, axes=ax_ica, show=False)
    ax_ica.set_title(f'ICA Comp {ica_idx}', fontsize=11)

    # Нижний ряд: Autoencoder
    ax_ae = axes[1, i] if top_n > 1 else axes[1]
    mne.viz.plot_topomap(matched_ae_pattern, mne_info, axes=ax_ae, show=False)
    
    # Выделяем зеленым цветом сильные совпадения (> 0.8)
    color = 'green' if score > 0.8 else 'black'
    ax_ae.set_title(f'AE Comp {ae_idx}\nCorr: {score:.3f}', fontsize=11, color=color)

plt.tight_layout()
plt.show()

# %%
# =============================================================================
# ВИЗУАЛИЗАЦИЯ ВСЕХ СОПОСТАВЛЕННЫХ ПАТТЕРНОВ
# =============================================================================
import math

# Собираем все совпадения в список и сортируем по убыванию сходства
matches = [(row_ind[i], col_ind[i], similarity_matrix[row_ind[i], col_ind[i]]) for i in range(len(row_ind))]
matches.sort(key=lambda x: x[2], reverse=True)

n_matches = len(matches)
n_cols = 5  # Количество пар в одном ряду (можно изменить)
n_pair_rows = math.ceil(n_matches / n_cols)
n_rows = n_pair_rows * 2  # Каждой паре нужно 2 ряда: сверху ICA, снизу AE

fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 2.5 * n_rows))
fig.suptitle('Сравнение всех пространственных паттернов: ICA (верх) vs Autoencoder (низ)', fontsize=16, y=1.02)

# Убедимся, что axes всегда двумерный массив
if n_rows == 1:
    axes = np.expand_dims(axes, axis=0)
if n_cols == 1:
    axes = np.expand_dims(axes, axis=1)

mne_info = raw.info

for i, (ica_idx, ae_idx, score) in enumerate(matches):
    # Вычисляем индексы для сетки
    pair_row = i // n_cols
    col = i % n_cols
    
    row_ica = pair_row * 2
    row_ae = pair_row * 2 + 1
    
    ax_ica = axes[row_ica, col]
    ax_ae = axes[row_ae, col]
    
    # Корректируем знак (полярность) AE паттерна
    sign = np.sign(np.dot(ica_patterns_norm[:, ica_idx], ae_patterns_norm[:, ae_idx]))
    matched_ae_pattern = ae_patterns[:, ae_idx] * sign

    # Отрисовка ICA
    mne.viz.plot_topomap(ica_patterns[:, ica_idx], mne_info, axes=ax_ica, show=False)
    ax_ica.set_title(f'ICA Comp {ica_idx}', fontsize=11)

    # Отрисовка Autoencoder
    mne.viz.plot_topomap(matched_ae_pattern, mne_info, axes=ax_ae, show=False)
    
    # Выделение цвета по порогу сходства
    color = 'green' if score > 0.8 else ('orange' if score > 0.5 else 'black')
    ax_ae.set_title(f'AE Comp {ae_idx}\nCorr: {score:.3f}', fontsize=11, color=color)

# Отключаем пустые оси, если количество паттернов не кратно n_cols
for i in range(n_matches, n_cols * n_pair_rows):
    pair_row = i // n_cols
    col = i % n_cols
    axes[pair_row * 2, col].axis('off')
    axes[pair_row * 2 + 1, col].axis('off')

plt.tight_layout()
plt.show()

# %%
# =============================================================================
# СРАВНЕНИЕ ПРОСТРАНСТВЕННЫХ ПАТТЕРНОВ: ICA vs AUTOENCODER (РАЗДЕЛЬНЫЕ ГРАФИКИ)
# =============================================================================
from scipy.optimize import linear_sum_assignment
import mne
import numpy as np
import matplotlib.pyplot as plt

print("Извлечение и нормализация паттернов...")

# 1. Извлекаем паттерны ICA (матрица смешивания)
ica_patterns = ica.get_components()

# 2. Извлекаем паттерны автоэнкодера (матрица A)
ae_patterns = np.column_stack(found_patterns)

# 3. Нормализуем столбцы для корректного расчета косинусного сходства
ica_patterns_norm = ica_patterns / (np.linalg.norm(ica_patterns, axis=0, keepdims=True) + 1e-12)
ae_patterns_norm = ae_patterns / (np.linalg.norm(ae_patterns, axis=0, keepdims=True) + 1e-12)

# 4. Вычисляем матрицу косинусного сходства (абсолютные значения)
similarity_matrix = np.abs(ica_patterns_norm.T @ ae_patterns_norm)

# 5. Оптимальное сопоставление (Венгерский алгоритм)
row_ind, col_ind = linear_sum_assignment(-similarity_matrix)

# =============================================================================
# ГРАФИК 1: СТАНДАРТНОЕ ОТОБРАЖЕНИЕ ICA
# =============================================================================
print("Отрисовка компонент ICA...")
ica.plot_components(title="ICA Components")

# =============================================================================
# ГРАФИК 2: ОТОБРАЖЕНИЕ AUTOENCODER ПАТТЕРНОВ (ОТСОРТИРОВАННЫХ ПО ICA)
# =============================================================================
print("Отрисовка сопоставленных компонент Автоэнкодера...")

# Собираем совпадения в список
matches = [(row_ind[i], col_ind[i], similarity_matrix[row_ind[i], col_ind[i]]) for i in range(len(row_ind))]

# СОРТИРУЕМ ПО ИНДЕКСУ ICA, чтобы порядок следования картинок AE совпадал с графиком ICA
matches.sort(key=lambda x: x[0])

n_components = len(matches)
n_cols = 5
n_rows = int(np.ceil(n_components / n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.5, n_rows * 2.5))
fig.suptitle('Autoencoder Components (отсортированы в порядке ICA)', fontsize=16, y=1.02)

axes = axes.flatten()
mne_info = raw.info

for i, (ica_idx, ae_idx, score) in enumerate(matches):
    ax = axes[i]
    
    # Корректируем знак (полярность) AE паттерна для визуального совпадения с ICA
    sign = np.sign(np.dot(ica_patterns_norm[:, ica_idx], ae_patterns_norm[:, ae_idx]))
    matched_ae_pattern = ae_patterns[:, ae_idx] * sign

    # Отрисовка топограммы
    mne.viz.plot_topomap(matched_ae_pattern, mne_info, axes=ax, show=False)
    
    # Выделяем цветом сильные совпадения
    color = 'green' if score > 0.8 else ('orange' if score > 0.5 else 'black')
    
    # Заголовок: номер AE компоненты + с какой ICA компонентой совпала + корреляция
    ax.set_title(f'AE Comp {ae_idx}\n(ICA {ica_idx} | r={score:.2f})', fontsize=10, color=color)

# Прячем пустые оси, если компонент меньше, чем ячеек в сетке
for j in range(i + 1, len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.show()
