# -*- coding: utf-8 -*-
"""
Created on Wed Oct 22 17:07:11 2025

@author: anton
"""

import mne
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import butter, filtfilt
from scipy.linalg import eigh
from scipy.linalg import inv, null_space

import sys
import os

lib_directory = os.path.abspath("C:/Users/ansbel/Documents/GitHub/pyRiemann") 
if lib_directory not in sys.path:
    sys.path.insert(0, lib_directory)
from pyriemann.estimation import Covariances

from topological_spatial_filter import fit_filters

# %%
fpath = "C:/Users/ansbel/Documents/GitHub/TriCo/data/external/music_listening/part1/eeg/10_07_g1_2223_raw.fif"
# fpath = "C:/Users/ansbel/Documents/GitHub/TriCo/data/external/music_listening/part2/eeg/Tumyalis_clear.fif"

raw = mne.io.read_raw_fif(fpath,preload=True)
sfreq = raw.info['sfreq']

# %%
raw.plot()

# %%
new_descriptions = [
    'RS_EC_1', 'RS_EO_1', '2Hz', '05Hz', '4Hz', '1Hz', '3Hz',
    'NoRy_1', 'Waltz_1', 'Waltz_2', 'NoRy_2', 'NoRy_3', 'Waltz_3',
    'NoRy_4', 'Waltz_4', 'NoRy_5', 'Waltz_5', 'RS_EC_2', 'RS_EO_2',
    # 'Waltz_6', 'Waltz_7', 'Waltz_8'
]

descriptions = raw.annotations.description

# Индексы значимых аннотаций (не BAD, не EDGE)
significant_mask = np.array([('BAD' not in d and 'EDGE' not in d) for d in descriptions])
significant_indices = np.where(significant_mask)[0]

# Проверка соответствия
if len(significant_indices) != len(new_descriptions):
    raise ValueError(
        f"Несоответствие: {len(significant_indices)} значимых, ожидается {len(new_descriptions)}"
    )

# Создаём новый список описаний
new_desc = list(descriptions)
for idx, label in zip(significant_indices, new_descriptions):
    new_desc[idx] = label

# %%
# Создаём новый объект Annotations с обновлёнными описаниями
old_annot = raw.annotations
new_annot = mne.Annotations(
    onset=old_annot.onset,
    duration=old_annot.duration,
    # duration=120,
    description=np.array(new_desc, dtype='U20'),  # тип с запасом по длине
    orig_time=old_annot.orig_time
)

# Применяем новые аннотации к raw_ica
raw.set_annotations(new_annot)

# Проверяем изменение
print(raw.annotations.description)

# %%
# Получаем данные после ICA (все каналы EEG)
raw_clean = raw.copy().pick_types(eeg=True)
data = raw.get_data()

# Списки для хранения отфильтрованных и обрезанных кусков
signal_pieces = []
noise_pieces = []
times_pieces = []

# Параметры фильтров
b_signal, a_signal = butter(3, np.array([15, 25]) / (int(sfreq) / 2), btype='band')
b_broad, a_broad = butter(3, np.array([13, 30]) / (int(sfreq) / 2), btype='band')
b_stop, a_stop = butter(3, np.array([14.5, 25.5]) / (int(sfreq) / 2), btype='stop')

# Длительность для обрезания краёв (в секундах)
crop_duration = 0.5  # 0.5 секунды с каждой стороны
crop_samples = int(crop_duration * sfreq)

# Проходим по аннотациям
for annot in raw.annotations:
    desc = annot['description']
    if desc == 'BAD_':
    # if desc == 'BAD_' or desc == 'RS_EC_1' or desc == 'RS_EC_2':
        continue
    duration = annot['duration']
    if duration < 2.0:  # слишком короткий блок - пропускаем
        continue
    print(desc)

    tmin = annot['onset']
    tmax = tmin + duration
    if tmax > raw.times[-1]:
        tmax = raw.times[-1]

    # Вырезаем кусок данных
    start_idx = int(tmin * sfreq)
    end_idx = int(tmax * sfreq)
    seg_data = data[:, start_idx:end_idx]

    # Фильтрация сигнала (15–25 Гц)
    seg_signal = filtfilt(b_signal, a_signal, seg_data, axis=1)

    # Фильтрация шума: широкополосный 13–27 Гц, затем режекция
    seg_noise_broad = filtfilt(b_broad, a_broad, seg_data, axis=1)
    seg_noise = filtfilt(b_stop, a_stop, seg_noise_broad, axis=1)

    # Обрезаем края для удаления переходных процессов
    if seg_signal.shape[1] > 2 * crop_samples:
        seg_signal_cropped = seg_signal[:, crop_samples:-crop_samples]
        seg_noise_cropped = seg_noise[:, crop_samples:-crop_samples]
    else:
        # Если блок слишком короткий, не обрезаем (или пропускаем)
        seg_signal_cropped = seg_signal
        seg_noise_cropped = seg_noise

    # Добавляем в общий список
    signal_pieces.append(seg_signal_cropped)
    noise_pieces.append(seg_noise_cropped)

# Сшиваем все куски в один длинный массив
if signal_pieces:
    signal_concatenated = np.concatenate(signal_pieces, axis=1)
    noise_concatenated = np.concatenate(noise_pieces, axis=1)
    print(f"Сшито {len(signal_pieces)} блоков, общая длина: {signal_concatenated.shape[1]} отсчётов.")
else:
    raise ValueError("Нет подходящих блоков для SSD!")

# Вычисляем ковариации по сшитому сигналу
C_signal = np.cov(signal_concatenated)
C_noise = np.cov(noise_concatenated)

# Регуляризация шумовой ковариации
reg_coeff = 1e-5
C_noise_reg = C_noise + reg_coeff * np.trace(C_noise) * np.eye(C_noise.shape[0])

# Обобщённая проблема собственных значений
eigvals, eigvecs = eigh(C_signal, C_noise_reg)
idx_sorted = np.argsort(eigvals)[::-1]
W_ssd = eigvecs
component_variances = np.diag(W_ssd.T @ C_signal @ W_ssd)
valid_components = component_variances > 0
n_components_ssd = len(valid_components[valid_components==True])

W_ssd = W_ssd[:, valid_components]
variances_filtered = component_variances[valid_components]
W_ssd = W_ssd / np.sqrt(variances_filtered)

A_ssd = C_signal @ W_ssd

print(f"SSD выполнено на сшитых блоках, получено {n_components_ssd} компонент.")

# %%
Wsize = 2
Ssize = 0.5
overlap = Wsize - Ssize

X_windows_band = []
X_windows_ssd = []
X_windows_unfilt = []

labels = []
trials = []
trial_id = 1

raw_band = raw.copy().filter(l_freq=15, h_freq=25).pick_types(eeg=True)
raw_unfilt = raw.copy().pick_types(eeg=True)

for annot in raw.annotations:
    desc = annot['description']

    base_cond = desc

    if annot['duration'] < Wsize or desc == 'BAD_':
    # if annot['duration'] < Wsize or desc == 'BAD_' or desc == 'RS_EC_1' or desc == 'RS_EC_2':
        continue
    print(desc)

    tmin = annot['onset']
    tmax = tmin + annot['duration']
    if tmax > raw.times[-1]:
        tmax = raw.times[-1]

    raw_crop = raw_band.copy().crop(tmin=tmin, tmax=tmax)
    raw_crop_unfilt = raw_unfilt.copy().crop(tmin=tmin, tmax=tmax)

    epochs_band = mne.make_fixed_length_epochs(
        raw_crop,
        duration=Wsize,
        overlap=overlap,
        preload=True,
        reject_by_annotation=True,
        verbose=False
    )
    epochs_band.drop_bad(verbose=False)

    epochs_ssd = mne.make_fixed_length_epochs(
        raw_crop,
        duration=Wsize,
        overlap=overlap,
        preload=True,
        reject_by_annotation=True,
        verbose=False
    )
    epochs_ssd.drop_bad(verbose=False)

    epochs_unfilt = mne.make_fixed_length_epochs(
        raw_crop_unfilt,
        duration=Wsize,
        overlap=overlap,
        preload=True,
        reject_by_annotation=True,
        verbose=False
    )
    epochs_unfilt.drop_bad(verbose=False)

    if epochs_band:
        if len(epochs_band) > 0 and len(epochs_band) == len(epochs_unfilt):
            X_windows_band.append(epochs_band.get_data(copy=False))
            X_windows_unfilt.append(epochs_unfilt.get_data(copy=False))
            labels.extend([base_cond] * len(epochs_band))
            trials.extend([trial_id] * len(epochs_band))
            trial_id += 1

if len(X_windows_band) > 0:
    X_windows_band = np.concatenate(X_windows_band, axis=0)
    X_windows_unfilt = np.concatenate(X_windows_unfilt, axis=0)
    labels = np.array(labels)
    trials = np.array(trials)
else:
    raise ValueError("После удаления артефактов не осталось ни одного чистого окна!")

# %%
# =====================================================================
# ПРОЕКЦИЯ ЭПОХ В SSD-ПРОСТРАНСТВО
# =====================================================================
# X_windows_band имеет размер (n_windows, n_channels, n_times)
n_windows, n_ch, n_times = X_windows_band.shape
X_windows_ssd_proj = np.zeros((n_windows, n_components_ssd, n_times))
for i in range(n_windows):
    X_windows_ssd_proj[i] = W_ssd.T @ X_windows_band[i]   # (n_comp, time)

print(f"Эпохи спроецированы в SSD-пространство, размер: {X_windows_ssd_proj.shape}")

# %%
covmats_band = Covariances(estimator='oas').fit_transform(X_windows_band)
covmats_ssd = Covariances(estimator='oas').fit_transform(X_windows_ssd_proj)
covmats = Covariances(estimator='oas').fit_transform(X_windows_band)

# %%
from pyriemann.geometry.distance import pairwise_distance

dist_matrix_init = pairwise_distance(covmats_ssd, metric='euclid')
# dist_matrix_init = pairwise_distance(covmats, metric='riemann')
N_neig = 20

# %%
import numpy as np
import umap
import matplotlib as mpl
from matplotlib.lines import Line2D

print("Вычисление 3D UMAP для матрицы расстояний...")
# Считаем 3D вложение (random_state можно убрать, если хочешь разную генерацию)
umap_3d = umap.UMAP(n_components=3, n_neighbors=20, metric='precomputed')
coords_3d = umap_3d.fit_transform(dist_matrix_init)

# %%
# Сохраняем уникальные условия в порядке их появления
unique_labels_ordered = []
for lab in labels:
    if lab not in unique_labels_ordered:
        unique_labels_ordered.append(lab)

# Берем широкую палитру, чтобы хватило на все твои 19 условий (например, tab20)
cond_cmap = mpl.colormaps['tab20'] 
label_to_color = {lab: cond_cmap(i % 20) for i, lab in enumerate(unique_labels_ordered)}

# --- Создаем фигуру ---
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

ax.tick_params(axis='x')
ax.tick_params(axis='y')
ax.tick_params(axis='z')

legend_handles = []

# Для красивого смещения текста над центроидом найдем разброс по Z
z_range = coords_3d[:, 2].max() - coords_3d[:, 2].min()

# --- Отрисовка точек и центроидов ---
for idx, lab in enumerate(unique_labels_ordered):
    cond_num = idx + 1  # Номер условия: 1, 2, 3...
    mask = (labels == lab)
    points = coords_3d[mask]
    color = label_to_color[lab]

    # 1. Облако точек (полупрозрачное)
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
               c=[color], s=15, alpha=0.6)
    
    # 2. Вычисляем координаты центроида
    centroid = points.mean(axis=0)
    
    # 3. Рисуем центроид (крупный крестик или звезда)
    ax.scatter(centroid[0], centroid[1], centroid[2], 
               c=[color], s=200, marker='X', linewidths=1.5, zorder=5)
    
    # 4. Подписываем центроид цифрой (чуть выше самого маркера)
    ax.text(centroid[0], centroid[1], centroid[2] + z_range * 0.05, 
            str(cond_num), fontsize=14, fontweight='bold', 
            ha='center', va='bottom', zorder=6)
    
    # 5. Добавляем элемент для легенды
    legend_handles.append(
        Line2D([0], [0], marker='o', color='none', markerfacecolor=color, 
               markersize=10, label=f'({cond_num}) {lab}')
    )

ax.set_title('3D Топология состояний')
ax.set_xlabel('UMAP 1')
ax.set_ylabel('UMAP 2')
ax.set_zlabel('UMAP 3')

# Добавляем легенду за пределами графика справа
leg = ax.legend(handles=legend_handles, loc='center left', bbox_to_anchor=(1.05, 0.5))

# %%
# =============================================================================
# ОТРИСОВКА МАТРИЦЫ РАССТОЯНИЙ С БЛОКАМИ УСЛОВИЙ И ПОДПИСЯМИ
# =============================================================================
import matplotlib.pyplot as plt
import numpy as np

# 1. Определяем границы блоков по меткам условий
unique_labels, label_indices = np.unique(labels, return_inverse=True)
change_idx = np.where(np.diff(label_indices) != 0)[0] + 1
block_bounds = np.concatenate([[0], change_idx, [len(labels)]])

# 2. Центры блоков и их метки (берём метку из первого окна каждого блока)
block_centers = [(block_bounds[i] + block_bounds[i+1]) / 2 for i in range(len(block_bounds)-1)]
block_labels = [labels[block_bounds[i]] for i in range(len(block_bounds)-1)]

# 3. Создаём фигуру и axis
fig, ax = plt.subplots(figsize=(10, 8))

# Находим границы (отбрасываем 2% экстремальных значений с обеих сторон)
vmin = np.percentile(dist_matrix_init, 5)
vmax = np.percentile(dist_matrix_init, 95)

# 4. Показываем матрицу расстояний с vmin и vmax
im = ax.imshow(dist_matrix_init, cmap='viridis', aspect='auto',
               vmin=vmin, vmax=vmax,
               extent=[0, dist_matrix_init.shape[1],
                       0, dist_matrix_init.shape[0]])
# 5. Чёрные пунктирные линии по границам блоков
for bound in block_bounds[1:-1]:
    ax.axvline(x=bound, color='black', linestyle='--', linewidth=1.5)
    ax.axhline(y=bound, color='black', linestyle='--', linewidth=1.5)

# 6. Подписи осей — реальные названия условий по центрам блоков
ax.set_xticks(block_centers)
ax.set_xticklabels(block_labels, rotation=45, ha='right', fontsize=8)
ax.set_yticks(block_centers)
ax.set_yticklabels(block_labels, fontsize=8)

# 7. Прозрачный фон
fig.patch.set_alpha(0.0)
ax.patch.set_alpha(0.0)

# 9. Колорбар с белым текстом и прозрачным фоном
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Distance')

plt.tight_layout()
plt.show()

# %%
from topological_spatial_filter import fit_filters

current_dim = covmats_ssd.shape[1]          # исходная размерность после SSD
covmats_current = covmats_ssd.copy()
# current_dim = covmats.shape[1]          # исходная размерность после SSD
# covmats_current = covmats.copy()

filters_full = []        
patterns_full = []      
losses = []
umap_coords_history = []
power_history = []
dims_history = []
scales_history = []

B_cumulative = np.eye(current_dim)
N_epochs = covmats_ssd.copy().shape[0]

dist_matrix_current = dist_matrix_init.copy()
distances_history = [dist_matrix_current]

N_NEIG = 20
N = 10
for comp in range(N):
    print(f"\n=== Компонент {comp+1} (текущая размерность {current_dim}) ===")

    # 1. поиск фильтра w
    w_opt, scales_opt, final_losses, loss_history, ind_losses = fit_filters(
        C=covmats_current,
        D_matrix=dist_matrix_current,
        N_dim=2,
        K_restarts=1,
        n_neighbors=N_NEIG,
        epochs=500,
        lr=0.05
    )
    print(ind_losses)
    scales_history.append(scales_opt)
    losses.append(loss_history)
    w_current = w_opt[0, 0, :]
    a_current = np.mean(covmats_current,0) @ w_current

    p_comp = np.zeros(N_epochs)
    for i in range(N_epochs):
        p_comp[i] = np.log(w_current.T @ covmats_current[i] @ w_current)

    power_history.append(p_comp)

    filters_full.append(B_cumulative @ w_current)
    patterns_full.append(B_cumulative @ a_current)

    umap_step = umap.UMAP(n_components=2, n_neighbors=N_NEIG, metric='precomputed')
    coords = umap_step.fit_transform(dist_matrix_current)

    umap_coords_history.append(coords)
    dims_history.append(current_dim)

    if comp < N - 1:
        Pu = null_space(a_current.reshape(1, -1))

        covmats_new = np.zeros((N_epochs, current_dim-1, current_dim-1))
        for i in range(N_epochs):
            c_new = Pu.T @ covmats_current[i] @ Pu
            covmats_new[i] = c_new + 1e-6 * np.trace(c_new) * np.eye(current_dim - 1)

        B_cumulative = B_cumulative @ Pu
        current_dim -= 1
        covmats_current = covmats_new
        
        dist_matrix_current = pairwise_distance(covmats_current, metric='riemann')
        distances_history.append(dist_matrix_current)

    print(f"  Потери: {final_losses[-1]:.4f}")

# %%
aligned_umap_coords = []
target_coords = None

for coords in umap_coords_history:
    if coords is None:
        aligned_umap_coords.append(None)
        continue

    if target_coords is None:
        # первое вложение оставляем как есть
        aligned_umap_coords.append(coords)
        target_coords = coords
    else:
        # Центрируем оба набора
        mean_src = np.mean(coords, axis=0)
        mean_tgt = np.mean(target_coords, axis=0)
        s0 = coords - mean_src
        t0 = target_coords - mean_tgt

        # Поиск оптимального поворота/отражения (алгоритм Кабша)
        H = s0.T @ t0
        U, S, Vt = np.linalg.svd(H)
        R = U @ Vt

        # Оптимальный масштаб
        scale = np.sum(t0 * (s0 @ R)) / (np.sum(s0 * s0) + 1e-8)

        # Применяем трансформацию и возвращаем на место целевого центра
        aligned_coords = (s0 @ R) * scale + mean_tgt

        aligned_umap_coords.append(aligned_coords)
        target_coords = aligned_coords   # следующая итерация будет выравниваться к этому

umap_coords_history = aligned_umap_coords
# ==============================================================================

# %%
# ========== График 1: кривые обучения ==========
plt.figure(figsize=(10, 6))
for idx, loss in enumerate(losses):
    plt.plot(loss, label=f'Компонент {idx+1}')
plt.xlabel('Эпоха оптимизации')
plt.ylabel('Loss')
plt.title('Эволюция функции потерь для каждого фильтра')
plt.legend()
plt.grid(True)
plt.show()

# ========== График 2: UMAP на каждом шаге дефляции ==========
n_steps = len(umap_coords_history)
cols = int(np.ceil(n_steps / 2))
fig, axes = plt.subplots(2, cols, figsize=(4*cols, 8))
axes = axes.flatten()
for step, (coords, dim) in enumerate(zip(umap_coords_history, dims_history)):
    ax = axes[step]
    if coords is None:
        ax.text(0.5, 0.5, f'Шаг {step}\n(одномерное пространство)', ha='center', va='center')
        ax.set_title(f'Шаг {step}')
    else:
        p_vals = power_history[step]
        ax.scatter(coords[:, 0], coords[:, 1], c=p_vals, cmap='plasma', s=10)
        ax.set_title(f'Шаг {step} (размерность {dim})')
        ax.grid(True, linestyle='--', alpha=0.6)

for ax in axes[n_steps:]:
    ax.set_visible(False)
plt.suptitle('Эволюция топологии касательного пространства по шагам дефляции', fontsize=14)
plt.tight_layout()
plt.show()

# %%
from matplotlib.gridspec import GridSpec
import matplotlib as mpl
import numpy as np

# Функция для настройки осей UMAP: оставляет сетку, но убирает цифры
def format_umap_axes(ax):
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(axis='both', which='both', length=0)
    ax.grid(True, linestyle='--', alpha=0.5, zorder=0)

comp_idx = 0  # выберите нужный компонент

w_comp = filters_full[comp_idx]
a_comp = patterns_full[comp_idx]

A_pattern = A_ssd @ a_comp     # Паттерн (Forward Model)
W_sensor = W_ssd @ w_comp      # Фильтр (Inverse Model)

p_vals = power_history[comp_idx]

# UMAP дефлированного пространства для данного этапа
umap_undefl = umap_coords_history[comp_idx]
umap_defl = umap_coords_history[comp_idx+1]

# Порядок условий: сохраняем порядок появления
unique_labels_ordered = []
for lab in labels:
    if lab not in unique_labels_ordered:
        unique_labels_ordered.append(lab)

cmap = mpl.colormaps['viridis']
label_to_color = {lab: cmap(i) for i, lab in enumerate(unique_labels_ordered)}

# ========== Новый макет ==========
fig = plt.figure(figsize=(16, 10))
gs = GridSpec(2, 3, figure=fig, width_ratios=[1, 1, 1], height_ratios=[1, 1])

# --- Левая колонка: сверху фильтр, снизу паттерн ---
ax_filt = fig.add_subplot(gs[0, 0])
mne.viz.plot_topomap(W_sensor, raw.info, axes=ax_filt, show=False)
ax_filt.set_title('Пространственный фильтр')

ax_patt = fig.add_subplot(gs[1, 0])
mne.viz.plot_topomap(A_pattern, raw.info, axes=ax_patt, show=False)
ax_patt.set_title('Паттерн')

# --- Правая верхняя часть: два вложения в ряд ---
# UMAP исходного пространства (до дефляции)
ax_umap_orig = fig.add_subplot(gs[0, 1])
sc1 = ax_umap_orig.scatter(umap_undefl[:, 0], umap_undefl[:, 1],
                           c=p_vals, cmap='plasma', s=15, zorder=2)
plt.colorbar(sc1, ax=ax_umap_orig, label='z-score log-power')
ax_umap_orig.set_title('UMAP до дефляции')
format_umap_axes(ax_umap_orig) # Применяем сетку без цифр

# UMAP дефлированного пространства (после дефляции)
ax_umap_defl = fig.add_subplot(gs[0, 2])
if umap_defl is not None:
    sc2 = ax_umap_defl.scatter(umap_defl[:, 0], umap_defl[:, 1],
                               c=p_vals, cmap='plasma', s=15, zorder=2)
    plt.colorbar(sc2, ax=ax_umap_defl, label='z-score log-power')
    ax_umap_defl.set_title('UMAP после дефляции')
else:
    ax_umap_defl.text(0.5, 0.5, 'Размерность < 2\n(нет вложения)',
                      ha='center', va='center', transform=ax_umap_defl.transAxes)
    ax_umap_defl.set_title('UMAP после дефляции')
format_umap_axes(ax_umap_defl) # Применяем сетку без цифр

# --- Правая нижняя часть: динамика мощности (объединяем ячейки 1:3) ---
ax_env = fig.add_subplot(gs[1, 1:])
window_idx = np.arange(len(p_vals))
ax_env.plot(window_idx, p_vals, color='black', lw=0.8, zorder=2)

# Цветная заливка по условиям и метки на оси X
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
        # Заливка фона
        ax_env.axvspan(s, e-1, facecolor=label_to_color[lab], alpha=0.2, zorder=0)
        
        # Разделительная пунктирная линия в начале блока
        ax_env.axvline(x=s, color='gray', linestyle='--', alpha=0.7, zorder=1)
        
        # Метка ставится точно в начало блока (s)
        xtick_positions.append(s)
        xtick_labels.append(lab)

ax_env.set_xticks(xtick_positions)
ax_env.set_xticklabels(xtick_labels, rotation=45, ha='center', fontsize=8)
ax_env.set_xlabel('Номер окна')
ax_env.set_ylabel('Source log-power')
ax_env.set_title('Динамика мощности источника')

# Включаем сетку по оси Y для графика мощности
ax_env.grid(True, axis='y', linestyle=':', alpha=0.6, zorder=0)

plt.suptitle(f'Компонента {comp_idx+1}', fontsize=16)
plt.tight_layout()
plt.show()

# %%
from matplotlib.gridspec import GridSpec
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import mne

# Функция для настройки осей UMAP
def format_umap_axes(ax):
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(axis='both', which='both', length=0)
    ax.grid(True)

# Выбираем компоненты, которые хотим показать (например, первые 3)
selected_comps = [0, 1, 2, 3, 4, 5]
N_comps = len(selected_comps)

# Выбираем контрастную палитру для синего фона (теплые яркие цвета)
umap_cmap = 'plasma' # Альтернативы: 'spring', 'autumn', 'hot'

# Подготовка меток условий
unique_labels_ordered = []
for lab in labels:
    if lab not in unique_labels_ordered:
        unique_labels_ordered.append(lab)

# Палитра для заливки условий (используем пастельные/контрастные тона)
cond_cmap = mpl.colormaps['Set3']
label_to_color = {lab: cond_cmap(i % 12) for i, lab in enumerate(unique_labels_ordered)}

# ========== МАКЕТ ПОСТЕРА ==========
# Создаем сетку: N строк, 3 колонки (UMAP, Паттерн, Активность)
fig = plt.figure(figsize=(18, 4.5 * N_comps))
# Третья колонка (динамика) делается шире остальных
gs = GridSpec(N_comps, 3, figure=fig, width_ratios=[1, 1, 2.5], wspace=0.2, hspace=0.4)

for row_idx, comp_idx in enumerate(selected_comps):
    
    # 1. Данные для текущего компонента
    a_comp = patterns_full[comp_idx]
    A_pattern = A_ssd @ a_comp     
    
    p_vals = power_history[comp_idx]
    
    # UMAP: для 1-й строки это исходное пространство (до дефляции), 
    # для последующих - пространство после предыдущих дефляций
    umap_coords = umap_coords_history[comp_idx]
    # umap_coords = np.zeros_like(umap_coords_history[comp_idx])
    # umap_coords[:,0] = power_history[comp_idx]
    # umap_coords[:,1] = power_history[comp_idx+1]
    
    # --- КОЛОНКА 1: UMAP Вложение ---
    ax_umap = fig.add_subplot(gs[row_idx, 0])
    if umap_coords is not None:
        sc = ax_umap.scatter(umap_coords[:, 0], umap_coords[:, 1],
                             c=p_vals, cmap=umap_cmap, s=15, zorder=2, alpha=0.9)
        cb = plt.colorbar(sc, ax=ax_umap)
        cb.set_label('log-power', color='white')
        # title_str = 'Исходное вложение' if comp_idx == 0 else f'Вложение после дефляции {comp_idx}'
        # ax_umap.set_title(title_str, pad=10)
        title_str = 'Соответсвующее вложение' if comp_idx == 0 else ''
        ax_umap.set_title(title_str)
    else:
        ax_umap.text(0.5, 0.5, 'Размерность < 2', ha='center', va='center')
    format_umap_axes(ax_umap)

    # --- КОЛОНКА 2: Паттерн ---
    ax_patt = fig.add_subplot(gs[row_idx, 1])
    # MNE топомапы в dark_background могут вести себя специфично.
    # Если голова рисуется черным, можно настроить параметры outlines.
    mne.viz.plot_topomap(A_pattern, raw.info, axes=ax_patt, show=False, 
                         contours=4, sphere=None)
    ax_patt.set_title(f'Паттерн {comp_idx + 1}', pad=10)

    # --- КОЛОНКА 3: Динамика мощности ---
    ax_env = fig.add_subplot(gs[row_idx, 2])
    window_idx = np.arange(len(p_vals))
    # Линия графика белая, чтобы выделяться на синем фоне
    ax_env.plot(window_idx, p_vals, lw=1.2, zorder=2)

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
            # Заливка фона условия
            ax_env.axvspan(s, e-1, facecolor=label_to_color[lab], alpha=0.3, zorder=0)
            # Разделитель
            ax_env.axvline(x=s, linestyle='--', alpha=0.5, zorder=1)
            
            xtick_positions.append(s)
            xtick_labels.append(lab)

    # Метки только на нижней строке, чтобы не захламлять график
    if row_idx == N_comps - 1:
        ax_env.set_xticks(xtick_positions)
        ax_env.set_xticklabels(xtick_labels, rotation=45, ha='center', fontsize=9)
        ax_env.set_xlabel('Номер окна')
    else:
        ax_env.set_xticks(xtick_positions)
        ax_env.set_xticklabels([])
        
    ax_env.set_ylabel('log-power')
    
    ax_env.grid(True, axis='y', linestyle=':', alpha=0.4)
    ax_env.tick_params(axis='both')

# %%s
# Предполагаем, что у вас уже есть:
# raw_ica — исходный Raw после ICA (или любой другой)
# W_ssd — матрица SSD (n_channels, n_components_ssd)
# filters_full — массив (n_filters, n_components_ssd)

# 1. Берём только EEG-каналы (чтобы размерность совпадала с W_ssd)
raw_eeg = raw.copy().pick_types(eeg=True)
data = raw_eeg.get_data()          # (n_channels, n_samples)
sfreq = raw_eeg.info['sfreq']

# 2. Матрица проекции из сенсорного пространства в компоненты
P = W_ssd @ np.array(filters_full).T         # (n_channels, n_filters)
components = P.T @ data            # (n_filters, n_samples)

# 3. Создаём info для новых каналов
n_filters = components.shape[0]
ch_names = [f'Comp_{i+1:02d}' for i in range(n_filters)]
ch_types = ['eeg'] * n_filters
info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

# 4. Создаём RawArray
raw_components = mne.io.RawArray(components, info)

# 5. Переносим аннотации (если нужно)
new_ann = mne.Annotations(raw_eeg.annotations.onset,raw_eeg.annotations.duration,raw_eeg.annotations.description)
raw_components.set_annotations(new_ann)

# 6. Визуализация
raw_components.copy().filter(l_freq=15, h_freq=25).plot(
    picks='all',  # <--- Ключевое исправление
    n_channels=n_filters,
    scalings='auto',
    title='Выделенные компоненты'
)
# Или по отдельности:
# raw_components.plot_psd()