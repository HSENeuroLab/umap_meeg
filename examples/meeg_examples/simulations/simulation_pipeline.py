# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 21:26:29 2026

@author: ansbel
"""

# -*- coding: utf-8 -*-
import sys
import os
import mne
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, hilbert
from scipy.linalg import inv, null_space

# Исправленный чистый путь без смеси слэшей
target_dir = r"C:\Users\ansbel\Documents\GitHub\umap_meeg\examples\meeg_examples\simulations"

if target_dir not in sys.path:
    sys.path.insert(0, target_dir)

from signal_simulation import generate_distributed_sources

lib_directory = os.path.abspath("C:/Users/ansbel/Documents/GitHub/pyRiemann") 
if lib_directory not in sys.path:
    sys.path.insert(0, lib_directory)
from pyriemann.estimation import Covariances

from topological_spatial_filter import fit_filters

from pyriemann.geometry.distance import pairwise_distance
import numpy as np
import umap
import matplotlib as mpl
from matplotlib.lines import Line2D

# %%
# =============================================================================
# 1. ЗАГРУЗКА МОДЕЛИ И INFO
# =============================================================================
work_dir = os.path.dirname(__file__)
fwd_fname = os.path.join(work_dir, 'fsaverage-fwd.fif')
info_fname = os.path.join(work_dir, 'fsaverage-info.fif')

print("Загрузка прямой модели и информации о каналах...")
fwd = mne.read_forward_solution(fwd_fname, verbose=False)
info = mne.io.read_info(info_fname, verbose=False)

# Извлекаем матрицу G. 
# MNE хранит ее в fwd['sol']['data']. Размер будет (64, 20484 * 3)
G = fwd['sol']['data']

# =============================================================================
# 2. НАСТРОЙКА ПАРАМЕТРОВ И ГЕНЕРАЦИЯ
# =============================================================================
Fs = info['sfreq']  # Берем частоту дискретизации из оригинальных данных (в eegbci это 160 Гц)
Ts = 500.0           # Длительность симуляции: 30 секунд
Nsrc = 100          # Общее количество активных источников в мозге
Ndistr = 9          # Из них 2 - целевые
flanker = 1.0       # 1 секунда "фланкеров" для фильтра

print(f"Генерация {Ts} секунд данных на частоте {Fs} Гц...")
X_s, X_bg, X_n, z, GA, S, labels = generate_distributed_sources(G, Nsrc, Ndistr, flanker, Ts, Fs)

# =============================================================================
# 3. СМЕШИВАНИЕ СИГНАЛОВ (SNR)
# =============================================================================
# Здесь мы определяем, насколько сильно выделяется целевой сигнал на фоне шума.
# Вы можете менять эти коэффициенты для усложнения/упрощения задачи.
# %%
SNR = 5
gamma = 0.1

X = SNR*X_s + X_bg + gamma * X_n / np.linalg.norm(X_s,'fro');

# =============================================================================
# 4. УПАКОВКА В MNE RAW И ВИЗУАЛИЗАЦИЯ
# =============================================================================
# Упаковываем матрицу (N_channels, N_samples) обратно в объект MNE
raw = mne.io.RawArray(X, info)

# Если в info были проекторы (average reference), их можно применить
if raw.info['projs']:
    raw.apply_proj()

raw.plot(duration=10.0, n_channels=64, scalings='auto')

# %%
# =============================================================================
# 5. ЭПОХИРОВАНИЕ И РАЗМЕТКА
# =============================================================================
Wsize = 2
Ssize = 0.5
overlap = Wsize - Ssize

epochs = mne.make_fixed_length_epochs(
    raw,
    duration=Wsize,
    overlap=overlap,
    preload=True
)
epochs_data = epochs.get_data(copy=False)

# Считаем ковариации
from pyriemann.estimation import Covariances
from pyriemann.geometry.distance import pairwise_distance

covmats = Covariances(estimator='oas').fit_transform(epochs_data)
dist_matrix_init = pairwise_distance(covmats, metric='riemann')

# --- Переводим посекундные метки (labels) в метки для эпох ---
n_samples_window = int(Wsize * Fs)
n_samples_step = int(Ssize * Fs)

epoch_labels = []
for i in range(len(epochs)):
    start = int(i * n_samples_step)
    # Берем метку, которая приходится на середину окна
    mid_point = start + (n_samples_window // 2)
    epoch_labels.append(labels[mid_point])

epoch_labels = np.array(epoch_labels)
unique_labels = sorted(np.unique(epoch_labels))

print(f"Уникальные состояния в данных: {unique_labels}")

# %%
# =============================================================================
# ОТБЕЛИВАНИЕ (SPATIAL WHITENING) - КЛЮЧ К ПОДАВЛЕНИЮ ШУМА
# =============================================================================
from pyriemann.estimation import Covariances
from pyriemann.geometry.distance import pairwise_distance
from scipy.linalg import eigh

epochs_data = epochs.get_data(copy=False)
covmats = Covariances(estimator='oas').fit_transform(epochs_data)

# 1. Считаем среднюю ковариационную матрицу по всем эпохам
C_mean = np.mean(covmats, axis=0)

# 2. Вычисляем матрицу отбеливания
evals, evecs = eigh(C_mean)
# Инвертируем и извлекаем корень (C^(-1/2)). Ограничиваем снизу для стабильности.
W_white = evecs @ np.diag(1.0 / np.sqrt(evals + 1e-6)) @ evecs.T
# Обратная матрица (C^(1/2)) для восстановления паттернов обратно в сенсоры
W_unwhite = evecs @ np.diag(np.sqrt(evals + 1e-6)) @ evecs.T

# 3. Отбеливаем все ковариационные матрицы
covmats_white = np.zeros_like(covmats)
for i in range(covmats.shape[0]):
    covmats_white[i] = W_white.T @ covmats[i] @ W_white

# Дистанции считаем уже по отбеленным матрицам
dist_matrix_init = pairwise_distance(covmats_white, metric='riemann')
N_neig = 20

# %%
print("Вычисление 3D UMAP для матрицы расстояний...")
umap_3d = umap.UMAP(n_components=3, n_neighbors=20, metric='precomputed')
coords_3d = umap_3d.fit_transform(dist_matrix_init)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.tick_params(axis='x')
ax.tick_params(axis='y')
ax.tick_params(axis='z')

legend_handles = []

ax.scatter(coords_3d[:, 0], coords_3d[:, 1], coords_3d[:, 2], 
           s=15, alpha=0.6)
    
ax.set_title('3D Топология состояний')
ax.set_xlabel('UMAP 1')
ax.set_ylabel('UMAP 2')
ax.set_zlabel('UMAP 3')

fig, ax = plt.subplots()

vmin = np.percentile(dist_matrix_init, 5)
vmax = np.percentile(dist_matrix_init, 95)

im = ax.imshow(dist_matrix_init, cmap='viridis', aspect='auto',
               vmin=vmin, vmax=vmax,
               extent=[0, dist_matrix_init.shape[1],
                       0, dist_matrix_init.shape[0]])

fig.patch.set_alpha(0.0)
ax.patch.set_alpha(0.0)

cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Distance')

plt.tight_layout()
plt.show()

# %%
from topological_spatial_filter import fit_filters
from scipy.linalg import null_space
import numpy as np
import umap

original_dim = covmats_white.shape[1]
current_dim = original_dim
covmats_current = covmats_white.copy()

filters_full = []        
patterns_full = []      
losses = []
umap_coords_history = []
power_history = []
dims_history = []
scales_history = []

# Хранилище всех найденных паттернов в ОТБЕЛЕННОМ пространстве
A_all_white = [] 
# Проектор из текущего нуль-пространства в 64D отбеленное
Pu_global = np.eye(original_dim) 

N_epochs = covmats_white.shape[0]
dist_matrix_current = dist_matrix_init.copy()
distances_history = [dist_matrix_current]

N_NEIG = 20
N_ITERATIONS = 3  
N_DIM = 5         

for comp in range(N_ITERATIONS):
    print(f"\n=== Итерация дефляции {comp+1} (текущая размерность {current_dim}) ===")

    # 1. Поиск N_DIM фильтров в ТЕКУЩЕМ отбеленном подпространстве
    w_opt, scales_opt, final_losses, loss_history, ind_losses = fit_filters(
        C=covmats_current,
        D_matrix=dist_matrix_current,
        N_dim=N_DIM,
        K_restarts=1,
        n_neighbors=N_NEIG,
        epochs=250,
        lr=0.05
    )
    print("Индивидуальные потери:", ind_losses)
    scales_history.append(scales_opt)
    losses.append(loss_history)
    
    umap_step = umap.UMAP(n_components=2, n_neighbors=N_NEIG, metric='precomputed')
    coords = umap_step.fit_transform(dist_matrix_current)

    W_current = w_opt[0, :, :] 
    C_mean_current = np.mean(covmats_current, axis=0)
    A_current = C_mean_current @ W_current.T

    # 2. Сохранение компонент и перевод обратно в сенсорное пространство
    for d in range(N_DIM):
        w_d = W_current[d, :]
        a_d = A_current[:, d]
        
        # Мощность
        p_comp = np.zeros(N_epochs)
        for i in range(N_epochs):
            p_comp[i] = np.log(w_d.T @ covmats_current[i] @ w_d)
        power_history.append(p_comp)
        
        # Переводим из текущего подпространства в 64D отбеленное пространство
        w_d_white = Pu_global @ w_d
        a_d_white = Pu_global @ a_d
        
        A_all_white.append(a_d_white)
        
        # Переводим из 64D отбеленного пространства в ИСХОДНОЕ сенсорное (для MNE)
        filters_full.append(W_white @ w_d_white)
        patterns_full.append(W_unwhite @ a_d_white)
        
        umap_coords_history.append(coords)
        dims_history.append(current_dim)

    # 3. Дефляция через глобальное нуль-пространство
    if comp < N_ITERATIONS - 1:
        A_mat = np.column_stack(A_all_white)
        Pu_global = null_space(A_mat.T)
        
        current_dim = Pu_global.shape[1]
        if current_dim <= 0:
            print("Размерность пространства полностью исчерпана.")
            break

        # Проецируем ИСХОДНЫЕ отбеленные ковариационные матрицы в новое нуль-пространство
        covmats_new = np.zeros((N_epochs, current_dim, current_dim))
        for i in range(N_epochs):
            c_new = Pu_global.T @ covmats_white[i] @ Pu_global
            covmats_new[i] = c_new + 1e-6 * np.trace(c_new) * np.eye(current_dim)

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
# =============================================================================
# ЖАДНЫЙ АЛГОРИТМ СРАВНЕНИЯ НАЙДЕННЫХ КОМПОНЕНТ С ИСТИННЫМИ (ПО ПАТТЕРНАМ)
# =============================================================================
n_epochs_mne = len(epochs)
P_true = np.zeros((Ndistr, n_epochs_mne))
z_targets = z[:Ndistr, :] 

# Усредняем истинную мощность внутри каждого окна
for i in range(n_epochs_mne):
    start = int(i * n_samples_step)
    end = start + n_samples_window
    P_true[:, i] = np.mean(z_targets[:, start:end], axis=1)

# Общее количество извлеченных компонент
total_found = len(patterns_full)

# Вычисление матрицы корреляций Пирсона ПО ПРОСТРАНСТВЕННЫМ ПАТТЕРНАМ
corr_matrix = np.zeros((total_found, Ndistr))
for i in range(total_found):
    for j in range(Ndistr):
        corr_matrix[i, j] = np.corrcoef(patterns_full[i], GA[:, j])[0, 1]

# Жадный алгоритм мэтчинга
abs_corr = np.abs(corr_matrix).copy()
matched_pairs = {} 

print("\n=== Результаты жадного мэтчинга (по паттернам) ===")
for _ in range(min(total_found, Ndistr)):
    i_fnd, j_tru = np.unravel_index(np.argmax(abs_corr), abs_corr.shape)
    
    if abs_corr[i_fnd, j_tru] < 0:
        break
        
    matched_pairs[i_fnd] = j_tru
    corr_val = corr_matrix[i_fnd, j_tru]
    print(f"Компонента {i_fnd+1} -> Истинный источник {j_tru+1} (Корреляция: {corr_val:.3f})")
    
    # Маскируем, чтобы не выбрать повторно
    abs_corr[i_fnd, :] = -1
    abs_corr[:, j_tru] = -1
    
# %%
# =============================================================================
# ФИНАЛЬНАЯ ВИЗУАЛИЗАЦИЯ С ИСТИННЫМИ ДАННЫМИ И МЕТКАМИ СОСТОЯНИЙ
# =============================================================================
from matplotlib.gridspec import GridSpec
import matplotlib as mpl
from matplotlib.lines import Line2D

def format_umap_axes(ax):
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(axis='both', which='both', length=0)
    ax.grid(True, linestyle='--', alpha=0.5, zorder=0)

# Цветовая палитра для классов
cmap_cat = plt.get_cmap('tab20')
label_colors = {lbl: cmap_cat(i) for i, lbl in enumerate(unique_labels)}
point_colors = [label_colors[lbl] for lbl in epoch_labels]

comp_idx = 0  # <--- МЕНЯЙТЕ ИНДЕКС ЗДЕСЬ (от 0 до total_found-1)

A_pattern_found = patterns_full[comp_idx]       
p_vals_found = power_history[comp_idx] 

has_true_match = comp_idx in matched_pairs
if has_true_match:
    j_tru = matched_pairs[comp_idx]
    A_pattern_true = GA[:, j_tru]
    p_vals_true = P_true[j_tru]  
else:
    A_pattern_true = np.zeros_like(A_pattern_found)
    p_vals_true = np.zeros_like(p_vals_found)

# Индекс итерации, на которой была найдена эта компонента
iter_idx = comp_idx // N_DIM
umap_undefl = umap_coords_history[comp_idx]
umap_defl = umap_coords_history[comp_idx + N_DIM] if (comp_idx + N_DIM) < len(umap_coords_history) else None

fig = plt.figure(figsize=(16, 10))
gs = GridSpec(2, 3, figure=fig, width_ratios=[1, 1, 1], height_ratios=[1, 1])

# --- Левая колонка: Паттерны ---
ax_filt = fig.add_subplot(gs[0, 0])
if has_true_match:
    sign_correction = np.sign(corr_matrix[comp_idx, j_tru])
    mne.viz.plot_topomap(A_pattern_true * sign_correction, raw.info, axes=ax_filt, show=False)
    ax_filt.set_title(f'ИСТИННЫЙ паттерн\n(Источник {j_tru+1})')
else:
    ax_filt.text(0.5, 0.5, 'Нет соответствия', ha='center', va='center')
    ax_filt.set_title('ИСТИННЫЙ паттерн')

ax_patt = fig.add_subplot(gs[1, 0])
mne.viz.plot_topomap(A_pattern_found, raw.info, axes=ax_patt, show=False)
ax_patt.set_title('НАЙДЕННЫЙ паттерн')

# --- Правая верхняя: UMAP с раскраской по состояниям ---
ax_umap_orig = fig.add_subplot(gs[0, 1])
ax_umap_orig.scatter(umap_undefl[:, 0], umap_undefl[:, 1], c=point_colors, s=15, alpha=0.8, zorder=2)
ax_umap_orig.set_title('UMAP до дефляции (по состояниям)')
format_umap_axes(ax_umap_orig)

ax_umap_defl = fig.add_subplot(gs[0, 2])
if umap_defl is not None:
    ax_umap_defl.scatter(umap_defl[:, 0], umap_defl[:, 1], c=point_colors, s=15, alpha=0.8, zorder=2)
    ax_umap_defl.set_title('UMAP после дефляции')
else:
    ax_umap_defl.text(0.5, 0.5, 'Размерность исчерпана', ha='center', va='center')
format_umap_axes(ax_umap_defl)

# Легенда для UMAP
legend_handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=label_colors[lbl], markersize=8, label=lbl) for lbl in unique_labels]
ax_umap_defl.legend(handles=legend_handles, loc='upper right', bbox_to_anchor=(1.3, 1))

# --- Правая нижняя: Динамика мощности ---
ax_env = fig.add_subplot(gs[1, 1:])
window_idx = np.arange(len(p_vals_found))

p_found_z = (p_vals_found - np.mean(p_vals_found)) / np.std(p_vals_found)
ax_env.plot(window_idx, p_found_z, color='black', lw=1.5, label='Найденная мощность', zorder=3)

if has_true_match:
    p_true_z = (p_vals_true - np.mean(p_vals_true)) / np.std(p_vals_true)
    label_true = 'Истинная мощность'
        
    ax_env.plot(window_idx, p_true_z, color='red', lw=1.2, alpha=0.7, label=label_true, zorder=2)
    ax_env.legend(loc='upper right')

# Добавляем заливку фона по состояниям
for lbl in unique_labels:
    mask = (epoch_labels == lbl)
    if not np.any(mask): continue
    
    changes = np.diff(np.concatenate(([0], mask.astype(int), [0])))
    starts, ends = np.where(changes == 1)[0], np.where(changes == -1)[0]
    
    for s, e in zip(starts, ends):
        ax_env.axvspan(s, e-1, facecolor=label_colors[lbl], alpha=0.2, zorder=0)

ax_env.set_xlabel('Номер окна')
ax_env.set_ylabel('Power (z-score)')
ax_env.set_title(f'Динамика мощности (Корр. паттернов: {corr_matrix[comp_idx, j_tru]:.3f})' if has_true_match else 'Динамика мощности')
ax_env.grid(True, axis='y', linestyle=':', alpha=0.6, zorder=0)

plt.suptitle(f'Компонента {comp_idx+1}', fontsize=16)
plt.tight_layout()
plt.show()

# %%

