# -*- coding: utf-8 -*-
import os
import sys
import mne
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
from scipy.linalg import eigh, null_space
from scipy.signal import hilbert, butter, filtfilt
import umap

# Предполагается, что скрипт лежит в папке, где доступен topological_spatial_filter
lib_directory = os.path.abspath("C:/Users/ansbel/Documents/GitHub/pyRiemann") 
if lib_directory not in sys.path:
    sys.path.insert(0, lib_directory)
    
from pyriemann.estimation import Covariances
from pyriemann.geometry.distance import pairwise_distance
from topological_spatial_filter import fit_filters

import warnings
warnings.filterwarnings("ignore")

# Функция для настройки осей UMAP
def format_umap_axes(ax):
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(axis='both', which='both', length=0)
    ax.grid(True, linestyle='--', alpha=0.5, zorder=0)

# %%
print("===============================================================")
print("TSF Анализ: Загрузка Center-Out, SSD, Суб-эпохирование и Дефляция")
print("===============================================================\n")

# -----------------------------------------------------------------
# 1. ПАРАМЕТРЫ АНАЛИЗА
# -----------------------------------------------------------------
fpath = "C:/Users/ansbel/Documents/GitHub/TriCo/data/external/center_out/sub1_center_out_epochs.fif"

freq_bands = {
    'Mu (9-14 Hz)': [9, 14],
    'Beta (15-25 Hz)': [9, 14]
}
selected_band_name = 'Beta (15-25 Hz)'
l_freq, h_freq = freq_bands[selected_band_name]

# Параметры скользящего окна
w_size_sec = 0.5   
w_step_sec = 0.25   

baseline_window = (-1.0, 0.0) # Окно для бейзлайна ERD/ERS (в секундах)
event_time = 0.0             # Время стимула/начала движения

# %%
# -----------------------------------------------------------------
# 2. ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ
# -----------------------------------------------------------------
print(f"Загрузка данных и фильтрация в диапазоне {selected_band_name}...")
epochs = mne.read_epochs(fpath, preload=True, verbose=False)
epochs = epochs.pick_types(eeg=True)

# Исходные данные для экстракции огибающих (уже отфильтрованные)
epochs_filt = epochs.copy().filter(l_freq, h_freq, verbose=False)
data = epochs_filt.get_data(copy=False)  
times = epochs_filt.times
info = epochs_filt.info
Fs = info['sfreq']

n_trials, n_channels, n_times = data.shape
w_size_samp = int(w_size_sec * Fs)
w_step_samp = int(w_step_sec * Fs)

# %%
# -----------------------------------------------------------------
# 3. SPATIO-SPECTRAL DECOMPOSITION (SSD)
# -----------------------------------------------------------------
print("\nРасчет SSD (Spatio-Spectral Decomposition)...")
raw_data = epochs.get_data(copy=True) # Сырые данные (trials, channels, times)
# Склеиваем все трайлы в один длинный массив
raw_concat = np.concatenate(raw_data, axis=1) # (channels, trials * times)

# Динамически задаем частоты для шума (широкополосный минус стоп-бенд)
l_broad, h_broad = l_freq - 2.0, h_freq + 2.0
l_stop, h_stop = l_freq - 0.5, h_freq + 0.5

b_sig, a_sig = butter(3, np.array([l_freq, h_freq]) / (Fs / 2), btype='band')
b_brd, a_brd = butter(3, np.array([l_broad, h_broad]) / (Fs / 2), btype='band')
b_stp, a_stp = butter(3, np.array([l_stop, h_stop]) / (Fs / 2), btype='stop')

# Фильтрация склеенных данных
sig_ssd = filtfilt(b_sig, a_sig, raw_concat, axis=1)
noise_broad = filtfilt(b_brd, a_brd, raw_concat, axis=1)
noise_ssd = filtfilt(b_stp, a_stp, noise_broad, axis=1)

# Вычисление ковариаций и решение GEVP
C_signal = np.cov(sig_ssd)
C_noise = np.cov(noise_ssd)
reg_coeff = 1e-5
C_noise_reg = C_noise + reg_coeff * np.trace(C_noise) * np.eye(C_noise.shape[0])

eigvals, eigvecs = eigh(C_signal, C_noise_reg)
idx_sorted = np.argsort(eigvals)[::-1]
W_ssd_full = eigvecs[:, idx_sorted]
eigvals = eigvals[idx_sorted]

# Фильтруем компоненты с положительной дисперсией
component_variances = np.diag(W_ssd_full.T @ C_signal @ W_ssd_full)
EPSILON = 1e-6
valid_components = [v > EPSILON for v in component_variances]
W_ssd = W_ssd_full[:, valid_components]
variances_filtered = component_variances[valid_components]
W_ssd = W_ssd / np.sqrt(variances_filtered)

A_ssd = C_signal @ W_ssd
n_components_ssd = W_ssd.shape[1]
print(f"SSD выполнено: получено {n_components_ssd} компонент (из {n_channels}).")

# %%
# -----------------------------------------------------------------
# 4. СУБ-ЭПОХИРОВАНИЕ (СКОЛЬЗЯЩЕЕ ОКНО) И ПРОЕКЦИЯ SSD
# -----------------------------------------------------------------
print("\nРазбиение трайлов на скользящие окна...")
windows_data = []
window_times = [] 

start_idx = 0
while start_idx + w_size_samp <= n_times:
    end_idx = start_idx + w_size_samp
    win_chunk = data[:, :, start_idx:end_idx]
    windows_data.append(win_chunk)
    
    center_time = times[start_idx + w_size_samp // 2]
    window_times.append(center_time)
    start_idx += w_step_samp

# Меняем порядок: (n_trials * n_windows, n_ch, w_size_samp)
data_reshaped = np.stack(windows_data, axis=1) 
all_windows = data_reshaped.reshape(-1, n_channels, w_size_samp)

# --- НОВЫЙ БЛОК: ФОРМИРОВАНИЕ МЕТОК КЛАССОВ ---
print("Формирование категориальных временных меток...")
unique_window_times = np.array(window_times)
trial_labels = np.zeros(len(unique_window_times), dtype=int)

# Метка 0 для бейзлайна. Для пост-стимула — уникальные ID.
current_post_stim_label = 1
for i, t in enumerate(unique_window_times):
    if t < event_time:
        trial_labels[i] = 0
    else:
        # trial_labels[i] = current_post_stim_label
        # current_post_stim_label += 1
        trial_labels[i] = 1

# Тиражируем метки на все трайлы (длина будет n_trials * n_windows)
labels = np.tile(trial_labels, n_trials)
time_labels = np.tile(window_times, n_trials) # Оставляем для графиков, если нужно
# ----------------------------------------------

print("Проекция окон в SSD-пространство...")
n_win, _, n_samp = all_windows.shape
all_windows_ssd = np.zeros((n_win, n_components_ssd, n_samp))
for i in range(n_win):
    all_windows_ssd[i] = W_ssd.T @ all_windows[i]

print(f"Вычисление {len(all_windows_ssd)} ковариационных матриц...")
covmats = Covariances(estimator='oas').fit_transform(all_windows_ssd)

# %%
# -----------------------------------------------------------------
# 5. РИМАНОВА ДЕФЛЯЦИЯ (БЕЗ ОТБЕЛИВАНИЯ)
# -----------------------------------------------------------------
# Параметры TSF и Дефляции
n_iters = 2        # Количество итераций дефляции
N_dim = 5            # Сколько компонент извлекаем за одну итерацию
n_neighbors = 25
total_components = n_iters * N_dim

found_filters = []
found_patterns = []
umap_coords_history = []

C_current = covmats.copy()
A_ssd_accumulated = []
Q_acc = np.eye(n_components_ssd)

for it in range(n_iters):
    print(f"\n  -> Итерация дефляции {it + 1}/{n_iters} ...")
    
    C_mean = np.mean(C_current, axis=0)
    
    # 5.1 Сохраняем топологию текущего касательного пространства (UMAP)
    print("     Вычисление UMAP для текущего подпространства...")
    dist_matrix = pairwise_distance(C_current, metric='riemann')
    reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, metric='precomputed')
    umap_coords = reducer.fit_transform(dist_matrix, labels=labels)
    umap_coords_history.append(umap_coords)

    # 5.2 TSF Оптимизация (многомерная)
    print("     Оптимизация фильтров...")
    w_opt, scales_opt, final_losses_sorted, loss_history_sorted, individual_losses_sorted = fit_filters(
        C=C_current, 
        D_matrix=dist_matrix, 
        N_dim=N_dim,
        labels=labels,          
        unknown_label=-1,       
        K_restarts=1, 
        n_neighbors=n_neighbors, 
        epochs=500, 
        lr=0.05, 
        verbose=True
    )
    
    # Транспонируем (D_curr, N_dim)
    W_cur = w_opt[0].T 
    
    # Паттерны в текущем подпространстве
    A_cur = C_mean @ W_cur 
        
    # 5.3 Переводим фильтры и паттерны в пространство SSD, а затем в глобальное пространство исходных сенсоров
    W_ssd_space = Q_acc @ W_cur             # Форма (N_comp_ssd, N_dim)
    A_ssd_space = Q_acc @ A_cur             # Форма (N_comp_ssd, N_dim)
    
    W_global = W_ssd @ W_ssd_space          # Форма (N_channels, N_dim)
    A_global = A_ssd @ A_ssd_space          # Форма (N_channels, N_dim)
    
    # Сохраняем каждую компоненту отдельно (уже в глобальном пространстве)
    for d in range(N_dim):
        found_filters.append(W_global[:, d])
        found_patterns.append(A_global[:, d])
        
    A_ssd_accumulated.append(A_ssd_space)
    
    # 5.4 Дефляция: Проекция ИСХОДНЫХ SSD-данных на накопленное нуль-пространство
    A_stacked = np.hstack(A_ssd_accumulated)
    Q_acc = null_space(A_stacked.T) 
    
    C_current = np.zeros((covmats.shape[0], Q_acc.shape[1], Q_acc.shape[1]))
    for i in range(covmats.shape[0]):
        C_current[i] = Q_acc.T @ covmats[i] @ Q_acc

# %%
# -----------------------------------------------------------------
# 6. РАСЧЕТ ПРОФИЛЕЙ (ERD/ERS И МОЩНОСТЬ ОКОН)
# -----------------------------------------------------------------
print("\nРасчет временных профилей (ERD/ERS) и мощности окон...")
erd_profiles_mean = []
erd_profiles_std = []  
window_powers = [] # Переименовали для логичности

base_mask = (times >= baseline_window[0]) & (times <= baseline_window[1])

for w_idx, w_glob in enumerate(found_filters):
    # --- 1. Расчет профиля мощности (непрерывные данные) ---
    # Применяем сенсорный фильтр к сырым исходным данным (trials, channels, time)
    S = np.tensordot(w_glob, data, axes=(0, 1))
    
    # Квадрат огибающей Гильберта — это физическая мощность сигнала
    env = np.abs(hilbert(S, axis=1))
    
    base_power = np.mean(env[:, base_mask], axis=1, keepdims=True)
    erd = (env - base_power) / base_power * 100
    
    erd_profiles_mean.append(np.mean(erd, axis=0))
    erd_profiles_std.append(np.std(erd, axis=0))
    
    # --- 2. Расчет активации для скользящих окон (многообразие) ---
    w_ssd = np.linalg.pinv(W_ssd) @ w_glob
    
    p_comp = np.zeros(covmats.shape[0])
    for i in range(covmats.shape[0]):
        # Здесь вычисляем дисперсию (мощность) пространственно-отфильтрованного окна.
        # Она строго > 0.
        power = w_ssd.T @ covmats[i] @ w_ssd
        
        # Если цветовой градиент на UMAP будет "съеден" выбросами даже после обрезки 95 перцентиля,
        # раскомментируйте логарифм:
        # p_comp[i] = np.log(power)
        
        p_comp[i] = power
        
    # ПОЛНОСТЬЮ УБРАНА z-нормализация p_comp_z!
    # Передаем сырую мощность (или ее логарифм) напрямую.
    window_powers.append(p_comp)

# %%
# -----------------------------------------------------------------
# 6.5. ВЫРАВНИВАНИЕ UMAP-ВЛОЖЕНИЙ (АЛГОРИТМ КАБША / PROCRUSTES)
# -----------------------------------------------------------------
print("Выравнивание UMAP-координат между итерациями дефляции...")
aligned_umap_coords = []
target_coords = None

for coords in umap_coords_history:
    if coords is None:
        aligned_umap_coords.append(None)
        continue

    if target_coords is None:
        # Первое вложение оставляем без изменений, оно служит базовым якорем
        aligned_umap_coords.append(coords)
        target_coords = coords
    else:
        # 1. Центрируем оба набора точек
        mean_src = np.mean(coords, axis=0)
        mean_tgt = np.mean(target_coords, axis=0)
        s0 = coords - mean_src
        t0 = target_coords - mean_tgt

        # 2. Поиск оптимального поворота/отражения (SVD)
        H = s0.T @ t0
        U, S_svd, Vt = np.linalg.svd(H)
        R = U @ Vt

        # 3. Расчет оптимального масштаба
        scale = np.sum(t0 * (s0 @ R)) / (np.sum(s0 * s0) + 1e-8)

        # 4. Применяем трансформацию (поворот + масштаб) и возвращаем к центру целевого вложения
        aligned_coords = (s0 @ R) * scale + mean_tgt

        aligned_umap_coords.append(aligned_coords)
        # Следующая итерация будет выравниваться к только что выровненной текущей
        target_coords = aligned_coords   

# Перезаписываем историю вложений выровненными координатами
umap_coords_history = aligned_umap_coords

# %%
# -----------------------------------------------------------------
# 7. БЛОЧНАЯ ВИЗУАЛИЗАЦИЯ (UMAP, ТОПОГРАФИЯ, ДИНАМИКА)
# -----------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import mne

print("Построение итогового дашборда...")

umap_cmap = 'plasma' 

TO_PLOT = 5
fig = plt.figure(figsize=(18, 4.5 * TO_PLOT))
gs = GridSpec(TO_PLOT, 3, figure=fig, width_ratios=[1, 1.2, 2.5], wspace=0.2, hspace=0.4)

for comp_idx in range(TO_PLOT):
    iter_num = comp_idx // N_dim
    
    A_pattern = found_patterns[comp_idx]
    mean_erd = erd_profiles_mean[comp_idx]
    std_erd = erd_profiles_std[comp_idx]
    
    # ИСПРАВЛЕНО: используем window_powers вместо старого window_log_powers
    p_vals = window_powers[comp_idx] 
    
    # Отсекаем 5% верхних выбросов для адекватного цветового масштаба
    vmax_val = np.percentile(p_vals, 95)
    vmin_val = np.min(p_vals) 
    
    # Берем уже ВЫРОВНЕННЫЕ координаты UMAP
    umap_coords = umap_coords_history[iter_num]

    # --- КОЛОНКА 1: Паттерн ---
    ax_patt = fig.add_subplot(gs[comp_idx, 0])
    mne.viz.plot_topomap(A_pattern, info, axes=ax_patt, show=False, contours=4)
    ax_patt.set_title(f'Паттерн {comp_idx + 1}', pad=10, fontsize=13, fontweight='bold')

    # --- КОЛОНКА 2: UMAP Вложение (Раскраска по активации) ---
    ax_umap = fig.add_subplot(gs[comp_idx, 1])
    
    sort_idx = np.argsort(p_vals)
    
    # ИСПРАВЛЕНО: добавлена прозрачность alpha=0.4 и убраны края точек edgecolors='none'
    sc = ax_umap.scatter(umap_coords[sort_idx, 0], umap_coords[sort_idx, 1],
                         c=p_vals[sort_idx], cmap=umap_cmap, s=20, zorder=2, 
                         alpha=0.4, edgecolors='none',
                         vmin=vmin_val, vmax=vmax_val)
    
    cb = plt.colorbar(sc, ax=ax_umap)
    title_str = 'Исходное вложение' if iter_num == 0 else f'Вложение после дефляции {iter_num}'
    ax_umap.set_title(title_str, fontsize=12)
    format_umap_axes(ax_umap)

    # --- КОЛОНКА 3: Динамика мощности ERD/ERS с отклонениями ---
    ax_env = fig.add_subplot(gs[comp_idx, 2])
    
    ax_env.plot(times, mean_erd, color='teal', lw=2, zorder=3, label='Средняя огибающая (абс.)')
    
    ax_env.fill_between(times, mean_erd - std_erd, mean_erd + std_erd, 
                        color='teal', alpha=0.2, zorder=2, label='±1 Std Dev')
    
    ax_env.axvspan(baseline_window[0], baseline_window[1], color='gray', alpha=0.2, zorder=0, label='Baseline')
    ax_env.axvline(event_time, color='red', linestyle='--', alpha=0.7, zorder=1, label='Стимул')
    
    # Линию нуля можно оставить для красоты или как ориентир бейзлайна, если данные центрированы
    ax_env.axhline(0, color='black', linewidth=1, zorder=1)
    
    ax_env.grid(True, axis='both', linestyle=':', alpha=0.6)
    ax_env.set_xlim([times[0], times[-1]])
    
    if comp_idx == TO_PLOT - 1:
        ax_env.set_xlabel('Время (с)', fontsize=12)
    else:
        ax_env.set_xticklabels([])

plt.suptitle(f'TSF Анализ Center-Out + SSD | Диапазон: {selected_band_name}', fontsize=18, y=0.98)
plt.savefig('tsf_ssd_center_out_dashboard.png', dpi=300, bbox_inches='tight')
plt.show()