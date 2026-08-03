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
    
lib_directory = os.path.abspath("C:/Users/ansbel/Documents/GitHub/pyRiemann") 
if lib_directory not in sys.path:
    sys.path.insert(0, lib_directory)
from pyriemann.estimation import Covariances
from pyriemann.utils.base import invsqrtm
from pyriemann.geometry.distance import pairwise_distance

from topological_spatial_filter import TopologicalSpatialFilter 

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
print("TSF Анализ: Мультиспектральная Топология и Широкополосная Дефляция")
print("===============================================================\n")

# -----------------------------------------------------------------
# 1. ПАРАМЕТРЫ АНАЛИЗА
# -----------------------------------------------------------------
fpath = "Z:/asbelokopytov/center_out/eeg/patients/Patient_3_CenterOut_OFF_EEG_clean_epochs.fif"

freq_bands = [[8, 12], [10, 14], [12, 16], [14, 18], [16, 20], [18, 22], [20, 24]]

# Определяем границы целевого широкого диапазона
l_freq_broad = freq_bands[0][0]
h_freq_broad = freq_bands[-1][1]

w_size_sec = 0.5
w_step_sec = 0.25

baseline_window = (-1.0, 0.0) # Окно для бейзлайна ERD/ERS
event_time = 0.0              # Время стимула/начала движения

label_mode = 'categorical_binary' 

# %%
# -----------------------------------------------------------------
# 2. ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ
# -----------------------------------------------------------------
print("Загрузка данных...")
epochs_all = mne.read_epochs(fpath, preload=True, verbose=False)

conditions = ['c1d4', 'c3d4', 'c1d2', 'c3d2']
epochs_list = [epochs_all[c] for c in conditions]
epochs = mne.concatenate_epochs(epochs_list)

trial_cond_labels = np.concatenate([[cond] * len(ep) for cond, ep in zip(conditions, epochs_list)])

epochs = epochs.pick_types(eeg=True)
data_raw = epochs.get_data(copy=True)  
times = epochs.times
info = epochs.info
Fs = info['sfreq']

n_trials, n_channels, n_times = data_raw.shape
w_size_samp = int(w_size_sec * Fs)
w_step_samp = int(w_step_sec * Fs)

raw_concat = np.concatenate(data_raw, axis=1)

# Для Секции 6 (ERD/ERS) нам нужны данные, отфильтрованные в широком диапазоне
epochs_filt_broad = epochs.copy().filter(l_freq_broad, h_freq_broad, verbose=False)
data = epochs_filt_broad.get_data(copy=False)

# %%
# -----------------------------------------------------------------
# 3. ПОДГОТОВКА ЦЕЛЕВЫХ МАТРИЦ (ШИРОКИЙ ДИАПАЗОН)
# -----------------------------------------------------------------
print(f"\nРасчет SSD и ковариаций для ЦЕЛЕВОГО диапазона ({l_freq_broad}-{h_freq_broad} Гц)...")

l_broad_b, h_broad_b = l_freq_broad - 2.0, h_freq_broad + 2.0
l_stop_b, h_stop_b = l_freq_broad - 0.5, h_freq_broad + 0.5

b_sig, a_sig = butter(3, np.array([l_freq_broad, h_freq_broad]) / (Fs / 2), btype='band')
b_brd, a_brd = butter(3, np.array([l_broad_b, h_broad_b]) / (Fs / 2), btype='band')
b_stp, a_stp = butter(3, np.array([l_stop_b, h_stop_b]) / (Fs / 2), btype='stop')

sig_ssd = filtfilt(b_sig, a_sig, raw_concat, axis=1)
noise_broad = filtfilt(b_brd, a_brd, raw_concat, axis=1)
noise_ssd = filtfilt(b_stp, a_stp, noise_broad, axis=1)

C_signal = np.cov(sig_ssd)
C_noise = np.cov(noise_ssd)
C_noise_reg = C_noise + 1e-5 * np.trace(C_noise) * np.eye(C_noise.shape[0])

eigvals, eigvecs = eigh(C_signal, C_noise_reg)
idx_sorted = np.argsort(eigvals)[::-1]
W_ssd_full = eigvecs[:, idx_sorted]
component_variances = np.diag(W_ssd_full.T @ C_signal @ W_ssd_full)

valid_components = [v > 1e-6 for v in component_variances]
target_W_ssd = W_ssd_full[:, valid_components]
target_W_ssd = target_W_ssd / np.sqrt(component_variances[valid_components])
target_A_ssd = C_signal @ target_W_ssd
n_components_ssd = target_W_ssd.shape[1]

# Суб-эпохирование целевого диапазона
windows_data = []
window_times = [] 
start_idx = 0
while start_idx + w_size_samp <= n_times:
    end_idx = start_idx + w_size_samp
    windows_data.append(data[:, :, start_idx:end_idx])
    window_times.append(times[start_idx + w_size_samp // 2])
    start_idx += w_step_samp

data_reshaped = np.stack(windows_data, axis=1) 
all_windows = data_reshaped.reshape(-1, n_channels, w_size_samp)

# Проекция в SSD и отбеливание целевых ковариаций
all_windows_ssd = np.zeros((all_windows.shape[0], n_components_ssd, w_size_samp))
for i in range(all_windows.shape[0]):
    all_windows_ssd[i] = target_W_ssd.T @ all_windows[i]

target_covmats = Covariances(estimator='oas').fit_transform(all_windows_ssd)
C_avg = np.mean(target_covmats, axis=0)                     
target_C_avg_invsqrt = invsqrtm(C_avg)                      
target_C_avg_sqrt = np.linalg.inv(target_C_avg_invsqrt)            
target_covmats_white = target_C_avg_invsqrt @ target_covmats @ target_C_avg_invsqrt

print(f"Целевые матрицы готовы (SSD компонент: {n_components_ssd}).")

# %%
# -----------------------------------------------------------------
# 4. РАСЧЕТ D_MATRICES ДЛЯ УЗКИХ ДИАПАЗОНОВ
# -----------------------------------------------------------------
print("\nНезависимый расчет матриц расстояний (D_matrices) для узких диапазонов...")
D_matrices = []

for l_f, h_f in freq_bands:
    print(f"  -> Обработка узкого диапазона {l_f}-{h_f} Гц")
    
    l_broad_n, h_broad_n = l_f - 2.0, h_f + 2.0
    l_stop_n, h_stop_n = l_f - 0.5, h_f + 0.5

    b_sig, a_sig = butter(3, np.array([l_f, h_f]) / (Fs / 2), btype='band')
    b_brd, a_brd = butter(3, np.array([l_broad_n, h_broad_n]) / (Fs / 2), btype='band')
    b_stp, a_stp = butter(3, np.array([l_stop_n, h_stop_n]) / (Fs / 2), btype='stop')

    sig_ssd_n = filtfilt(b_sig, a_sig, raw_concat, axis=1)
    noise_broad_n = filtfilt(b_brd, a_brd, raw_concat, axis=1)
    noise_ssd_n = filtfilt(b_stp, a_stp, noise_broad_n, axis=1)

    C_signal_n = np.cov(sig_ssd_n)
    C_noise_n = np.cov(noise_ssd_n)
    C_noise_reg_n = C_noise_n + 1e-5 * np.trace(C_noise_n) * np.eye(C_noise_n.shape[0])

    eigvals_n, eigvecs_n = eigh(C_signal_n, C_noise_reg_n)
    idx_sorted_n = np.argsort(eigvals_n)[::-1]
    W_ssd_n = eigvecs_n[:, idx_sorted_n]
    comp_vars_n = np.diag(W_ssd_n.T @ C_signal_n @ W_ssd_n)
    
    valid_n = [v > 1e-6 for v in comp_vars_n]
    W_ssd_n = W_ssd_n[:, valid_n]
    W_ssd_n = W_ssd_n / np.sqrt(comp_vars_n[valid_n])
    
    # Фильтрация и суб-эпохирование узкого диапазона
    d_filt_n = filtfilt(b_sig, a_sig, data_raw, axis=2)
    windows_data_n = []
    start_idx = 0
    while start_idx + w_size_samp <= n_times:
        windows_data_n.append(d_filt_n[:, :, start_idx:start_idx + w_size_samp])
        start_idx += w_step_samp
        
    all_windows_n = np.stack(windows_data_n, axis=1).reshape(-1, n_channels, w_size_samp)
    
    # Проекция и отбеливание
    all_windows_ssd_n = np.zeros((all_windows_n.shape[0], W_ssd_n.shape[1], w_size_samp))
    for i in range(all_windows_n.shape[0]):
        all_windows_ssd_n[i] = W_ssd_n.T @ all_windows_n[i]
        
    covs_n = Covariances(estimator='oas').fit_transform(all_windows_ssd_n)
    C_avg_n = np.mean(covs_n, axis=0)
    C_avg_invsqrt_n = invsqrtm(C_avg_n)
    covs_white_n = C_avg_invsqrt_n @ covs_n @ C_avg_invsqrt_n
    
    # Риманова матрица расстояний для текущего узкого диапазона
    D_mat = pairwise_distance(covs_white_n, metric='riemann')
    D_matrices.append(D_mat)

# Генерация меток
print(f"\nФормирование меток для режима: {label_mode}...")
if label_mode == 'categorical_binary':
    unique_window_times = np.array(window_times)
    trial_labels = np.zeros(len(unique_window_times), dtype=int)
    for i, t in enumerate(unique_window_times):
        trial_labels[i] = 0 if t < event_time else -1
    labels = np.tile(trial_labels, n_trials)
    target_metric = 'categorical'

# %%
# -----------------------------------------------------------------
# 5. РИМАНОВА ДЕФЛЯЦИЯ С ИСПОЛЬЗОВАНИЕМ НОВОГО КЛАССА
# -----------------------------------------------------------------
n_iters = 3
N_dim = 3
n_neighbors = 30

found_filters = []
found_patterns = []
umap_coords_history = []

# Инициализируем переменные целевыми (широкополосными) матрицами
C_current = target_covmats_white.copy()
A_ssd_accumulated = []
Q_acc = np.eye(n_components_ssd)

# =================================================================
# ШАГ 1: ОБУЧЕНИЕ ТОПОЛОГИИ ПО МУЛЬТИСПЕКТРАЛЬНЫМ РАССТОЯНИЯМ
# =================================================================
print("\nОбучение целевой топологии на множестве узких диапазонов...")
tsf = TopologicalSpatialFilter(
    N_dim=N_dim,
    K_restarts=1,
    n_neighbors=n_neighbors,
    target_metric=target_metric,
    target_weight=0.5,
    epochs=500,
    lr=0.05,
    verbose=True
)
tsf.fit(D_matrices=D_matrices, y=labels)


# =================================================================
# ШАГ 2: ЦИКЛ ДЕФЛЯЦИИ ДЛЯ ЦЕЛЕВЫХ МАТРИЦ
# =================================================================
for it in range(n_iters):
    print(f"\n  -> Итерация дефляции {it + 1}/{n_iters} ...")
    
    dist_matrix_current = pairwise_distance(C_current, metric='riemann')
    
    print("     Вычисление классического UMAP для дашборда...")
    reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, 
                        metric='precomputed', target_metric=target_metric)
    umap_coords = reducer.fit_transform(dist_matrix_current, y=labels)
    umap_coords_history.append(umap_coords)

    print("     Оптимизация пространственных фильтров (transform)...")
    # Нейросеть с нуля находит фильтры для широкополосного C_current под топологию узких диапазонов
    y_emb = tsf.transform(C_current)
    
    # Извлекаем найденные веса
    W_cur = tsf.w_opt_.T  
    
    C_mean_current = np.mean(C_current, axis=0)
    A_cur = C_mean_current @ W_cur   
    
    W_ssd_white = Q_acc @ W_cur      
    A_ssd_white = Q_acc @ A_cur
    
    W_ssd_orig = target_C_avg_invsqrt @ W_ssd_white   
    A_ssd_orig = target_C_avg_sqrt @ A_ssd_white      
    
    W_global = target_W_ssd @ W_ssd_orig
    A_global = target_A_ssd @ A_ssd_orig
    
    for d in range(N_dim):
        found_filters.append(W_global[:, d])
        found_patterns.append(A_global[:, d])
    
    A_ssd_accumulated.append(A_ssd_white)
    
    A_stacked = np.hstack(A_ssd_accumulated)
    Q_acc = null_space(A_stacked.T) 
    
    # Обрезаем матрицы для следующей итерации
    C_current = np.zeros((target_covmats_white.shape[0], Q_acc.shape[1], Q_acc.shape[1]))
    for i in range(target_covmats_white.shape[0]):
        C_current[i] = Q_acc.T @ target_covmats_white[i] @ Q_acc
        
# %%
# -----------------------------------------------------------------
# 6. РАСЧЕТ ПРОФИЛЕЙ (ERD/ERS ПО УСЛОВИЯМ)
# -----------------------------------------------------------------
print("\nРасчет временных профилей (ERD/ERS) по условиям...")
erd_profiles_dict = [] 
window_powers = [] 

base_mask = (times >= baseline_window[0]) & (times <= baseline_window[1])

for w_idx, w_glob in enumerate(found_filters):
    # Применяем найденный глобальный фильтр к широкополосным данным (data)
    S = np.tensordot(w_glob, data, axes=(0, 1))
    env = np.abs(hilbert(S, axis=1))
    
    comp_erd = {}
    for cond in conditions:
        cond_mask = trial_cond_labels == cond
        env_cond = env[cond_mask]
        
        base_power = np.mean(env_cond[:, base_mask], axis=1, keepdims=True)
        erd_cond = (env_cond - base_power) / base_power * 100
        
        comp_erd[cond] = {
            'mean': np.mean(erd_cond, axis=0),
            'std': np.std(erd_cond, axis=0),
            'n_trials': np.sum(cond_mask)
        }
        
    erd_profiles_dict.append(comp_erd)
    
    # Мощность окон для UMAP-раскраски
    w_ssd = np.linalg.pinv(target_W_ssd) @ w_glob
    p_comp = np.zeros(target_covmats.shape[0])
    for i in range(target_covmats.shape[0]):
        p_comp[i] = w_ssd.T @ target_covmats[i] @ w_ssd
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
        aligned_umap_coords.append(coords)
        target_coords = coords
    else:
        mean_src = np.mean(coords, axis=0)
        mean_tgt = np.mean(target_coords, axis=0)
        s0 = coords - mean_src
        t0 = target_coords - mean_tgt

        H = s0.T @ t0
        U, S_svd, Vt = np.linalg.svd(H)
        R = U @ Vt

        scale = np.sum(t0 * (s0 @ R)) / (np.sum(s0 * s0) + 1e-8)
        aligned_coords = (s0 @ R) * scale + mean_tgt

        aligned_umap_coords.append(aligned_coords)
        target_coords = aligned_coords   

umap_coords_history = aligned_umap_coords

# %%
# -----------------------------------------------------------------
# 7. БЛОЧНАЯ ВИЗУАЛИЗАЦИЯ (UMAP, ТОПОГРАФИЯ, ДИНАМИКА ПО УСЛОВИЯМ)
# -----------------------------------------------------------------
print("Построение итоговых дашбордов...")

umap_cmap = 'plasma' 
cond_colors = {conditions[0]: 'tab:blue', conditions[1]: 'tab:orange', 
               conditions[2]: 'tab:green', conditions[3]: 'tab:red'}

n_comps = len(found_filters)
comps_per_fig = 3 
n_figs = int(np.ceil(n_comps / comps_per_fig))

for fig_idx in range(n_figs):
    start_comp = fig_idx * comps_per_fig
    end_comp = min(start_comp + comps_per_fig, n_comps)
    comps_in_this_fig = end_comp - start_comp
    
    fig = plt.figure(figsize=(18, 5.0 * comps_in_this_fig))
    gs = GridSpec(comps_in_this_fig, 3, figure=fig, width_ratios=[1, 1.2, 2.5], wspace=0.2, hspace=0.4)
    
    for local_idx, comp_idx in enumerate(range(start_comp, end_comp)):
        iter_num = comp_idx // N_dim
        A_pattern = found_patterns[comp_idx]
        p_vals = window_powers[comp_idx] 
        
        vmax_val = np.percentile(p_vals, 95)
        vmin_val = np.min(p_vals) 
        umap_coords = umap_coords_history[iter_num]

        # --- КОЛОНКА 1: Паттерн ---
        ax_patt = fig.add_subplot(gs[local_idx, 0])
        mne.viz.plot_topomap(A_pattern, info, axes=ax_patt, show=False, contours=4)
        ax_patt.set_title(f'Паттерн {comp_idx + 1}', pad=10, fontsize=13, fontweight='bold')

        # --- КОЛОНКА 2: UMAP Вложение ---
        ax_umap = fig.add_subplot(gs[local_idx, 1])
        sort_idx = np.argsort(p_vals)
        sc = ax_umap.scatter(umap_coords[sort_idx, 0], umap_coords[sort_idx, 1],
                             c=p_vals[sort_idx], cmap=umap_cmap, s=20, zorder=2, 
                             alpha=0.4, edgecolors='none', vmin=vmin_val, vmax=vmax_val)
        cb = plt.colorbar(sc, ax=ax_umap)
        title_str = 'Исходное вложение' if iter_num == 0 else f'Вложение после дефляции {iter_num}'
        ax_umap.set_title(title_str, fontsize=12)
        format_umap_axes(ax_umap)

        # --- КОЛОНКА 3: Динамика мощности ERD/ERS (По условиям) ---
        ax_env = fig.add_subplot(gs[local_idx, 2])
        
        ax_env.axvspan(baseline_window[0], baseline_window[1], color='gray', alpha=0.2, zorder=0, label='Baseline')
        ax_env.axvline(event_time, color='black', linestyle='--', alpha=0.7, zorder=1, label='Стимул')
        ax_env.axhline(0, color='black', linewidth=1, zorder=1)
        
        ymin_list, ymax_list = [], []
        
        for cond in conditions:
            mean_erd = erd_profiles_dict[comp_idx][cond]['mean']
            se_erd = erd_profiles_dict[comp_idx][cond]['std'] / np.sqrt(erd_profiles_dict[comp_idx][cond]['n_trials'])
            
            ax_env.plot(times, mean_erd, color=cond_colors[cond], lw=2, zorder=3, label=cond)
            ax_env.fill_between(times, mean_erd - se_erd, mean_erd + se_erd, 
                                color=cond_colors[cond], alpha=0.15, zorder=2)
            
            ymin_list.append(np.min(mean_erd - se_erd))
            ymax_list.append(np.max(mean_erd + se_erd))
            
        ymin, ymax = np.min(ymin_list), np.max(ymax_list)
        y_range = ymax - ymin
        
        for i_c, cond in enumerate(conditions):
            mean_erd = erd_profiles_dict[comp_idx][cond]['mean']
            se_erd = erd_profiles_dict[comp_idx][cond]['std'] / np.sqrt(erd_profiles_dict[comp_idx][cond]['n_trials'])
            sig_mask = np.abs(mean_erd) > (1.96 * se_erd)
            
            band_y = ymin - (0.05 * y_range) - (i_c * 0.03 * y_range)
            ax_env.scatter(times[sig_mask], np.full(np.sum(sig_mask), band_y), 
                           color=cond_colors[cond], s=8, marker='s', zorder=4)

        ax_env.grid(True, axis='both', linestyle=':', alpha=0.6)
        ax_env.set_xlim([times[0], times[-1]])
        
        if local_idx == 0:
            ax_env.legend(loc='upper right', fontsize=9, ncol=2)
        
        if local_idx == comps_in_this_fig - 1:
            ax_env.set_xlabel('Время (с)', fontsize=12)
        else:
            ax_env.set_xticklabels([])

    plt.suptitle(f'TSF Анализ (Условия) | Диапазон: {l_freq_broad}-{h_freq_broad} Гц | Фигура {fig_idx + 1}/{n_figs}', fontsize=16, y=0.95)
    plt.show()

# %%