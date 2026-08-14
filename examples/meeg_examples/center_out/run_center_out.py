# -*- coding: utf-8 -*-
import os
import sys
import mne
import numpy as np
import matplotlib.pyplot as plt
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
fpath = "Z:/asbelokopytov/center_out/eeg/patients/Patient_3_CenterOut_OFF_EEG_clean_epochs.fif"
# fpath = "Z:/asbelokopytov/center_out/eeg/healthy/Control_10_CenterOut_epochs.fif"

freq_bands = {
    'Mu': [9, 14],
    'Beta': [15, 25]
}
selected_band_name = 'Beta'
l_freq, h_freq = freq_bands[selected_band_name]

# Параметры скользящего окна
w_size_sec = 0.5
w_step_sec = 0.25

baseline_window = (-1.0, 0.0) # Окно для бейзлайна ERD/ERS (в секундах)
event_time = 0.0              # Время стимула/начала движения

label_mode = 'categorical_binary' 

# %%
# -----------------------------------------------------------------
# 2. ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ
# -----------------------------------------------------------------
print(f"Загрузка данных и фильтрация в диапазоне {selected_band_name}...")
epochs_all = mne.read_epochs(fpath, preload=True, verbose=False)

# %%
# Список ваших условий
conditions = ['c1d4', 'c3d4', 'c1d2', 'c3d2']
epochs_list = [epochs_all[c] for c in conditions]

# Склеиваем эпохи в один объект
epochs = mne.concatenate_epochs(epochs_list)

# Создаем вектор меток условий для каждого трайла (чтобы потом разделить ERD)
trial_cond_labels = np.concatenate([[cond] * len(ep) for cond, ep in zip(conditions, epochs_list)])

# Оставляем только ЭЭГ
epochs = epochs.pick_types(eeg=True)

epochs_filt = epochs.copy().filter(l_freq, h_freq, verbose=False)
data = epochs_filt.get_data(copy=False)  
times = epochs_filt.times
info = epochs_filt.info
Fs = info['sfreq']

n_trials, n_channels, n_times = data.shape
w_size_samp = int(w_size_sec * Fs)
w_step_samp = int(w_step_sec * Fs)

# %%
from mne.preprocessing import ICA

ica = ICA(n_components=0.999, random_state=97, method='fastica')
ica.fit(epochs)
ica.plot_components()

# %%
# -----------------------------------------------------------------
# 3. SPATIO-SPECTRAL DECOMPOSITION (SSD)
# -----------------------------------------------------------------
print("\nРасчет SSD (Spatio-Spectral Decomposition)...")
raw_data = epochs.get_data(copy=True)
raw_concat = np.concatenate(raw_data, axis=1)

l_broad, h_broad = l_freq - 2.0, h_freq + 2.0
l_stop, h_stop = l_freq - 0.5, h_freq + 0.5

b_sig, a_sig = butter(3, np.array([l_freq, h_freq]) / (Fs / 2), btype='band')
b_brd, a_brd = butter(3, np.array([l_broad, h_broad]) / (Fs / 2), btype='band')
b_stp, a_stp = butter(3, np.array([l_stop, h_stop]) / (Fs / 2), btype='stop')

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
W_ssd = W_ssd_full[:, valid_components]
variances_filtered = component_variances[valid_components]
W_ssd = W_ssd / np.sqrt(variances_filtered)

A_ssd = C_signal @ W_ssd
n_components_ssd = W_ssd.shape[1]
print(f"SSD выполнено: получено {n_components_ssd} компонент.")

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

data_reshaped = np.stack(windows_data, axis=1) 
all_windows = data_reshaped.reshape(-1, n_channels, w_size_samp)

print(f"Формирование меток для режима: {label_mode}...")
if label_mode == 'categorical_binary':
    unique_window_times = np.array(window_times)
    trial_labels = np.zeros(len(unique_window_times), dtype=int)
    for i, t in enumerate(unique_window_times):
        trial_labels[i] = -1 if t < event_time else -1
    labels = np.tile(trial_labels, n_trials)
    target_metric = 'categorical'

print("Проекция окон в SSD-пространство...")
n_win, _, n_samp = all_windows.shape
all_windows_ssd = np.zeros((n_win, n_components_ssd, n_samp))
for i in range(n_win):
    all_windows_ssd[i] = W_ssd.T @ all_windows[i]

covmats_ssd = Covariances().fit_transform(all_windows_ssd / np.std(all_windows_ssd))
covmats = Covariances().fit_transform(all_windows / np.std(all_windows))

# print("Отбеливание ковариационных матриц по среднему арифметическому...")
# C_avg = np.mean(covmats_ssd, axis=0)                     
# C_avg_invsqrt = invsqrtm(C_avg)                       
# C_avg_sqrt = np.linalg.inv(C_avg_invsqrt)             
# covmats_white = C_avg_invsqrt @ covmats_ssd @ C_avg_invsqrt

# %%
# -----------------------------------------------------------------
# 5. РИМАНОВА ДЕФЛЯЦИЯ (БЕЗ ОТБЕЛИВАНИЯ)
# -----------------------------------------------------------------
C_current = covmats_ssd.copy()
dist_matrix = pairwise_distance(C_current, metric='riemann')
dist_matrix_init = dist_matrix.copy()

# %%
import tensorflow as tf
from umap.parametric_umap import ParametricUMAP
import numpy as np
import mne

# Предполагается, что covmats уже существует, форма (n_epochs, M, M)
M = covmats.shape[1]

# =============================================================================
# 1. ПОДГОТОВКА ДАННЫХ (ТОЛЬКО ВЕРХНИЙ ТРЕУГОЛЬНИК)
# =============================================================================
# Получаем индексы верхнего треугольника (включая главную диагональ)
triu_row, triu_col = np.triu_indices(M)

# Вытягиваем только уникальные значения
X_cov_triu = covmats[:, triu_row, triu_col].astype(np.float32)
input_dim = X_cov_triu.shape[1] # Это будет M * (M + 1) / 2

print(f"Размерность входа уменьшена с {M*M} до {input_dim}")

# =============================================================================
# 2. УМНЫЕ ИНДЕКСЫ ДЛЯ БЫСТРОЙ РЕКОНСТРУКЦИИ В TENSORFLOW
# =============================================================================
# А) Индексы, чтобы собрать симметричную матрицу M x M из triu-вектора (для Loss)
idx_sym = np.zeros((M, M), dtype=int)
idx_sym[triu_row, triu_col] = np.arange(input_dim)
idx_sym = np.maximum(idx_sym, idx_sym.T) # Отражаем симметрично
idx_sym_flat = idx_sym.flatten()         # Плоский массив индексов

# Б) Индексы, чтобы вытащить triu-вектор из плоской матрицы M x M (для Декодера)
flat_triu_indices = np.ravel_multi_index((triu_row, triu_col), (M, M))

# =============================================================================
# 3. ДЕКОДЕР С ДИНАМИЧЕСКИМ ШУМОМ (TWO-HEADED)
# =============================================================================
class SpatialPatternDecoder(tf.keras.layers.Layer):
    def __init__(self, M_channels, N_patterns, flat_triu_indices, **kwargs):
        super().__init__(**kwargs)
        self.M_channels = M_channels
        self.N_patterns = N_patterns
        self.flat_triu_indices = flat_triu_indices

    def build(self, input_shape):
        self.A = self.add_weight(
            shape=(self.M_channels, self.N_patterns),
            initializer=tf.keras.initializers.RandomNormal(stddev=0.1),
            trainable=True,
            name="A_patterns"
        )

    def call(self, z_combined):
        # 1. Расщепляем вектор на мощности источников и мощности шума
        z_signal = z_combined[:, :self.N_patterns]
        z_noise = z_combined[:, self.N_patterns:]
        
        # 2. Нормализуем столбцы A
        A_norm = tf.math.l2_normalize(self.A, axis=0)
        
        # 3. Сигнальная часть (A * P * A^T)
        P = tf.exp(z_signal)
        C_signal = tf.einsum("mf,bf,lf->bml", A_norm, P, A_norm)
        
        # 4. Шумовая часть (Диагональная матрица D_t)
        # Используем softplus (или exp), чтобы дисперсия шума была положительной
        noise_variance = tf.exp(z_noise) 
        noise_diag = tf.linalg.diag(noise_variance)
        
        # 5. Итоговая ковариация
        C_recon = C_signal + noise_diag
        
        # 6. ВАЖНО: Выдаем наружу только верхний треугольник, 
        # чтобы размерность совпала со входом энкодера для UMAP
        flat_C = tf.reshape(C_recon, [-1, self.M_channels * self.M_channels])
        C_recon_triu = tf.gather(flat_C, self.flat_triu_indices, axis=1)
        
        return C_recon_triu
    
# =============================================================================
# 4. РИМАНОВА ФУНКЦИЯ ПОТЕРЬ (С ВОССТАНОВЛЕНИЕМ СИММЕТРИИ)
# =============================================================================
def riemannian_distance_loss(y_true_triu, y_pred_triu):
    y_true_triu = tf.cast(y_true_triu, tf.float32)
    y_pred_triu = tf.cast(y_pred_triu, tf.float32)

    C_true_flat = tf.gather(y_true_triu, idx_sym_flat, axis=1)
    C_pred_flat = tf.gather(y_pred_triu, idx_sym_flat, axis=1)
    
    C_true = tf.reshape(C_true_flat, [-1, M, M])
    C_pred = tf.reshape(C_pred_flat, [-1, M, M])
    
    smoothing_factor = 1e-2  
    eye_M = tf.eye(M, dtype=tf.float32)
    
    C_true_reg = C_true + smoothing_factor * eye_M
    C_pred_reg = C_pred + smoothing_factor * eye_M

    eps = 1e-7

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

# =============================================================================
# 5. СБОРКА И ОБУЧЕНИЕ МОДЕЛИ
# =============================================================================
N_patterns = 15  # Количество полезных источников
N_noise = M      # Оси шума (по одной на каждый канал)
N_dim = N_patterns + N_noise  # Итоговая размерность вложения!

# Энкодер теперь выдает объединенный вектор: [полезные_мощности, шум_сенсоров]
encoder = tf.keras.Sequential([
    tf.keras.layers.InputLayer(shape=(input_dim,)),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(N_dim, activation="linear", name="z_combined")
])

decoder = tf.keras.Sequential([
    tf.keras.layers.InputLayer(shape=(N_dim,)),
    SpatialPatternDecoder(M_channels=M, N_patterns=N_patterns, 
                          flat_triu_indices=flat_triu_indices, name="spatial_decoder")
])

print(f"Обучение ParametricUMAP: N_dim={N_dim} ({N_patterns} источников + {N_noise} осей шума)")

embedder = ParametricUMAP(
    encoder=encoder,
    decoder=decoder,
    n_components=N_dim, 
    dims=(input_dim,),
    metric="precomputed",
    parametric_reconstruction=True,
    autoencoder_loss=True,
    parametric_reconstruction_loss_fcn=riemannian_distance_loss,
    verbose=True
)

# Обучаем модель! 
# Теперь X_cov_triu используется и как вход энкодера, и как таргет для реконструкции.
embedder.fit(X_cov_triu, precomputed_distances=dist_matrix_init)

# =============================================================================
# 6. КЛИКЕР: ПРОЕКЦИЯ ДЛЯ ИНТЕРФЕЙСА (ХИТРОСТЬ!)
# =============================================================================
# В embedder.embedding_ сейчас лежат N_patterns + M осей.
# UMAP строил свой граф с учетом шума (как ты и просил).
# Но для визуализации состояний мозга нам логично рисовать 2D проекцию 
# ТОЛЬКО на основе полезных мощностей (отбрасывая оси шума).

print("Проекция ТОЛЬКО полезных мощностей в 2D для кликера...")
z_signal_only = embedder.embedding_[:, :N_patterns] # Берем только первые 20 осей

reducer_2d = umap.UMAP(n_components=2)
umap_coords = reducer_2d.fit_transform(z_signal_only)

# %%
# =============================================================================
# 6. ИЗВЛЕЧЕНИЕ ПАРАМЕТРОВ И ОТРИСОВКА
# =============================================================================
import matplotlib as mpl

# 1. Извлекаем сырые значения z (логарифмы мощностей) 
# Теперь выход энкодера — это и есть наше пространство источников!
z_values = embedder.encoder.predict(X_cov_triu)   # Размерность: (n_windows, N_patterns)

# 2. Переводим в реальные мощности
powers = np.exp(z_values[:, :N_patterns])

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
# =============================================================================
# 7. РАСЧЕТ ПРОФИЛЕЙ ERD/ERS: МОДЕЛЬ И КЛАССИЧЕСКИЙ ФИЛЬТР
# =============================================================================
print("\nИзвлечение временных профилей ERD/ERS (Модель vs Классический фильтр)...")

unique_window_times = np.array(window_times)
num_windows_per_trial = len(unique_window_times)
window_conditions = np.repeat(trial_cond_labels, num_windows_per_trial)

# 1. Мощности нейросети
powers_reshaped = powers.reshape(n_trials, num_windows_per_trial, N_patterns)

erd_profiles_dict = []       # Для модели
erd_filt_profiles_dict = []  # Для классического фильтра

base_mask = (unique_window_times >= baseline_window[0]) & (unique_window_times <= baseline_window[1])

for comp_idx in range(N_patterns):
    comp_erd = {}
    comp_filt_erd = {}
    
    # Извлекаем фильтр Хауфе для текущей компоненты
    w = found_filters[comp_idx]
    
    # Считаем мощность классическим способом для каждого окна: p = w^T * C * w
    # covmats имеет форму (n_windows_total, n_channels, n_channels)
    p_filt = np.einsum('i,nij,j->n', w, covmats, w)
    p_filt_reshaped = p_filt.reshape(n_trials, num_windows_per_trial)
    
    for cond in conditions:
        cond_mask = (trial_cond_labels == cond)
        
        # --- ERD/ERS ДЛЯ МОДЕЛИ ---
        cond_powers = powers_reshaped[cond_mask, :, comp_idx]
        base_power = np.mean(cond_powers[:, base_mask], axis=1, keepdims=True)
        erd_cond = (cond_powers - base_power) / base_power * 100
        
        comp_erd[cond] = {
            'mean': np.mean(erd_cond, axis=0),
            'std': np.std(erd_cond, axis=0),
            'n_trials': np.sum(cond_mask)
        }
        
        # --- ERD/ERS ДЛЯ ФИЛЬТРА ХАУФЕ ---
        cond_filt_powers = p_filt_reshaped[cond_mask, :]
        base_filt_power = np.mean(cond_filt_powers[:, base_mask], axis=1, keepdims=True)
        erd_filt_cond = (cond_filt_powers - base_filt_power) / base_filt_power * 100
        
        comp_filt_erd[cond] = {
            'mean': np.mean(erd_filt_cond, axis=0),
            'std': np.std(erd_filt_cond, axis=0),
            'n_trials': np.sum(cond_mask)
        }
        
    erd_profiles_dict.append(comp_erd)
    erd_filt_profiles_dict.append(comp_filt_erd)

# %%
# =============================================================================
# 7.5 РАСЧЕТ И ВИЗУАЛИЗАЦИЯ ДИНАМИКИ ШУМА (СЕНСОРНЫЕ ОСЦИЛЛЯЦИИ ПО КАНАЛАМ)
# =============================================================================
print("\nИзвлечение временных профилей аппаратного/декоррелированного шума по каждому каналу...")

# 1. Извлекаем сырые значения шума из латентного пространства
z_noise_raw = z_values[:, N_patterns:]

# 2. Переводим через softplus (как в SpatialPatternDecoder)
powers_noise = np.log(1 + np.exp(z_noise_raw))

# 3. Меняем форму под трайлы и окна
# Размерность: (n_trials, num_windows_per_trial, M_channels)
powers_noise_reshaped = powers_noise.reshape(n_trials, num_windows_per_trial, M)

# 4. Настройка сетки графиков (Subplots)
n_cols = 5  # Количество колонок в сетке
n_rows = int(np.ceil(M / n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 3 * n_rows), sharex=True)
axes = axes.flatten()  # Переводим в 1D массив для удобного цикла

# 5. Отрисовка динамики для КАЖДОГО канала
for ch_idx in range(M):
    ax = axes[ch_idx]
    ch_name = info.ch_names[ch_idx] if 'info' in locals() else f"Ch {ch_idx+1}"
    
    # Отрисовка стимула и бейзлайна
    ax.axvspan(baseline_window[0], baseline_window[1], color='gray', alpha=0.2, zorder=0)
    ax.axvline(event_time, color='black', linestyle='--', alpha=0.7, zorder=1)
    
    # Отрисовка профилей по условиям
    for cond in conditions:
        cond_mask = (trial_cond_labels == cond)
        # Извлекаем шум только для ТЕКУЩЕГО канала (ch_idx)
        cond_noise = powers_noise_reshaped[cond_mask, :, ch_idx]
        
        mean_noise = np.mean(cond_noise, axis=0)
        se_noise = np.std(cond_noise, axis=0) / np.sqrt(np.sum(cond_mask))
        
        ax.plot(unique_window_times, mean_noise, color=cond_colors[cond], lw=1.5, zorder=3, label=cond)
        ax.fill_between(unique_window_times, mean_noise - se_noise, mean_noise + se_noise, 
                        color=cond_colors[cond], alpha=0.15, zorder=2)
    
    # Оформление каждого маленького графика
    ax.set_title(f"Канал: {ch_name}", fontsize=11, fontweight='bold')
    ax.grid(True, linestyle=':', alpha=0.6)
    
    # Оставляем легенду только на первом графике
    if ch_idx == 0:
        ax.legend(loc='upper right', fontsize=8)

# 6. Отключаем пустые оси, если M не кратно n_cols
for empty_ax_idx in range(M, len(axes)):
    axes[empty_ax_idx].axis('off')
    
# 7. Подписи для нижнего ряда
for ax in axes[-n_cols:]:
    if ax.axis() != (0.0, 1.0, 0.0, 1.0):  # Если ось не выключена
        ax.set_xlabel('Время (с)', fontsize=10)

plt.suptitle('Динамика локального шума $D_t$ независимо по каждому каналу', fontsize=16, y=0.98, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# %%
# =============================================================================
# 8. СТАТИЧНЫЙ ДАШБОРД СО СРАВНЕНИЕМ ОГИБАЮЩИХ (4 КОЛОНКИ)
# =============================================================================
print("\nПостроение сравнительных дашбордов...")

umap_cmap = 'plasma' 
cond_colors = {
    conditions[0]: 'tab:blue', 
    conditions[1]: 'tab:orange', 
    conditions[2]: 'tab:green', 
    conditions[3]: 'tab:red'
}

n_comps = len(found_patterns)
comps_per_fig = 3 
n_figs = int(np.ceil(n_comps / comps_per_fig))

for fig_idx in range(n_figs):
    start_comp = fig_idx * comps_per_fig
    end_comp = min(start_comp + comps_per_fig, n_comps)
    comps_in_this_fig = end_comp - start_comp
    
    # Немного расширяем фигуру под 4 колонки
    fig = plt.figure(figsize=(22, 5.0 * comps_in_this_fig))
    gs = GridSpec(comps_in_this_fig, 4, figure=fig, width_ratios=[1, 1.2, 2.5, 2.5], wspace=0.2, hspace=0.4)
    
    for local_idx, comp_idx in enumerate(range(start_comp, end_comp)):
        A_pattern = found_patterns[comp_idx]
        p_vals = powers[:, comp_idx] 
        vmax_val = np.percentile(p_vals, 95)
        vmin_val = np.min(p_vals) 
        
        # --- КОЛОНКА 1: Пространственный паттерн ---
        ax_patt = fig.add_subplot(gs[local_idx, 0])
        mne.viz.plot_topomap(A_pattern, info, axes=ax_patt, show=False, contours=4)
        ax_patt.set_title(f'Паттерн {comp_idx + 1}', pad=10, fontsize=13, fontweight='bold')

        # --- КОЛОНКА 2: UMAP Пространство ---
        ax_umap = fig.add_subplot(gs[local_idx, 1])
        sort_idx = np.argsort(p_vals)
        sc = ax_umap.scatter(umap_coords[sort_idx, 0], umap_coords[sort_idx, 1],
                             c=p_vals[sort_idx], cmap=umap_cmap, s=20, zorder=2, 
                             alpha=0.6, edgecolors='none', vmin=vmin_val, vmax=vmax_val)
        cb = plt.colorbar(sc, ax=ax_umap)
        cb.set_label('Мощность $e^z$', rotation=270, labelpad=15)
        ax_umap.set_title('Проекция мощности', fontsize=12)
        format_umap_axes(ax_umap)

        # Синхронизация лимитов Y для графиков ERD
        ymin_list, ymax_list = [], []
        for cond in conditions:
            # Границы для модели
            m1 = erd_profiles_dict[comp_idx][cond]['mean']
            se1 = erd_profiles_dict[comp_idx][cond]['std'] / np.sqrt(erd_profiles_dict[comp_idx][cond]['n_trials'])
            ymin_list.append(np.min(m1 - se1))
            ymax_list.append(np.max(m1 + se1))
            # Границы для классического фильтра
            m2 = erd_filt_profiles_dict[comp_idx][cond]['mean']
            se2 = erd_filt_profiles_dict[comp_idx][cond]['std'] / np.sqrt(erd_filt_profiles_dict[comp_idx][cond]['n_trials'])
            ymin_list.append(np.min(m2 - se2))
            ymax_list.append(np.max(m2 + se2))
            
        ymin, ymax = np.min(ymin_list), np.max(ymax_list)
        y_range = ymax - ymin
        ylim_bottom = ymin - (0.2 * y_range) # Оставляем место внизу для полосок значимости
        ylim_top = ymax + (0.1 * y_range)

        # Функция для отрисовки профиля ERD на переданной оси
        def plot_erd(ax, profiles_dict, title_text):
            ax.axvspan(baseline_window[0], baseline_window[1], color='gray', alpha=0.2, zorder=0, label='Baseline')
            ax.axvline(event_time, color='black', linestyle='--', alpha=0.7, zorder=1, label='Стимул')
            ax.axhline(0, color='black', linewidth=1, zorder=1)
            
            for i_c, cond in enumerate(conditions):
                mean_erd = profiles_dict[comp_idx][cond]['mean']
                se_erd = profiles_dict[comp_idx][cond]['std'] / np.sqrt(profiles_dict[comp_idx][cond]['n_trials'])
                
                ax.plot(unique_window_times, mean_erd, color=cond_colors[cond], lw=2, zorder=3, label=cond)
                ax.fill_between(unique_window_times, mean_erd - se_erd, mean_erd + se_erd, 
                                color=cond_colors[cond], alpha=0.15, zorder=2)
                
            ax.grid(True, axis='both', linestyle=':', alpha=0.6)
            ax.set_xlim([unique_window_times[0], unique_window_times[-1]])
            ax.set_title(title_text, fontsize=12, fontweight='bold')
            
            if local_idx == 0: ax.legend(loc='upper right', fontsize=9, ncol=2)
            if local_idx == comps_in_this_fig - 1:
                ax.set_xlabel('Время центра окна (с)', fontsize=12)
            else:
                ax.set_xticklabels([])

        # --- КОЛОНКА 3: ERD Модели (Нейросеть) ---
        ax_env_model = fig.add_subplot(gs[local_idx, 2])
        plot_erd(ax_env_model, erd_profiles_dict, "ERD/ERS: Нейросеть ($e^z$)")
        ax_env_model.set_ylabel('ERD/ERS (%)', fontsize=12)

        # --- КОЛОНКА 4: ERD Фильтра Хауфе (Классика) ---
        ax_env_filt = fig.add_subplot(gs[local_idx, 3])
        plot_erd(ax_env_filt, erd_filt_profiles_dict, "ERD/ERS: Фильтр ($w^T C w$)")
        ax_env_filt.set_yticklabels([]) # Убираем подписи оси Y, так как они идентичны колонке 3

    plt.suptitle(f'TSF Анализ | Сравнение Модели и Бимформера | Фиг. {fig_idx + 1}/{n_figs}', fontsize=16, y=0.95)
    plt.show()
    
# %%
# =============================================================================
# 9. ВИЗУАЛИЗАЦИЯ УСРЕДНЁННОЙ ДИНАМИКИ X_cov_flat (диагональные элементы)
# =============================================================================
print("\nПостроение усреднённой динамики исходных ковариационных матриц...")

# Переводим X_cov_flat в форму (n_trials, n_windows, n_features)
X_cov_reshaped = X_cov_flat.reshape(n_trials, num_windows_per_trial, -1)

# Индексы диагональных элементов ковариационной матрицы
diag_indices = [i * n_channels + i for i in range(n_channels)]

# Усредняем по трайлам для каждого условия
mean_cov_by_cond = {}
for cond in conditions:
    mask = (trial_cond_labels == cond)
    mean_cov_by_cond[cond] = np.mean(X_cov_reshaped[mask], axis=0)  # (n_windows, n_features)

# Визуализируем первые N_cov_plots диагональных элементов (можно изменить)
N_cov_plots = min(10, n_channels)  # сколько каналов показать
plot_indices = diag_indices[:N_cov_plots]  # первые N_cov_plots диагональных элементов

fig, axes = plt.subplots(N_cov_plots, 1, figsize=(14, 3 * N_cov_plots), sharex=True)
if N_cov_plots == 1:
    axes = [axes]  # чтобы не падало при одном subplot

for ax, feat_idx in zip(axes, plot_indices):
    for cond in conditions:
        ax.plot(unique_window_times, mean_cov_by_cond[cond][:, feat_idx],
                color=cond_colors[cond], lw=2, label=cond)
    # Подписи
    ax.set_ylabel(f'Диаг. {feat_idx // n_channels}')
    ax.axvline(event_time, color='black', linestyle='--', alpha=0.7)
    ax.axvspan(baseline_window[0], baseline_window[1], color='gray', alpha=0.2)
    ax.grid(True, linestyle=':', alpha=0.6)
    if ax is axes[0]:
        ax.legend(loc='upper right', fontsize=9, ncol=2)
    if ax is axes[-1]:
        ax.set_xlabel('Время центра окна (с)')

plt.suptitle('Усреднённая динамика диагональных элементов ковариационной матрицы по условиям', fontsize=14)
plt.tight_layout()
plt.show()    
