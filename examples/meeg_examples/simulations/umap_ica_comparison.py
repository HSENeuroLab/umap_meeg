import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from scipy.optimize import linear_sum_assignment
import mne
from mne.preprocessing import ICA

import tensorflow as tf
from pyriemann.estimation import Covariances
from pyriemann.geometry.distance import pairwise_distance
import umap
from umap.parametric_umap import ParametricUMAP

# Настройте пути под вашу систему
lib_directory = os.path.abspath("C:/Users/ansbel/Documents/GitHub/pyRiemann") 
if lib_directory not in sys.path:
    sys.path.insert(0, lib_directory)

sim_dir = os.path.abspath("C:/Users/ansbel/Documents/GitHub/umap_meeg/examples/meeg_examples/simulations/")
if sim_dir not in sys.path:
    sys.path.insert(0, sim_dir)

# Импорты ваших кастомных функций и слоев
from signal_simulation import generate_microstates
from signal_simulation import BiMapLayer, LogDiagScaleLayer, UnflattenSymmetricLayer

def evaluate_bss_performance(P_true, A_true, S_true, P_pred, A_pred, S_pred):
    """
    Оценивает качество найденных компонент по 3 метрикам.
    Выбор (мэтчинг) компонент производится по максимальной корреляции 
    сырых отфильтрованных сигналов (S).
    """
    # 1. Матрица кросс-корреляций СЫРЫХ временных рядов
    # S_pred форма: (Ncomp, n_times), S_true форма: (Ndistr, n_times)
    C_corr_S = np.corrcoef(S_pred, S_true)
    
    # Блок корреляций между предсказаниями и истиной
    corr_block_S = np.abs(C_corr_S[:S_pred.shape[0], S_pred.shape[0]:])
    
    # 2. Оптимальное сопоставление (Венгерский алгоритм) по сырому сигналу
    row_pred, col_true = linear_sum_assignment(-corr_block_S)
    
    matched_p_corr = []
    matched_a_sim = []
    matched_s_corr = []
    
    for r, c in zip(row_pred, col_true):
        # --- Сырой сигнал (то, по чему искали соответствие) ---
        matched_s_corr.append(corr_block_S[r, c])
        
        # --- Эпохальная мощность ---
        # P_pred: (n_epochs, Ncomp), P_true: (n_epochs, Ndistr)
        p_c = np.abs(np.corrcoef(P_pred[:, r], P_true[:, c])[0, 1])
        matched_p_corr.append(p_c)
        
        # --- Пространственные паттерны ---
        a_t = A_true[:, c]
        a_p = A_pred[:, r]
        cos_sim = np.abs(np.dot(a_t, a_p) / (np.linalg.norm(a_t) * np.linalg.norm(a_p)))
        matched_a_sim.append(cos_sim)
        
    return np.mean(matched_p_corr), np.mean(matched_a_sim), np.mean(matched_s_corr)

# =============================================================================
# 1. ЗАГРУЗКА И ВЫБОР ЭЭГ КАНАЛОВ
# =============================================================================
fwd_fname = 'C:/Users/ansbel/Documents/GitHub/umap_meeg/examples/meeg_examples/simulations/fsaverage-fwd.fif'
info_fname = 'C:/Users/ansbel/Documents/GitHub/umap_meeg/examples/meeg_examples/simulations/fsaverage-info.fif'

fwd = mne.read_forward_solution(fwd_fname, verbose=False)
info = mne.io.read_info(info_fname, verbose=False)
Fs = info['sfreq']

montage_25 = [
    'Fp1', 'Fpz', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FCz', 
    'T7', 'C3', 'Cz', 'C4', 'T8', 'CPz', 'P7', 'P3', 'Pz', 'P4', 'P8', 
    'POz', 'O1', 'Oz', 'O2', 'Iz'
]

picks = mne.pick_types(info, eeg=True, meg=False, selection=montage_25)
info_sub = mne.pick_info(info, sel=picks)
G_sub = fwd['sol']['data'][picks, :]
M_channels = len(picks)

# =============================================================================
# 2. ПАРАМЕТРЫ СИМУЛЯЦИИ И ЭКСПЕРИМЕНТА
# =============================================================================
Ts = 250          
Nsrc = 100         
Ndistr = 1
flanker = 1.0       
gamma = 0.1
Wsize = 1.0
Ssize = 0.5
overlap = Wsize - Ssize

snr_grid = np.logspace(-0.2, 1, 13)
n_snr = len(snr_grid)

# НАСТРОЙКА МОНТЕ-КАРЛО
n_mc_runs = 10

# Хранилище метрик размерности (Кол-во симуляций, Кол-во шагов SNR)
metrics = {
    'umap_p': np.zeros((n_mc_runs, n_snr)),
    'umap_a': np.zeros((n_mc_runs, n_snr)),
    'umap_s': np.zeros((n_mc_runs, n_snr)),
    'ica_p': np.zeros((n_mc_runs, n_snr)),
    'ica_a': np.zeros((n_mc_runs, n_snr)),
    'ica_s': np.zeros((n_mc_runs, n_snr))
}

# =============================================================================
# 3. ОСНОВНОЙ ЦИКЛ МОНТЕ-КАРЛО
# =============================================================================
for mc_idx in range(n_mc_runs):
    print(f"\n===================================================================")
    print(f" М О Н Т Е - К А Р Л О   И Т Е Р А Ц И Я   {mc_idx + 1} / {n_mc_runs}")
    print(f"===================================================================")
    
    # 1. Генерация АБСОЛЮТНО НОВЫХ базисных источников и их пространственных проекций
    print("Генерация новых базисных источников для текущей итерации...")
    X_s_base, X_bg, X_n, z, GA, S, activity = generate_microstates(
        G_sub, Nsrc, Ndistr=Ndistr, flanker=flanker, Ts=Ts, Fs=Fs,
        min_seg_dur=5, max_seg_dur=10.0, transition=0.1
    )

    # Истинные метрики для текущей генерации
    A_true = GA[:, :Ndistr]
    S_true_continuous = S[:Ndistr, :] 

    info_S = mne.create_info(ch_names=[f"Src_{i+1}" for i in range(S.shape[0])], sfreq=Fs, ch_types='misc')
    raw_S = mne.io.RawArray(S.copy(), info_S, verbose=False)
    epochs_S = mne.make_fixed_length_epochs(raw_S, duration=Wsize, overlap=overlap, preload=True, verbose=False)
    true_source_variances = np.var(epochs_S.get_data(), axis=2)
    ga_scales_sq = np.linalg.norm(A_true, axis=0)**2
    P_true = true_source_variances[:, :Ndistr] * ga_scales_sq

    # 2. Внутренний цикл по SNR
    for snr_idx, current_snr in enumerate(snr_grid):
        print(f"\n  --- Шаг SNR {snr_idx + 1}/{n_snr}: SNR = {current_snr:.3f} ---")
        
        X = current_snr * X_s_base + X_bg + gamma * X_n / np.linalg.norm(X_s_base, 'fro')
        X = X / np.std(X)
        
        raw = mne.io.RawArray(X, info_sub, verbose=False)
        epochs = mne.make_fixed_length_epochs(raw, duration=Wsize, overlap=overlap, preload=True, verbose=False)
        epochs_data = epochs.get_data()
        
        # -------------------------------------------------------------------------
        # АЛГОРИТМ 1: FAST ICA
        # -------------------------------------------------------------------------
        # print("    Обучение ICA...")
        ica = ICA(method='fastica', random_state=None, verbose=False)
        ica.fit(epochs, verbose=False)
        
        ica_sources_epochs = ica.get_sources(epochs).get_data()
        P_ica = np.var(ica_sources_epochs, axis=2)
        A_ica = ica.get_components() 
        S_ica_continuous = ica.get_sources(raw).get_data()
        
        ica_p, ica_a, ica_s = evaluate_bss_performance(P_true, A_true, S_true_continuous, P_ica, A_ica, S_ica_continuous)
        metrics['ica_p'][mc_idx, snr_idx] = ica_p
        metrics['ica_a'][mc_idx, snr_idx] = ica_a
        metrics['ica_s'][mc_idx, snr_idx] = ica_s

        # -------------------------------------------------------------------------
        # АЛГОРИТМ 2: PARAMETRIC UMAP + RIEMANNIAN
        # -------------------------------------------------------------------------
        # print("    Обучение Parametric UMAP...")
        tf.keras.backend.clear_session() 
        
        covmats = Covariances(estimator='cov').fit_transform(epochs_data)
        mean_cov = np.mean(covmats, axis=0)
        eigvals, eigvecs = eigh(mean_cov)
        
        idx_sorted = np.argsort(eigvals)[::-1]
        eigvals, eigvecs = eigvals[idx_sorted], eigvecs[:, idx_sorted]
        valid_mask = eigvals > 1e-6
        eigvals, eigvecs = eigvals[valid_mask], eigvecs[:, valid_mask]
        
        Ww = eigvecs / np.sqrt(eigvals)[np.newaxis, :]
        n_components = Ww.shape[1]
        covmats_w = Ww.T @ covmats @ Ww
        dists_init = pairwise_distance(covmats_w, metric='riemann')
        
        idx_i, idx_j = np.triu_indices(n_components)
        multipliers = np.where(idx_i == idx_j, 1.0, np.sqrt(2.0)).astype(np.float32)
        X_cov_flat = (covmats_w[:, idx_i, idx_j] * multipliers).astype(np.float32)
        input_dim = int(X_cov_flat.shape[1])
        
        inputs_enc = tf.keras.Input(shape=(input_dim,), dtype=tf.float32)
        x = UnflattenSymmetricLayer(M=n_components)(inputs_enc)
        x = BiMapLayer(d_out=Ndistr, orth_weight=0.01, name="bimap_1")(x)
        z_latent = LogDiagScaleLayer(n_filters=Ndistr, epsilon=1e-4)(x)
        encoder = tf.keras.Model(inputs=inputs_enc, outputs=z_latent)
        
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=50, min_delta=1e-4)
        embedder = ParametricUMAP(
            encoder=encoder, n_components=Ndistr, dims=(input_dim,), 
            metric="precomputed", n_neighbors=20, verbose=False,
            keras_fit_kwargs={"callbacks": [early_stopping], "verbose": 0}
        )
        
        embedder.n_training_epochs = 3 
        embedder.fit(X_cov_flat, precomputed_distances=dists_init)
        
        W_bimap = embedder.encoder.get_layer("bimap_1").get_weights()[0] 
        P_umap = np.zeros((len(covmats_w), Ndistr))
        for i, C in enumerate(covmats_w):
            C_filtered = W_bimap @ C @ W_bimap.T
            P_umap[i] = np.diag(C_filtered)
            
        W_filters_ssd = W_bimap.T 
        C_avg_ssd = np.mean(covmats_w, axis=0)
        A_ssd = C_avg_ssd @ W_filters_ssd
        A_umap = np.linalg.pinv(Ww.T) @ A_ssd 
        
        W_full = W_bimap @ Ww.T 
        S_umap_continuous = W_full @ raw.get_data() 
        
        umap_p, umap_a, umap_s = evaluate_bss_performance(P_true, A_true, S_true_continuous, P_umap, A_umap, S_umap_continuous)
        metrics['umap_p'][mc_idx, snr_idx] = umap_p
        metrics['umap_a'][mc_idx, snr_idx] = umap_a
        metrics['umap_s'][mc_idx, snr_idx] = umap_s

# =============================================================================
# 4. ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ МОНТЕ-КАРЛО С ДОВЕРИТЕЛЬНЫМИ ИНТЕРВАЛАМИ
# =============================================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

def plot_mc_metric(ax, snr, metric_ica, metric_umap, title, ylabel):
    # Усреднение и стандартное отклонение по нулевой оси (оси Монте-Карло)
    ica_mean = np.mean(metric_ica, axis=0)
    ica_std = np.std(metric_ica, axis=0)
    umap_mean = np.mean(metric_umap, axis=0)
    umap_std = np.std(metric_umap, axis=0)
    
    # Отрисовка ICA
    ax.plot(snr, ica_mean, 'o-', label='FastICA', color='blue', lw=2)
    ax.fill_between(snr, ica_mean - ica_std, ica_mean + ica_std, color='blue', alpha=0.15)
    
    # Отрисовка UMAP
    ax.plot(snr, umap_mean, 's-', label='Parametric UMAP', color='red', lw=2)
    ax.fill_between(snr, umap_mean - umap_std, umap_mean + umap_std, color='red', alpha=0.15)
    
    ax.set_xscale('log')
    ax.set_title(title, fontsize=13)
    ax.set_xlabel('Current SNR', fontsize=11)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(fontsize=11, loc='lower right')
    
# График 1: Огибающие Мощности (P)
plot_mc_metric(axes[0], snr_grid, metrics['ica_p'], metrics['umap_p'], 
               'Огибающие Мощности (P)', 'Abs Pearson Correlation')

# График 2: Сырые Временные Ряды (S)
plot_mc_metric(axes[1], snr_grid, metrics['ica_s'], metrics['umap_s'], 
               'Сырые Временные Ряды Источника (S)', '')

# График 3: Пространственные Паттерны (A)
plot_mc_metric(axes[2], snr_grid, metrics['ica_a'], metrics['umap_a'], 
               'Пространственные Паттерны (A)', 'Abs Cosine Similarity')

plt.tight_layout()
plt.show()