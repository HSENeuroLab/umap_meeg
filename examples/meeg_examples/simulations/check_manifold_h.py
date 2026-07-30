# -*- coding: utf-8 -*-
import sys
import os
import mne
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.spatial import procrustes
from scipy.linalg import eigh
from scipy.signal import butter, filtfilt, hilbert
from sklearn.preprocessing import StandardScaler
import matplotlib.gridspec as gridspec

# Настройка путей
work_dir = os.path.dirname(os.path.abspath(__file__))
if work_dir not in sys.path:
    sys.path.insert(0, work_dir)

repo_root = os.path.abspath(os.path.join(work_dir, "..", "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# Замените путь к pyRiemann на свой
lib_directory = os.path.abspath("C:/Users/ansbel/Documents/GitHub/pyRiemann") 
if lib_directory not in sys.path:
    sys.path.insert(0, lib_directory)

from pyriemann.estimation import Covariances
from pyriemann.geometry.distance import pairwise_distance
from topological_spatial_filter import fit_filters

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) # Отключаем варнинги MNE

def generate_manifold_sources(G, Nsrc, Ndistr, flanker, Ts, Fs,
                              manifold='spiral', noise_power=0.1, rho=0.85):
    N = int(Ts * Fs)
    flanker_samples = int(flanker * Fs)

    b, a = butter(5, [8 / (Fs / 2), 12 / (Fs / 2)], btype='bandpass')
    b_lp, a_lp = butter(5, 0.5 / (Fs / 2), btype='lowpass')

    Gx, Gy, Gz = G[:, 0::3], G[:, 1::3], G[:, 2::3]
    Nsens, Nsites = Gx.shape
    GA = np.zeros((Nsens, Nsrc))
    src_inds = np.random.permutation(Nsites)
    for i in range(Nsrc):
        r = np.random.rand(3); r /= np.linalg.norm(r)
        GA[:, i] = Gx[:, src_inds[i]]*r[0] + Gy[:, src_inds[i]]*r[1] + Gz[:, src_inds[i]]*r[2]

    t = np.linspace(0, 1, N)
    n_cycles = 10
    z_target = np.zeros((Ndistr, N))

    if manifold == 'spiral':
        theta = t * (n_cycles * 2 * np.pi) 
        x = theta * np.cos(theta)
        y = theta * np.sin(theta)
        x = (x - x.mean()) / x.std()
        y = (y - y.mean()) / y.std()
        true_manifold = np.column_stack((x, y))
        for k in range(Ndistr):
            if k == 0: f = np.sin(x) * np.cos(y)
            elif k == 1: f = np.cos(x) * np.sin(y)
            elif k == 2: f = np.sin(x + y)
            elif k == 3: f = np.cos(x - y)
            else: f = np.sin(2*x) + np.cos(2*y)
            f -= f.min()
            z_target[k] = f / f.std() + 0.5

    elif manifold == 's_curve':
        x = t * 2 - 1
        y = np.sin(2 * np.pi * n_cycles * t) 
        x = (x - x.mean()) / x.std()
        y = (y - y.mean()) / y.std()
        true_manifold = np.column_stack((x, y))
        for k in range(Ndistr):
            if k == 0: f = np.sin(x) * np.cos(y)
            elif k == 1: f = np.cos(x) * np.sin(y)
            elif k == 2: f = np.sin(x + y)
            elif k == 3: f = np.cos(x - y)
            else: f = np.sin(2*x) + np.cos(2*y)
            f -= f.min()
            z_target[k] = f / f.std() + 0.5

    elif manifold == 'random':
        for k in range(Ndistr):
            noise_mod = np.random.randn(N + 2*flanker_samples)
            lp_noise_full = filtfilt(b_lp, a_lp, noise_mod)
            lp_noise = lp_noise_full[flanker_samples : -flanker_samples] if flanker_samples > 0 else lp_noise_full.copy()
            lp_noise /= lp_noise.std()
            z_target[k] = lp_noise - np.min(lp_noise) + 0.5
        
        true_manifold = z_target.T
        true_manifold = (true_manifold - true_manifold.mean(axis=0)) / true_manifold.std(axis=0)

    else:
        raise ValueError("Unknown manifold type")

    z_target += np.random.randn(Ndistr, N) * noise_power

    raw_noise = np.random.randn(Nsrc, N + 2*flanker_samples)
    shared_carrier = np.random.randn(N + 2*flanker_samples)
    
    for k in range(Ndistr):
        raw_noise[k] = np.sqrt(1 - rho**2) * raw_noise[k] + rho * shared_carrier

    S_full = filtfilt(b, a, raw_noise, axis=1)
    S = S_full[:, flanker_samples : -flanker_samples] if flanker_samples > 0 else S_full.copy()

    z = np.ones((Nsrc, N))
    for k in range(Nsrc):
        analytic = hilbert(S[k, :])
        env = np.abs(analytic)
        S_norm = S[k, :] / env

        if k < Ndistr:
            amp_mod = z_target[k, :]
        else:
            noise_mod = np.random.randn(N + 2*flanker_samples)
            lp_noise_full = filtfilt(b_lp, a_lp, noise_mod)
            lp_noise = lp_noise_full[flanker_samples : -flanker_samples] if flanker_samples > 0 else lp_noise_full.copy()
            lp_noise /= lp_noise.std()
            amp_mod = lp_noise - np.min(lp_noise) + 0.05

        S[k, :] = S_norm * amp_mod
        sigma_s = np.std(S[k, :])
        S[k, :] /= sigma_s
        z[k, :] = (amp_mod / sigma_s)**2 

    X_s = GA[:, :Ndistr] @ S[:Ndistr, :]
    X_bg = GA[:, Ndistr:] @ S[Ndistr:, :]

    X_n = np.random.randn(Nsens, N)
    X_n -= X_n.mean(axis=1, keepdims=True)
    X_n /= np.std(X_n, axis=1, keepdims=True)

    return X_s, X_bg, X_n, z_target, GA, S, true_manifold


def procrustes_align(source, target):
    _, aligned, _ = procrustes(target, source)
    return aligned

def match_sources_greedy(est_signals, true_signals):
    """Глобальный жадный алгоритм сопоставления"""
    n_est = est_signals.shape[0]
    n_true = true_signals.shape[0]
    
    corr_matrix = np.zeros((n_est, n_true))
    for i in range(n_est):
        for j in range(n_true):
            corr_matrix[i, j] = np.abs(np.corrcoef(est_signals[i], true_signals[j])[0, 1])
            
    matched_est_indices = [-1] * n_true
    available_est = set(range(n_est))
    available_true = set(range(n_true))
    
    while available_true and available_est:
        max_c, best_e, best_t = -1, -1, -1
        for e in available_est:
            for t in available_true:
                if corr_matrix[e, t] > max_c:
                    max_c = corr_matrix[e, t]
                    best_e, best_t = e, t
                    
        matched_est_indices[best_t] = best_e
        available_est.remove(best_e)
        available_true.remove(best_t)
        
    return matched_est_indices

def run_single_hypothesis_check():
    print("===============================================================")
    print("Эксперимент: Детальная отрисовка (Многообразие, Паттерны, Огибающие)")
    print("===============================================================\n")

    fwd_fname = os.path.join(work_dir, 'fsaverage-fwd.fif')
    info_fname = os.path.join(work_dir, 'fsaverage-info.fif')

    fwd = mne.read_forward_solution(fwd_fname, verbose=False)
    info = mne.io.read_info(info_fname, verbose=False)
    G = fwd['sol']['data']
    Fs = info['sfreq']

    Ts = 200
    Nsrc = 50
    Ndistr = 2               
    flanker = 1.0
    gamma = 0.1
    manifold_type = 's_curve' 
    noise_power = 0.05
    correlation_rho = 0.99   

    current_snr = 10.0       
    Wsize = 1.0
    Ssize = 0.1
    overlap = Wsize - Ssize

    print(f"Генерация данных (Модель огибающих: {manifold_type}, rho: {correlation_rho}, SNR: {current_snr})...")
    
    X_s, X_bg, X_n, z_target, GA, S, true_manifold = generate_manifold_sources(
        G, Nsrc=Nsrc, Ndistr=Ndistr, flanker=flanker,
        Ts=Ts, Fs=Fs, manifold=manifold_type, noise_power=noise_power, rho=correlation_rho
    )

    X = current_snr * X_s + X_bg + gamma * X_n / np.linalg.norm(X_s, 'fro')

    raw = mne.io.RawArray(X, info, verbose=False)
    epochs = mne.make_fixed_length_epochs(
        raw, duration=Wsize, overlap=overlap, preload=True, verbose=False
    )
    epochs_data = epochs.get_data()
    n_epochs = len(epochs)
    n_samples_step = int(Ssize * Fs)
    n_samples_window = int(Wsize * Fs)

    true_epoch_coords = np.zeros((n_epochs, 2))
    for i in range(n_epochs):
        start = int(i * n_samples_step)
        end = start + n_samples_window
        true_epoch_coords[i] = true_manifold[start:end].mean(axis=0)

    true_norm = true_epoch_coords - true_epoch_coords.mean(axis=0)
    true_norm /= true_norm.std(axis=0)
    
    true_patterns = GA[:, :2]
    true_envelopes = z_target[:2, :]

    # =================================================================
    # TSF 
    # =================================================================
    print("Запуск TSF...")
    covmats = Covariances(estimator='oas').fit_transform(epochs_data)
    C_mean_global = np.mean(covmats, axis=0)
    evals, evecs = eigh(C_mean_global)
    W_white = evecs @ np.diag(1.0 / np.sqrt(evals + 1e-6)) @ evecs.T
    W_unwhite = evecs @ np.diag(np.sqrt(evals + 1e-6)) @ evecs.T

    covmats_white = np.zeros_like(covmats)
    for i in range(covmats.shape[0]):
        covmats_white[i] = W_white.T @ covmats[i] @ W_white

    dist_matrix = pairwise_distance(covmats_white, metric='riemann')
    w_opt, scales_opt, _, _, _ = fit_filters(
        C=covmats_white, D_matrix=dist_matrix, N_dim=2,
        K_restarts=3, n_neighbors=15, epochs=300, lr=0.05, verbose=False
    )
    
    w_1 = w_opt[0]           
    scales_1 = scales_opt[0] 
    
    y_tsf = np.zeros((n_epochs, 2))
    for d in range(2):
        for i in range(n_epochs):
            power = w_1[d] @ covmats_white[i] @ w_1[d]
            y_tsf[i, d] = scales_1[d] * np.log(power + 1e-8)
            
    tsf_patterns_raw = (W_unwhite @ w_1.T).T  
    S_tsf_raw = w_1 @ W_white.T @ X  
    env_tsf_raw = np.abs(hilbert(S_tsf_raw, axis=1))

    # Жадное сопоставление TSF
    tsf_idx = match_sources_greedy(tsf_patterns_raw, true_patterns.T)
    tsf_patterns_ordered = tsf_patterns_raw[tsf_idx]
    env_tsf_ordered = env_tsf_raw[tsf_idx]
    
    # Коррекция знака
    for i in range(2):
        if np.corrcoef(tsf_patterns_ordered[i], true_patterns[:, i])[0, 1] < 0:
            tsf_patterns_ordered[i] *= -1

    y_tsf_ordered = y_tsf[:, tsf_idx]
    tsf_aligned = procrustes_align(y_tsf_ordered, true_norm)

    # Вычисление метрик TSF
    corr_tsf_manif = np.mean([abs(pearsonr(true_norm[:, d], tsf_aligned[:, d])[0]) for d in range(2)])
    corr_tsf_p1 = np.abs(np.corrcoef(tsf_patterns_ordered[0], true_patterns[:, 0])[0, 1])
    corr_tsf_p2 = np.abs(np.corrcoef(tsf_patterns_ordered[1], true_patterns[:, 1])[0, 1])
    corr_tsf_e1 = np.abs(np.corrcoef(env_tsf_ordered[0], true_envelopes[0])[0, 1])
    corr_tsf_e2 = np.abs(np.corrcoef(env_tsf_ordered[1], true_envelopes[1])[0, 1])


    # =================================================================
    # ICA
    # =================================================================
    print("Запуск FastICA...")
    ica = mne.preprocessing.ICA(n_components=15, method='fastica', random_state=42)
    ica.fit(raw, verbose=False)
    
    ica_patterns_raw = ica.get_components().T 
    S_ica_raw = ica.get_sources(raw).get_data() 
    
    # Жадное сопоставление ICA
    ica_idx = match_sources_greedy(ica_patterns_raw, true_patterns.T)
    ica_patterns_ordered = ica_patterns_raw[ica_idx]
    env_ica_ordered = np.abs(hilbert(S_ica_raw[ica_idx], axis=1))
    
    # Коррекция знака
    for i in range(2):
        if np.corrcoef(ica_patterns_ordered[i], true_patterns[:, i])[0, 1] < 0:
            ica_patterns_ordered[i] *= -1

    ica_epochs = mne.make_fixed_length_epochs(
        ica.get_sources(raw), duration=Wsize, overlap=overlap, preload=True, verbose=False
    )
    ica_power_all = np.log(np.var(ica_epochs.get_data(), axis=2) + 1e-8) 
    y_ica_ordered = np.zeros((n_epochs, 2))
    for d, idx in enumerate(ica_idx):
        y_ica_ordered[:, d] = ica_power_all[:, idx]
        
    y_ica_scaled = StandardScaler().fit_transform(y_ica_ordered)
    ica_aligned = procrustes_align(y_ica_scaled, true_norm)

    # Вычисление метрик ICA
    corr_ica_manif = np.mean([abs(pearsonr(true_norm[:, d], ica_aligned[:, d])[0]) for d in range(2)])
    corr_ica_p1 = np.abs(np.corrcoef(ica_patterns_ordered[0], true_patterns[:, 0])[0, 1])
    corr_ica_p2 = np.abs(np.corrcoef(ica_patterns_ordered[1], true_patterns[:, 1])[0, 1])
    corr_ica_e1 = np.abs(np.corrcoef(env_ica_ordered[0], true_envelopes[0])[0, 1])
    corr_ica_e2 = np.abs(np.corrcoef(env_ica_ordered[1], true_envelopes[1])[0, 1])

    # =========================================================================
    # ГРАНДИОЗНАЯ ВИЗУАЛИЗАЦИЯ (4 РЯДА)
    # =========================================================================
    print("Построение графиков...")
    fig = plt.figure(figsize=(16, 18))
    gs = gridspec.GridSpec(4, 3, height_ratios=[1.2, 1, 1, 1.2], hspace=0.35)

    colors = np.linspace(0, 1, true_norm.shape[0])

    # ---- РЯД 1: МНОГООБРАЗИЯ ----
    ax_m1 = fig.add_subplot(gs[0, 0])
    ax_m1.scatter(true_norm[:, 0], true_norm[:, 1], c=colors, cmap='viridis', s=30)
    ax_m1.set_title('Истинное многообразие', fontsize=14, fontweight='bold')
    
    ax_m2 = fig.add_subplot(gs[0, 1])
    ax_m2.scatter(tsf_aligned[:, 0], tsf_aligned[:, 1], c=colors, cmap='viridis', s=30)
    ax_m2.set_title(f'TSF (Корр: {corr_tsf_manif:.2f})', fontsize=14, fontweight='bold')
    
    ax_m3 = fig.add_subplot(gs[0, 2])
    ax_m3.scatter(ica_aligned[:, 0], ica_aligned[:, 1], c=colors, cmap='viridis', s=30)
    ax_m3.set_title(f'ICA (Корр: {corr_ica_manif:.2f})', fontsize=14, fontweight='bold')

    for ax in [ax_m1, ax_m2, ax_m3]:
        ax.set_xticks([]); ax.set_yticks([])
        ax.grid(True, linestyle='--', alpha=0.5)

    # ---- РЯД 2: ТОПОГРАФИИ (ИСТОЧНИК 1) ----
    ax_p11 = fig.add_subplot(gs[1, 0]); ax_p11.set_title("Истинный Паттерн 1")
    ax_p12 = fig.add_subplot(gs[1, 1]); ax_p12.set_title(f"TSF Паттерн 1\n(Корр: {corr_tsf_p1:.2f})")
    ax_p13 = fig.add_subplot(gs[1, 2]); ax_p13.set_title(f"ICA Паттерн 1\n(Корр: {corr_ica_p1:.2f})")
    
    mne.viz.plot_topomap(true_patterns[:, 0], info, axes=ax_p11, show=False, sphere='eeglab')
    mne.viz.plot_topomap(tsf_patterns_ordered[0], info, axes=ax_p12, show=False, sphere='eeglab')
    mne.viz.plot_topomap(ica_patterns_ordered[0], info, axes=ax_p13, show=False, sphere='eeglab')

    # ---- РЯД 3: ТОПОГРАФИИ (ИСТОЧНИК 2) ----
    ax_p21 = fig.add_subplot(gs[2, 0]); ax_p21.set_title("Истинный Паттерн 2")
    ax_p22 = fig.add_subplot(gs[2, 1]); ax_p22.set_title(f"TSF Паттерн 2\n(Корр: {corr_tsf_p2:.2f})")
    ax_p23 = fig.add_subplot(gs[2, 2]); ax_p23.set_title(f"ICA Паттерн 2\n(Корр: {corr_ica_p2:.2f})")

    mne.viz.plot_topomap(true_patterns[:, 1], info, axes=ax_p21, show=False, sphere='eeglab')
    mne.viz.plot_topomap(tsf_patterns_ordered[1], info, axes=ax_p22, show=False, sphere='eeglab')
    mne.viz.plot_topomap(ica_patterns_ordered[1], info, axes=ax_p23, show=False, sphere='eeglab')

    # ---- РЯД 4: ОГИБАЮЩИЕ (ПЕРВЫЕ 2000 ОТСЧЕТОВ) ----
    plot_samples = int(20 * Fs) # Покажем первые 20 секунд для наглядности
    time_vec = np.arange(plot_samples) / Fs

    def normalize(x):
        return (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-8)

    ax_e1 = fig.add_subplot(gs[3, 0])
    ax_e1.plot(time_vec, normalize(true_envelopes[0, :plot_samples]), label='Источник 1 (x)', lw=2)
    ax_e1.plot(time_vec, normalize(true_envelopes[1, :plot_samples]), label='Источник 2 (y)', lw=2)
    ax_e1.set_title('Истинные Огибающие', fontsize=12)
    ax_e1.legend(loc='upper right'); ax_e1.grid(True)

    ax_e2 = fig.add_subplot(gs[3, 1])
    ax_e2.plot(time_vec, normalize(env_tsf_ordered[0, :plot_samples]), lw=2, label=f'Ис 1 (Корр: {corr_tsf_e1:.2f})')
    ax_e2.plot(time_vec, normalize(env_tsf_ordered[1, :plot_samples]), lw=2, label=f'Ис 2 (Корр: {corr_tsf_e2:.2f})')
    ax_e2.set_title('TSF Огибающие', fontsize=12)
    ax_e2.legend(loc='upper right'); ax_e2.grid(True)

    ax_e3 = fig.add_subplot(gs[3, 2])
    ax_e3.plot(time_vec, normalize(env_ica_ordered[0, :plot_samples]), lw=2, label=f'Ис 1 (Корр: {corr_ica_e1:.2f})')
    ax_e3.plot(time_vec, normalize(env_ica_ordered[1, :plot_samples]), lw=2, label=f'Ис 2 (Корр: {corr_ica_e2:.2f})')
    ax_e3.set_title('ICA Огибающие', fontsize=12)
    ax_e3.legend(loc='upper right'); ax_e3.grid(True)

    plt.suptitle('Глубокий анализ извлечения сетей (TSF vs ICA)', fontsize=20, y=0.98)
    
    plt.savefig('visual_analysis_full.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Сохранено как 'visual_analysis_full.png'")

if __name__ == '__main__':
    run_single_hypothesis_check()