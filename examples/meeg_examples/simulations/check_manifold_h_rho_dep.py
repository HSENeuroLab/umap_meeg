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
    n_cycles = 25
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
    """
    Глобальный жадный алгоритм: вычисляет полную матрицу корреляций, 
    находит абсолютный максимум, фиксирует пару, ищет следующий максимум.
    Возвращает:
      - matched_est_indices: список индексов найденных компонент, 
                             упорядоченный так же, как true_signals.
      - matched_corrs: список значений корреляции для каждой пары.
    """
    n_est = est_signals.shape[0]
    n_true = true_signals.shape[0]
    
    corr_matrix = np.zeros((n_est, n_true))
    for i in range(n_est):
        for j in range(n_true):
            corr_matrix[i, j] = np.abs(np.corrcoef(est_signals[i], true_signals[j])[0, 1])
            
    matched_est_indices = [-1] * n_true
    matched_corrs = [0.0] * n_true
    
    available_est = set(range(n_est))
    available_true = set(range(n_true))
    
    while available_true and available_est:
        max_c = -1
        best_e = -1
        best_t = -1
        
        # Ищем глобальный максимум среди оставшихся свободных пар
        for e in available_est:
            for t in available_true:
                if corr_matrix[e, t] > max_c:
                    max_c = corr_matrix[e, t]
                    best_e = e
                    best_t = t
                    
        # Фиксируем лучшую пару
        matched_est_indices[best_t] = best_e
        matched_corrs[best_t] = max_c
        available_est.remove(best_e)
        available_true.remove(best_t)
        
    return matched_est_indices, matched_corrs


def run_rho_dependence_experiment():
    print("===============================================================")
    print("Эксперимент: Детальный анализ индивидуальных источников (TSF vs ICA)")
    print("===============================================================\n")

    fwd_fname = os.path.join(work_dir, 'fsaverage-fwd.fif')
    info_fname = os.path.join(work_dir, 'fsaverage-info.fif')

    fwd = mne.read_forward_solution(fwd_fname, verbose=False)
    info = mne.io.read_info(info_fname, verbose=False)
    G = fwd['sol']['data']
    Fs = info['sfreq']

    Ts = 500
    Nsrc = 50
    Ndistr = 2               
    flanker = 1.0
    gamma = 0.1
    manifold_type = 'spiral' # Можно поменять на 'random' или 'spiral'
    noise_power = 0.1
    current_snr = 10.0       
    Wsize = 1
    Ssize = 0.5
    overlap = Wsize - Ssize

    rhos = np.linspace(0.0, 0.99, 15)
    n_mc_iters = 1

    keys = ['patt_1', 'patt_2', 'sig_1', 'sig_2', 'env_1', 'env_2', 'manif_1', 'manif_2']
    results = {
        'tsf': {k: [] for k in keys},
        'ica': {k: [] for k in keys}
    }

    for rho in rhos:
        print(f"\n---> Тестирование rho = {rho:.2f} ...")
        
        iter_tsf = {k: [] for k in keys}
        iter_ica = {k: [] for k in keys}

        for iteration in range(n_mc_iters):
            print(f"     Итерация {iteration+1}/{n_mc_iters}")

            X_s, X_bg, X_n, z_target, GA, S, true_manifold = generate_manifold_sources(
                G, Nsrc=Nsrc, Ndistr=Ndistr, flanker=flanker,
                Ts=Ts, Fs=Fs, manifold=manifold_type, noise_power=noise_power, rho=rho
            )

            X = current_snr * X_s + X_bg + gamma * X_n / np.linalg.norm(X_s, 'fro')

            raw = mne.io.RawArray(X, info, verbose=False)
            epochs = mne.make_fixed_length_epochs(
                raw, duration=Wsize, overlap=overlap, preload=True, verbose=False
            )
            epochs_data = epochs.get_data() # copy=False убрано
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
            true_signals = S[:2, :]
            true_envelopes = z_target[:2, :]

            # ================= TSF =================
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
                K_restarts=5, n_neighbors=15, epochs=300, lr=0.05, verbose=True
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
            
            # Жадное сопоставление TSF (основано на паттернах)
            tsf_idx, tsf_patt_corrs = match_sources_greedy(tsf_patterns_raw, true_patterns.T)
            
            # Упорядочиваем компоненты TSF так, чтобы они четко соответствовали Источникам 1 и 2
            y_tsf_ordered = y_tsf[:, tsf_idx]
            tsf_aligned = procrustes_align(y_tsf_ordered, true_norm)
            
            S_tsf_ordered = S_tsf_raw[tsf_idx]
            env_tsf_ordered = env_tsf_raw[tsf_idx]

            # Вычисление остальных метрик строго для упорядоченных пар
            tsf_sig_corrs = [np.abs(np.corrcoef(S_tsf_ordered[i], true_signals[i])[0, 1]) for i in range(2)]
            tsf_env_corrs = [np.abs(np.corrcoef(env_tsf_ordered[i], true_envelopes[i])[0, 1]) for i in range(2)]
            tsf_manif_corrs = [np.abs(pearsonr(true_norm[:, i], tsf_aligned[:, i])[0]) for i in range(2)]

            iter_tsf['patt_1'].append(tsf_patt_corrs[0]); iter_tsf['patt_2'].append(tsf_patt_corrs[1])
            iter_tsf['sig_1'].append(tsf_sig_corrs[0]);  iter_tsf['sig_2'].append(tsf_sig_corrs[1])
            iter_tsf['env_1'].append(tsf_env_corrs[0]);  iter_tsf['env_2'].append(tsf_env_corrs[1])
            iter_tsf['manif_1'].append(tsf_manif_corrs[0]); iter_tsf['manif_2'].append(tsf_manif_corrs[1])

            # ================= ICA =================
            ica = mne.preprocessing.ICA(n_components=15, method='fastica', random_state=42)
            try:
                ica.fit(raw, verbose=False)
            except Exception:
                pass 
            
            ica_patterns_all = ica.get_components().T 
            S_ica_all = ica.get_sources(raw).get_data() 
            
            # Жадное сопоставление ICA: выбираем 2 лучшие компоненты из 15
            ica_idx, ica_patt_corrs = match_sources_greedy(ica_patterns_all, true_patterns.T)
            
            S_ica_ordered = S_ica_all[ica_idx]
            env_ica_ordered = np.abs(hilbert(S_ica_ordered, axis=1))
            
            ica_epochs = mne.make_fixed_length_epochs(
                ica.get_sources(raw), duration=Wsize, overlap=overlap, preload=True, verbose=False
            )
            ica_power_all = np.log(np.var(ica_epochs.get_data(), axis=2) + 1e-8) 
            
            y_ica_ordered = np.zeros((n_epochs, 2))
            for d, idx in enumerate(ica_idx):
                y_ica_ordered[:, d] = ica_power_all[:, idx]
                
            y_ica_scaled = StandardScaler().fit_transform(y_ica_ordered)
            ica_aligned = procrustes_align(y_ica_scaled, true_norm)
            
            # Вычисление остальных метрик строго для упорядоченных пар
            ica_sig_corrs = [np.abs(np.corrcoef(S_ica_ordered[i], true_signals[i])[0, 1]) for i in range(2)]
            ica_env_corrs = [np.abs(np.corrcoef(env_ica_ordered[i], true_envelopes[i])[0, 1]) for i in range(2)]
            ica_manif_corrs = [np.abs(pearsonr(true_norm[:, i], ica_aligned[:, i])[0]) for i in range(2)]

            iter_ica['patt_1'].append(ica_patt_corrs[0]); iter_ica['patt_2'].append(ica_patt_corrs[1])
            iter_ica['sig_1'].append(ica_sig_corrs[0]);  iter_ica['sig_2'].append(ica_sig_corrs[1])
            iter_ica['env_1'].append(ica_env_corrs[0]);  iter_ica['env_2'].append(ica_env_corrs[1])
            iter_ica['manif_1'].append(ica_manif_corrs[0]); iter_ica['manif_2'].append(ica_manif_corrs[1])

        for key in keys:
            results['tsf'][key].append(np.mean(iter_tsf[key]))
            results['ica'][key].append(np.mean(iter_ica[key]))

    # =========================================================================
    # ПОСТРОЕНИЕ 8-ПАНЕЛЬНОГО ГРАФИКА
    # =========================================================================
    print("\nПостроение 8-панельного итогового графика...")
    
    fig, axes = plt.subplots(4, 2, figsize=(15, 18))
    
    plot_map = [
        ('patt_1', 'Пространственный Паттерн (Источник 1)'), ('patt_2', 'Пространственный Паттерн (Источник 2)'),
        ('sig_1', 'Временной Ряд (Источник 1)'), ('sig_2', 'Временной Ряд (Источник 2)'),
        ('env_1', 'Огибающая Амплитуды (Источник 1)'), ('env_2', 'Огибающая Амплитуды (Источник 2)'),
        ('manif_1', 'Координата Изменчивости (Ось X)'), ('manif_2', 'Координата Изменчивости (Ось Y)')
    ]

    for idx, (key, title) in enumerate(plot_map):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]
        
        ax.plot(rhos, results['tsf'][key], 'o-', label='TSF (Наш Метод)', color='blue', linewidth=2.5)
        ax.plot(rhos, results['ica'][key], 's-', label='ICA (Baseline)', color='orange', linewidth=2.5)
        
        ax.set_title(title, fontsize=13, fontweight='bold')
        # ИСПРАВЛЕНО: Добавлен префикс r для raw string, чтобы LaTeX рендерился корректно
        ax.set_xlabel(r'Фазовая синхронизация ($\rho$)', fontsize=11)
        if col == 0:
            ax.set_ylabel('Абс. Корреляция Пирсона', fontsize=11)
            
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, linestyle='--', alpha=0.6)
        if idx == 0:
            ax.legend(loc='lower left')

    plt.suptitle(f'Устойчивость извлечения каждого источника (Модель: {manifold_type})', fontsize=18, y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    
    plt.savefig('detailed_rho_results.png', dpi=300)
    plt.show()
    print("График успешно сохранен как 'detailed_rho_results.png'.")

if __name__ == '__main__':
    run_rho_dependence_experiment()