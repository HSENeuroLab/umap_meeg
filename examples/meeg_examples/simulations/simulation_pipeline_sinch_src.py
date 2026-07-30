# -*- coding: utf-8 -*-
import sys
import os
import mne
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.spatial import procrustes
from scipy.linalg import eigh
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Add necessary paths
work_dir = os.path.dirname(os.path.abspath(__file__))
if work_dir not in sys.path:
    sys.path.insert(0, work_dir)

repo_root = os.path.abspath(os.path.join(work_dir, "..", "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

lib_directory = os.path.abspath("C:/Users/ansbel/Documents/GitHub/pyRiemann") 
if lib_directory not in sys.path:
    sys.path.insert(0, lib_directory)

from signal_simulation import generate_manifold_sources
from pyriemann.estimation import Covariances
from pyriemann.geometry.distance import pairwise_distance
from topological_spatial_filter import fit_filters

def procrustes_align(source, target):
    """Прокрустово выравнивание source к target (обе матрицы N x D)."""
    _, aligned, _ = procrustes(target, source)
    return aligned

def run_manifold_experiment():
    print("Эксперимент: Восстановление многообразия (TSF с отбеливанием vs ICA)")

    # 1. Загрузка прямой модели
    fwd_fname = os.path.join(work_dir, 'fsaverage-fwd.fif')
    info_fname = os.path.join(work_dir, 'fsaverage-info.fif')

    fwd = mne.read_forward_solution(fwd_fname, verbose=False)
    info = mne.io.read_info(info_fname, verbose=False)
    G = fwd['sol']['data']
    Fs = info['sfreq']

    # Параметры симуляции
    Ts = 400.0
    Nsrc = 100
    Ndistr = 4               # 4 целевых источника, образующих многообразие
    flanker = 1.0
    gamma = 0.1
    manifold_type = 'spiral'
    noise_power = 0.1
    correlation_rho = 0.85   # Фазовая синхронизация для слома ICA

    # Параметры эксперимента
    snr_logs = np.arange(-0.2, 2.01, 0.1)
    n_mc_iterations = 10      # Увеличьте до 5-10 для гладких графиков

    Wsize = 1.0
    Ssize = 0.5
    overlap = Wsize - Ssize

    # Сохранение метрик
    mean_tsf_manifold_corr, std_tsf_manifold_corr = [], []
    mean_ica_manifold_corr, std_ica_manifold_corr = [], []
    mean_tsf_patt_corr, std_tsf_patt_corr = [], []
    mean_ica_patt_corr, std_ica_patt_corr = [], []

    # Для визуализации scatter-plot (сохраним лучший результат последнего SNR)
    last_true_manifold = None
    last_tsf_manifold = None
    last_ica_manifold = None

    for snr_log in snr_logs:
        current_snr = 10 ** snr_log
        print(f"\nSNR = 10^{snr_log:.2f} ({current_snr:.4f})")

        iter_tsf_manif, iter_ica_manif = [], []
        iter_tsf_patt, iter_ica_patt = [], []

        for iteration in range(n_mc_iterations):
            print(f"  Итерация {iteration+1}/{n_mc_iterations}")

            # Генерация данных (ВАЖНО: функция должна принимать rho)
            X_s, X_bg, X_n, z, GA, S, true_manifold = generate_manifold_sources(
                G, Nsrc=Nsrc, Ndistr=Ndistr, flanker=flanker,
                Ts=Ts, Fs=Fs, manifold=manifold_type, noise_power=noise_power, rho=correlation_rho
            )

            X = current_snr * X_s + X_bg + gamma * X_n / np.linalg.norm(X_s, 'fro')

            # Эпохирование
            raw = mne.io.RawArray(X, info, verbose=False)
            epochs = mne.make_fixed_length_epochs(
                raw, duration=Wsize, overlap=overlap, preload=True, verbose=False
            )
            epochs_data = epochs.get_data(copy=False)
            n_epochs = len(epochs)
            n_samples_step = int(Ssize * Fs)
            n_samples_window = int(Wsize * Fs)

            # Истинные координаты
            true_epoch_coords = np.zeros((n_epochs, 2))
            for i in range(n_epochs):
                start = int(i * n_samples_step)
                end = start + n_samples_window
                true_epoch_coords[i] = true_manifold[start:end].mean(axis=0)

            # Истинные паттерны
            true_patterns = GA[:, :Ndistr]

            # =================================================================
            # TSF (С ПРОСТРАНСТВЕННЫМ ОТБЕЛИВАНИЕМ)
            # =================================================================
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
                K_restarts=1, n_neighbors=30, epochs=300, lr=0.05, verbose=False
            )
            
            w_1 = w_opt[0]           
            scales_1 = scales_opt[0] 
            y_tsf = np.zeros((n_epochs, 2))
            tsf_patterns_found = []

            for d in range(2):
                # Координаты многообразия
                for i in range(n_epochs):
                    power = w_1[d] @ covmats_white[i] @ w_1[d]
                    y_tsf[i, d] = scales_1[d] * np.log(power + 1e-8)
                
                # Физические паттерны
                a_d_sensor = W_unwhite @ w_1[d]
                tsf_patterns_found.append(a_d_sensor)

            # =================================================================
            # ICA + PCA BASELINE
            # =================================================================
            ica = mne.preprocessing.ICA(n_components=15, method='fastica', random_state=42)
            ica.fit(raw, verbose=False)
            ica_patterns = ica.get_components()

            ica_raw = ica.get_sources(raw)
            ica_epochs = mne.make_fixed_length_epochs(
                ica_raw, duration=Wsize, overlap=overlap, preload=True, verbose=False
            )
            ica_epochs_data = ica_epochs.get_data(copy=False)
            
            ica_power = np.log(np.var(ica_epochs_data, axis=2) + 1e-8) 
            scaler = StandardScaler()
            ica_power_scaled = scaler.fit_transform(ica_power)
            pca = PCA(n_components=2)
            y_ica = pca.fit_transform(ica_power_scaled)

            # =================================================================
            # ОЦЕНКА МЕТРИК
            # =================================================================
            # 1. Многообразие (Прокруст)
            true_norm = true_epoch_coords - true_epoch_coords.mean(axis=0)
            true_norm /= true_norm.std(axis=0)

            tsf_aligned = procrustes_align(y_tsf, true_norm)
            corr_tsf_manif = np.mean([abs(pearsonr(true_norm[:, d], tsf_aligned[:, d])[0]) for d in range(2)])

            ica_aligned = procrustes_align(y_ica, true_norm)
            corr_ica_manif = np.mean([abs(pearsonr(true_norm[:, d], ica_aligned[:, d])[0]) for d in range(2)])

            # 2. Пространственные Паттерны (Максимальная корреляция с любым из 4-х целевых)
            tsf_patt_scores = []
            for p_tsf in tsf_patterns_found:
                best_match = max([np.abs(np.corrcoef(p_tsf, true_patterns[:, t])[0, 1]) for t in range(Ndistr)])
                tsf_patt_scores.append(best_match)
            
            # Для честности берем 2 лучшие компоненты ICA
            ica_patt_scores = []
            for c in range(ica.n_components_):
                best_match = max([np.abs(np.corrcoef(ica_patterns[:, c], true_patterns[:, t])[0, 1]) for t in range(Ndistr)])
                ica_patt_scores.append(best_match)
            ica_patt_scores.sort(reverse=True)

            iter_tsf_manif.append(corr_tsf_manif)
            iter_ica_manif.append(corr_ica_manif)
            iter_tsf_patt.append(np.mean(tsf_patt_scores))
            iter_ica_patt.append(np.mean(ica_patt_scores[:2])) # Топ-2 ICA

            # Сохраняем последние координаты для визуализации
            if snr_log == snr_logs[-1] and iteration == n_mc_iterations - 1:
                last_true_manifold = true_norm
                last_tsf_manifold = tsf_aligned
                last_ica_manifold = ica_aligned

        # Агрегация Монте-Карло
        mean_tsf_manifold_corr.append(np.mean(iter_tsf_manif)); std_tsf_manifold_corr.append(np.std(iter_tsf_manif))
        mean_ica_manifold_corr.append(np.mean(iter_ica_manif)); std_ica_manifold_corr.append(np.std(iter_ica_manif))
        
        mean_tsf_patt_corr.append(np.mean(iter_tsf_patt)); std_tsf_patt_corr.append(np.std(iter_tsf_patt))
        mean_ica_patt_corr.append(np.mean(iter_ica_patt)); std_ica_patt_corr.append(np.std(iter_ica_patt))

    # =========================================================================
    # ГРАФИК 1: КОЛИЧЕСТВЕННЫЕ МЕТРИКИ
    # =========================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.errorbar(snr_logs, mean_tsf_manifold_corr, yerr=std_tsf_manifold_corr, fmt='-o', color='blue', label='TSF')
    ax1.errorbar(snr_logs, mean_ica_manifold_corr, yerr=std_ica_manifold_corr, fmt='-s', color='green', label='ICA + PCA')
    ax1.set_title('Восстановление геометрии многообразия')
    ax1.set_xlabel('log10(SNR)')
    ax1.set_ylabel('Средняя корреляция Пирсона (Прокруст)')
    ax1.legend()
    ax1.grid(True)
    ax1.set_ylim(0, 1.05)

    ax2.errorbar(snr_logs, mean_tsf_patt_corr, yerr=std_tsf_patt_corr, fmt='-o', color='blue', label='TSF')
    ax2.errorbar(snr_logs, mean_ica_patt_corr, yerr=std_ica_patt_corr, fmt='-s', color='green', label='ICA (Топ-2)')
    ax2.set_title('Точность локализации (Пространственные Паттерны)')
    ax2.set_xlabel('log10(SNR)')
    ax2.set_ylabel('Абсолютная корреляция Пирсона')
    ax2.legend()
    ax2.grid(True)
    ax2.set_ylim(0, 1.05)

    plt.suptitle(f'Эксперимент BSS: Топология vs Независимость (rho={correlation_rho})')
    plt.tight_layout()
    plt.savefig('manifold_metrics.png', dpi=300)
    print("Графики метрик сохранены в manifold_metrics.png")

    # =========================================================================
    # ГРАФИК 2: ВИЗУАЛИЗАЦИЯ SCATTER-PLOT (для максимального SNR)
    # =========================================================================
    fig2, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Раскрасим точки градиентом от начала к концу эпох, чтобы было видно "движение"
    colors = np.linspace(0, 1, last_true_manifold.shape[0])

    axes[0].scatter(last_true_manifold[:, 0], last_true_manifold[:, 1], c=colors, cmap='viridis', s=20)
    axes[0].set_title('Истинное многообразие (Спираль)')
    
    axes[1].scatter(last_tsf_manifold[:, 0], last_tsf_manifold[:, 1], c=colors, cmap='viridis', s=20)
    axes[1].set_title('Восстановлено TSF (Наш метод)')
    
    axes[2].scatter(last_ica_manifold[:, 0], last_ica_manifold[:, 1], c=colors, cmap='viridis', s=20)
    axes[2].set_title('Восстановлено ICA + PCA')

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(True, linestyle='--', alpha=0.5)

    plt.suptitle('Визуальное сравнение извлеченных 2D-вложений')
    plt.tight_layout()
    plt.savefig('manifold_scatter.png', dpi=300)
    print("Scatter-plots сохранены в manifold_scatter.png")

if __name__ == '__main__':
    run_manifold_experiment()