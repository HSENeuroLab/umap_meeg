# -*- coding: utf-8 -*-
import sys
import os
import mne
import numpy as np
import matplotlib.pyplot as plt

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

from signal_simulation import generate_correlated_sources
from pyriemann.estimation import Covariances
from pyriemann.geometry.distance import pairwise_distance
from topological_spatial_filter import fit_filters

def run_snr_experiment():
    print("Начинаем эксперимент с КОРРЕЛИРОВАННЫМИ источниками (Раздельные кривые)...")

    # 1. ЗАГРУЗКА ПРЯМОЙ МОДЕЛИ
    fwd_fname = os.path.join(work_dir, 'fsaverage-fwd.fif')
    info_fname = os.path.join(work_dir, 'fsaverage-info.fif')

    fwd = mne.read_forward_solution(fwd_fname, verbose=False)
    info = mne.io.read_info(info_fname, verbose=False)
    G = fwd['sol']['data']
    Fs = info['sfreq']

    # ПАРАМЕТРЫ СИМУЛЯЦИИ
    Ts = 400.0           
    Nsrc = 100           
    Ndistr = 2           
    flanker = 1.0        
    gamma = 0.1          
    correlation_rho = 0.85 

    # ПАРАМЕТРЫ ЭКСПЕРИМЕНТА
    snr_logs = np.arange(-0.2, 1.01, 0.1)  
    n_mc_iterations = 2  

    Wsize = 1.0
    Ssize = 0.5
    overlap = Wsize - Ssize

    # РАЗДЕЛЬНОЕ сохранение результатов для TSF (Источник 1 и Источник 2)
    mean_tsf_patt_1, std_tsf_patt_1 = [], []
    mean_tsf_patt_2, std_tsf_patt_2 = [], []
    mean_tsf_pow_1, std_tsf_pow_1 = [], []
    mean_tsf_pow_2, std_tsf_pow_2 = [], []

    # РАЗДЕЛЬНОЕ сохранение результатов для ICA
    mean_ica_patt_1, std_ica_patt_1 = [], []
    mean_ica_patt_2, std_ica_patt_2 = [], []
    mean_ica_pow_1, std_ica_pow_1 = [], []
    mean_ica_pow_2, std_ica_pow_2 = [], []

    for snr_log in snr_logs:
        current_snr = 10 ** snr_log
        print(f"\nТестируем SNR = 10^{snr_log:.2f} ({current_snr:.4f})")

        iter_tsf_patt_1, iter_tsf_patt_2 = [], []
        iter_tsf_pow_1, iter_tsf_pow_2 = [], []
        
        iter_ica_patt_1, iter_ica_patt_2 = [], []
        iter_ica_pow_1, iter_ica_pow_2 = [], []

        for iteration in range(n_mc_iterations):
            print(f"  Итерация {iteration + 1}/{n_mc_iterations}...")

            X_s, X_bg, X_n, z, GA, S = generate_correlated_sources(
                G, Nsrc, Ndistr, flanker, Ts, Fs, rho=correlation_rho
            )

            X = current_snr * X_s + X_bg + gamma * X_n / np.linalg.norm(X_s, 'fro')

            raw = mne.io.RawArray(X, info, verbose=False)
            epochs = mne.make_fixed_length_epochs(
                raw, duration=Wsize, overlap=overlap, preload=True, verbose=False
            )
            epochs_data = epochs.get_data(copy=False)

            # БАЗОВАЯ ИСТИНА
            true_patterns = GA[:, :Ndistr] 
            true_z = z[:Ndistr, :]         

            n_epochs_mne = len(epochs)
            n_samples_window = int(Wsize * Fs)
            n_samples_step = int(Ssize * Fs)

            p_true_epochs = np.zeros((Ndistr, n_epochs_mne))
            for k in range(Ndistr):
                for i in range(n_epochs_mne):
                    start = int(i * n_samples_step)
                    end = start + n_samples_window
                    p_true_epochs[k, i] = np.mean(true_z[k, start:end])

            # TOPOLOGICAL SPATIAL FILTER (БЕЗ ОТБЕЛИВАНИЯ)
            covmats = Covariances(estimator='oas').fit_transform(epochs_data)
            dist_matrix = pairwise_distance(covmats, metric='riemann')

            Ndim = 2 
            w_opt, _, _, _, _ = fit_filters(
                C=covmats, D_matrix=dist_matrix, N_dim=Ndim,                
                K_restarts=1, n_neighbors=30, epochs=300, lr=0.05, verbose=False
            )

            w_found_2d = w_opt[0]
            mean_C = np.mean(covmats, axis=0)
            
            patterns_found = []
            power_found = []

            for d in range(Ndim):
                w_d = w_found_2d[d, :]
                patterns_found.append(mean_C @ w_d)
                
                p_d = np.zeros(covmats.shape[0])
                for i in range(covmats.shape[0]):
                    p_d[i] = np.log(w_d.T @ covmats[i] @ w_d)
                power_found.append(p_d)

            # ICA BASELINE
            ica = mne.preprocessing.ICA(n_components=0.999, method='fastica', random_state=42)
            ica.fit(raw, verbose=False)
            
            ica_patterns = ica.get_components() 
            ica_raw = ica.get_sources(raw)
            ica_epochs = mne.make_fixed_length_epochs(
                ica_raw, duration=Wsize, overlap=overlap, preload=True, verbose=False
            )
            ica_epochs_data = ica_epochs.get_data(copy=False)
            ica_power = np.log(np.var(ica_epochs_data, axis=2) + 1e-8) 

            # СРАВНЕНИЕ (Сохраняем источники раздельно)
            tsf_patt_scores, tsf_pow_scores = [], []
            ica_patt_scores, ica_pow_scores = [], []

            for target_idx in range(Ndistr):
                best_tsf_patt = max([np.abs(np.corrcoef(p_tsf, true_patterns[:, target_idx])[0, 1]) for p_tsf in patterns_found])
                best_tsf_pow = max([np.abs(np.corrcoef(p_pow, p_true_epochs[target_idx])[0, 1]) for p_pow in power_found])
                tsf_patt_scores.append(best_tsf_patt)
                tsf_pow_scores.append(best_tsf_pow)

                best_ica_patt = max([np.abs(np.corrcoef(ica_patterns[:, c], true_patterns[:, target_idx])[0, 1]) for c in range(ica.n_components_)])
                best_ica_pow = max([np.abs(np.corrcoef(ica_power[:, c], p_true_epochs[target_idx])[0, 1]) for c in range(ica.n_components_)])
                ica_patt_scores.append(best_ica_patt)
                ica_pow_scores.append(best_ica_pow)

            # Распределение по спискам для Источника 1 (индекс 0) и Источника 2 (индекс 1)
            iter_tsf_patt_1.append(tsf_patt_scores[0])
            iter_tsf_patt_2.append(tsf_patt_scores[1])
            iter_tsf_pow_1.append(tsf_pow_scores[0])
            iter_tsf_pow_2.append(tsf_pow_scores[1])

            iter_ica_patt_1.append(ica_patt_scores[0])
            iter_ica_patt_2.append(ica_patt_scores[1])
            iter_ica_pow_1.append(ica_pow_scores[0])
            iter_ica_pow_2.append(ica_pow_scores[1])

        # Агрегация по Монте-Карло для TSF
        mean_tsf_patt_1.append(np.mean(iter_tsf_patt_1)); std_tsf_patt_1.append(np.std(iter_tsf_patt_1))
        mean_tsf_patt_2.append(np.mean(iter_tsf_patt_2)); std_tsf_patt_2.append(np.std(iter_tsf_patt_2))
        
        mean_tsf_pow_1.append(np.mean(iter_tsf_pow_1)); std_tsf_pow_1.append(np.std(iter_tsf_pow_1))
        mean_tsf_pow_2.append(np.mean(iter_tsf_pow_2)); std_tsf_pow_2.append(np.std(iter_tsf_pow_2))

        # Агрегация по Монте-Карло для ICA
        mean_ica_patt_1.append(np.mean(iter_ica_patt_1)); std_ica_patt_1.append(np.std(iter_ica_patt_1))
        mean_ica_patt_2.append(np.mean(iter_ica_patt_2)); std_ica_patt_2.append(np.std(iter_ica_patt_2))
        
        mean_ica_pow_1.append(np.mean(iter_ica_pow_1)); std_ica_pow_1.append(np.std(iter_ica_pow_1))
        mean_ica_pow_2.append(np.mean(iter_ica_pow_2)); std_ica_pow_2.append(np.std(iter_ica_pow_2))


    # =========================================================================
    # ПОСТРОЕНИЕ ГРАФИКОВ (4 линии на каждый сабплот)
    # =========================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # График 1: Паттерны
    # TSF
    ax1.errorbar(snr_logs, mean_tsf_patt_1, yerr=std_tsf_patt_1, fmt='-o', capsize=5, color='b', label='TSF (Ист. 1)')
    ax1.errorbar(snr_logs, mean_tsf_patt_2, yerr=std_tsf_patt_2, fmt='--o', capsize=5, color='b', alpha=0.5, label='TSF (Ист. 2)')
    # ICA
    ax1.errorbar(snr_logs, mean_ica_patt_1, yerr=std_ica_patt_1, fmt='-s', capsize=5, color='g', label='ICA (Ист. 1)')
    ax1.errorbar(snr_logs, mean_ica_patt_2, yerr=std_ica_patt_2, fmt='--s', capsize=5, color='g', alpha=0.5, label='ICA (Ист. 2)')
    
    ax1.set_title('Качество восстановления паттерна (Раздельно по источникам)')
    ax1.set_xlabel('log10(SNR)')
    ax1.set_ylabel('Абсолютная корреляция Пирсона')
    ax1.legend()
    ax1.grid(True)
    ax1.set_ylim(0, 1.05)

    # График 2: Мощности
    # TSF
    ax2.errorbar(snr_logs, mean_tsf_pow_1, yerr=std_tsf_pow_1, fmt='-o', capsize=5, color='r', label='TSF (Ист. 1)')
    ax2.errorbar(snr_logs, mean_tsf_pow_2, yerr=std_tsf_pow_2, fmt='--o', capsize=5, color='r', alpha=0.5, label='TSF (Ист. 2)')
    # ICA
    ax2.errorbar(snr_logs, mean_ica_pow_1, yerr=std_ica_pow_1, fmt='-s', capsize=5, color='orange', label='ICA (Ист. 1)')
    ax2.errorbar(snr_logs, mean_ica_pow_2, yerr=std_ica_pow_2, fmt='--s', capsize=5, color='orange', alpha=0.5, label='ICA (Ист. 2)')
    
    ax2.set_title('Качество восстановления динамики мощности (Раздельно)')
    ax2.set_xlabel('log10(SNR)')
    ax2.set_ylabel('Абсолютная корреляция Пирсона')
    ax2.legend()
    ax2.grid(True)
    ax2.set_ylim(0, 1.05)

    plt.suptitle(f'Эксперимент 2: Детализация по источникам (rho={correlation_rho})')
    plt.tight_layout()
    plt.savefig('exp2_snr_results_detailed.png', dpi=300)
    print("Результаты сохранены в exp2_snr_results_detailed.png")

if __name__ == '__main__':
    run_snr_experiment()