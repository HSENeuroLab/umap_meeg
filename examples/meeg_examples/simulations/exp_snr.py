# -*- coding: utf-8 -*-
import sys
import os
import mne
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import null_space

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

from signal_simulation import generate_distributed_sources
from pyriemann.estimation import Covariances
from pyriemann.geometry.distance import pairwise_distance
from topological_spatial_filter import fit_filters
import umap

def run_snr_experiment():
    """
    Эксперимент 1: Оценка качества восстановления (по паттерну и мощности)
    при различных уровнях отношения сигнал/шум (SNR).
    Сравнение Topological Spatial Filter (TSF) и FastICA.
    """
    print("Начинаем эксперимент по исследованию влияния SNR...")

    # 1. ЗАГРУЗКА ПРЯМОЙ МОДЕЛИ
    fwd_fname = os.path.join(work_dir, 'fsaverage-fwd.fif')
    info_fname = os.path.join(work_dir, 'fsaverage-info.fif')

    fwd = mne.read_forward_solution(fwd_fname, verbose=False)
    info = mne.io.read_info(info_fname, verbose=False)
    G = fwd['sol']['data']
    Fs = info['sfreq']

    # ПАРАМЕТРЫ СИМУЛЯЦИИ
    Ts = 400.0           # Длительность симуляции в секундах
    Nsrc = 100           # Общее количество источников
    Ndistr = 1           # Ищем ОДИН целевой источник
    flanker = 1.0        # Фланкеры для фильтрации
    gamma = 0.1          # Фиксированный уровень сенсорного шума

    # ПАРАМЕТРЫ ЭКСПЕРИМЕНТА
    snr_logs = np.arange(-0.2, 1.01, 0.1)  # десятичный логарифм SNR от -0.2 до 1.0
    n_mc_iterations = 5  # количество итераций Монте-Карло

    # Для эпохирования
    Wsize = 1.0
    Ssize = 0.5
    overlap = Wsize - Ssize

    # Сохранение результатов для TSF (вашего метода)
    mean_pattern_corr = []
    std_pattern_corr = []
    mean_power_corr = []
    std_power_corr = []

    # Сохранение результатов для ICA
    mean_ica_pattern_corr = []
    std_ica_pattern_corr = []
    mean_ica_power_corr = []
    std_ica_power_corr = []

    for snr_log in snr_logs:
        current_snr = 10 ** snr_log
        print(f"\nТестируем SNR = 10^{snr_log:.2f} ({current_snr:.4f})")

        iter_pattern_corrs = []
        iter_power_corrs = []
        
        iter_ica_pattern_corrs = []
        iter_ica_power_corrs = []

        for iteration in range(n_mc_iterations):
            print(f"  Итерация {iteration + 1}/{n_mc_iterations}...")

            # Генерация данных
            X_s, X_bg, X_n, z, GA, S = generate_distributed_sources(
                G, Nsrc, Ndistr, flanker, Ts, Fs
            )

            # Смешивание с текущим SNR
            X = current_snr * X_s + X_bg + gamma * X_n / np.linalg.norm(X_s, 'fro')

            # Эпохирование
            raw = mne.io.RawArray(X, info, verbose=False)
            epochs = mne.make_fixed_length_epochs(
                raw, duration=Wsize, overlap=overlap, preload=True, verbose=False
            )
            epochs_data = epochs.get_data(copy=False)

            # =================================================================
            # БАЗОВАЯ ИСТИНА (GROUND TRUTH)
            # =================================================================
            true_pattern = GA[:, 0]
            true_z = z[0, :]

            n_epochs_mne = len(epochs)
            n_samples_window = int(Wsize * Fs)
            n_samples_step = int(Ssize * Fs)

            p_true_epochs = np.zeros(n_epochs_mne)
            for i in range(n_epochs_mne):
                start = int(i * n_samples_step)
                end = start + n_samples_window
                p_true_epochs[i] = np.mean(true_z[start:end])


            # =================================================================
            # 1. TOPOLOGICAL SPATIAL FILTER (ВАШ МЕТОД)
            # =================================================================
            covmats = Covariances(estimator='oas').fit_transform(epochs_data)
            dist_matrix = pairwise_distance(covmats, metric='riemann')

            Ndim = 2
            w_opt, _, final_losses, _, _ = fit_filters(
                C=covmats,
                D_matrix=dist_matrix,
                N_dim=Ndim,                
                K_restarts=1,
                n_neighbors=30,
                epochs=300,             
                lr=0.05,
                verbose=False
            )

            w_found_3d = w_opt[0]
            mean_C = np.mean(covmats, axis=0)
            patterns_found = []
            power_found = []

            for d in range(Ndim):
                w_d = w_found_3d[d, :]
                patterns_found.append(mean_C @ w_d)
                
                p_d = np.zeros(covmats.shape[0])
                for i in range(covmats.shape[0]):
                    p_d[i] = np.log(w_d.T @ covmats[i] @ w_d)
                power_found.append(p_d)

            best_pattern_corr = 0
            best_power_corr = 0

            for d in range(Ndim):
                patt_corr = np.abs(np.corrcoef(patterns_found[d], true_pattern)[0, 1])
                pow_corr = np.abs(np.corrcoef(power_found[d], p_true_epochs)[0, 1])

                if patt_corr > best_pattern_corr:
                    best_pattern_corr = patt_corr
                    best_power_corr = pow_corr

            iter_pattern_corrs.append(best_pattern_corr)
            iter_power_corrs.append(best_power_corr)

            # =================================================================
            # 2. ICA BASELINE (ФОНОВОЕ СРАВНЕНИЕ)
            # =================================================================
            # Ищем 15 компонент, чтобы не перегружать вычисления, но захватить основную дисперсию
            ica = mne.preprocessing.ICA(n_components=15, method='fastica', random_state=42)
            ica.fit(raw, verbose=False)
            
            # Матрица смешивания (mixing matrix) содержит пространственные паттерны ICA
            ica_patterns = ica.get_components() 
            
            # Извлекаем временные ряды источников ICA и режем их на эпохи
            ica_raw = ica.get_sources(raw)
            ica_epochs = mne.make_fixed_length_epochs(
                ica_raw, duration=Wsize, overlap=overlap, preload=True, verbose=False
            )
            ica_epochs_data = ica_epochs.get_data(copy=False) # Размер: (n_epochs, n_components, n_samples)
            
            best_ica_patt_corr = 0
            best_ica_pow_corr = 0

            for c in range(ica.n_components_):
                # Корреляция паттерна ICA с истинным Leadfield
                c_patt_corr = np.abs(np.corrcoef(ica_patterns[:, c], true_pattern)[0, 1])
                
                # Считаем логарифм дисперсии (мощность) ICA компоненты в каждом окне
                c_power = np.log(np.var(ica_epochs_data[:, c, :], axis=1) + 1e-8)
                c_pow_corr = np.abs(np.corrcoef(c_power, p_true_epochs)[0, 1])
                
                # Жадный выбор лучшей компоненты ICA (ориентируемся по паттерну)
                if c_patt_corr > best_ica_patt_corr:
                    best_ica_patt_corr = c_patt_corr
                    best_ica_pow_corr = c_pow_corr
            
            iter_ica_pattern_corrs.append(best_ica_patt_corr)
            iter_ica_power_corrs.append(best_ica_pow_corr)

        # Сохранение агрегированных метрик для TSF
        mean_pattern_corr.append(np.mean(iter_pattern_corrs))
        std_pattern_corr.append(np.std(iter_pattern_corrs))
        mean_power_corr.append(np.mean(iter_power_corrs))
        std_power_corr.append(np.std(iter_power_corrs))
        
        # Сохранение агрегированных метрик для ICA
        mean_ica_pattern_corr.append(np.mean(iter_ica_pattern_corrs))
        std_ica_pattern_corr.append(np.std(iter_ica_pattern_corrs))
        mean_ica_power_corr.append(np.mean(iter_ica_power_corrs))
        std_ica_power_corr.append(np.std(iter_ica_power_corrs))


    # =========================================================================
    # ПОСТРОЕНИЕ ГРАФИКОВ СО СРАВНЕНИЕМ
    # =========================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # График 1: Паттерны
    ax1.errorbar(snr_logs, mean_pattern_corr, yerr=std_pattern_corr, fmt='-o', capsize=5, color='b', label='TSF (Наш метод)')
    ax1.errorbar(snr_logs, mean_ica_pattern_corr, yerr=std_ica_pattern_corr, fmt='--s', capsize=5, color='g', alpha=0.7, label='FastICA')
    ax1.set_title('Качество восстановления паттерна (Leadfield)')
    ax1.set_xlabel('log10(SNR)')
    ax1.set_ylabel('Абсолютная корреляция Пирсона')
    ax1.legend()
    ax1.grid(True)
    ax1.set_ylim(0, 1.05)

    # График 2: Мощности
    ax2.errorbar(snr_logs, mean_power_corr, yerr=std_power_corr, fmt='-o', capsize=5, color='r', label='TSF (Наш метод)')
    ax2.errorbar(snr_logs, mean_ica_power_corr, yerr=std_ica_power_corr, fmt='--s', capsize=5, color='orange', alpha=0.7, label='FastICA')
    ax2.set_title('Качество восстановления динамики мощности')
    ax2.set_xlabel('log10(SNR)')
    ax2.set_ylabel('Абсолютная корреляция Пирсона')
    ax2.legend()
    ax2.grid(True)
    ax2.set_ylim(0, 1.05)

    plt.suptitle('Эксперимент 1: Влияние SNR (Topological Filter vs. ICA)')
    plt.tight_layout()
    plt.savefig('exp1_snr_results_with_ica.png', dpi=300)
    print("Результаты сохранены в exp1_snr_results_with_ica.png")

if __name__ == '__main__':
    run_snr_experiment()