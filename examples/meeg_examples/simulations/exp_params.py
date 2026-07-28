import os
import sys
import numpy as np
import mne
import matplotlib.pyplot as plt
from scipy.linalg import null_space

# Add necessary paths
work_dir = os.path.dirname(os.path.abspath(__file__))
if work_dir not in sys.path:
    sys.path.insert(0, work_dir)

repo_root = os.path.abspath(os.path.join(work_dir, "..", "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import sys
import os
lib_directory = os.path.abspath("C:/Users/ansbel/Documents/GitHub/pyRiemann") 
if lib_directory not in sys.path:
    sys.path.insert(0, lib_directory)

from signal_simulation import generate_distributed_sources
from pyriemann.estimation import Covariances
from pyriemann.geometry.distance import pairwise_distance
from topological_spatial_filter import fit_filters

def run_params_experiment():
    """
    Эксперимент 2: Оценка качества восстановления при варьировании числа соседей (n_neighbors)
    и метрики расстояния для ковариационных матриц (Riemannian vs euclid).
    """
    print("Начинаем эксперимент по исследованию гиперпараметров...")

    # ЗАГРУЗКА ПРЯМОЙ МОДЕЛИ
    fwd_fname = os.path.join(work_dir, 'fsaverage-fwd.fif')
    info_fname = os.path.join(work_dir, 'fsaverage-info.fif')

    fwd = mne.read_forward_solution(fwd_fname, verbose=False)
    info = mne.io.read_info(info_fname, verbose=False)
    G = fwd['sol']['data']
    Fs = info['sfreq']

    # ПАРАМЕТРЫ СИМУЛЯЦИИ (Фиксированный SNR для этого эксперимента)
    Ts = 300.0
    Nsrc = 100
    Ndistr = 1
    flanker = 1.0
    gamma = 0.1
    # current_snr = 10 ** 0.4 # Фиксированный SNR (умеренный)
    current_snr = 5

    # ПАРАМЕТРЫ ЭКСПЕРИМЕНТА
    neighbors_list = [5, 10, 20, 50, 100, 150, 200]
    metrics = ['riemann', 'euclid']
    n_mc_iterations = 3  # Количество итераций для усреднения

    # Для эпохирования
    Wsize = 2.0
    Ssize = 0.5
    overlap = Wsize - Ssize

    results_pattern = {m: {'mean': [], 'std': []} for m in metrics}
    results_power = {m: {'mean': [], 'std': []} for m in metrics}

    for metric in metrics:
        print(f"\n--- Тестируем метрику: {metric} ---")

        for n_neighbors in neighbors_list:
            print(f"  n_neighbors = {n_neighbors}...")

            iter_pattern_corrs = []
            iter_power_corrs = []

            for iteration in range(n_mc_iterations):
                # Генерация данных
                X_s, X_bg, X_n, z, GA, S = generate_distributed_sources(
                    G, Nsrc, Ndistr, flanker, Ts, Fs
                )

                # Смешивание
                X = current_snr * X_s + X_bg + gamma * X_n / np.linalg.norm(X_s, 'fro')

                # Эпохирование
                raw = mne.io.RawArray(X, info, verbose=False)
                epochs = mne.make_fixed_length_epochs(
                    raw, duration=Wsize, overlap=overlap, preload=True, verbose=False
                )
                epochs_data = epochs.get_data(copy=False)

                # Ковариационные матрицы
                covmats = Covariances(estimator='oas').fit_transform(epochs_data)

                # Расстояния
                dist_matrix = pairwise_distance(covmats, metric=metric)

                # Обучение фильтров
                w_opt, _, _, _, _ = fit_filters(
                    C=covmats,
                    D_matrix=dist_matrix,
                    N_dim=3,
                    K_restarts=1,
                    n_neighbors=n_neighbors,
                    epochs=300,
                    lr=0.05,
                    verbose=False
                )

                w_found_3d = w_opt[0]
                mean_C = np.mean(covmats, axis=0)
                patterns_found = []
                power_found = []

                for d in range(3):
                    w_d = w_found_3d[d, :]
                    a_d = mean_C @ w_d
                    patterns_found.append(a_d)

                    p_d = np.zeros(covmats.shape[0])
                    for i in range(covmats.shape[0]):
                        p_d[i] = np.log(w_d.T @ covmats[i] @ w_d)
                    power_found.append(p_d)

                true_pattern = GA[:, 0]
                true_z = z[0, :]

                n_epochs_mne = len(epochs)
                sfreq = info['sfreq']
                n_samples_window = int(Wsize * sfreq)
                n_samples_step = int(Ssize * sfreq)

                p_true_epochs = np.zeros(n_epochs_mne)
                for i in range(n_epochs_mne):
                    start = int(i * n_samples_step)
                    end = start + n_samples_window
                    p_true_epochs[i] = np.mean(true_z[start:end])

                best_pattern_corr = 0
                best_power_corr = 0

                for d in range(3):
                    patt_corr = np.abs(np.corrcoef(patterns_found[d], true_pattern)[0, 1])
                    pow_corr = np.abs(np.corrcoef(power_found[d], p_true_epochs)[0, 1])
                    if patt_corr > best_pattern_corr:
                        best_pattern_corr = patt_corr
                        best_power_corr = pow_corr

                iter_pattern_corrs.append(best_pattern_corr)
                iter_power_corrs.append(best_power_corr)

            results_pattern[metric]['mean'].append(np.mean(iter_pattern_corrs))
            results_pattern[metric]['std'].append(np.std(iter_pattern_corrs))
            results_power[metric]['mean'].append(np.mean(iter_power_corrs))
            results_power[metric]['std'].append(np.std(iter_power_corrs))

    # Построение графиков
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    colors = {'riemann': 'b', 'euclid': 'r'}
    labels = {'riemann': 'Riemannian', 'euclid': 'euclid'}

    for metric in metrics:
        ax1.errorbar(neighbors_list, results_pattern[metric]['mean'], yerr=results_pattern[metric]['std'],
                     fmt='-o', capsize=5, color=colors[metric], label=labels[metric])
        ax2.errorbar(neighbors_list, results_power[metric]['mean'], yerr=results_power[metric]['std'],
                     fmt='-o', capsize=5, color=colors[metric], label=labels[metric])

    ax1.set_title('Влияние числа соседей на паттерн')
    ax1.set_xlabel('n_neighbors')
    ax1.set_ylabel('Абсолютная корреляция Пирсона')
    ax1.legend()
    ax1.grid(True)
    ax1.set_ylim(0, 1.05)

    ax2.set_title('Влияние числа соседей на мощность')
    ax2.set_xlabel('n_neighbors')
    ax2.set_ylabel('Абсолютная корреляция Пирсона')
    ax2.legend()
    ax2.grid(True)
    ax2.set_ylim(0, 1.05)

    plt.suptitle('Эксперимент 2: Влияние гиперпараметров (соседи и метрика) на поиск источников')
    plt.tight_layout()
    plt.savefig('exp2_params_results.png', dpi=300)
    print("Результаты сохранены в exp2_params_results.png")

if __name__ == '__main__':
    run_params_experiment()
