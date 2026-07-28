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

from signal_simulation import generate_distributed_sources
from pyriemann.estimation import Covariances
from pyriemann.geometry.distance import pairwise_distance
from topological_spatial_filter import fit_filters
import umap

def run_snr_experiment():
    """
    Эксперимент 1: Оценка качества восстановления (по паттерну и мощности)
    при различных уровнях отношения сигнал/шум (SNR).
    Используется метод Монте-Карло для усреднения результатов.
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
    Ts = 300.0           # Длительность симуляции в секундах
    Nsrc = 100           # Общее количество источников
    Ndistr = 1           # Ищем ОДИН целевой источник
    flanker = 1.0        # Фланкеры для фильтрации
    gamma = 0.1          # Фиксированный уровень сенсорного шума

    # ПАРАМЕТРЫ ЭКСПЕРИМЕНТА
<<<<<<< Updated upstream
    snr_logs = np.arange(-1.4, 1.01, 0.2)  # десятичный логарифм SNR от -1.4 до 1.0
    n_mc_iterations = 2  # количество итераций Монте-Карло (уменьшено для скорости)
=======
    snr_logs = np.arange(-0.2, 1.01, 0.1)  # десятичный логарифм SNR от -0.2 до 1.0
    n_mc_iterations = 10  # количество итераций Монте-Карло
>>>>>>> Stashed changes

    # Для эпохирования
    Wsize = 2.0
    Ssize = 0.5
    overlap = Wsize - Ssize

    # Сохранение результатов (UMAP)
    mean_pattern_corr = []
    std_pattern_corr = []
    mean_power_corr = []
    std_power_corr = []

    # Сохранение результатов (ICA)
    mean_pattern_corr_ica = []
    std_pattern_corr_ica = []
    mean_power_corr_ica = []
    std_power_corr_ica = []

    for snr_log in snr_logs:
        current_snr = 10 ** snr_log
        print(f"\nТестируем SNR = 10^{snr_log:.2f} ({current_snr:.4f})")

        iter_pattern_corrs = []
        iter_power_corrs = []
        iter_pattern_corrs_ica = []
        iter_power_corrs_ica = []

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

            # Ковариационные матрицы
            covmats = Covariances(estimator='oas').fit_transform(epochs_data)

            # Расстояния и UMAP топология
            dist_matrix = pairwise_distance(covmats, metric='riemann')

            # Обучение фильтров (ищем 3D компоненты)
            w_opt, _, final_losses, _, _ = fit_filters(
                C=covmats,
                D_matrix=dist_matrix,
                N_dim=3,                # Вложение в 3D
                K_restarts=1,
                n_neighbors=20,
                epochs=300,             # Можно уменьшить для скорости, но лучше оставить 300-500
                lr=0.05,
                verbose=False
            )

            w_found_3d = w_opt[0] # Размер (3, N_channels)

            # Находим паттерны для этих трех компонент
            # a = E[C] * w
            mean_C = np.mean(covmats, axis=0)
            patterns_found = []
            power_found = []

            for d in range(3):
                w_d = w_found_3d[d, :]
                a_d = mean_C @ w_d
                patterns_found.append(a_d)

                # Считаем мощность
                p_d = np.zeros(covmats.shape[0])
                for i in range(covmats.shape[0]):
                    p_d[i] = np.log(w_d.T @ covmats[i] @ w_d)
                power_found.append(p_d)

            # Истинный паттерн и мощность целевого источника
            true_pattern = GA[:, 0]
            true_z = z[0, :]

            # Усреднение истинной мощности (z) по эпохам
            n_epochs_mne = len(epochs)
            sfreq = info['sfreq']
            n_samples_window = int(Wsize * sfreq)
            n_samples_step = int(Ssize * sfreq)

            p_true_epochs = np.zeros(n_epochs_mne)
            for i in range(n_epochs_mne):
                start = int(i * n_samples_step)
                end = start + n_samples_window
                p_true_epochs[i] = np.mean(true_z[start:end])

            # Находим компоненту UMAP с лучшей абсолютной корреляцией паттерна
            best_pattern_corr = 0
            best_power_corr = 0

            for d in range(3):
                # Корреляция паттерна
                patt_corr = np.abs(np.corrcoef(patterns_found[d], true_pattern)[0, 1])
                # Корреляция мощности
                pow_corr = np.abs(np.corrcoef(power_found[d], p_true_epochs)[0, 1])

                if patt_corr > best_pattern_corr:
                    best_pattern_corr = patt_corr
                    best_power_corr = pow_corr

            iter_pattern_corrs.append(best_pattern_corr)
            iter_power_corrs.append(best_power_corr)

<<<<<<< Updated upstream
            # ==============================
            # BASELINE: СРАВНЕНИЕ С ICA
            # ==============================
            from mne.preprocessing import ICA
            # Запускаем ICA на сырых данных
            ica = ICA(n_components=15, method='picard', random_state=42, max_iter=200)
            # Чтобы не сыпались лишние варнинги
            mne.set_log_level('ERROR')
            ica.fit(raw)
            mne.set_log_level('INFO')
=======
            # =================================================================
            # 2. ICA BASELINE (ФОНОВОЕ СРАВНЕНИЕ)
            # =================================================================
            # Ищем 15 компонент, чтобы не перегружать вычисления, но захватить основную дисперсию
            ica = mne.preprocessing.ICA(n_components=0.999, method='fastica', random_state=42)
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
>>>>>>> Stashed changes

            # Паттерны ICA - столбцы матрицы в сенсорном пространстве
            ica_patterns = ica.get_components()

            best_pattern_corr_ica = 0
            best_power_corr_ica = 0

            # Получаем источники ICA (компоненты во времени)
            ica_sources = ica.get_sources(raw).get_data() # (n_components, n_samples)

            for c in range(ica_patterns.shape[1]):
                patt_ica = ica_patterns[:, c]
                source_ica = ica_sources[c, :]

                # Считаем мощность компоненты ICA в эпохах (как огибающая)
                from scipy.signal import hilbert
                # Огибающая мощности ICA компоненты
                env_ica = np.abs(hilbert(source_ica))**2

                p_ica_epochs = np.zeros(n_epochs_mne)
                for i in range(n_epochs_mne):
                    start = int(i * n_samples_step)
                    end = start + n_samples_window
                    p_ica_epochs[i] = np.mean(env_ica[start:end])
                # Приводим к логарифму, как у UMAP
                p_ica_epochs = np.log(p_ica_epochs + 1e-8)

                corr_patt_ica = np.abs(np.corrcoef(patt_ica, true_pattern)[0, 1])
                corr_pow_ica = np.abs(np.corrcoef(p_ica_epochs, p_true_epochs)[0, 1])

                if corr_patt_ica > best_pattern_corr_ica:
                    best_pattern_corr_ica = corr_patt_ica
                    best_power_corr_ica = corr_pow_ica

            iter_pattern_corrs_ica.append(best_pattern_corr_ica)
            iter_power_corrs_ica.append(best_power_corr_ica)

        mean_pattern_corr.append(np.mean(iter_pattern_corrs))
        std_pattern_corr.append(np.std(iter_pattern_corrs))
        mean_power_corr.append(np.mean(iter_power_corrs))
        std_power_corr.append(np.std(iter_power_corrs))

        mean_pattern_corr_ica.append(np.mean(iter_pattern_corrs_ica))
        std_pattern_corr_ica.append(np.std(iter_pattern_corrs_ica))
        mean_power_corr_ica.append(np.mean(iter_power_corrs_ica))
        std_power_corr_ica.append(np.std(iter_power_corrs_ica))

    # Построение графиков
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.errorbar(snr_logs, mean_pattern_corr, yerr=std_pattern_corr, fmt='-o', capsize=5, color='b', label='UMAP (fit_filters)')
    ax1.errorbar(snr_logs, mean_pattern_corr_ica, yerr=std_pattern_corr_ica, fmt='--s', capsize=5, color='g', label='ICA (picard)')
    ax1.set_title('Качество восстановления паттерна (Leadfield)')
    ax1.set_xlabel('log10(SNR)')
    ax1.set_ylabel('Абсолютная корреляция Пирсона')
    ax1.grid(True)
    ax1.set_ylim(0, 1.05)

    ax1.legend()

    ax2.errorbar(snr_logs, mean_power_corr, yerr=std_power_corr, fmt='-o', capsize=5, color='r', label='UMAP (fit_filters)')
    ax2.errorbar(snr_logs, mean_power_corr_ica, yerr=std_power_corr_ica, fmt='--s', capsize=5, color='orange', label='ICA (picard)')
    ax2.set_title('Качество восстановления мощности сигнала')
    ax2.legend()
    ax2.set_xlabel('log10(SNR)')
    ax2.set_ylabel('Абсолютная корреляция Пирсона')
    ax2.grid(True)
    ax2.set_ylim(0, 1.05)

    plt.suptitle('Эксперимент 1: Влияние SNR на топологический поиск источников')
    plt.tight_layout()
    plt.savefig('exp1_snr_results.png', dpi=300)
    print("Результаты сохранены в exp1_snr_results.png")

if __name__ == '__main__':
    run_snr_experiment()
