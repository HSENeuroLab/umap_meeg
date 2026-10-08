# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import mne

from pyriemann.estimation import Covariances
from pyriemann.utils.distance import distance
from scipy.signal import butter, filtfilt

# Укажите свои пути к библиотекам
lib_directory = os.path.abspath("C:/Users/ansbel/Documents/GitHub/pyRiemann")
if lib_directory not in sys.path:
    sys.path.insert(0, lib_directory)

sim_dir = os.path.abspath("C:/Users/ansbel/Documents/GitHub/umap_meeg/examples/meeg_examples/simulations/")
if sim_dir not in sys.path:
    sys.path.insert(0, sim_dir)

# =============================================================================
# 1. ФУНКЦИЯ MONTE-CARLO СИМУЛЯЦИИ
# =============================================================================
def run_monte_carlo_riemann(G, Fs, P_bases, max_power=10.0,
                            n_iters=30, n_steps=20, epoch_len=2.0):
    N = int(epoch_len * Fs)
    Nsens, Nsites = G.shape[0], G.shape[1] // 3
    b, a = butter(5, [15 / (Fs / 2), 25 / (Fs / 2)], btype='bandpass')

    Nsrc = 100  # 1 таргет + 99 фоновых

    common_powers = np.linspace(0.0, max_power, n_steps)

    results_dist = {pb: np.zeros((n_iters, n_steps)) for pb in P_bases}
    results_powers = {pb: common_powers for pb in P_bases}

    cov_est = Covariances(estimator='lwf')

    print(f"Запуск симуляции Монте-Карло: {n_iters} итераций (метрика: riemann)...")

    for it in range(n_iters):
        if (it + 1) % 5 == 0:
            print(f"Итерация {it + 1}/{n_iters}")

        # 1. Выбираем 100 случайных локаций
        src_inds = np.random.choice(Nsites, Nsrc, replace=False)
        GA = np.zeros((Nsens, Nsrc))
        for i in range(Nsrc):
            r = np.random.randn(3)
            r /= np.linalg.norm(r)
            idx = src_inds[i]
            GA[:, i] = G[:, idx*3]*r[0] + G[:, idx*3+1]*r[1] + G[:, idx*3+2]*r[2]

        # 2. Генерируем временные ряды
        S = filtfilt(b, a, np.random.randn(Nsrc, N), axis=1)
        S /= np.std(S, axis=1, keepdims=True)
        sens_noise = 0.0001 * np.random.randn(Nsens, N)

        # 3. Фоновый сигнал
        X_bg = GA[:, 1:] @ S[1:, :] + sens_noise

        # 4. По каждому базовому состоянию
        for pb in P_bases:
            X_base = X_bg + GA[:, 0:1] @ (S[0:1, :] * np.sqrt(pb))
            C_base = cov_est.fit_transform(X_base[np.newaxis, ...])[0]

            for step, P_test in enumerate(common_powers):
                X_test = X_bg + GA[:, 0:1] @ (S[0:1, :] * np.sqrt(P_test))
                C_test = cov_est.fit_transform(X_test[np.newaxis, ...])[0]

                results_dist[pb][it, step] = distance(C_base, C_test, metric="riemann")

    return results_powers, results_dist

# =============================================================================
# 2. ИНИЦИАЛИЗАЦИЯ И РАСЧЕТ
# =============================================================================
fwd_fname = 'C:/Users/ansbel/Documents/GitHub/umap_meeg/examples/meeg_examples/simulations/fsaverage-fwd.fif'
info_fname = 'C:/Users/ansbel/Documents/GitHub/umap_meeg/examples/meeg_examples/simulations/fsaverage-info.fif'

fwd = mne.read_forward_solution(fwd_fname, verbose=False)
info = mne.io.read_info(info_fname, verbose=False)
Fs = info['sfreq']

picks = mne.pick_types(info, eeg=True, meg=False)
info_sub = mne.pick_info(info, sel=picks)
G_sub = fwd['sol']['data'][picks, :]

P_base_list = [0.0, 0.1, 1.0, 2.0, 4.0]
max_target_power = 10.0

n_iters = 100
n_steps = 25

res_powers, res_dist = run_monte_carlo_riemann(
    G_sub, Fs, P_bases=P_base_list,
    max_power=max_target_power, n_iters=n_iters, n_steps=n_steps
)

# =============================================================================
# 3. ВИЗУАЛИЗАЦИЯ: ТОЛЬКО P >= P_base И СОХРАНЕНИЕ
# =============================================================================
mean_curves = {}

fig, ax = plt.subplots(figsize=(10, 6))
colors = plt.cm.tab10(np.linspace(0, 1, len(P_base_list)))

for idx, pb in enumerate(P_base_list):
    color = colors[idx]
    powers = res_powers[pb]
    distances = res_dist[pb]

    mean_dist = np.mean(distances, axis=0)
    std_dist  = np.std(distances,  axis=0)

    # Оставляем только правую ветвь: P > P_base (строго больше, чтобы не дублировать ноль)
    mask = powers > pb
    p_gt = powers[mask]
    m_gt = mean_dist[mask]
    s_gt = std_dist[mask]
    
    # Дополняем массивы точкой точного нуля в P = P_base
    p_use = np.insert(p_gt, 0, pb)
    m_use = np.insert(m_gt, 0, 0.0)
    s_use = np.insert(s_gt, 0, 0.0)

    mean_curves[pb] = {
        "powers": p_use,
        "mean":   m_use,
        "std":    s_use,
    }

    label_str = f"$P_{{base}} = {pb}$"
    ax.plot(p_use, m_use, color=color, linewidth=2.5, label=label_str)
    ax.fill_between(p_use, np.clip(m_use - s_use, 0, None), m_use + s_use,
                    color=color, alpha=0.15)
    ax.plot(p_use[0], m_use[0], 'o', color=color, markersize=6, zorder=5)
    ax.axvline(x=pb, color=color, linestyle='--', alpha=0.5)

ax.axhline(0, color='black', linewidth=1.2, linestyle='-', zorder=1)
ax.set_title("Риманово расстояние (Монте-Карло): $d_R(P)$, только $P \\geq P_{base}$", fontsize=13)
ax.set_xlabel("Целевая мощность источника ($P$)", fontsize=11)
ax.set_ylabel("Риманово расстояние (неотрицательное)", fontsize=11)
ax.grid(True, linestyle=':', alpha=0.7)
ax.set_xlim(0, max_target_power)
ax.set_ylim(bottom=0)
ax.legend(fontsize=10, loc="upper left")

plt.suptitle(f"Монте-Карло: риманово расстояние ($n_{{iter}} = {n_iters}$)", fontsize=15, y=0.98)
plt.tight_layout()
plt.show()

save_path = "riemann_mean_curves.npz"
payload = {}
for pb in P_base_list:
    key = f"pb_{pb}".replace(".", "p")
    payload[f"{key}_powers"] = mean_curves[pb]["powers"]
    payload[f"{key}_mean"]   = mean_curves[pb]["mean"]
    payload[f"{key}_std"]    = mean_curves[pb]["std"]

np.savez(save_path, **payload)
print(f"Средние кривые сохранены в {save_path}")

# %%
# =============================================================================
# 4. ПРОИЗВОДНЫЕ ПО КАЖДОЙ ИТЕРАЦИИ + ДВЕ ПАНЕЛИ
# =============================================================================
CI = 1.96
derivs_per_iter = {}

for pb in P_base_list:
    powers    = res_powers[pb]
    distances = res_dist[pb]

    mask = powers > pb
    P_gt = powers[mask]
    d_gt = distances[:, mask]

    # Гарантируем, что точка P = P_base (где d_R строго = 0) является началом сетки
    P_use = np.insert(P_gt, 0, pb)
    zeros = np.zeros((distances.shape[0], 1))
    d_use = np.hstack([zeros, d_gt])

    x_rel = P_use - pb  # Сдвиг по X: 0 = P_base

    # Производная per-iteration по реальной оси P
    grad = np.gradient(d_use, P_use, axis=1)

    n       = grad.shape[0]
    mean_g  = np.mean(grad, axis=0)
    std_g   = np.std(grad,  axis=0, ddof=1)
    half_ci = CI * std_g / np.sqrt(n)

    derivs_per_iter[pb] = (x_rel, grad, mean_g, half_ci)

    mean_curves[pb]["P_use"]      = P_use
    mean_curves[pb]["x_rel"]      = x_rel
    mean_curves[pb]["deriv_mean"] = mean_g
    mean_curves[pb]["deriv_ci"]   = half_ci
    mean_curves[pb]["deriv_all"]  = grad

# Две панели
fig, (ax_top, ax_bot) = plt.subplots(
    2, 1, figsize=(11, 12), sharex=False,
    gridspec_kw={'height_ratios': [1, 1]}
)

colors = plt.cm.tab10(np.linspace(0, 1, len(P_base_list)))

all_lows  = np.concatenate([(derivs_per_iter[pb][2] - derivs_per_iter[pb][3]) for pb in P_base_list])
all_highs = np.concatenate([(derivs_per_iter[pb][2] + derivs_per_iter[pb][3]) for pb in P_base_list])
y_top_lim = all_highs.max() * 1.10
y_bot_lim = min(0.0, all_lows.min() * 1.10)

for idx, pb in enumerate(P_base_list):
    color = colors[idx]
    label_str = f"$P_{{base}} = {pb}$"

    # ---------- ВЕРХНЯЯ ПАНЕЛЬ ----------
    p_use = mean_curves[pb]["powers"]
    m_use = mean_curves[pb]["mean"]
    s_use = mean_curves[pb]["std"]

    ax_top.plot(p_use, m_use, color=color, linewidth=2.5, label=label_str)
    ax_top.fill_between(p_use, np.clip(m_use - s_use, 0, None), m_use + s_use,
                        color=color, alpha=0.15)
    ax_top.axvline(x=pb, color=color, linestyle='--', alpha=0.5)
    ax_top.plot(p_use[0], m_use[0], 'o', color=color, markersize=6, zorder=5)

    # ---------- НИЖНЯЯ ПАНЕЛЬ ----------
    x_rel, grad_arr, mean_g, half_ci = derivs_per_iter[pb]

    ax_bot.plot(x_rel, mean_g, color=color, linewidth=2.5, label=label_str)
    ax_bot.fill_between(x_rel, mean_g - half_ci, mean_g + half_ci,
                        color=color, alpha=0.20)
    ax_bot.plot(x_rel[0], mean_g[0], 'o', color=color, markersize=6, zorder=5)

ax_top.axhline(0, color='black', linewidth=1.2, zorder=1)
ax_top.grid(True, linestyle=':', alpha=0.7)
ax_top.set_xlim(0, max_target_power)
ax_top.set_ylim(bottom=0)
ax_top.legend(fontsize=10, loc="upper left")

ax_bot.axhline(0, color='black', linewidth=1.2, zorder=1)
ax_bot.axvline(0, color='black', linewidth=1.0, alpha=0.6, zorder=1)
ax_bot.set_xlabel("$P - P_{base}$", fontsize=11)
ax_bot.set_ylabel("$d d_R / dP$", fontsize=11)
ax_bot.grid(True, linestyle=':', alpha=0.7)
ax_bot.set_xlim(0, max_target_power - min(P_base_list))
ax_bot.set_ylim(y_bot_lim, y_top_lim)
ax_bot.legend(fontsize=10, loc="upper right")

plt.tight_layout()
plt.show()