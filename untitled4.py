# -*- coding: utf-8 -*-
"""
Эксперимент: Сравнение UMAP (Риманова дефляция) и ICA при различных SNR.
Сценарий: 2 микросостояния, по 1 целевому источнику на каждое.
"""
import os
import sys
import numpy as np 
import matplotlib.pyplot as plt
from scipy.linalg import eigh, null_space
from scipy.optimize import linear_sum_assignment
import scipy.signal
from scipy.signal import butter, filtfilt, hilbert

import mne
import umap
from umap.parametric_umap import ParametricUMAP
import tensorflow as tf

# Подключение pyRiemann
lib_directory = os.path.abspath("C:/Users/ansbel/Documents/GitHub/pyRiemann") 
if lib_directory not in sys.path:
    sys.path.insert(0, lib_directory)
from pyriemann.estimation import Covariances
from pyriemann.geometry.distance import pairwise_distance

# =============================================================================
# 1. ФУНКЦИИ ГЕНЕРАЦИИ ДАННЫХ
# =============================================================================
def make_microstate_schedule(N, Fs, n_states, dwell_time, mode='round_robin', smooth_s=0.05):
    dwell_samples = max(1, int(dwell_time * Fs))
    n_intervals = int(np.ceil(N / dwell_samples))
    A = np.zeros((n_states, N))

    if mode == 'round_robin':
        order = np.arange(n_intervals) % n_states
    elif mode == 'random':
        order = np.random.randint(0, n_states, size=n_intervals)
        for i in range(1, n_intervals):
            while order[i] == order[i-1]:
                order[i] = np.random.randint(0, n_states)
    else:
        raise ValueError(mode)

    for i, s in enumerate(order):
        start = i * dwell_samples
        end   = min(start + dwell_samples, N)
        A[s, start:end] = 1.0

    if smooth_s > 0:
        w = max(1, int(smooth_s * Fs))
        kernel = np.hanning(w + 1)
        kernel /= kernel.sum()
        A = np.apply_along_axis(lambda x: np.convolve(x, kernel, mode='same'), axis=1, arr=A)
    return A

def generate_microstate_sources(
    G, Nsrc, N_microstates=2, sources_per_microstate=1,
    flanker=25.0, Ts=600, Fs=100, dwell_time=5.0,
    snr_active=2.0, snr_inactive=0.1, schedule='round_robin',
    smooth_s=0.05, transition_freq=0.5
):
    N = int(Ts * Fs)
    flanker_samples = int(flanker * Fs)
    Ndistr = N_microstates * sources_per_microstate
    
    b, a = butter(5, [8/(Fs/2), 12/(Fs/2)], btype='bandpass')
    b_lp, a_lp = butter(5, transition_freq / (Fs/2), btype='lowpass')

    Gx, Gy, Gz = G[:, 0::3], G[:, 1::3], G[:, 2::3]
    Nsens, Nsites = Gx.shape
    GA = np.zeros((Nsens, Nsrc))
    src_inds = np.random.permutation(Nsites)
    for i in range(Nsrc):
        r = np.random.rand(3); r /= np.linalg.norm(r)
        GA[:, i] = Gx[:, src_inds[i]]*r[0] + Gy[:, src_inds[i]]*r[1] + Gz[:, src_inds[i]]*r[2]

    raw_noise = np.random.randn(Nsrc, N + 2*flanker_samples)
    S_full = filtfilt(b, a, raw_noise, axis=1)
    S = (S_full[:, flanker_samples:-flanker_samples] if flanker_samples > 0 else S_full.copy())

    raw_env = np.random.randn(Nsrc, N + 2*flanker_samples)
    env_full = filtfilt(b_lp, a_lp, raw_env, axis=1)
    if flanker_samples > 0:
        env_full = env_full[:, flanker_samples:-flanker_samples]

    env_pos = env_full - env_full.min(axis=1, keepdims=True) + 0.05
    env_unit = env_pos / np.sqrt(np.mean(env_pos**2, axis=1, keepdims=True))

    A = make_microstate_schedule(N, Fs, N_microstates, dwell_time, mode=schedule, smooth_s=smooth_s) 

    z = np.zeros((Nsrc, N))
    for k in range(Nsrc):
        analytic = hilbert(S[k, :])
        S_norm   = S[k, :] / (np.abs(analytic) + 1e-12) 

        if k < Ndistr:
            mu = k // sources_per_microstate
            a_k = A[mu]
            snr_k = snr_inactive + (snr_active - snr_inactive) * a_k
            scale = np.sqrt(np.clip(snr_k, 1e-12, None))
        else:
            scale = 1.0

        S[k, :] = S_norm * env_unit[k] * scale
        z[k, :] = (env_unit[k]**2) * (scale**2 if np.isscalar(scale) else scale**2)

    X_s  = GA[:, :Ndistr]  @ S[:Ndistr, :]
    X_bg = GA[:, Ndistr:]  @ S[Ndistr:, :]

    X_n = np.random.randn(Nsens, N)
    X_n = X_n - X_n.mean(axis=1, keepdims=True)
    X_n = X_n / X_n.std(axis=1, keepdims=True)

    return X_s, X_bg, X_n, z, GA, S, A

# =============================================================================
# 2. СЛОИ НЕЙРОСЕТИ UMAP
# =============================================================================
def build_unvec_matrix(M):
    D = M * (M + 1) // 2
    W = np.zeros((D, M * M), dtype=np.float32)
    idx_i, idx_j = np.triu_indices(M)
    for k, (i, j) in enumerate(zip(idx_i, idx_j)):
        if i == j:
            W[k, i * M + j] = 1.0
        else:
            W[k, i * M + j] = 1.0 / np.sqrt(2.0)
            W[k, j * M + i] = 1.0 / np.sqrt(2.0)
    return W

@tf.keras.utils.register_keras_serializable()
class UnflattenSymmetricLayer(tf.keras.layers.Layer):
    def __init__(self, M, **kwargs):
        super().__init__(**kwargs)
        self.M = int(M)
    def build(self, input_shape):
        w_init = build_unvec_matrix(self.M)
        self.W_UNVEC = self.add_weight(shape=w_init.shape, initializer=tf.keras.initializers.Constant(w_init), trainable=False, name='W_UNVEC')
        super().build(input_shape)
    def call(self, inputs):
        C_flat = tf.matmul(inputs, self.W_UNVEC)
        return tf.reshape(C_flat, [-1, self.M, self.M])
    def get_config(self):
        config = super().get_config()
        config.update({"M": self.M})
        return config

@tf.keras.utils.register_keras_serializable()
class BiMapLayer(tf.keras.layers.Layer):
    def __init__(self, d_out, **kwargs):
        super().__init__(**kwargs)
        self.d_out = int(d_out)
    def build(self, input_shape):
        self.d_in = int(input_shape[-1])
        init_val = np.eye(self.d_out, self.d_in, dtype=np.float32) + np.random.normal(loc=0.0, scale=0.01, size=(self.d_out, self.d_in)).astype(np.float32)
        self.W = self.add_weight(shape=(self.d_out, self.d_in), initializer=tf.keras.initializers.Constant(init_val), trainable=True, name="W_bimap")
        super().build(input_shape)
    def call(self, inputs):
        W_norm = tf.math.l2_normalize(self.W, axis=1)    
        X_out = tf.einsum('ij,bjk,lk->bil', W_norm, inputs, W_norm)
        return 0.5 * (X_out + tf.transpose(X_out, perm=[0, 2, 1]))
    def get_config(self):
        config = super().get_config()
        config.update({"d_out": self.d_out})
        return config

@tf.keras.utils.register_keras_serializable()
class LogDiagScaleLayer(tf.keras.layers.Layer):
    def __init__(self, n_filters, epsilon=1e-9, off_diag_penalty=1.0, **kwargs):
        super().__init__(**kwargs)
        self.n_filters = int(n_filters)
        self.epsilon = epsilon
        self.off_diag_penalty = float(off_diag_penalty)
    def call(self, inputs):
        diags = tf.linalg.diag_part(inputs)
        if self.off_diag_penalty > 0.0 and self.n_filters > 1:
            inv_std = tf.math.rsqrt(tf.maximum(diags, self.epsilon))
            R = inputs * tf.expand_dims(inv_std, -1) * tf.expand_dims(inv_std, -2)
            log_det_R = tf.linalg.logdet(R + self.epsilon * tf.eye(self.n_filters, dtype=inputs.dtype))
            pairs_count = tf.maximum(tf.cast(self.n_filters, inputs.dtype) * (self.n_filters - 1.0) / 2.0, 1.0) 
            self.add_loss(self.off_diag_penalty * tf.reduce_mean(-log_det_R) / pairs_count)
        return tf.math.log(tf.maximum(diags, self.epsilon))
    def get_config(self):
        config = super().get_config()
        config.update({"n_filters": self.n_filters, "epsilon": self.epsilon, "off_diag_penalty": self.off_diag_penalty})
        return config

# =============================================================================
# 3. ОСНОВНОЙ ЭКСПЕРИМЕНТ (ПЕРЕБОР SNR)
# =============================================================================
# Загрузка Leadfield
fwd_fname = 'C:/Users/ansbel/Documents/GitHub/umap_meeg/examples/meeg_examples/simulations/fsaverage-fwd.fif'
info_fname = 'C:/Users/ansbel/Documents/GitHub/umap_meeg/examples/meeg_examples/simulations/fsaverage-info.fif'
fwd = mne.read_forward_solution(fwd_fname, verbose=False)
info = mne.io.read_info(info_fname, verbose=False)
picks = mne.pick_types(info, eeg=True, meg=False)
info_sub = mne.pick_info(info, sel=picks)
G_sub = fwd['sol']['data'][picks, :] 
Fs = info['sfreq']

# Параметры симуляции
Ts = 300  # Снизим время до 5 минут для ускорения эксперимента
Nsrc = 100
N_microstates = 2
sources_per_microstate = 1
Ndistr = N_microstates * sources_per_microstate
Wsize, Ssize = 2.0, 0.5
overlap = Wsize - Ssize
gamma = 0.1 # Уровень сенсорного шума

snr_values = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]  # Сетка SNR для тестирования
results_power_umap = []
results_pattern_umap = []
results_power_ica = []
results_pattern_ica = []

print(f"\nНачинаем эксперимент. Тестируемые значения SNR: {snr_values}\n")

for current_snr in snr_values:
    print(f"\n{'='*60}\n ОЦЕНКА ДЛЯ SNR_ACTIVE = {current_snr}\n{'='*60}")
    
    # --- 3.1 Генерация данных ---
    X_s, X_bg, X_n, z, GA, S, gt_states = generate_microstate_sources(
        G_sub, Nsrc, N_microstates=N_microstates, sources_per_microstate=sources_per_microstate,
        flanker=25.0, Ts=Ts, Fs=Fs, dwell_time=20.0,
        snr_active=current_snr, snr_inactive=0.01, schedule='round_robin', smooth_s=0.05
    )
    X = X_s + X_bg + gamma * X_n / np.linalg.norm(X_s, 'fro')
    X = X / np.std(X)

    # --- 3.2 Эпохирование и Истинные мощности ---
    raw = mne.io.RawArray(X, info_sub, verbose=False)
    epochs = mne.make_fixed_length_epochs(raw, duration=Wsize, overlap=overlap, preload=True, verbose=False)
    epochs_data = epochs.get_data(copy=False) 

    # Истинные мощности таргетов
    info_S = mne.create_info([f"Src_{i}" for i in range(S.shape[0])], Fs, 'misc')
    epochs_S = mne.make_fixed_length_epochs(mne.io.RawArray(S, info_S, verbose=False), duration=Wsize, overlap=overlap, preload=True, verbose=False)
    epochs_S_data = epochs_S.get_data(copy=False)
    
    GA_target = GA[:, :Ndistr]
    ga_scales_sq = np.linalg.norm(GA_target, axis=0)**2
    P_true = np.var(epochs_S_data, axis=2)[:, :Ndistr] * ga_scales_sq

    # --- 3.3 Препроцессинг (Отбеливание) ---
    covmats = Covariances(estimator='cov').fit_transform(epochs_data)
    eigvals, eigvecs = eigh(np.mean(covmats, axis=0))
    idx_sorted = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[idx_sorted], eigvecs[:, idx_sorted]
    valid_mask = eigvals > 1e-6
    eigvals, eigvecs = eigvals[valid_mask], eigvecs[:, valid_mask]
    Ww = eigvecs / np.sqrt(eigvals)[np.newaxis, :]
    n_components = Ww.shape[1]
    covmats_w = Ww.T @ covmats @ Ww

    # --- 3.4 МЕТОД 1: ИТЕРАТИВНЫЙ UMAP ---
    print("-> Запуск алгоритма UMAP (Дефляция)...")
    # ИСПРАВЛЕНИЕ: n_jobs=1 для отключения многопроцессорности и избежания PicklingError
    dists_init = pairwise_distance(covmats_w, metric='riemann')
    n_iterations = Ndistr  # 2 итерации
    Npatt = 1
    W_bimap_list = []
    
    for it in range(n_iterations):
        if it == 0:
            V = np.eye(n_components)
            covmats_reduced = covmats_w.copy()
        else:
            W_stacked = np.vstack(W_bimap_list)
            V = null_space(W_stacked).T
            M_current = V.shape[0]
            covmats_reduced = np.zeros((len(covmats_w), M_current, M_current))
            for i in range(len(covmats_w)):
                covmats_reduced[i] = V @ covmats_w[i] @ V.T

        M_current = V.shape[0]
        idx_i, idx_j = np.triu_indices(M_current)
        multipliers = np.where(idx_i == idx_j, 1.0, np.sqrt(2.0)).astype(np.float32)
        X_cov_flat = (covmats_reduced[:, idx_i, idx_j] * multipliers).astype(np.float32)
        
        tf.keras.backend.clear_session() 
        inputs_enc = tf.keras.Input(shape=(X_cov_flat.shape[1],), dtype=tf.float32)
        x = UnflattenSymmetricLayer(M=M_current)(inputs_enc)
        x = BiMapLayer(d_out=Npatt, name=f"bimap_{it}")(x)
        z_latent = LogDiagScaleLayer(n_filters=Npatt, epsilon=1e-9)(x)
        encoder = tf.keras.Model(inputs=inputs_enc, outputs=z_latent)

        embedder = ParametricUMAP(
            encoder=encoder, n_components=Npatt, dims=(X_cov_flat.shape[1],), 
            metric="precomputed", n_neighbors=20, verbose=False,
            keras_fit_kwargs={"verbose": 0}
        )
        embedder.n_training_epochs = 3 
        embedder.loss_report_frequency = embedder.n_training_epochs * 100

        embedder.fit(X_cov_flat, precomputed_distances=dists_init)
        
        w_reduced = encoder.get_layer(f"bimap_{it}").get_weights()[0]
        w_orig = w_reduced @ V
        w_orig = w_orig / np.linalg.norm(w_orig, axis=1, keepdims=True)
        W_bimap_list.append(w_orig)

    # Оценка UMAP
    W_bimap_stacked = np.vstack(W_bimap_list)
    
    # Получение паттернов UMAP (A = Sigma_X @ W_sensor)
    W_filters_sensor = Ww @ W_bimap_stacked.T
    A_umap = np.mean(covmats, axis=0) @ W_filters_sensor
    A_true = GA[:, :Ndistr]

    P_umap = np.zeros((len(covmats_w), n_iterations))
    for i, C in enumerate(covmats_w):
        P_umap[i] = np.diag(W_bimap_stacked @ C @ W_bimap_stacked.T)
        
    C_corr_umap = np.corrcoef(P_umap.T, P_true.T)
    corr_block_umap = np.abs(C_corr_umap[:n_iterations, n_iterations:])
    row_u, col_u = linear_sum_assignment(-corr_block_umap)
    
    P_true_norm = P_true / (P_true.std(axis=0) + 1e-9)
    P_umap_norm = P_umap / (P_umap.std(axis=0) + 1e-9)
    
    corrs_p_u = [np.corrcoef(P_true_norm[:, col_u[i]], P_umap_norm[:, row_u[i]])[0, 1] for i in range(len(row_u))]
    corrs_a_u = [np.abs(np.corrcoef(A_true[:, col_u[i]], A_umap[:, row_u[i]])[0, 1]) for i in range(len(row_u))]
    
    mean_corr_power_umap = np.mean(corrs_p_u)
    mean_corr_pattern_umap = np.mean(corrs_a_u)
    
    print(f"   Средняя корреляция мощностей UMAP: {mean_corr_power_umap:.3f}")
    print(f"   Средняя корреляция паттернов UMAP: {mean_corr_pattern_umap:.3f}")
    
    results_power_umap.append(mean_corr_power_umap)
    results_pattern_umap.append(mean_corr_pattern_umap)


    # --- 3.5 МЕТОД 2: FAST ICA ---
    print("-> Запуск алгоритма FastICA...")
    from mne.preprocessing import ICA
    ica = ICA(method='fastica', max_iter=500, random_state=42)
    ica.fit(raw, verbose=False)
    
    A_ica = ica.get_components()
    
    ica_sources_raw = ica.get_sources(raw)
    epochs_ica = mne.make_fixed_length_epochs(ica_sources_raw, duration=Wsize, overlap=overlap, preload=True, verbose=False)
    P_ica = np.var(epochs_ica.get_data(copy=False), axis=2)
    
    n_ica = P_ica.shape[1]
    C_corr_ica = np.corrcoef(P_ica.T, P_true.T)
    corr_block_ica = np.abs(C_corr_ica[:n_ica, n_ica:])
    
    row_i, col_i = linear_sum_assignment(-corr_block_ica)
    P_ica_norm = P_ica / (P_ica.std(axis=0) + 1e-9)
    
    corrs_p_i = [np.corrcoef(P_true_norm[:, col_i[j]], P_ica_norm[:, row_i[j]])[0, 1] for j in range(len(row_i))]
    corrs_a_i = [np.abs(np.corrcoef(A_true[:, col_i[j]], A_ica[:, row_i[j]])[0, 1]) for j in range(len(row_i))]
    
    mean_corr_power_ica = np.mean(corrs_p_i)
    mean_corr_pattern_ica = np.mean(corrs_a_i)
    
    print(f"   Средняя корреляция мощностей ICA:  {mean_corr_power_ica:.3f}")
    print(f"   Средняя корреляция паттернов ICA:  {mean_corr_pattern_ica:.3f}")
    
    results_power_ica.append(mean_corr_power_ica)
    results_pattern_ica.append(mean_corr_pattern_ica)

# =============================================================================
# 4. ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ ЭКСПЕРИМЕНТА
# =============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# График 1: Мощности (Огибающие)
ax1.plot(snr_values, results_power_umap, marker='o', linewidth=2.5, color='blue', label='UMAP (Riemannian Deflation)')
ax1.plot(snr_values, results_power_ica, marker='s', linewidth=2.5, color='green', label='FastICA')
ax1.set_title('Восстановление огибающих (Мощности)', fontsize=13, fontweight='bold')
ax1.set_xlabel('Активный SNR (snr_active)', fontsize=12)
ax1.set_ylabel('Средняя корреляция (Pearson r)', fontsize=12)
ax1.set_xscale('log')
ax1.set_ylim([0.0, 1.05])
ax1.set_xticks(snr_values)
ax1.set_xticklabels([str(v) for v in snr_values])
ax1.grid(True, which='both', linestyle='--', alpha=0.6)
ax1.legend(loc='lower right', fontsize=11)

# График 2: Паттерны (Топографии)
ax2.plot(snr_values, results_pattern_umap, marker='o', linewidth=2.5, color='blue', label='UMAP (Riemannian Deflation)')
ax2.plot(snr_values, results_pattern_ica, marker='s', linewidth=2.5, color='green', label='FastICA')
ax2.set_title('Восстановление паттернов (Топографии)', fontsize=13, fontweight='bold')
ax2.set_xlabel('Активный SNR (snr_active)', fontsize=12)
ax2.set_ylabel('Средняя корреляция (Abs Pearson r)', fontsize=12)
ax2.set_xscale('log')
ax2.set_ylim([0.0, 1.05])
ax2.set_xticks(snr_values)
ax2.set_xticklabels([str(v) for v in snr_values])
ax2.grid(True, which='both', linestyle='--', alpha=0.6)
ax2.legend(loc='lower right', fontsize=11)

plt.tight_layout()
plt.show()

print("\nЭксперимент завершен!")