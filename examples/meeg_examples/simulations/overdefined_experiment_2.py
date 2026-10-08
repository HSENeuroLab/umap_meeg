# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 14:32:53 2026

@author: ansbel
"""
import os
import sys

lib_directory = os.path.abspath("C:/Users/ansbel/Documents/GitHub/pyRiemann") 
if lib_directory not in sys.path:
    sys.path.insert(0, lib_directory)
from pyriemann.estimation import Covariances
from pyriemann.geometry.distance import pairwise_distance

sim_dir = os.path.abspath("C:/Users/ansbel/Documents/GitHub/umap_meeg/examples/meeg_examples/simulations/")
if sim_dir not in sys.path:
    sys.path.insert(0, sim_dir)
from scipy.linalg import eigh

import mne

import numpy as np 
import matplotlib.pyplot as plt

import umap
from umap.parametric_umap import ParametricUMAP

import tensorflow as tf

import scipy.signal
from scipy.signal import butter, filtfilt, hilbert

def make_microstate_schedule(N, Fs, n_states, dwell_time,
                             mode='round_robin', smooth_s=0.05):
    """
    Возвращает A формы (n_states, N), A[m,t] ∈ [0,1] — насколько микростат m
    активен в момент t.
      mode='round_robin' — строго по кругу (0,1,2,...,0,1,2,...)
      mode='random'      — случайная последовательность без повторов подряд
    smooth_s — длительность сглаживания фронтов (сек); 0 → ступеньки.
    """
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
        A = np.apply_along_axis(lambda x: np.convolve(x, kernel, mode='same'),
                                axis=1, arr=A)
        # sum по состояниям снова ~1 в каждый момент, но не идеально
        # — можно не нормировать, масштаб задаётся snr_inactive/snr_active
    return A

def generate_microstate_sources(
    G, Nsrc, N_microstates=2, sources_per_microstate=10,
    flanker=25.0, Ts=600, Fs=100,
    dwell_time=5.0,                
    snr_active=2.0,                
    snr_inactive=0.1,              
    schedule='round_robin',        
    smooth_s=0.05,                 
    transition_freq=0.5,           
    source_corr=0,               # <--- НОВЫЙ ПАРАМЕТР: степень корреляции (0.0 - 1.0)
    use_sine_carrier=True        
):
    """
    Генерирует источники, где источники внутри одного микросостояния могут
    быть скоррелированы по фазе и огибающей на заданную величину source_corr.
    """
    N = int(Ts * Fs)
    flanker_samples = int(flanker * Fs)
    Ndistr = N_microstates * sources_per_microstate
    assert Ndistr <= Nsrc, "N_microstates*sources_per_microstate > Nsrc"

    # --- Фильтры ----------------------------------------------------------
    b, a = butter(5, [8/(Fs/2), 12/(Fs/2)], btype='bandpass')
    b_lp, a_lp = butter(5, transition_freq / (Fs/2), btype='lowpass')

    # --- Forward model ---------------------------------------------------
    Gx, Gy, Gz = G[:, 0::3], G[:, 1::3], G[:, 2::3]
    Nsens, Nsites = Gx.shape
    GA = np.zeros((Nsens, Nsrc))
    src_inds = np.random.permutation(Nsites)
    for i in range(Nsrc):
        r = np.random.rand(3); r /= np.linalg.norm(r)
        GA[:, i] = Gx[:, src_inds[i]]*r[0] + Gy[:, src_inds[i]]*r[1] + Gz[:, src_inds[i]]*r[2]

    # --- Несущие (band-pass, гауссовские, zero-mean) ----------------------
    raw_noise = np.random.randn(Nsrc, N + 2*flanker_samples)
    S_full = filtfilt(b, a, raw_noise, axis=1)
    S = (S_full[:, flanker_samples:-flanker_samples]
         if flanker_samples > 0 else S_full.copy())

    # --- Огибающие (low-pass, гауссовские, zero-mean) ---------------------
    raw_env = np.random.randn(Nsrc, N + 2*flanker_samples)
    env_full = filtfilt(b_lp, a_lp, raw_env, axis=1)
    if flanker_samples > 0:
        env_full = env_full[:, flanker_samples:-flanker_samples]

    # =====================================================================
    # --- ВНЕДРЕНИЕ КОРРЕЛЯЦИИ --------------------------------------------
    # =====================================================================
    if sources_per_microstate > 1 and source_corr > 0.0:
        rho = source_corr
        blend_w = np.sqrt(1.0 - rho**2)
        
        for mu in range(N_microstates):
            base_idx = mu * sources_per_microstate
            for j in range(1, sources_per_microstate):
                idx = base_idx + j
                
                # Смешиваем несущие сигналы
                S[idx, :] = rho * S[base_idx, :] + blend_w * S[idx, :]
                
                # Смешиваем огибающие
                env_full[idx, :] = rho * env_full[base_idx, :] + blend_w * env_full[idx, :]

    # =====================================================================

    # --- Нелинейные преобразования огибающей ---
    # Dähne-подобное смещение: делаем строго положительной
    env_pos = env_full - env_full.min(axis=1, keepdims=True) + 0.05
    # Нормируем по RMS, чтобы E[env_unit²] = 1
    env_unit = env_pos / np.sqrt(np.mean(env_pos**2, axis=1, keepdims=True))

    # --- Расписание микросостояний ----------------------------------------
    A = make_microstate_schedule(
        N, Fs, N_microstates, dwell_time,
        mode=schedule, smooth_s=smooth_s
    )

    # --- Синтез источников -------------------------------------------------
    z = np.zeros((Nsrc, N))
    for k in range(Nsrc):
        
        if use_sine_carrier:
            # Оригинальный метод: принудительно делаем идеальный синус (kurtosis = -1.5)
            analytic = hilbert(S[k, :])
            S_norm = S[k, :] / (np.abs(analytic) + 1e-12)
        else:
            # Наш метод: оставляем гауссовский узкополосный шум (kurtosis = 0)
            S_norm = S[k, :] / np.std(S[k, :])

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

    gt_states = A
    return X_s, X_bg, X_n, z, GA, S, gt_states

# =============================================================================
# 1. ЗАГРУЗКА И ВЫБОР ЭЭГ КАНАЛОВ (МОНТАЖ 10-20)
# =============================================================================
fwd_fname = 'C:/Users/ansbel/Documents/GitHub/umap_meeg/examples/meeg_examples/simulations/fsaverage-fwd.fif'
info_fname = 'C:/Users/ansbel/Documents/GitHub/umap_meeg/examples/meeg_examples/simulations/fsaverage-info.fif'

fwd = mne.read_forward_solution(fwd_fname, verbose=False)
info = mne.io.read_info(info_fname, verbose=False)
Fs = info['sfreq']

picks = mne.pick_types(info, eeg=True, meg=False)
info_sub = mne.pick_info(info, sel=picks)
G_sub = fwd['sol']['data'][picks, :]  # Форма: (M_channels, N_vertices * 3)

M_channels = len(picks)

# =============================================================================
# 2. СИМУЛЯЦИЯ СИГНАЛОВ
# =============================================================================
Ts = 600          
Nsrc = 100         
N_microstates = 2               
sources_per_microstate = 2
Ndistr = N_microstates * sources_per_microstate  

flanker = 25.0        
gamma = 0.1          

Wsize = 2
Ssize = 0.5
overlap = Wsize - Ssize

print("Генерация сетевых источников...")

X_s, X_bg, X_n, z, GA, S, gt_states = generate_microstate_sources(
    G_sub, Nsrc,
    N_microstates=N_microstates,
    sources_per_microstate=sources_per_microstate,
    flanker=25.0, Ts=Ts, Fs=Fs,
    dwell_time=20.0,         
    snr_active=1.0,          
    snr_inactive=0.01,        
    schedule='round_robin',
    smooth_s=0.05,
)

gamma = 0.1
X = X_s + X_bg + gamma * X_n
X = X / np.std(X)

# =============================================================================
# 3. УПАКОВКА В MNE И РАСЧЕТ КОВАРИАЦИЙ
# =============================================================================
print("Создание Raw и нарезка на эпохи...")
raw = mne.io.RawArray(X, info_sub)

epochs = mne.make_fixed_length_epochs(
    raw, 
    duration=Wsize, 
    overlap=overlap, 
    preload=True, 
    verbose=False
)
epochs_data = epochs.get_data(copy=False) 

covmats = Covariances(estimator='cov').fit_transform(epochs_data)
mean_cov = np.mean(covmats, axis=0)

# 2. Спектральное разложение
eigvals, eigvecs = eigh(mean_cov)

# 3. Сортировка по убыванию
idx_sorted = np.argsort(eigvals)[::-1]
eigvals = eigvals[idx_sorted]
eigvecs = eigvecs[:, idx_sorted]

# 4. Отсечение сингулярных / зашумленных компонент
valid_mask = eigvals > 1e-6
eigvals = eigvals[valid_mask]
eigvecs = eigvecs[:, valid_mask]

# 5. Вычисление матрицы отбеливания Ww = V * Lambda^(-1/2)
Ww = eigvecs / np.sqrt(eigvals)[np.newaxis, :]
n_components = Ww.shape[1]

# 6. Векторизованное преобразование ковариационных матриц
covmats_w = Ww.T @ covmats @ Ww

# 7. Проверка: средняя ковариация должна быть строго единичной
assert np.allclose(np.mean(covmats_w, axis=0), np.eye(n_components), atol=1e-5), "Отбеливание некорректно!"

# %%
# =============================================================================
# 3.5 ИЗВЛЕЧЕНИЕ ДИСПЕРСИИ ИСТИННЫХ ИСТОЧНИКОВ И ИХ МАСШТАБИРОВАНИЕ
# =============================================================================
print("Нарезка истинных источников S на эпохи и расчет дисперсий...")

# 1. Создаем фейковое info для источников (чтобы MNE мог их "проглотить")
info_S = mne.create_info(
    ch_names=[f"Src_{i+1}" for i in range(S.shape[0])], 
    sfreq=Fs, 
    ch_types='misc'
)

raw_S = mne.io.RawArray(S, info_S)

# 2. Нарезаем на эпохи СТРОГО ТАК ЖЕ, как резали X
epochs_S = mne.make_fixed_length_epochs(
    raw_S, 
    duration=Wsize, 
    overlap=overlap, 
    preload=True, 
    verbose=False
)
epochs_S_data = epochs_S.get_data(copy=False)  # Форма: (n_epochs, Nsrc, n_samples)

# 3. Считаем дисперсию (мощность) сигнала по оси времени (axis=2) внутри каждой эпохи
true_source_variances = np.var(epochs_S_data, axis=2)  # Форма: (n_epochs, Nsrc)

# 4. Нас интересуют только целевые источники (первые Ndistr штук)
z_true_epochs =  true_source_variances[:, :Ndistr]

# --- НОВОЕ: ПЕРЕВОД В АБСОЛЮТНУЮ ФИЗИЧЕСКУЮ МОЩНОСТЬ НА СЕНСОРАХ ---
# Извлекаем паттерны целевых источников из истинной матрицы смешивания
GA_target = GA[:, :Ndistr]

# Вычисляем квадрат L2-нормы для каждого столбца (||A_i||^2)
GA_scales_squared = np.linalg.norm(GA_target, axis=0)**2

# Умножаем дисперсии на квадраты норм топографий
z_true_epochs = z_true_epochs * GA_scales_squared

plt.plot(true_source_variances[:,:sources_per_microstate])
plt.plot(-true_source_variances[:,sources_per_microstate:sources_per_microstate*2])

# %%
from mne.preprocessing import ICA
ica = ICA(method='fastica')

# 3. Обучение ICA на отфильтрованных данных
ica.fit(raw)

# %%
# 4. Визуализация компонент (для ручного поиска артефактов глаз/сердца)
ica.plot_components()  # Карты топографии компонент
# ica.plot_sources(raw)  # Временные ряды компонент

# %%
# =============================================================================
# 2. ИЗВЛЕЧЕНИЕ ЭПОХИРОВАННЫХ ИСТОЧНИКОВ И ИХ МОЩНОСТИ
# =============================================================================
print("Извлечение источников ICA и расчет их дисперсии на эпохах...")
# Извлекаем непрерывные временные ряды найденных компонент
ica_sources_raw = ica.get_sources(raw)

# Нарезаем на эпохи СТРОГО ТАК ЖЕ, как мы резали X и S
epochs_ica = mne.make_fixed_length_epochs(
    ica_sources_raw, 
    duration=Wsize, 
    overlap=overlap, 
    preload=True, 
    verbose=False
)
epochs_ica_data = epochs_ica.get_data(copy=False) # (n_epochs, n_components, n_samples)

# Считаем мощность (дисперсию) каждой компоненты на каждой эпохе
P_ica = np.var(epochs_ica_data, axis=2)  # (n_epochs, n_components)

# Паттерны (топограммы) ICA
# get_components() возвращает матрицу смешивания размера (n_channels, n_components)
A_ica = ica.get_components() 

# =============================================================================
# 3. БЕЗОПАСНОЕ СОГЛАСОВАНИЕ (ICA VS ИСТИННЫЕ ИСТОЧНИКИ)
# =============================================================================
P_true = true_source_variances[:, :Ndistr]  # Только таргетные!
n_ica = P_ica.shape[1]
n_true = P_true.shape[1]

# Используем P_true.T, а не true_source_variances.T
C_corr_ica = np.corrcoef(P_ica.T, P_true.T)
corr_block_ica = np.abs(C_corr_ica[:n_ica, n_ica:])  # Размеры: (n_ica, Ndistr)

# Венгерский алгоритм
row_ica, col_ica = linear_sum_assignment(-corr_block_ica)

# Сортируем пары по индексу истинного источника
sort_idx_ica = np.argsort(col_ica)
matched_ica_idx = row_ica[sort_idx_ica]
matched_true_idx_ica = col_ica[sort_idx_ica]
num_matched_ica = len(matched_true_idx_ica)

P_ica_matched = P_ica[:, matched_ica_idx]
A_ica_matched = A_ica[:, matched_ica_idx]

print(f"Из {Ndistr} истинных источников ICA сопоставлено {num_matched_ica} компонент.")

# =============================================================================
# 4. ТОПОГРАММЫ И КОРРЕЛЯЦИИ
# =============================================================================
fig, axes = plt.subplots(2, num_matched_ica, figsize=(4 * num_matched_ica, 7))
if num_matched_ica == 1: axes = np.expand_dims(axes, axis=1)

A_true = GA[:, :Ndistr]    
for i in range(num_matched_ica):
    t_idx = matched_true_idx_ica[i]
    
    ax_t = axes[0, i]
    mne.viz.plot_topomap(A_true[:, t_idx], info_sub, axes=ax_t, show=False, cmap='RdBu_r')
    ax_t.set_title(f'True Source {t_idx+1}')

    ax_l = axes[1, i]
    mne.viz.plot_topomap(A_ica_matched[:, i], info_sub, axes=ax_l, show=False, cmap='RdBu_r')
    ax_l.set_title(f'ICA Comp {matched_ica_idx[i]}')

plt.tight_layout()
plt.show()

# --- Графики мощностей ---
P_true_matched_ica = P_true[:, matched_true_idx_ica]

P_true_n_ica = P_true_matched_ica / (P_true_matched_ica.std(axis=0, keepdims=True) + 1e-9)
P_pred_n_ica = P_ica_matched / (P_ica_matched.std(axis=0, keepdims=True) + 1e-9)

correlations_ica = [np.corrcoef(P_true_n_ica[:, i], P_pred_n_ica[:, i])[0, 1] for i in range(num_matched_ica)]

fig, axes = plt.subplots(num_matched_ica, 1, figsize=(12, 3 * num_matched_ica), sharex=True)
if num_matched_ica == 1: axes = [axes]

fig.suptitle(f'[ICA Baseline] Мощности: совпало {num_matched_ica} из {Ndistr} источников', y=1.02, fontweight='bold')

for i in range(num_matched_ica):
    ax = axes[i]
    ax.plot(P_true_n_ica[:, i], 'b-', lw=2.5, alpha=0.6, label='Истинная')
    ax.plot(P_pred_n_ica[:, i], 'g', lw=1.5, label='Предсказанная (ICA)')
    ax.set_title(f'True {matched_true_idx_ica[i]+1} vs ICA {matched_ica_idx[i]} (corr = {correlations_ica[i]:.3f})')
    ax.set_ylabel('Норм. мощность')
    ax.legend(loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.5)

plt.xlabel('Окна')
plt.tight_layout()
plt.show()

# %%
dists_init = pairwise_distance(covmats_w, metric='riemann')

# %%
reducer = umap.UMAP(n_components=2, n_neighbors=20, metric='precomputed')
coords = reducer.fit_transform(dists_init)

# %%
plt.figure(figsize=(8, 6))
plt.scatter(coords[:, 0], coords[:, 1], s=5, cmap='Spectral')

plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.colorbar()
plt.show()

# %%
def build_unvec_matrix(M):
    """Создает матрицу для быстрого преобразования вектора обратно в симметричную матрицу"""
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
    """Развёртка плоских векторов обратно в симметричные матрицы."""
    def __init__(self, M, **kwargs):
        super().__init__(**kwargs)
        self.M = int(M)

    def build(self, input_shape):
        w_init = build_unvec_matrix(self.M)
        self.W_UNVEC = self.add_weight(
            shape=w_init.shape,
            initializer=tf.keras.initializers.Constant(w_init),
            trainable=False,  
            name='W_UNVEC'
        )
        super().build(input_shape)

    def call(self, inputs):
        # Дифференцируемое восстановление (batch, D) -> (batch, M, M)
        C_flat = tf.matmul(inputs, self.W_UNVEC)
        return tf.reshape(C_flat, [-1, self.M, self.M])

    def get_config(self):
        config = super().get_config()
        config.update({"M": self.M})
        return config

@tf.keras.utils.register_keras_serializable()
class BiMapLayer(tf.keras.layers.Layer):
    """
    Билинейное преобразование: X_out = W_norm · X · W_norm^T
    """
    def __init__(self, d_out, **kwargs):
        super().__init__(**kwargs)
        self.d_out = int(d_out)

    def build(self, input_shape):
        self.d_in = int(input_shape[-1])
        
        initializer = tf.keras.initializers.Orthogonal()
        init_val = initializer(shape=(self.d_out, self.d_in)).numpy()
        
        self.W = self.add_weight(
            shape=(self.d_out, self.d_in),
            initializer=tf.keras.initializers.Constant(init_val),
            trainable=True,
            name="W_bimap"
        )
        super().build(input_shape)
            
    def call(self, inputs):
        W_norm = tf.math.l2_normalize(self.W, axis=1)    
        X_out = tf.einsum('ij,bjk,lk->bil', W_norm, inputs, W_norm)
        X_out = 0.5 * (X_out + tf.transpose(X_out, perm=[0, 2, 1]))

        return X_out

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_out": self.d_out,
        })
        return config

@tf.keras.utils.register_keras_serializable()
class LogDiagScaleLayer(tf.keras.layers.Layer):
    """
    z_i = log( scale_i * max( (W C W^T)_{ii}, eps ) )
    Штраф за внедиагональные элементы рассчитывается на основе матрицы корреляции.
    Масштаб строго больше нуля благодаря функции softplus, выступает как усилитель дисперсии.
    """
    def __init__(self, n_filters, epsilon=1e-9, off_diag_penalty=1.0, **kwargs):
        super().__init__(**kwargs)
        self.n_filters = int(n_filters)
        self.epsilon = epsilon
        self.off_diag_penalty = float(off_diag_penalty)

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs):
        diags = tf.linalg.diag_part(inputs)
        
        if self.off_diag_penalty > 0.0 and self.n_filters > 1:
            inv_std = tf.math.rsqrt(tf.maximum(diags, self.epsilon))
            inv_std_col = tf.expand_dims(inv_std, axis=-1)
            inv_std_row = tf.expand_dims(inv_std, axis=-2)
            R = inputs * inv_std_col * inv_std_row
            
            eps_eye = self.epsilon * tf.eye(self.n_filters, dtype=inputs.dtype)
            log_det_R = tf.linalg.logdet(R + eps_eye)
            
            N = tf.cast(self.n_filters, dtype=inputs.dtype)
            pairs_count = tf.maximum(N * (N - 1.0) / 2.0, 1.0) 
            
            pham_penalty = self.off_diag_penalty * tf.reduce_mean(-log_det_R) / pairs_count
            self.add_loss(pham_penalty)
                
        return tf.math.log(tf.maximum(diags, self.epsilon))

    def get_config(self):
        config = super().get_config()
        config.update({
            "n_filters": self.n_filters,
            "epsilon": self.epsilon,
            "off_diag_penalty": self.off_diag_penalty
        })
        return config

# %%
Npatt = 1 

idx_i, idx_j = np.triu_indices(n_components)
multipliers = np.where(idx_i == idx_j, 1.0, np.sqrt(2.0)).astype(np.float32)
X_cov_flat = (covmats_w[:, idx_i, idx_j] * multipliers).astype(np.float32)
input_dim = int(X_cov_flat.shape[1])

inputs_enc = tf.keras.Input(shape=(input_dim,), dtype=tf.float32)
x = UnflattenSymmetricLayer(M=n_components)(inputs_enc)
x = BiMapLayer(d_out=Npatt, name="bimap_1")(x)
z_latent = LogDiagScaleLayer(n_filters=Npatt, epsilon=1e-9)(x)
encoder = tf.keras.Model(inputs=inputs_enc, outputs=z_latent)

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=50, min_delta=1e-4)
embedder = ParametricUMAP(
    encoder=encoder, n_components=Npatt, dims=(input_dim,), 
    metric="precomputed", n_neighbors=20, verbose=True,
    keras_fit_kwargs={"callbacks": [early_stopping], "verbose": 1}
)

embedder.n_training_epochs = 3 
embedder.loss_report_frequency = embedder.n_training_epochs * 100
embedder.fit(X_cov_flat, precomputed_distances=dists_init)

# %%
# =============================================================================
# ВИЗУАЛИЗАЦИЯ 
# =============================================================================
import matplotlib.pyplot as plt

history = embedder._history

# Создаем фигуру с двумя подграфиками (для Recon и для UMAP)
fig, ax1 = plt.subplots(1, 1)
# --- График 2: Истинный лосс графа UMAP ---
ax1.plot(history['umap_loss'], label='UMAP Graph Loss (Cross-Entropy)', color='purple', linewidth=2)
    
ax1.set_title('Качество кодера')
ax1.set_xlabel('Шаги оценки')
ax1.set_ylabel('Cross-Entropy')
ax1.legend()
ax1.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

# %%
# Матрица фильтров в пространстве SSD (каждый столбец — отдельный фильтр)
W_bimap = embedder.encoder.get_layer("bimap_1").get_weights()[0]

# Фильтры в исходном пространстве сенсоров: X_latent = W_bimap @ W_ssd.T @ X_sensor
W_filters_sensor = Ww @ W_bimap.T

# =============================================================================
# 2. РАСЧЕТ ИСТИННЫХ МОЩНОСТЕЙ ИСТОЧНИКОВ
# =============================================================================
# Считаем физические мощности напрямую для каждого окна: diag(W C W^T)
powers_true = np.zeros((len(covmats_w), Npatt))
for i, C in enumerate(covmats_w):
    C_filtered = W_bimap @ C @ W_bimap.T
    powers_true[i] = np.diag(C_filtered)

A_sensor_raw =  np.mean(covmats, axis=0) @ W_filters_sensor

# Нормировка физических паттернов для визуализации
scales = np.linalg.norm(A_sensor_raw, axis=0)
A_global_norm = A_sensor_raw / scales

# =============================================================================
# 4. СОРТИРОВКА ПО СРЕДНЕЙ МОЩНОСТИ
# =============================================================================
power_variances = np.mean(powers_true, axis=0)
sort_idx = np.argsort(power_variances)[::-1]

# Сортируем мощности, паттерны и фильтры по убыванию значимости
powers = powers_true[:, sort_idx]
A_sorted = A_global_norm[:, sort_idx]
W_sorted = W_filters_sensor[:, sort_idx]

print(f"Порядок компонент по убыванию средней мощности: {sort_idx}")

# =============================================================================
# 5. СПИСКИ ДЛЯ ОТРИСОВКИ MNE
# =============================================================================
found_filters = [W_sorted[:, i] for i in range(Npatt)]
found_patterns = [A_sorted[:, i] for i in range(Npatt)]

# %%
from scipy.linalg import null_space

n_iterations = sources_per_microstate * N_microstates
Npatt = 1 

W_bimap_list = [] 
history_list = []

# Списки для сохранения результатов каждой итерации
covmats_reduced_list = []  
# dists_list = []            

for it in range(n_iterations):
    print(f"\n{'='*50}\n ИТЕРАЦИЯ {it + 1} ИЗ {n_iterations}\n{'='*50}")
    
    # 1. ПОДГОТОВКА ПОДПРОСТРАНСТВА И РАССТОЯНИЙ
    if it == 0:
        # Пропускаем лишние вычисления: берем уже готовые исходные данные
        V = np.eye(n_components)
        covmats_reduced = covmats_w.copy()
        dists_current = dists_init.copy()
    else:
        # Строим базис нуль-пространства из всех ранее найденных фильтров
        W_stacked = np.vstack(W_bimap_list)
        V = null_space(W_stacked).T
        M_current = V.shape[0]
        
        # Проецируем ковариации в меньшую размерность
        covmats_reduced = np.zeros((len(covmats_w), M_current, M_current))
        for i in range(len(covmats_w)):
            covmats_reduced[i] = V @ covmats_w[i] @ V.T

        # Идеальный перерасчет топологии без регуляризации
        # dists_current = pairwise_distance(covmats_reduced, metric='riemann')
        # dists_current = pairwise_distance(covmats_reduced, metric='riemann')
    
    # Сохраняем спроецированные матрицы и их дистанции в списки
    covmats_reduced_list.append(covmats_reduced.copy())
    # dists_list.append(dists_current.copy())

    M_current = V.shape[0]
    
    # 2. ВЕКТОРИЗАЦИЯ УМЕНЬШЕННЫХ КОВАРИАЦИЙ
    idx_i, idx_j = np.triu_indices(M_current)
    multipliers = np.where(idx_i == idx_j, 1.0, np.sqrt(2.0)).astype(np.float32)
    X_cov_flat = (covmats_reduced[:, idx_i, idx_j] * multipliers).astype(np.float32)
    input_dim = int(X_cov_flat.shape[1])

    # 3. ИНИЦИАЛИЗАЦИЯ И ОБУЧЕНИЕ МОДЕЛИ
    tf.keras.backend.clear_session() # Очищаем память
    
    inputs_enc = tf.keras.Input(shape=(input_dim,), dtype=tf.float32)
    x = UnflattenSymmetricLayer(M=M_current)(inputs_enc)
    layer_name = f"bimap_{it}"
    x = BiMapLayer(d_out=Npatt, name=layer_name)(x)
    z_latent = LogDiagScaleLayer(n_filters=Npatt, epsilon=1e-9)(x)
    encoder = tf.keras.Model(inputs=inputs_enc, outputs=z_latent)

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=50, min_delta=1e-4)
    embedder = ParametricUMAP(
        encoder=encoder, n_components=Npatt, dims=(input_dim,), 
        metric="precomputed", n_neighbors=20, verbose=True,
        keras_fit_kwargs={"callbacks": [early_stopping], "verbose": 1}
    )

    embedder.n_training_epochs = 3
    embedder.loss_report_frequency = embedder.n_training_epochs * 100

    embedder.fit(X_cov_flat, precomputed_distances=dists_current)
    history_list.append(embedder._history)
    
    # 4. ИЗВЛЕЧЕНИЕ ФИЛЬТРА И ВОЗВРАТ В ИСХОДНОЕ ПРОСТРАНСТВО
    # w_reduced имеет размер (1, M_current)
    w_reduced = encoder.get_layer(layer_name).get_weights()[0]
    
    # Проецируем обратно в исходное отбеленное пространство: (1, M_current) @ (M_current, M) -> (1, M)
    w_orig = w_reduced @ V
    
    # Нормализуем
    w_orig = w_orig / np.linalg.norm(w_orig, axis=1, keepdims=True)
    W_bimap_list.append(w_orig)

# %%
# =============================================================================
# ПОСТПРОЦЕССИНГ
# =============================================================================
W_bimap_stacked = np.vstack(W_bimap_list)
W_filters_sensor = Ww @ W_bimap_stacked.T

# Дальнейший код для мощностей, сортировки и отрисовки

# 2. РАСЧЕТ ИСТИННЫХ МОЩНОСТЕЙ ИСТОЧНИКОВ
powers = np.zeros((len(covmats_w), n_iterations * Npatt))
for i, C in enumerate(covmats_w):
    # Применяем матрицу ко всем ИСХОДНЫМ отбеленным ковариациям
    C_filtered = W_bimap_stacked @ C @ W_bimap_stacked.T
    powers[i] = np.diag(C_filtered)

# Расчет паттернов: A = Sigma_mean * W
A_sensor_raw = np.mean(covmats, axis=0) @ W_filters_sensor

# Нормировка физических паттернов для визуализации
scales = np.linalg.norm(A_sensor_raw, axis=0)
A_global_norm = A_sensor_raw / scales

found_filters = [W_filters_sensor[:, i] for i in range(n_iterations * Npatt)]
found_patterns = [A_global_norm[:, i] for i in range(n_iterations * Npatt)]

# %%
import matplotlib.pyplot as plt
import numpy as np

# Вычисляем матрицу ковариации один раз, чтобы использовать для графиков и текста
cov_matrix = np.cov(found_filters @ raw.get_data()) 

fig, ax = plt.subplots(figsize=(6, 6)) # Увеличим размер,  чтобы цифры поместились
im = ax.imshow(cov_matrix, cmap='viridis')

# Добавляем colorbar
plt.colorbar(im)

# Добавляем числа в каждую ячейку

rows, cols = cov_matrix.shape
for i in range(rows):
    for j in range(cols):
        value = cov_matrix[i, j]
        
        # Форматирование: '.2f' для обычных чисел (0.12), '.2e' если порядок очень мал (1.23e-05)
        text_label = f"{value:.2f}" 
        
        # Меняем цвет текста в зависимости от яркости ячейки, чтобы его было видноS
        # (выбираем белый для темных ячеек, черный для светлых)
        text_color = "white" if im.norm(value) < 0.5 else "black"
        
        ax.text(j, i, text_label,
                ha="center", va="center", 
                color=text_color, fontsize=9)

plt.title("Матрица ковариации с числовыми значениями")
plt.show()

# %%
from scipy.optimize import linear_sum_assignment

# =============================================================================
# 6. БЕЗОПАСНОЕ СОГЛАСОВАНИЕ (КОГДА Npatt != Ndistr)
# =============================================================================
# Убедитесь, что вы используете P_learned = powers.copy() (так как выше вы назвали переменную powers)
P_learned = powers.copy()  # (n_epochs, 12)
A_true = GA[:, :Ndistr]    # (Nsens, 6)
ga_scales_sq = np.linalg.norm(A_true, axis=0) ** 2            
P_true = true_source_variances[:, :Ndistr] * ga_scales_sq    # (n_epochs, 6)

n_learned = P_learned.shape[1] # В вашем случае 12
n_true = P_true.shape[1]       # В вашем случае 6

# Матрица корреляций (18x18)
C_corr = np.corrcoef(P_learned.T, P_true.T)                  

# ИЗМЕНЕНО: Правильно вырезаем блок корреляций между Learned и True (размер 12x6)
corr_block = np.abs(C_corr[:n_learned, n_learned:])                  

# Венгерский алгоритм
row, col = linear_sum_assignment(-corr_block)

# Сортируем пары по индексу истинного источника для красоты вывода
sort_idx = np.argsort(col)
matched_learned_idx = row[sort_idx]
matched_true_idx = col[sort_idx]
num_matched = len(matched_true_idx)

P_learned_matched = P_learned[:, matched_learned_idx]
A_learned_matched = A_global_norm[:, matched_learned_idx]
W_learned_matched = W_filters_sensor[:, matched_learned_idx]

print(f"Из {Ndistr} истинных источников алгоритм успешно сопоставил {num_matched} фильтров.")

# =============================================================================
# 7. ТОПОГРАММЫ (Только для совпавших пар)
# =============================================================================
fig, axes = plt.subplots(2, num_matched, figsize=(4 * num_matched, 7))
if num_matched == 1: axes = np.expand_dims(axes, axis=1) 

# ИЗМЕНЕНО: Заменили жесткое range(6) на range(num_matched)
for i in range(num_matched):
    t_idx = matched_true_idx[i]
    
    ax_t = axes[0, i]
    mne.viz.plot_topomap(A_true[:, t_idx], info_sub, axes=ax_t, show=False, cmap='RdBu_r')
    ax_t.set_title(f'True Source {t_idx+1}')

    ax_l = axes[1, i]
    mne.viz.plot_topomap(A_learned_matched[:, i], info_sub, axes=ax_l, show=False, cmap='RdBu_r')
    ax_l.set_title(f'Learned {matched_learned_idx[i]+1}')

plt.tight_layout()
plt.show()

# =============================================================================
# 8. КОРРЕЛЯЦИИ (Только для совпавших пар)
# =============================================================================
P_true_matched = P_true[:, matched_true_idx]
P_true_n = P_true_matched / (P_true_matched.std(axis=0, keepdims=True) + 1e-9)
P_pred_n = P_learned_matched / (P_learned_matched.std(axis=0, keepdims=True) + 1e-9)

correlations = [np.corrcoef(P_true_n[:, i], P_pred_n[:, i])[0, 1] for i in range(num_matched)]

fig, axes = plt.subplots(num_matched, 1, figsize=(12, 3 * num_matched), sharex=True)
if num_matched == 1: axes = [axes]

fig.suptitle(f'Мощности: совпало {num_matched} из {Ndistr} источников', y=1.02)

# ИЗМЕНЕНО: Заменили жесткое range(6) на range(num_matched)
for i in range(num_matched):
    ax = axes[i]
    ax.plot(P_true_n[:, i], 'b-', lw=2.5, alpha=0.6, label='Истинная')
    ax.plot(P_pred_n[:, i], 'r', lw=1.5, label='Предсказанная')
    ax.set_title(f'True {matched_true_idx[i]+1} vs Learned {matched_learned_idx[i]+1} (corr = {correlations[i]:.3f})')
    ax.set_ylabel('Норм. мощность')
    ax.legend(loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.5)

plt.xlabel('Окна')
plt.tight_layout()
plt.show()

# %%

