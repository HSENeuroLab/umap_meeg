# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 21:11:08 2026

@author: ansbel
"""

import numpy as np
from scipy.signal import butter, filtfilt, hilbert

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

import mne

import numpy as np 
import matplotlib.pyplot as plt

import umap
from umap.parametric_umap import ParametricUMAP

import tensorflow as tf

# %%
# -*- coding: utf-8 -*-
import numpy as np
from scipy.signal import butter, filtfilt, hilbert
import os
import sys
import mne

# =============================================================================
# ИСПРАВЛЕННАЯ ФУНКЦИЯ ГЕНЕРАЦИИ
# =============================================================================
def generate_distributed_sources(G, Nsrc, Ndistr, flanker, Ts, Fs):
    N = int(Ts * Fs)
    flanker_samples = int(flanker * Fs)

    b, a = butter(5, [8 / (Fs / 2), 12 / (Fs / 2)], btype='bandpass')
    b_lp, a_lp = butter(5, 0.5 / (Fs / 2), btype='lowpass')

    Gx = G[:, 0::3]
    Gy = G[:, 1::3]
    Gz = G[:, 2::3]
    Nsens, Nsites = Gx.shape

    GA = np.zeros((Nsens, Nsrc))
    orientations = np.zeros((3, Nsrc)) # ДОБАВЛЕНО: сохраняем ориентации
    src_indsA = np.random.permutation(Nsites)
    
    for i in range(Nsrc):
        src_idx = src_indsA[i]
        r = np.random.rand(3)
        r = r / np.linalg.norm(r)
        orientations[:, i] = r # Сохраняем
        GA[:, i] = Gx[:, src_idx]*r[0] + Gy[:, src_idx]*r[1] + Gz[:, src_idx]*r[2]

    raw_noise = np.random.randn(Nsrc, N + 2 * flanker_samples)
    S_full = filtfilt(b, a, raw_noise, axis=1)
    S = S_full[:, flanker_samples : -flanker_samples] if flanker_samples > 0 else S_full.copy()

    z = np.zeros((Nsrc, N))

    for k in range(Nsrc):
        analytic_signal = hilbert(S[k, :])
        carrier_env = np.abs(analytic_signal)
        
        # ДОБАВЛЕНО: Защита от деления на ноль (1e-9)
        S_norm = S[k, :] / (carrier_env + 1e-9)

        noise_mod = np.random.randn(N + 2 * flanker_samples)
        lp_noise_full = filtfilt(b_lp, a_lp, noise_mod)
        lp_noise = lp_noise_full[flanker_samples : -flanker_samples] if flanker_samples > 0 else lp_noise_full.copy()

        lp_noise = lp_noise / np.std(lp_noise)
        amp_mod = lp_noise - np.min(lp_noise) + 0.05
        S[k, :] = S_norm * amp_mod

        sigma_s = np.std(S[k, :])
        S[k, :] = S[k, :] / sigma_s
        z[k, :] = (amp_mod / sigma_s)**2

    X_s = GA[:, :Ndistr] @ S[:Ndistr, :]
    X_bg = GA[:, Ndistr:] @ S[Ndistr:, :]

    X_n = np.random.randn(Nsens, N)
    X_n = X_n - np.mean(X_n, axis=1, keepdims=True)
    X_n = X_n / np.std(X_n, axis=1, keepdims=True)

    # ДОБАВЛЕНО: возвращаем позиции и ориентации
    return X_s, X_bg, X_n, z, GA, S, src_indsA[:Nsrc], orientations

# %%
# =============================================================================
# 1. ЗАГРУЗКА И ВЫБОР ПОДМНОЖЕСТВА КАНАЛОВ (РАЗРЕЖЕННЫЙ МОНТАЖ)
# =============================================================================
fwd_fname = 'C:/Users/ansbel/Documents/GitHub/umap_meeg/examples/meeg_examples/simulations/fsaverage-fwd.fif'
info_fname = 'C:/Users/ansbel/Documents/GitHub/umap_meeg/examples/meeg_examples/simulations/fsaverage-info.fif'

fwd = mne.read_forward_solution(fwd_fname, verbose=False)
info = mne.io.read_info(info_fname, verbose=False)
Fs = info['sfreq']

# Урезанный список (14 базовых каналов, равномерно покрывающих скальп)
sparse_montage = info.ch_names

valid_channels = [ch for ch in sparse_montage if ch in info.ch_names]

if len(valid_channels) < 5:
    eeg_picks = mne.pick_types(info, eeg=True, meg=False)
    step = max(1, len(eeg_picks) // 14)
    picks = eeg_picks[::step][:14]
else:
    picks = mne.pick_channels(info.ch_names, include=valid_channels)

info_sub = mne.pick_info(info, sel=picks)
G_sub = fwd['sol']['data'][picks, :]
print(f"Выбрано каналов: {len(picks)}")

# =============================================================================
# 2. СИМУЛЯЦИЯ СИГНАЛОВ
# =============================================================================
Ts_train = 250.0  # Время для обучения
Ts_test = 600.0   # Время для теста
Ts = Ts_train + Ts_test  # Итого: 850.0 секунд

Nsrc = 100           
Ndistr = 1          # Целевых источников
flanker = 1.0        
gamma = 0.1          
current_snr = 10 ** 0.4

Wsize = 1.0
Ssize = 1.0  # Шаг окна (Step size)
overlap = Wsize - Ssize  # 0.0

print(f"Генерация источников (всего {Ts} сек)...")
X_s, X_bg, X_n, z, GA, S, positions_idx, orientations = generate_distributed_sources(
    G_sub, Nsrc, Ndistr, flanker, Ts, Fs
)

# Масштаб шума
signal_std = np.std(X_s)
X = current_snr * X_s + X_bg + gamma * X_n * signal_std
X = X / np.std(X)

# =============================================================================
# 3. УПАКОВКА В MNE И РАСЧЕТ КОЭФФИЦИЕНТОВ ПО ОКНАМ
# =============================================================================
from pyriemann.estimation import Covariances

print("Создание Raw и нарезка на эпохи...")
raw_eeg = mne.io.RawArray(X, info_sub)
info_z = mne.create_info(Ndistr, Fs, ch_types='misc')
raw_z = mne.io.RawArray(z[:Ndistr, :], info_z)

# Нарезаем ЭЭГ
epochs_eeg = mne.make_fixed_length_epochs(raw_eeg, duration=Wsize, overlap=overlap, preload=True, verbose=False)
epochs_data = epochs_eeg.get_data(copy=False) 

# Считаем ковариации
print("Вычисление ковариационных матриц...")
covmats = Covariances(estimator='cov').fit_transform(epochs_data)

# Нарезаем z (мощность)
epochs_z = mne.make_fixed_length_epochs(raw_z, duration=Wsize, overlap=overlap, preload=True, verbose=False)
z_epochs = epochs_z.get_data(copy=False).mean(axis=2) 

print(f"Готово! Размерность covmats (Входы): {covmats.shape}")
print(f"Размерность z_epochs (Таргеты): {z_epochs.shape}")

# %%
# =============================================================================
# 4. ХРОНОЛОГИЧЕСКОЕ РАЗБИЕНИЕ И ЧЕСТНОЕ ОТБЕЛИВАНИЕ (С РЕГУЛЯРИЗАЦИЕЙ)
# =============================================================================
from pyriemann.utils.base import invsqrtm
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Считаем, сколько эпох попадает в обучающую выборку
Ts_train = 250.0
Ssize = 1.0
n_train_epochs = int(Ts_train / Ssize)

# ТАРГЕТЫ: Берем сырую мощность
y_targets = z_epochs.copy()

# 1. Хронологическое разделение матриц ДО отбеливания
X_train_raw = covmats[:n_train_epochs]
X_test_raw = covmats[n_train_epochs:]

y_train = y_targets[:n_train_epochs] - np.mean(y_targets[:n_train_epochs])
y_test = y_targets[n_train_epochs:] - np.mean(y_targets[n_train_epochs:])

print(f"Train raw shape: {X_train_raw.shape}")
print(f"Test raw shape: {X_test_raw.shape}")

# 2. Обучаем отбеливание ТОЛЬКО на тренировочной выборке
print("Вычисляем среднее по обучающей выборке...")
C_avg_train = np.mean(X_train_raw, axis=0)

# ---> РЕГУЛЯРИЗАЦИЯ (Diagonal Loading) <---
alpha = 1e-9  
M = C_avg_train.shape[0]
trace_avg = np.trace(C_avg_train) / M
C_avg_train_reg = C_avg_train + alpha * trace_avg * np.eye(M)

# Корень извлекается безопасно
C_avg_invsqrt = invsqrtm(C_avg_train_reg)

# 3. Применяем фильтр отбеливания ко всем данным
print("Отбеливаем Train и Test...")
X_train = np.array([C_avg_invsqrt @ c @ C_avg_invsqrt for c in X_train_raw])
X_test = np.array([C_avg_invsqrt @ c @ C_avg_invsqrt for c in X_test_raw])

print("Данные готовы к подаче в нейросеть!")


# =============================================================================
# 5. СБОРКА И ОБУЧЕНИЕ ВАШЕЙ CUSTOM SPD-NET
# =============================================================================
class LearnableRotationLayer(tf.keras.layers.Layer):
    """
    Обучаемое ортогональное преобразование.
    """
    def __init__(self, reg_weight=0.001, **kwargs):
        super().__init__(**kwargs)
        self.reg_weight = reg_weight

    def build(self, input_shape):
        M = int(input_shape[-1])
        self.W = self.add_weight(
            shape=(M, M),
            initializer=tf.keras.initializers.Identity(),
            trainable=True,
            name="W_rotation"
        )
        self.eye = tf.constant(np.eye(M, dtype=np.float32))

    def call(self, C):
        C_w = tf.einsum('im,bmn,jn->bij', self.W, C, self.W)
        W_WT = tf.matmul(self.W, self.W, transpose_b=True)
        ortho_loss = tf.reduce_mean(tf.square(W_WT - self.eye))
        self.add_loss(self.reg_weight * ortho_loss)
        return C_w

class LogDiagLayer(tf.keras.layers.Layer):
    """
    ВАША ИДЕЯ: Извлечение диагонали (мощностей компонент) и их логарифмирование.
    Это эквивалентно логарифму матрицы для диагональных матриц, но работает мгновенно.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, C_w):
        # 1. Извлекаем диагональные элементы (batch_size, M)
        diag_elements = tf.linalg.diag_part(C_w)
        # 2. Безопасный логарифм
        log_diag = tf.math.log(tf.maximum(diag_elements, 1e-9))
        return log_diag

def pearson_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    y_true_centered = y_true - tf.reduce_mean(y_true, axis=0, keepdims=True)
    y_pred_centered = y_pred - tf.reduce_mean(y_pred, axis=0, keepdims=True)
    cov = tf.reduce_sum(y_true_centered * y_pred_centered, axis=0)
    std_true = tf.sqrt(tf.reduce_sum(tf.square(y_true_centered), axis=0))
    std_pred = tf.sqrt(tf.reduce_sum(tf.square(y_pred_centered), axis=0))
    corr = cov / tf.maximum(std_true * std_pred, 1e-7)
    return 1.0 - tf.reduce_mean(tf.abs(corr))

# --- АРХИТЕКТУРА СЕТИ ---
M_channels = X_train.shape[1]
inputs = tf.keras.Input(shape=(M_channels, M_channels))

# 1. Вращение W^T * C * W
rotated_C = LearnableRotationLayer(reg_weight=0.001, name="rotation")(inputs)

# 2. Log(Diag(C))
log_diags = LogDiagLayer(name="log_diag")(rotated_C)

# 3. Линейная комбинация логарифмов дисперсий!
# Dense слой сам выучит, какие компоненты важны для предсказания таргета
linear_comb = tf.keras.layers.Dense(1, activation='linear', use_bias=True, name="linear_weights")(log_diags)

# 4. Возврат в линейный масштаб (экспонента), так как таргет у нас без логарифма
outputs = tf.keras.layers.Activation(tf.math.exp, name="exp_activation")(linear_comb)

model = tf.keras.Model(inputs=inputs, outputs=outputs)

# --- ОБУЧЕНИЕ ---
cosine_decay = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=0.02,
    decay_steps=300,
    alpha=0.001
)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=cosine_decay), loss=pearson_loss)

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=250, 
    batch_size=64,
    verbose=1,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=50, restore_best_weights=True, monitor='val_loss')
    ]
)

# =============================================================================
# 6. ВИЗУАЛИЗАЦИЯ И ИЗВЛЕЧЕНИЕ ПАТТЕРНА
# =============================================================================
y_pred = model.predict(X_test)
r, _ = pearsonr(y_test[:, 0], y_pred[:, 0])
print(f"\nАбсолютная корреляция Пирсона на тесте: |R| = {np.abs(r):.4f}\n")

# --- ИЗВЛЕЧЕНИЕ ПРОСТРАНСТВЕННОГО ПАТТЕРНА ---
W_rotation = model.get_layer("rotation").get_weights()[0]  # (M, M)
w_dense = model.get_layer("linear_weights").get_weights()[0] # (M, 1)

# Ищем компоненту, которой сеть (Dense слой) дала самый большой вес по модулю
best_comp_idx = np.argmax(np.abs(w_dense[:, 0]))

# Извлекаем соответствующий ей пространственный фильтр из матрицы вращения
w_best_filter = W_rotation[:, best_comp_idx]

# Восстанавливаем паттерн
A_est = C_avg_train_reg @ C_avg_invsqrt @ w_best_filter
A_true = GA[:, 0]

# Нормализация для графиков
A_est_norm = A_est / np.max(np.abs(A_est))
A_true_norm = A_true / np.max(np.abs(A_true))
if np.corrcoef(A_true_norm, A_est_norm)[0, 1] < 0:
    A_est_norm = -A_est_norm

# --- ГРАФИКИ ТОПОМАП ---
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
mne.viz.plot_topomap(A_true_norm, info_sub, axes=axes[0], show=False, cmap='RdBu_r', vlim=(-1, 1))
axes[0].set_title("Истинный паттерн")

mne.viz.plot_topomap(A_est_norm, info_sub, axes=axes[1], show=False, cmap='RdBu_r', vlim=(-1, 1))
axes[1].set_title(f"Найденный паттерн (Log-Diag Net)\n|R| = {np.abs(r):.2f}")

plt.suptitle("Сравнение пространственных паттернов", fontsize=14)
plt.tight_layout()
plt.show()

# --- ВРЕМЕННОЙ РЯД ---
plt.figure(figsize=(12, 4))
y_test_z = (y_test[:, 0] - np.mean(y_test[:, 0])) / np.std(y_test[:, 0])
y_pred_z = (y_pred[:, 0] - np.mean(y_pred[:, 0])) / np.std(y_pred[:, 0])

if r < 0:
    y_pred_z = -y_pred_z

plt.plot(y_test_z[:200], label='Истинная мощность (z-score)', alpha=0.8, linewidth=2)
plt.plot(y_pred_z[:200], label='Предсказание Log-Diag сети', alpha=0.8, linestyle='--')
plt.xlabel('Окна (Epochs)')
plt.legend()
plt.show()

# %%
# --- ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ ---
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

y_pred = model.predict(X_test)
r, p_val = pearsonr(y_test[:, 0], y_pred[:, 0])
print(f"\nАбсолютная корреляция Пирсона на тесте: |R| = {np.abs(r):.4f}\n")

# --- РАСЧЕТ ПРОСТРАНСТВЕННОГО ПАТТЕРНА ---
W_rotation = model.get_layer("rotation").get_weights()[0]  # (M, M)
w_spoc = model.get_layer("spoc_filter").get_weights()[0]   # (1, M)

# Эффективный фильтр с учетом ротации
w_eff = W_rotation.T @ w_spoc[0]

# Восстанавливаем паттерн в исходном пространстве
A_est = C_avg_train_reg @ C_avg_invsqrt @ w_eff
A_true = GA[:, 0]

# Нормализация
A_est_norm = A_est / np.max(np.abs(A_est))
A_true_norm = A_true / np.max(np.abs(A_true))

if np.corrcoef(A_true_norm, A_est_norm)[0, 1] < 0:
    A_est_norm = -A_est_norm

# --- ГРАФИКИ ---
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

mne.viz.plot_topomap(A_true_norm, info_sub, axes=axes[0], show=False, cmap='RdBu_r', vlim=(-1, 1))
axes[0].set_title("Заданный паттерн (Истинный)")

mne.viz.plot_topomap(A_est_norm, info_sub, axes=axes[1], show=False, cmap='RdBu_r', vlim=(-1, 1))
axes[1].set_title(f"Найденный паттерн SPOC\n(|R| = {np.abs(r):.2f})")

plt.suptitle("Сравнение пространственных паттернов", fontsize=14)
plt.tight_layout()
plt.show()

# Отрисовка временного ряда
plt.figure(figsize=(12, 4))
y_test_z = (y_test[:, 0] - np.mean(y_test[:, 0])) / np.std(y_test[:, 0])
y_pred_z = (y_pred[:, 0] - np.mean(y_pred[:, 0])) / np.std(y_pred[:, 0])

if r < 0:
    y_pred_z = -y_pred_z

plt.plot(y_test_z[:200], label='Истинная мощность (z-score)', alpha=0.8, linewidth=2)
plt.plot(y_pred_z[:200], label='Предсказание SPOC-сети', alpha=0.8, linestyle='--')
plt.xlabel('Окна (Epochs)')
plt.legend()
plt.show()