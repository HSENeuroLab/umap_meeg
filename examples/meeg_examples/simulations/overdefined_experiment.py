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
from signal_simulation import generate_bursty_sources

import mne

import numpy as np 
import matplotlib.pyplot as plt

import umap
from umap.parametric_umap import ParametricUMAP

import tensorflow as tf

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
sparse_montage = [
    'Fp1', 'Fp2', 
    'F3', 'Fz', 'F4', 
    'C3', 'Cz', 'C4', 
    'P3', 'Pz', 'P4', 
    'O1', 'Oz', 'O2'
]

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
Ts = 600.0          
Nsrc = 100           
Ndistr = 3          # 25 целевых источников > 14 каналов (система переопределена)
flanker = 1.0        
gamma = 0.1          
current_snr = 10 ** 0.6

Wsize = 2.0
Ssize = 0.1
overlap = Wsize - Ssize

print("Генерация источников...")
X_s, X_bg, X_n, z, GA, S, positions_idx, orientations = generate_bursty_sources(
    G_sub, Nsrc, Ndistr, flanker, Ts, Fs, 
    block_duration=5.0, 
    burst_prob=0.15 
)

X = current_snr * X_s + X_bg + gamma * X_n / np.linalg.norm(X_s, 'fro')
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

print("Вычисление ковариационных матриц...")
covmats = Covariances(estimator='cov').fit_transform(epochs_data)

print(f"Готово! Размерность covmats: {covmats.shape}")

# %%
dist_matrix_init = pairwise_distance(Covariances(estimator='oas').fit_transform(epochs_data), metric='riemann')

# %%
covmats = Covariances(estimator='oas').fit_transform(epochs_data)

import numpy as np

n_ch_white = covmats.shape[1]

# =============================================================================
# ПОДГОТОВКА ДАННЫХ (УНИКАЛЬНЫЕ ЭЛЕМЕНТЫ)
# =============================================================================
M_channels = covmats.shape[1]

# Индексы верхней треугольной матрицы
idx_i, idx_j = np.triu_indices(M_channels)

# Множители: 1.0 для диагонали, sqrt(2) для внедиагональных
multipliers = np.where(idx_i == idx_j, 1.0, np.sqrt(2.0)).astype(np.float32)

# Вытаскиваем уникальные элементы и сразу масштабируем
X_cov_flat = (covmats[:, idx_i, idx_j] * multipliers).astype(np.float32)

input_dim = int(X_cov_flat.shape[1]) # Теперь размерность M*(M+1)/2
print(f"Новая размерность входа: {input_dim}")

class SpatialPatternDecoder(tf.keras.layers.Layer):
    def __init__(self, M_channels, N_patterns, **kwargs):
        super().__init__(**kwargs)
        self.M_channels = int(M_channels)
        self.N_patterns = int(N_patterns)
        
    def build(self, input_shape):
        self.A = self.add_weight(
            shape=(self.M_channels, self.N_patterns),
            initializer=tf.keras.initializers.Orthogonal(gain=1.0),
            trainable=True,
            name="A_patterns"
        )
        
        # 1. Диагональный аппаратный шум
        self.noise_log = self.add_weight(
            shape=(self.M_channels,),
            initializer=tf.keras.initializers.Constant(-4.0),
            trainable=True,
            name="sensor_noise"
        )
                
        i, j = np.triu_indices(self.M_channels)
        self.flat_indices = tf.constant(i * self.M_channels + j, dtype=tf.int32)
        self.multipliers = tf.constant(np.where(i == j, 1.0, np.sqrt(2.0)), dtype=tf.float32)

    def call(self, z):
        A_norm = tf.math.l2_normalize(self.A, axis=0)
        
        # 
        P = tf.math.softplus(z)
        
        self.add_loss(1e-3 * tf.reduce_mean(P))

        C_signal = tf.einsum("mf,bf,lf->bml", A_norm, P, A_norm)
        
        # ИЗМЕНЕНИЕ 2: Формируем полную модель шума
        noise_diag = tf.linalg.diag(tf.math.softplus(self.noise_log))
        
        C_recon = C_signal + noise_diag
        
        C_recon_flat = tf.reshape(C_recon, [-1, self.M_channels * self.M_channels])
        vecs = tf.gather(C_recon_flat, self.flat_indices, axis=1)
        
        return vecs * self.multipliers
    
def build_unvec_matrix(M):
    """Создает матрицу для быстрого преобразования вектора обратно в симметричную матрицу"""
    D = M * (M + 1) // 2
    W = np.zeros((D, M * M), dtype=np.float32)
    idx_i, idx_j = np.triu_indices(M)
    for k, (i, j) in enumerate(zip(idx_i, idx_j)):
        if i == j:
            W[k, i * M + j] = 1.0
        else:
            # Делим на sqrt(2), чтобы снять примененный ранее масштаб
            W[k, i * M + j] = 1.0 / np.sqrt(2.0)
            W[k, j * M + i] = 1.0 / np.sqrt(2.0)
    return W

# Инициализируем константу один раз
W_UNVEC = tf.constant(build_unvec_matrix(n_ch_white), dtype=tf.float32)

def reconstruct_sym_matrix(vecs, M):
    """Дифференцируемое восстановление (batch, D) -> (batch, M, M)"""
    C_flat = tf.matmul(vecs, W_UNVEC)
    return tf.reshape(C_flat, [-1, M, M])

def riemannian_distance_loss(y_true_flat, y_pred_flat):
    y_true_flat = tf.cast(y_true_flat, tf.float32)
    y_pred_flat = tf.cast(y_pred_flat, tf.float32)

    M = n_ch_white  
    
    C_true = reconstruct_sym_matrix(y_true_flat, M)
    C_pred = reconstruct_sym_matrix(y_pred_flat, M)
    
    C_true = 0.5 * (C_true + tf.transpose(C_true, [0, 2, 1]))
    C_pred = 0.5 * (C_pred + tf.transpose(C_pred, [0, 2, 1]))

    # 1. Отсекаем построение градиентного графа для истинных данных
    C_true = tf.stop_gradient(C_true)
        
    # 2. Безопасный порог для float32 (машинный эпсилон ~1.19e-7)
    eps = 1e-4

    # Шаг 1: C_true^{-1/2}
    eigvals_true, eigvecs_true = tf.linalg.eigh(C_true)
    inv_sqrt_eigvals = 1.0 / tf.sqrt(tf.maximum(eigvals_true, eps))
    C_true_invsqrt = tf.einsum('bij,bj,bkj->bik', eigvecs_true, inv_sqrt_eigvals, eigvecs_true)

    # Шаг 2: C_true^{-1/2} * C_pred * C_true^{-1/2}
    C_mid = tf.einsum('bij,bjk,bkl->bil', C_true_invsqrt, C_pred, C_true_invsqrt)
    C_mid = 0.5 * (C_mid + tf.transpose(C_mid, perm=[0, 2, 1])) 

    # Шаг 3: Собственные значения полученной матрицы
    eigvals_mid, _ = tf.linalg.eigh(C_mid) 
    log_eigvals_mid = tf.math.log(tf.maximum(eigvals_mid, eps))

    # Шаг 4: Считаем дистанцию напрямую по собственным значениям    
    # dist_sq = tf.reduce_sum(tf.square(log_eigvals_mid), axis=1) / tf.cast(M * M, tf.float32)
    dist_sq = tf.sqrt(tf.reduce_sum(tf.square(log_eigvals_mid), axis=1))

    return tf.reduce_mean(dist_sq)

# =============================================================================
# ПОДГОТОВКА ДАННЫХ
# =============================================================================
N_patterns = int(3)
N_dim = int(N_patterns)

print(f"Обучение ParametricUMAP: N_dim={N_dim}, N_patterns={N_patterns}")

# =============================================================================
# ЭНКОДЕР И ДЕКОДЕР
# =============================================================================

encoder = tf.keras.Sequential([
    tf.keras.Input(shape=(int(input_dim),), dtype=tf.float32),
    
    # Блок 1 (LayerNormalization + ELU)
    tf.keras.layers.Dense(512, use_bias=False),
    tf.keras.layers.LayerNormalization(epsilon=1e-6),
    tf.keras.layers.Activation("elu"),
    tf.keras.layers.Dropout(0.2),
    
    # Блок 2
    tf.keras.layers.Dense(512, use_bias=False),
    tf.keras.layers.LayerNormalization(epsilon=1e-6),
    tf.keras.layers.Activation("elu"),
    tf.keras.layers.Dropout(0.2),
    
    # Блок 3
    tf.keras.layers.Dense(512, use_bias=False),
    tf.keras.layers.LayerNormalization(epsilon=1e-6),
    tf.keras.layers.Activation("elu"),
    tf.keras.layers.Dropout(0.1),
    
    # Латентное пространство (z_powers)
    # Здесь оставляем linear, так как экспонента берется внутри SpatialPatternDecoder
    tf.keras.layers.Dense(
        int(N_dim), 
        activation="linear", 
        use_bias=True, 
        name="z_powers"
    )
])

decoder = tf.keras.Sequential([
    tf.keras.Input(shape=(int(N_dim),), dtype=tf.float32),    
    # Слой выравнивания латентного пространства
    tf.keras.layers.Dense(int(N_dim), activation="linear", use_bias=False, name="latent_alignment"),
    SpatialPatternDecoder(
        M_channels=int(n_ch_white),
        N_patterns=int(N_patterns),
        name="spatial_decoder"
    )
])

from sklearn.model_selection import train_test_split

# =============================================================================
# РАЗДЕЛЕНИЕ НА ОБУЧАЮЩУЮ И ВАЛИДАЦИОННУЮ ВЫБОРКИ
# =============================================================================
# Индексы для сплита (откладываем 10% данных на валидацию)
indices = np.arange(len(X_cov_flat))
train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)

X_train = X_cov_flat[train_idx]
X_val = X_cov_flat[val_idx]

# Важно: для графа UMAP нужна матрица расстояний ТОЛЬКО между обучающими окнами
dist_matrix_train = dist_matrix_init[train_idx][:, train_idx]

print(f"Обучающая выборка: {len(X_train)} окон. Валидационная: {len(X_val)} окон.")

# =============================================================================
# ПОДГОТОВКА ДАННЫХ И КОЛЛБЕКОВ
# =============================================================================
n_neighbors = 20 
loss_weight = 1.0

# =============================================================================
# ИНИЦИАЛИЗАЦИЯ И ЗАПУСК PARAMETRIC UMAP
# =============================================================================
embedder = ParametricUMAP(
    encoder=encoder,
    decoder=decoder,
    n_components=N_dim, 
    dims=(input_dim,),
    metric="precomputed", 
    n_neighbors=n_neighbors,
    parametric_reconstruction=True,
    autoencoder_loss=True, 
    parametric_reconstruction_loss_fcn=riemannian_distance_loss, 
    parametric_reconstruction_loss_weight=loss_weight,
    reconstruction_validation=X_val, 
    # keras_fit_kwargs=keras_fit_args, 
    verbose=True
)

embedder.loss_report_frequency = 200
embedder.n_training_epochs = 3      

embedder.fit(X_train, precomputed_distances=dist_matrix_train)

covmats = Covariances(estimator='cov').fit_transform(epochs_data)

# %%
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# =============================================================================
# 1. ИЗВЛЕЧЕНИЕ ПАРАМЕТРОВ СЕТИ И СОРТИРОВКА
# =============================================================================
z_raw = embedder.encoder.predict(X_cov_flat)

# Прогоняем их через слои декодера ДО SpatialPatternDecoder, 
# чтобы учесть выравнивание (latent_alignment) и получить корректные z для мощностей
alignment_layer = embedder.decoder.get_layer("latent_alignment")
z_aligned = alignment_layer(z_raw).numpy() 

# Переводим в реальные мощности
# powers = np.exp(z_aligned) 
powers = tf.math.softplus(z_aligned).numpy()

# Извлекаем матрицу паттернов и нормируем 
spatial_decoder_layer = embedder.decoder.get_layer("spatial_decoder")
A_learned_raw = spatial_decoder_layer.get_weights()[0]
A_global = A_learned_raw / np.linalg.norm(A_learned_raw, axis=0) 

# Считаем дисперсию (разброс) каждого из N_patterns вдоль всех эпох (axis=0)
power_variances = np.var(powers, axis=0)

# Получаем индексы сортировки по убыванию [::-1]
sort_idx = np.argsort(power_variances)[::-1]

# Сортируем массив мощностей и столбцы матрицы прямой модели A_global
powers = powers[:, sort_idx]
A_global = A_global[:, sort_idx]

print(f"Топ-5 компонент по дисперсии мощности (оригинальные индексы): {sort_idx[:5]}")

# =============================================================================
# 2. ПОДГОТОВКА ПАТТЕРНОВ И ФИЛЬТРОВ ХАУФЕ
# =============================================================================
# ТЕПЕРЬ A_global - ЭТО ОТСОРТИРОВАННЫЕ ПАТТЕРНЫ В ПРОСТРАНСТВЕ СЕНСОРОВ
found_patterns = [A_global[:, i] for i in range(N_patterns)]

# Фильтры Хауфе (с Тихоновской регуляризацией)
C_global_mean = np.mean(covmats, axis=0)

# Коэффициент регуляризации 
alpha = 1e-4  

# Добавляем единичную матрицу, масштабированную на след C_global_mean
I = np.eye(C_global_mean.shape[0])
C_global_reg = C_global_mean + alpha * np.trace(C_global_mean) * I

# Обращаем регуляризованную матрицу
C_global_inv = np.linalg.inv(C_global_reg)

# Фильтры автоматически отсортированы
found_filters = [C_global_inv @ A_global[:, i] for i in range(N_patterns)]

# %%
# =============================================================================
# 3. ЖАДНЫЙ МЕТЧИНГ И ВИЗУАЛИЗАЦИЯ
# =============================================================================
def greedy_matching(similarity_matrix):
    """
    Жадный алгоритм соответствия. 
    Находит глобальный максимум, фиксирует пару, обнуляет строку и столбец, и повторяет.
    """
    sim = similarity_matrix.copy()
    n_learned, n_true = sim.shape
    matches = {}
    
    for _ in range(min(n_learned, n_true)):
        # Находим индексы максимального элемента в оставшейся матрице
        best_learned, best_true = np.unravel_index(np.argmax(sim), sim.shape)
        
        # Если максимум уже меньше 0 (все распределено), прерываем
        if sim[best_learned, best_true] < 0:
            break
            
        matches[best_learned] = best_true
        
        # Исключаем эту строку и столбец из дальнейшего поиска
        sim[best_learned, :] = -1
        sim[:, best_true] = -1
        
    return matches

# --- 1. ВЫЧИСЛЕНИЕ МАТРИЦЫ СХОДСТВА ДЛЯ ВСЕХ ПАТТЕРНОВ ---
# Нормируем все выученные и все истинные паттерны
A_learned_all = np.array(found_patterns) # Форма: (N_patterns, M_channels)
A_true_all = GA[:, :Ndistr]              # Форма: (M_channels, Ndistr)

A_learned_norm = A_learned_all / np.linalg.norm(A_learned_all, axis=1, keepdims=True)
A_true_norm = A_true_all / np.linalg.norm(A_true_all, axis=0, keepdims=True)

# Матрица абсолютного косинусного сходства (N_patterns x Ndistr)
spatial_sim_matrix = np.abs(A_learned_norm @ A_true_norm)

# Применяем жадный алгоритм
match_dict = greedy_matching(spatial_sim_matrix)

# --- 2. ВЫБОР КОМПОНЕНТЫ ДЛЯ ВИЗУАЛИЗАЦИИ ---
# Выбираем индекс выученного паттерна (0 — самая сильная компонента)
comp_idx = 0              

if comp_idx not in match_dict:
    raise ValueError(f"Компонента {comp_idx} не получила пары в процессе метчинга.")

best_match_idx = match_dict[comp_idx]
best_sim = spatial_sim_matrix[comp_idx, best_match_idx]

A_pattern = found_patterns[comp_idx]
A_true_matched = A_true_all[:, best_match_idx]

print(f"Жадный алгоритм: Выученная компонента {comp_idx+1} совпала с Истинным источником №{best_match_idx+1}")
print(f"Косинусное сходство паттернов: {best_sim:.4f}")

# --- 3. РАСЧЕТ ПРОСТРАНСТВЕННОГО ФИЛЬТРА ОТ ИСТИННОГО ПАТТЕРНА ---
W_true = C_global_inv @ A_true_matched

# --- 4. ИЗВЛЕЧЕНИЕ 3 ТИПОВ МОЩНОСТЕЙ ---
p_vals = powers[:, comp_idx]
filtered_powers_true = np.array([W_true.T @ C @ W_true for C in covmats])

# Усредняем истинный сигнал z внутри окон
epoch_samples = int(Wsize * Fs)
z_true_full = z[best_match_idx, :]
true_powers = np.array([
    np.mean(z_true_full[event[0] : event[0] + epoch_samples]) 
    for event in epochs.events
])

# Нормализация
p_vals_norm = p_vals / np.std(p_vals)
filtered_powers_true_norm = filtered_powers_true / np.std(filtered_powers_true)
true_powers_norm = true_powers / np.std(true_powers)

# --- 5. РАСЧЕТ КОРРЕЛЯЦИЙ ---
corr_nn = np.corrcoef(true_powers_norm, p_vals_norm)[0, 1]
corr_filt = np.corrcoef(true_powers_norm, filtered_powers_true_norm)[0, 1]

print(f"Корреляция (Истинная vs Нейросеть): {corr_nn:.4f}")
print(f"Корреляция (Истинная vs Фильтр Хауфе): {corr_filt:.4f}")

# =============================================================================
# ОТРИСОВКА
# =============================================================================
fig = plt.figure(figsize=(14, 8))
gs = GridSpec(2, 2, figure=fig, width_ratios=[1, 3])

# --- Истинный паттерн (совпавший) ---
ax_true = fig.add_subplot(gs[0, 0])
mne.viz.plot_topomap(A_true_matched, raw.info, axes=ax_true, show=False)
ax_true.set_title(f'Истинный паттерн (Ист. {best_match_idx+1})\nСходство: {best_sim:.2f}')

# --- Выученный сетью паттерн ---
ax_patt = fig.add_subplot(gs[1, 0])
mne.viz.plot_topomap(A_pattern, raw.info, axes=ax_patt, show=False)
ax_patt.set_title(f'Выученный паттерн (Комп. {comp_idx+1})')

# --- Динамика 3 мощностей ---
ax_env = fig.add_subplot(gs[:, 1])
window_idx = np.arange(len(p_vals_norm))

ax_env.plot(window_idx, true_powers_norm, color='red', lw=4, alpha=0.3, 
            label='Истинная симулированная мощность (z)')

ax_env.plot(window_idx, filtered_powers_true_norm, color='gray', lw=2, alpha=0.8,
            label=f'Фильтр Хауфе (от ИСТИННОГО паттерна) | r = {corr_filt:.3f}')

ax_env.plot(window_idx, p_vals_norm, color='black', lw=1.2, 
            label=f'Декодированная мощность (Нейросеть) | r = {corr_nn:.3f}')

ax_env.set_xlabel('Номер окна (эпохи)')
ax_env.set_ylabel('Мощность (scaled by std)')
ax_env.set_title('Сравнение динамики мощностей')
ax_env.grid(True, linestyle=':', alpha=0.6, zorder=0)
ax_env.legend(loc='upper right', fontsize=10)

plt.suptitle(f'Сравнение выученной компоненты {comp_idx+1} с истинным источником {best_match_idx+1}', fontsize=16)
plt.tight_layout()
plt.show()

# %%
from matplotlib.gridspec import GridSpec
import numpy as np

# =============================================================================
# 3. МЕТЧИНГ С ИСТИННЫМИ ИСТОЧНИКАМИ И ВИЗУАЛИЗАЦИЯ
# =============================================================================
print("Toha, the best coder in the world")  # AI slope  

# Выбираем индекс выученного паттерна (0 — самая сильная компонента после сортировки)
comp_idx = 0
A_pattern = found_patterns[comp_idx]

# --- МАКСИМАЛЬНЫЙ МЕТЧИНГ ---
# Истинные паттерны целевых источников (из симуляции)
A_true_all = GA[:, :Ndistr] 

# Считаем абсолютное косинусное сходство между выученным паттерном и всеми истинными
sims = np.abs(A_pattern @ A_true_all) / (np.linalg.norm(A_pattern) * np.linalg.norm(A_true_all, axis=0))

# Находим индекс истинного источника с максимальным сходством
best_match_idx = np.argmax(sims)
best_sim = sims[best_match_idx]
A_true_matched = A_true_all[:, best_match_idx]

print(f"Компонента {comp_idx+1} наилучшим образом совпала с истинным источником №{best_match_idx+1}")
print(f"Косинусное сходство паттернов: {best_sim:.4f}")

# --- РАСЧЕТ ПРОСТРАНСТВЕННОГО ФИЛЬТРА ОТ ИСТИННОГО ПАТТЕРНА ---
# Берем глобальную инвертированную ковариацию C_global_inv из предыдущего шага
W_true = C_global_inv @ A_true_matched

# --- ИЗВЛЕЧЕНИЕ 3 ТИПОВ МОЩНОСТЕЙ ---
# 1. Декодированная нейронкой мощность
p_vals = powers[:, comp_idx]

# 2. Мощность от фильтра, построенного на основе ИСТИННОГО паттерна
filtered_powers_true = np.array([W_true.T @ C @ W_true for C in covmats])

# 3. Истинная симулированная мощность (z) целевого источника
# Нам нужно усреднить сигнал z внутри тех же временных окон, что и эпохи ковариаций
epoch_samples = int(Wsize * Fs)
z_true_full = z[best_match_idx, :]

true_powers = np.array([
    np.mean(z_true_full[event[0] : event[0] + epoch_samples]) 
    for event in epochs.events
])

# Нормализация (делим на std, без вычета среднего, согласно оригиналу)
p_vals_norm = p_vals / np.std(p_vals)
filtered_powers_true_norm = filtered_powers_true / np.std(filtered_powers_true)
true_powers_norm = true_powers / np.std(true_powers)

# --- РАСЧЕТ КОРРЕЛЯЦИЙ ---
corr_nn = np.corrcoef(true_powers_norm, p_vals_norm)[0, 1]
corr_filt = np.corrcoef(true_powers_norm, filtered_powers_true_norm)[0, 1]

print(f"Корреляция (Истинная vs Нейросеть): {corr_nn:.4f}")
print(f"Корреляция (Истинная vs Фильтр Хауфе): {corr_filt:.4f}")

# =============================================================================
# ОТРИСОВКА
# =============================================================================
fig = plt.figure(figsize=(14, 8))
gs = GridSpec(2, 2, figure=fig, width_ratios=[1, 3])

# --- Истинный паттерн (совпавший) ---
ax_true = fig.add_subplot(gs[0, 0])
mne.viz.plot_topomap(A_true_matched, raw.info, axes=ax_true, show=False)
ax_true.set_title(f'Истинный паттерн (Ист. {best_match_idx+1})\nСходство: {best_sim:.2f}')

# --- Выученный сетью паттерн ---
ax_patt = fig.add_subplot(gs[1, 0])
mne.viz.plot_topomap(A_pattern, raw.info, axes=ax_patt, show=False)
ax_patt.set_title(f'Выученный паттерн (Комп. {comp_idx+1})')

# --- Динамика 3 мощностей ---
ax_env = fig.add_subplot(gs[:, 1])
window_idx = np.arange(len(p_vals_norm))

# Отрисовка с разной толщиной и прозрачностью для наглядности наложений
ax_env.plot(window_idx, true_powers_norm, color='red', lw=4, alpha=0.3, 
            label='Истинная симулированная мощность (z)')

# Добавляем коэффициенты корреляции прямо в подписи легенды
ax_env.plot(window_idx, filtered_powers_true_norm, color='gray', lw=2, alpha=0.8,
            label=f'Фильтр Хауфе (от ИСТИННОГО паттерна) | r = {corr_filt:.3f}')

ax_env.plot(window_idx, p_vals_norm, color='black', lw=1.2, 
            label=f'Декодированная мощность (Нейросеть) | r = {corr_nn:.3f}')

ax_env.set_xlabel('Номер окна (эпохи)')
ax_env.set_ylabel('Мощность (scaled by std)')
ax_env.set_title('Сравнение динамики мощностей')
ax_env.grid(True, linestyle=':', alpha=0.6, zorder=0)
ax_env.legend(loc='upper right', fontsize=10)

plt.suptitle(f'Сравнение выученной компоненты {comp_idx+1} с истинным источником {best_match_idx+1}', fontsize=16)
plt.tight_layout()
plt.show()