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
from signal_simulation import generate_bursty_sources, generate_distributed_sources, generate_complex_sources, generate_sources, generate_switching_microstates, generate_microstates
from scipy.linalg import eigh

import mne

import numpy as np 
import matplotlib.pyplot as plt

import umap
from umap.parametric_umap import ParametricUMAP

import tensorflow as tf
from sklearn.model_selection import train_test_split

# =============================================================================
# 1. ЗАГРУЗКА И ВЫБОР ЭЭГ КАНАЛОВ (МОНТАЖ 10-20)
# =============================================================================
fwd_fname = 'C:/Users/ansbel/Documents/GitHub/umap_meeg/examples/meeg_examples/simulations/fsaverage-fwd.fif'
info_fname = 'C:/Users/ansbel/Documents/GitHub/umap_meeg/examples/meeg_examples/simulations/fsaverage-info.fif'

fwd = mne.read_forward_solution(fwd_fname, verbose=False)
info = mne.io.read_info(info_fname, verbose=False)
Fs = info['sfreq']

# 25 каналов из вашей сетки, образующих симметричный монтаж 10-20:
montage_25 = [
    # Лобно-полюсные и сагиттальные вспомогательные
    'Fp1', 'Fpz', 'Fp2',
    # Лобные
    'F7', 'F3', 'Fz', 'F4', 'F8',
    # Лобно-центральные
    'FCz',
    # Височные и центральные
    'T7', 'C3', 'Cz', 'C4', 'T8',
    # Центрально-теменные
    'CPz',
    # Теменные
    'P7', 'P3', 'Pz', 'P4', 'P8',
    # Теменно-затылочные
    'POz',
    # Затылочные
    'O1', 'Oz', 'O2',
    # Нижний затылочный ориентир
    'Iz'
]

# Выбираем только те ЭЭГ каналы, которые есть в нашем списке
# Функция сама проигнорирует каналы из списка, которых нет в info
picks = mne.pick_types(info, eeg=True, meg=False, selection=montage_25)
info_sub = mne.pick_info(info, sel=picks)
G_sub = fwd['sol']['data'][picks, :]  # Форма: (M_channels, N_vertices * 3)

M_channels = len(picks)

# =============================================================================
# 2. СИМУЛЯЦИЯ СИГНАЛОВ
# =============================================================================
Ts = 600          
Nsrc = 100         
Ndistr = 3          
flanker = 1.0        
gamma = 0.1          
current_snr = 10 ** 0

Wsize = 2
Ssize = 0.5
overlap = Wsize - Ssize

print("Генерация источников...")
# X_s, X_bg, X_n, z, GA, S, gt_positions_idx, gt_orientations = generate_bursty_sources(
#     G_sub, Nsrc, Ndistr, flanker, Ts, Fs, 
#     block_duration=5.0, 
#     burst_prob=0.3 
# )
# X_s, X_bg, X_n, z, GA, S = generate_distributed_sources(
#     G_sub, Nsrc, Ndistr, flanker, Ts, Fs
# )
# X_s, X_bg, X_n, z, GA, S = generate_complex_sources(
#     G_sub, Nsrc, Ndistr, flanker, Ts, Fs, 
# )

# print("Генерация переключающихся микросостояний...")
# X_s, X_bg, X_n, z, GA, S, gt_states_samples = generate_switching_microstates(
#     G_sub, Nsrc, Ndistr=3, flanker=flanker, Ts=Ts, Fs=Fs, 
#     min_dur=0.8, max_dur=2.5, snr_target=4.0
# )
X_s, X_bg, X_n, z, GA, S, activity = generate_microstates(
    G_sub, Nsrc, Ndistr=1,
    flanker=flanker, Ts=Ts, Fs=Fs,
    min_seg_dur=5, max_seg_dur=10.0,
    transition=0.1,
    # rng=np.random.default_rng(42),
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
from mne.preprocessing import ICA
ica = ICA(method='fastica')

# 3. Обучение ICA на отфильтрованных данных
ica.fit(raw)

# 4. Визуализация компонент (для ручного поиска артефактов глаз/сердца)
ica.plot_components()  # Карты топографии компонент
# ica.plot_sources(raw)  # Временные ряды компонент

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

SS = S.copy()
SS[:Ndistr] *= 1
raw_S = mne.io.RawArray(SS, info_S)

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

plt.plot(true_source_variances[:,0])

# %%
dists_init = pairwise_distance(covmats_w, metric='riemann')

# %%
import matplotlib.pyplot as plt
import umap

reducer = umap.UMAP(n_components=2, n_neighbors=25, metric='precomputed')
coords = reducer.fit_transform(dists_init)

# %%
plt.figure(figsize=(8, 6))
plt.scatter(coords[:, 0], coords[:, 1], s=5, cmap='Spectral')

plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.colorbar() 
plt.show()

# %%
def init_identity_with_noise(shape, dtype=None):
    M, N = shape
    init = np.zeros((M, N), dtype=np.float32)
    # Первые M столбцов — единичные
    for i in range(min(M, N)):
        init[i, i] = 1.0
    # Остальные — случайные малые, чтобы после нормализации не были нулевыми
    if N > M:
        init[:, M:] = np.random.normal(0, 0.1, (M, N - M)).astype(np.float32)
    return tf.constant(init, dtype=dtype)
n_ch_white = covmats_w.shape[1]

# =============================================================================
# ПОДГОТОВКА ДАННЫХ (УНИКАЛЬНЫЕ ЭЛЕМЕНТЫ)
# =============================================================================
M_channels = covmats_w.shape[1]

# Индексы верхней треугольной матрицы
idx_i, idx_j = np.triu_indices(M_channels)

# Множители: 1.0 для диагонали, sqrt(2) для внедиагональных
multipliers = np.where(idx_i == idx_j, 1.0, np.sqrt(2.0)).astype(np.float32)

# Вытаскиваем уникальные элементы и сразу масштабируем
X_cov_flat = (covmats_w[:, idx_i, idx_j] * multipliers).astype(np.float32)

input_dim = int(X_cov_flat.shape[1]) # Теперь размерность M*(M+1)/2
print(f"Новая размерность входа: {input_dim}")

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
@tf.keras.utils.register_keras_serializable()
class BiMapLayer(tf.keras.layers.Layer):
    """
    Билинейное преобразование: X_out = W_norm · X · W_norm^T

    W_norm — строчно-нормированная версия обучаемой W:
        W_norm[i] = W[i] / ||W[i]||

    Это гарантирует ||w_i|| = 1 для всех фильтров, а штраф
    ||W_norm W_norm^T − I||_F^2 тогда отвечает именно за
    ортогональность направлений (углы между фильтрами).
    """
    def __init__(self, d_out, orth_weight=0.01, **kwargs):
        super().__init__(**kwargs)
        self.d_out = int(d_out)
        self.orth_weight = float(orth_weight)

    def build(self, input_shape):
        self.d_in = int(input_shape[-1])
        # Инициализация: первые d_out строк — единичные по соответствующим каналам
        init_val = np.eye(self.d_out, self.d_in, dtype=np.float32)
        self.W = self.add_weight(
            shape=(self.d_out, self.d_in),
            initializer=tf.keras.initializers.Constant(init_val),
            trainable=True,
            name="W_bimap"
        )
        super().build(input_shape)

    def call(self, inputs):
        # --- 1. Нормировка строк: ‖w_i‖ = 1 ---
        W_norm = tf.math.l2_normalize(self.W, axis=1)    # (d_out, d_in)

        # --- 2. Билинейное преобразование ---
        X_out = tf.einsum('ij,bjk,lk->bil', W_norm, inputs, W_norm)
        X_out = 0.5 * (X_out + tf.transpose(X_out, perm=[0, 2, 1]))

        # --- 3. Штраф на ортогональность направлений ---
        if self.orth_weight > 0.0:
            WWT = tf.matmul(W_norm, W_norm, transpose_b=True)  # (d_out, d_out)
            I = tf.eye(self.d_out)
            orth_penalty = tf.reduce_sum(tf.square(WWT - I))
            self.add_loss(self.orth_weight * orth_penalty)

        return X_out


@tf.keras.utils.register_keras_serializable()
class LogDiagScaleLayer(tf.keras.layers.Layer):
    """
    z_i = softplus(s_i) · log( max( (W C W^T)_{ii}, eps ) ) + b_i

    s_i, b_i — обучаемые параметры:
      * softplus(s_i) > 0 — масштаб динамического диапазона по оси i;
      * b_i — аддитивный сдвиг (компенсирует разницу в ||g_i||^2 между источниками).
    """
    def __init__(self, n_filters, epsilon=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.n_filters = int(n_filters)
        self.epsilon = epsilon

    def build(self, input_shape):
        # s_raw: после softplus(0) ≈ 0.69 — мягкий нейтральный старт
        self.s_raw = self.add_weight(
            shape=(self.n_filters,),
            initializer='zeros',
            trainable=True,
            name='log_scale_raw'
        )
        self.bias = self.add_weight(
            shape=(self.n_filters,),
            initializer='zeros',
            trainable=True,
            name='log_bias'
        )
        super().build(input_shape)

    def call(self, inputs):
        diags = tf.linalg.diag_part(inputs)                       # (batch, n_filters)
        z = tf.math.log(tf.maximum(diags, self.epsilon))          # log-power

        scale = tf.nn.softplus(self.s_raw)                         # > 0
        return scale * z + self.bias

@tf.keras.utils.register_keras_serializable()
class UnflattenSymmetricLayer(tf.keras.layers.Layer):
    """Развёртка плоских векторов обратно в симметричные матрицы."""
    def __init__(self, M, **kwargs):
        super().__init__(**kwargs)
        self.M = int(M)

    def call(self, inputs):
        return reconstruct_sym_matrix(inputs, self.M)
    
# =============================================================================
# 3. СБОРКА АРХИТЕКТУРЫ ЭНКОДЕРА (Диагонализирующий фильтр)
# =============================================================================

M_channels = covmats_w.shape[1]
N_patterns = 3

inputs_enc = tf.keras.Input(shape=(input_dim,), dtype=tf.float32, name="encoder_input")

x = UnflattenSymmetricLayer(M=M_channels, name="unflatten_to_sym")(inputs_enc)
x = BiMapLayer(d_out=N_patterns, orth_weight=0.01, name="bimap_1")(x)
z_latent = LogDiagScaleLayer(n_filters=N_patterns, epsilon=1e-4, name="log_diag_scale")(x)

encoder = tf.keras.Model(inputs=inputs_enc, outputs=z_latent, name="spdnet_diag_encoder")

# =============================================================================
# 5. ПОДГОТОВКА ДАННЫХ
# =============================================================================
M_channels = covmats_w.shape[1]
idx_i, idx_j = np.triu_indices(M_channels)
multipliers = np.where(idx_i == idx_j, 1.0, np.sqrt(2.0)).astype(np.float32)

X_cov_flat = (covmats_w[:, idx_i, idx_j] * multipliers).astype(np.float32)
input_dim = int(X_cov_flat.shape[1])
print(f"Обучение ParametricUMAP: N_dim={N_patterns}, N_patterns={N_patterns}")

# =============================================================================
# 7. ПОДГОТОВКА ПОЛНОЙ ВЫБОРКИ (БЕЗ РАЗДЕЛЕНИЯ)
# =============================================================================

# Используем все доступные данные без сплита
X_train = X_cov_flat
dist_matrix_train = dists_init

print(f"Размер выборки для обучения: {len(X_train)} окон.")

# =============================================================================
# 8. КОЛЛБЕКИ И ЗАПУСК
# =============================================================================

n_neighbors = 20
N_dim = N_patterns

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='loss',        
    patience=30,           
    min_delta=1e-5,        
    restore_best_weights=True
)

embedder = ParametricUMAP(
    encoder=encoder,
    n_components=N_dim,
    dims=(input_dim,),
    metric="precomputed",
    n_neighbors=n_neighbors,
    verbose=True,
    keras_fit_kwargs={"callbacks": [early_stopping]} 
)

embedder.loss_report_frequency = 100
embedder.n_training_epochs = 5

embedder.fit(X_train, precomputed_distances=dist_matrix_train)

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
import numpy as np

# =============================================================================
# 1. ИЗВЛЕЧЕНИЕ ВЕСОВ ФИЛЬТРОВ ИЗ МОДЕЛИ
# =============================================================================
# Получаем веса обученного слоя BiMap. Размерность: (N_patterns, n_components_ssd)
W_bimap = embedder.encoder.get_layer("bimap_1").get_weights()[0]

# Матрица фильтров в пространстве SSD (каждый столбец — отдельный фильтр)
W_filters_ssd = W_bimap.T 

# Фильтры в исходном пространстве сенсоров: X_latent = W_bimap @ W_ssd.T @ X_sensor
W_filters_sensor = Ww @ W_filters_ssd

# =============================================================================
# 2. РАСЧЕТ ИСТИННЫХ МОЩНОСТЕЙ ИСТОЧНИКОВ
# =============================================================================
# Считаем физические мощности напрямую для каждого окна: diag(W C W^T)
powers_true = np.zeros((len(covmats_w), N_patterns))
for i, C in enumerate(covmats_w):
    C_filtered = W_bimap @ C @ W_bimap.T
    powers_true[i] = np.diag(C_filtered)

# =============================================================================
# 3. ОЦЕНКА ПАТТЕРНОВ ЧЕРЕЗ C @ W (В СЕНСОРНОМ ПРОСТРАНСТВЕ)
# =============================================================================
# Усредненная ковариационная матрица в пространстве SSD
C_avg_ssd = np.mean(covmats_w, axis=0)

# Паттерны в пространстве SSD
A_ssd = C_avg_ssd @ W_filters_ssd

# Проекция паттернов обратно в исходное пространство сенсоров
A_sensor_raw = np.linalg.pinv(Ww.T) @ A_ssd

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
found_filters = [W_sorted[:, i] for i in range(N_patterns)]
found_patterns = [A_sorted[:, i] for i in range(N_patterns)]

# %%
from scipy.optimize import linear_sum_assignment

# =============================================================================
# 6. СОГЛАСОВАНИЕ ВЫУЧЕННЫХ ПАТТЕРНОВ С ИСТИННЫМИ ИСТОЧНИКАМИ
# =============================================================================

# Истинные топографии целевых источников и их масштабы
A_true = GA[:, :Ndistr]                              # (Nsens, Ndistr)
ga_scales_sq = np.linalg.norm(A_true, axis=0) ** 2   # ||g_i||^2

# Истинная линейная мощность источника в сенсорном пространстве (по эпохам)
# z_true_epochs уже считается в блоке 3.5, но убедимся, что это именно линейная мощность
P_true = true_source_variances[:, :Ndistr] * ga_scales_sq   # (n_epochs, Ndistr)

# Выученная линейная мощность фильтров в whitened-пространстве
P_learned = powers_true.copy()                       # (n_epochs, N_patterns)

# --- Согласование: какой learned-фильтр соответствует какому истинному источнику ---
# Считаем корреляционную матрицу между временными рядами мощностей
C_corr = np.corrcoef(P_learned.T, P_true.T)          # (Np+Nd, Np+Nd)
corr_block = np.abs(C_corr[:N_patterns, N_patterns:])  # (N_patterns, Ndistr)

# Оптимальное назначение (максимизируем сумму |corr|)
row, col = linear_sum_assignment(-corr_block)
# row[k] -> индекс learned, col[k] -> индекс истинного источника

# Применяем перестановку: learned-массивы выстраиваются под истинные источники
learned_order = row[np.argsort(col)]                # learned i <- источник i
P_learned_matched = P_learned[:, learned_order]
A_learned_matched = A_global_norm[:, learned_order]
W_learned_matched = W_filters_sensor[:, learned_order]

# =============================================================================
# 7. ТОПОГРАММЫ: истинные vs согласованные выученные
# =============================================================================
fig, axes = plt.subplots(2, Ndistr, figsize=(4 * Ndistr, 7))
for i in range(Ndistr):
    ax_t = axes[0, i]
    mne.viz.plot_topomap(A_true[:, i], info_sub, axes=ax_t, show=False, cmap='RdBu_r')
    ax_t.set_title(f'True source {i+1}')

    ax_l = axes[1, i]
    mne.viz.plot_topomap(A_learned_matched[:, i], info_sub, axes=ax_l,
                         show=False, cmap='RdBu_r')
    ax_l.set_title(f'Learned (matched) {i+1}')

plt.tight_layout()
plt.show()

# =============================================================================
# 8. КОРРЕЛЯЦИИ ВРЕМЕННЫХ РЯДОВ МОЩНОСТЕЙ
# =============================================================================
# По-столбцовая нормировка (каждую колонку — к единичной std)
P_true_n = P_true / P_true.std(axis=0, keepdims=True)
P_pred_n = P_learned_matched / P_learned_matched.std(axis=0, keepdims=True)

correlations = [
    np.corrcoef(P_true_n[:, i], P_pred_n[:, i])[0, 1]
    for i in range(Ndistr)
]

fig, axes = plt.subplots(Ndistr, 1, figsize=(12, 3 * Ndistr), sharex=True)
fig.suptitle('Мощность источника: истинная vs предсказанная (согласовано)', y=1.02)
for i in range(Ndistr):
    ax = axes[i] if Ndistr > 1 else axes
    ax.plot(P_true_n[:, i], 'b-', lw=2.5, alpha=0.6, label='Истинная')
    ax.plot(P_pred_n[:, i], 'r', lw=1.5, label='Предсказанная')
    ax.set_title(f'Источник {i+1} (corr = {correlations[i]:.3f})')
    ax.set_ylabel('Норм. мощность')
    ax.legend(loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.5)
plt.xlabel('Окна')
plt.tight_layout()
plt.show()

print("Корреляции по источникам:", correlations)

# %%
