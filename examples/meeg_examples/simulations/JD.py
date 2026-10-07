import numpy as np
import scipy.linalg as la
import tensorflow as tf

# ==========================================
# 1. Генерация батча ЭЭГ (100 эпох: 50/50)
# ==========================================
np.random.seed(42)
tf.random.set_seed(42)

n_channels = 10
n_trials_per_class = 50
n_total = n_trials_per_class * 2

# Истинные пространственные паттерны (коллинеарность ~0.98)
A_true = np.random.randn(n_channels, n_channels)
A_true[:, 1] = A_true[:, 0] + 0.15 * np.random.randn(n_channels)
A_true = A_true / np.linalg.norm(A_true, axis=0)

print(f"Косинусное сходство истинных источников 0 и 1: {np.abs(np.dot(A_true[:, 0], A_true[:, 1])):.4f}\n")

# Базовые мощности источников для двух классов
v1_mean = np.ones(n_channels)
v2_mean = np.ones(n_channels)
v1_mean[:6] = [10.0, 0.5, 5.0, 2.0, 1.0, 1.0]
v2_mean[:6] = [0.5, 10.0, 1.0, 1.0, 5.0, 2.0]

covs = np.zeros((n_total, n_channels, n_channels))

# Генерируем 100 ковариационных матриц с вариативностью (trial-to-trial variability)
for i in range(n_total):
    power_fluctuation = np.exp(np.random.randn(n_channels) * 0.3) 
    if i < n_trials_per_class:
        v = v1_mean * power_fluctuation
    else:
        v = v2_mean * power_fluctuation
        
    covs[i] = A_true @ np.diag(v) @ A_true.T + 0.5 * np.eye(n_channels)

# ==========================================
# 2. Истинные попарные Римановы расстояния и маски
# ==========================================
true_dists = np.zeros((n_total, n_total))
for i in range(n_total):
    for j in range(i + 1, n_total):
        eigvals = la.eigvals(covs[i], covs[j]).real
        d = np.sqrt(np.sum(np.log(eigvals[eigvals > 0])**2))
        true_dists[i, j] = d
        true_dists[j, i] = d

# Создаем метки классов
labels = np.concatenate([np.zeros(n_trials_per_class), np.ones(n_trials_per_class)])

# Маска для межклассовых пар (разные классы)
mask_inter = labels[:, None] != labels[None, :] 
mask_inter_t = tf.constant(mask_inter, dtype=tf.float32)

# Маска для внутриклассовых пар (одинаковые классы, исключая диагональ)
mask_intra = labels[:, None] == labels[None, :]
np.fill_diagonal(mask_intra, False) # Расстояние объекта с самим собой нас не интересует
mask_intra_t = tf.constant(mask_intra, dtype=tf.float32)

# ==========================================
# 3. Градиентный спуск (Metric Learning)
# ==========================================
covs_t = tf.constant(covs, dtype=tf.float32)
true_dists_t = tf.constant(true_dists, dtype=tf.float32)

W_gd = tf.Variable(tf.random.normal([n_channels, n_channels], stddev=0.1))
optimizer = tf.keras.optimizers.Adam(learning_rate=0.02)

# Гиперпараметр баланса: 
# 1.0 = только межклассовая дифференциация, 
# 0.0 = только сохранение внутриклассовой структуры
alpha = 0.8 

print(f"=== Обучение через Metric Learning (Вращение) | alpha = {alpha} ===")
for epoch in range(4001):
    with tf.GradientTape() as tape:
        # ЖЕСТКАЯ НОРМАЛИЗАЦИЯ: оптимизируем только вращение направлений!
        W_norm = tf.math.l2_normalize(W_gd, axis=0)
        
        # Проецируем весь батч сразу
        S = tf.einsum('ji,kjl,lm->kim', W_norm, covs_t, W_norm)
        V = tf.linalg.diag_part(S)
        log_V = tf.math.log(V)
        
        diff = tf.expand_dims(log_V, 1) - tf.expand_dims(log_V, 0)
        est_dists = tf.sqrt(tf.reduce_sum(tf.square(diff), axis=2))
        
        # Квадрат ошибки аппроксимации
        squared_errors = tf.square(est_dists - true_dists_t)
        
        # Разделяем лосс на межклассовый и внутриклассовый компоненты
        loss_inter = tf.reduce_sum(squared_errors * mask_inter_t) / tf.reduce_sum(mask_inter_t)
        
        # Если alpha == 1.0, можно избежать деления на ноль при вычислении loss_intra
        if alpha < 1.0:
            loss_intra = tf.reduce_sum(squared_errors * mask_intra_t) / tf.reduce_sum(mask_intra_t)
        else:
            loss_intra = 0.0
            
        # Итоговая взвешенная функция потерь
        loss = alpha * loss_inter + (1 - alpha) * loss_intra

    gradients = tape.gradient(loss, [W_gd])
    optimizer.apply_gradients(zip(gradients, [W_gd]))
    
    if epoch % 500 == 0:
        print(f"Эпоха {epoch:4d} | Weighted Loss: {loss.numpy():.4f}")

# ==========================================
# 4. Аналитический CSP
# ==========================================
# Классический CSP на усредненных ковариациях классов
C1_mean = np.mean(covs[:n_trials_per_class], axis=0)
C2_mean = np.mean(covs[n_trials_per_class:], axis=0)
_, W_csp = la.eigh(C1_mean, C2_mean)

# Оцениваем, как CSP аппроксимирует попарные расстояния всего датасета
S_csp = np.einsum('ji,kjl,lm->kim', W_csp, covs, W_csp)
V_csp = np.diagonal(S_csp, axis1=1, axis2=2)
log_V_csp = np.log(V_csp + 1e-8)
diff_csp = np.expand_dims(log_V_csp, 1) - np.expand_dims(log_V_csp, 0)
est_dists_csp = np.sqrt(np.sum(np.square(diff_csp), axis=2))

# Считаем взвешенный MSE для CSP, чтобы сравнение было честным
sq_err_csp = np.square(est_dists_csp - true_dists)
loss_inter_csp = np.sum(sq_err_csp * mask_inter) / np.sum(mask_inter)
loss_intra_csp = np.sum(sq_err_csp * mask_intra) / np.sum(mask_intra)
weighted_mse_csp = alpha * loss_inter_csp + (1 - alpha) * loss_intra_csp

# ==========================================
# 5. Сравнение и анализ
# ==========================================
# Извлекаем паттерны (используем pseudo-inverse для безопасности GD)
W_final = tf.math.l2_normalize(W_gd, axis=0).numpy()
A_gd = np.linalg.pinv(W_final).T
A_gd = A_gd / np.linalg.norm(A_gd, axis=0)

A_csp = np.linalg.inv(W_csp).T
A_csp = A_csp / np.linalg.norm(A_csp, axis=0)

def get_best_match(true_vec, found_matrix):
    sims = np.abs(true_vec @ found_matrix)
    return np.max(sims)

print("\n=== ИТОГОВЫЕ ВЫВОДЫ ===")
print(f"1. Аппроксимация взвешенной метрики (alpha = {alpha}):")
print(f"   CSP Loss: {weighted_mse_csp:.4f}  <-- Насколько сильно CSP исказил целевую геометрию")
print(f"   GD Loss:  {loss.numpy():.4f}  <-- Точность нашего метода")

print("\n2. Точность восстановления коллинеарных паттернов (от 0 до 1):")
print(f"   Источник 0 (CSP): {get_best_match(A_true[:, 0], A_csp):.4f}")
print(f"   Источник 0 (GD):  {get_best_match(A_true[:, 0], A_gd):.4f}")
print(f"   Источник 1 (CSP): {get_best_match(A_true[:, 1], A_csp):.4f}")
print(f"   Источник 1 (GD):  {get_best_match(A_true[:, 1], A_gd):.4f}")