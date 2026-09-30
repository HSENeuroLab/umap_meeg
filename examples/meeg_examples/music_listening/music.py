# -*- coding: utf-8 -*-
"""
Created on Wed Oct 22 17:07:11 2025

@author: anton
"""
import os
import sys
import mne
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import umap

from scipy.signal import butter, filtfilt
from scipy.linalg import eigh
from scipy.linalg import inv, null_space
from umap.parametric_umap import ParametricUMAP

lib_directory = os.path.abspath("C:/Users/ansbel/Documents/GitHub/pyRiemann") 
if lib_directory not in sys.path:
    sys.path.insert(0, lib_directory)
from pyriemann.estimation import Covariances
from pyriemann.utils.base import invsqrtm
from pyriemann.geometry.distance import pairwise_distance

# %%
fpath = 'C:/Users/ansbel/Downloads/Telegram Desktop/P1_AR_bars.fif'
raw = mne.io.read_raw_fif(fpath, preload=True)

# %%
raw.plot()

# %%
import mne

# Укажите путь к вашему .vhdr файлу
vhdr_path = "C:/Users/ansbel/Downloads/Telegram Desktop/pilot_01.vhdr"

# Загрузка данных в объект Raw
raw = mne.io.read_raw_brainvision(vhdr_path, preload=True)

# Просмотр базовой информации о записи
print(raw)
print(raw.info)

# Интерактивный просмотр графиков (если настроено графическое окружение)
raw.plot()

# %%
# Загружаем файл
data = np.load('C:/Users/ansbel/Downloads/Telegram Desktop/P3_AinC_vs_AinB_15-25Hz_csp.npz')
print(data.files)
array1 = data['patterns']

# %%
# %%
import numpy as np
import mne
import matplotlib.pyplot as plt
from mne.time_frequency import tfr_array_morlet

# --- 1. ЗАГРУЗКА ДАННЫХ ---
vhdr_path = "C:/Users/ansbel/Downloads/Telegram Desktop/pilot_01.vhdr"
raw = mne.io.read_raw_brainvision(vhdr_path, preload=True)

data = np.load('C:/Users/ansbel/Downloads/Telegram Desktop/P3_AinC_vs_AinB_15-25Hz_csp.npz')

# --- 2. ПОДГОТОВКА ДАННЫХ ---
# Берем только ЭЭГ каналы (важно сделать это до фильтров, чтобы посчитать ковариацию C)
picks = mne.pick_types(raw.info, eeg=True, meg=False)
eeg_data = raw.get_data(picks=picks) # Размерность: (n_channels, n_samples)

# --- 3. ПОЛУЧЕНИЕ ФИЛЬТРОВ ИЗ ПАТТЕРНОВ ЧЕРЕЗ C^-1 ---
if 'filters' in data.files:
    filters = data['filters']
else:
    patterns = data['patterns'] 
    # Предполагается, что patterns имеет размер (n_components, n_channels)
    
    # 1. Считаем матрицу ковариации C (размер n_channels x n_channels)
    C = np.cov(eeg_data)
    
    # 2. Находим обратную матрицу C^-1 
    # (Используем pinv вместо inv для вычислительной стабильности, если каналы сильно скоррелированы)
    C_inv = np.linalg.pinv(C)
    
    # 3. Применяем формулу W = C^-1 * A
    # Так как A (паттерны) в формуле представлены как векторы-столбцы, транспонируем patterns: patterns.T
    W_cols = np.dot(C_inv, patterns.T) # Размер: (n_channels, n_components)
    
    # Транспонируем обратно, чтобы фильтры имели размер (n_components, n_channels) для применения к данным
    filters = W_cols.T 
    
data.close()

# --- 4. ВЫДЕЛЕНИЕ КОМПОНЕНТ ---
# Умножаем матрицу фильтров на данные (W * X)
# Получаем компоненты размерностью (n_components, n_samples)
components = np.dot(filters, eeg_data)

# По желанию: упакуем компоненты обратно в объект Raw для удобного просмотра
n_components = components.shape[0]
info_csp = mne.create_info(
    ch_names=[f'CSP {i+1}' for i in range(n_components)], 
    sfreq=raw.info['sfreq'], 
    ch_types=['eeg'] * n_components
)
raw_csp = mne.io.RawArray(components, info_csp)
# raw_csp.plot(title="Выделенные CSP компоненты")

# %%
# --- 5. ЧАСТОТНО-ВРЕМЕННОЙ АНАЛИЗ (TFR) ---
# Возьмем первую компоненту (индекс 0) на всем отрезке
sfreq = raw.info['sfreq']
start_sec = 0
end_sec = 1940
start_samp = int(start_sec * sfreq)
end_samp = int(end_sec * sfreq)

# Извлекаем нужный фрагмент (размерность должна стать 3D для функции MNE: n_epochs, n_channels, n_times)
comp_segment = components[0, start_samp:end_samp]
comp_segment = comp_segment[np.newaxis, np.newaxis, :] 

# Настройки для вейвлет-преобразования (Morlet)
freqs = np.arange(2, 70, 1) # Диапазон частот от 2 до 30 Гц
n_cycles = freqs / 2.0      # Адаптивное количество циклов (больше для высоких частот)

# Вычисляем абсолютную мощность
power = tfr_array_morlet(
    comp_segment, 
    sfreq=sfreq, 
    freqs=freqs, 
    n_cycles=n_cycles, 
    output='power'
)

# Извлекаем 2D массив для нашей компоненты: размерность (n_freqs, n_times)
power_2d = power[0, 0]

# 1. Переводим мощность в логарифмический масштаб (в децибелы)
# Добавляем крошечную константу (eps), чтобы избежать ошибки логарифма от нуля
log_power = 10 * np.log10(power_2d)

# 2. Независимая нормировка по частотам (Z-score вдоль оси времени)
# Считаем среднее и стандартное отклонение для КАЖДОЙ частоты
mean_per_freq = np.mean(log_power, axis=1, keepdims=True)
std_per_freq = np.std(log_power, axis=1, keepdims=True)

# Применяем нормировку: (значение - среднее) / стандартное отклонение
norm_power = (log_power - mean_per_freq) / std_per_freq
# norm_power = log_power

# Отрисовка
plt.figure(figsize=(12, 6))

# Рисуем нормированную мощность. 
# Для z-score логично установить симметричные лимиты, например vmin=-3, vmax=3 
# (отражает отклонения в +-3 стандартных отклонения)
plt.imshow(norm_power, aspect='auto', origin='lower', 
           extent=[start_sec, end_sec, freqs[0], freqs[-1]], 
           cmap='RdBu_r', 
           vmin=-3, vmax=3
           )

plt.colorbar(label='Нормированная лог. мощность (Z-score)')
plt.xlabel('Время (с)')
plt.ylabel('Частота (Гц)')
plt.title('Частотно-временной график 1-й компоненты (лог. масштаб, норм. по частотам)')
plt.show()

# %%
import mne

# Если данные еще не загружены:
# fpath = 'C:/Users/ansbel/Downloads/Telegram Desktop/P01_bars.fif'
# raw = mne.io.read_raw_fif(fpath, preload=True)

# 1. Находим все уникальные аннотации и создаем словарь маппинга в ID
# Присвоим каждой категории свой числовой триггер: 
# AinB = 1, AinC = 2, BinA = 3, CinA = 4
custom_mapping = {}
for desc in set(raw.annotations.description):
    if desc.endswith('AinB'):
        custom_mapping[desc] = 1
    elif desc.endswith('AinC'):
        custom_mapping[desc] = 2
    elif desc.endswith('BinA'):
        custom_mapping[desc] = 3
    elif desc.endswith('CinA'):
        custom_mapping[desc] = 4

# 2. Извлекаем массив событий на основе нашего маппинга
# Все аннотации, не попавшие в словарь, будут проигнорированы
events, _ = mne.events_from_annotations(raw, event_id=custom_mapping)

# 3. Создаем словарь с понятными именами для самих эпох
epochs_event_id = {
    'AinB': 1, 
    'AinC': 2, 
    'BinA': 3, 
    'CinA': 4
}

# 4. Нарезаем эпохи
# tmin = 0.0 (начинаем ровно с онсета)
# tmax = 1.85 (длительность 1.85 сек)
# baseline = None (без бейзлайна)
epochs = mne.Epochs(raw.copy().filter(l_freq=25,h_freq=40), 
                    events=events, 
                    event_id=epochs_event_id, 
                    tmin=-0.1, 
                    tmax=1.85, 
                    baseline=(-0.1,0), 
                    preload=True)

# Проверяем результат
print(epochs)

# %%
evoked_ab = epochs["AinB"].average()
evoked_ac = epochs["AinC"].average()
evoked_ba = epochs["BinA"].average()
evoked_ca = epochs["CinA"].average()

# %%
import numpy as np
import scipy.linalg

# ====================================================================
# 5. КАСТОМНЫЙ CSP (МАКСИМИЗАЦИЯ ТАРГЕТА ОТНОСИТЕЛЬНО ДРУГОГО КЛАССА)
# ====================================================================
def manual_csp(X_target, X_other, reg_coef=1e-5):
    """
    X_target: массив эпох (n_epochs, n_channels, n_times) для класса, дисперсию которого максимизируем
    X_other: массив эпох для второго класса (например, минимизируем)
    reg_coef: коэффициент регуляризации (если 1e-9 все еще выдает ошибку, увеличьте до 1e-5)
    """
    def compute_epoch_covs(X):
        # Вычисляем ковариацию для каждой эпохи
        covs = np.zeros((X.shape[0], X.shape[1], X.shape[1]))
        for i in range(X.shape[0]):
            covs[i] = np.cov(X[i])
        return covs

    # Усредненные ковариационные матрицы по классам
    C_target = np.mean(compute_epoch_covs(X_target), axis=0)
    C_other = np.mean(compute_epoch_covs(X_other), axis=0)
    
    # Общая ковариация
    C_denom = C_target + C_other
    
    # Регуляризация для стабильности матриц (защита от LinAlgError)
    # Используем след матрицы для масштабирования шума
    C_denom_reg = C_denom.copy() + reg_coef * (np.trace(C_denom) / C_denom.shape[0]) * np.eye(C_denom.shape[0])
    
    # Решаем обобщенную задачу на собственные значения 
    # (максимизируем C_target относительно C_denom_reg)
    evals, evecs = scipy.linalg.eigh(C_target, C_denom_reg)
    
    # Сортируем по убыванию собственных значений
    # Индексы в начале = фильтры для X_target; в конце = фильтры для X_other
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]
    
    # Матрица пространственных фильтров
    filters = evecs.T
    
    # Нормализация весов фильтров 
    norms = np.linalg.norm(filters, axis=1, keepdims=True)
    filters = filters / norms
    
    # ПАТТЕРНЫ: В BCI паттерны вычисляются как псевдообратная матрица к фильтрам.
    # Это позволяет рисовать корректные топограммы распределения активности.
    patterns = np.linalg.pinv(filters)
    
    return filters, patterns, evals

def apply_filters(X, filters, n_components=4):
    """
    Применяет пространственные фильтры и возвращает дисперсию по каждой эпохе
    Аналог csp.transform(X) с log=False
    """
    # Берем первые n/2 (максимизируют target) и последние n/2 (минимизируют target)
    # Если нужно просто 4 компоненты, берем: 2 первых и 2 последних
    if n_components > 0:
        half = n_components // 2
        selected_filters = np.vstack([filters[:half], filters[-half:]])
    else:
        selected_filters = filters
        
    # Проецируем данные: (n_epochs, n_channels, n_times) -> (n_epochs, n_components, n_times)
    # np.tensordot удобно умножает матрицу фильтров на каждую эпоху
    X_proj = np.tensordot(X, selected_filters, axes=(1, 1)).transpose(0, 2, 1)
    
    # Считаем дисперсию (мощность) в каждой компоненте для каждой эпохи
    # Формат возврата: (n_epochs, n_components)
    variances = np.var(X_proj, axis=2)
    return variances


# ====================================================================
# ПРИМЕНЕНИЕ К ВАШИМ ДАННЫМ
# ====================================================================
# Разделяем данные на два массива (например, AinB и BinA)
X_target = epochs['BinA'].get_data(picks='eeg')
X_other = epochs['CinA'].get_data(picks='eeg')

# 1. Обучаем ручной CSP
filters, patterns, evals = manual_csp(X_target, X_other, reg_coef=1e-5)

# 2. Получаем дисперсии (например, для 4 компонент: 2 для AinB, 2 для BinA)
var_target = apply_filters(X_target, filters, n_components=4)
var_other = apply_filters(X_other, filters, n_components=4)

# 3. Сравниваем
print("=== Усредненные дисперсии в 4 выделенных компонентах ===")
mean_var_target = np.mean(var_target, axis=0)
mean_var_other = np.mean(var_other, axis=0)

for i in range(4):
    print(f"Компонента {i+1}:")
    print(f"  AinB: {mean_var_target[i]:.2e}")
    print(f"  BinA: {mean_var_other[i]:.2e}")
    print(f"  Отношение (AinB/BinA): {mean_var_target[i] / mean_var_other[i]:.2f}\n")    
    
# %%
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import mne

# 1. Подготовка данных для визуализации
info_eeg = epochs.copy().pick('eeg').info

# Выбираем те же 4 компоненты, что и при фильтрации данных:
# 2 первых (максимизируют дисперсию AinB) и 2 последних (максимизируют AinC)
half = 4  
selected_patterns = np.column_stack([patterns[:, :half], patterns[:, -half:]])

# Для корректной статистики и нормального распределения используем логарифм дисперсии
# var_target и var_other получены ранее через apply_filters
log_var_target = np.log(var_target)
log_var_other = np.log(var_other)

# Названия условий (должны совпадать с тем, что вы подали в X_target и X_other)
cond_target = 'BinA'
cond_other = 'CinA'

# 2. Создаем фигуру (2 строки, 4 колонки)
# Верхний ряд: Топограммы
# Нижний ряд: Barplots (Среднее + 95% CI)
fig, axes = plt.subplots(2, 4, figsize=(16, 8), gridspec_kw={'height_ratios': [1, 1.2]})

for i in range(4):
    # --- ВЕРХНИЙ РЯД: Топограммы ---
    ax_topo = axes[0, i]
    mne.viz.plot_topomap(selected_patterns[:, i], info_eeg, axes=ax_topo, show=False)
    
    # Подписываем логику компонент
    if i < half:
        ax_topo.set_title(f'Comp {i+1}\n(Max {cond_target})', fontweight='bold')
    else:
        ax_topo.set_title(f'Comp {i+1}\n(Max {cond_other})', fontweight='bold')

    # --- НИЖНИЙ РЯД: Распределение w^T C w ---
    ax_var = axes[1, i]
    
    # Собираем данные компоненты в DataFrame
    df_comp = pd.DataFrame({
        'Condition': [cond_target]*len(log_var_target) + [cond_other]*len(log_var_other),
        'Log Variance': np.concatenate([log_var_target[:, i], log_var_other[:, i]])
    })
    
    # 1. Рисуем индивидуальные эпохи (точки) заданными цветами
    sns.stripplot(data=df_comp, x='Condition', y='Log Variance', 
                  dodge=False, alpha=0.4, size=4, ax=ax_var, jitter=True,
                  palette=['#e74c3c', '#3498db'])
    
    # 2. Рисуем среднее и 95% CI черным маркером (ромбом) поверх точек
    sns.pointplot(data=df_comp, x='Condition', y='Log Variance', 
                  errorbar=('ci', 95), capsize=0.1, join=False, 
                  color='black', markers='D', errwidth=1.5, ax=ax_var)
    
    ax_var.set_xlabel('')
    if i == 0:
        # Математический смысл: мощность сигнала, спроецированного через пространственный фильтр
        ax_var.set_ylabel(r'Log Variance ($log(w^T C w)$)')
    else:
        ax_var.set_ylabel('')
        
plt.suptitle("CSP Spatial Patterns and Projected Variance", fontsize=16, y=1.02)
# plt.tight_layout()
plt.show()

# %%
# 2. Визуализация распределения мощностей (дисперсий) со средним и ДИ

# Для стабильной статистики и красивой визуализации дисперсию (мощность) в ЭЭГ 
# принято логарифмировать. Иначе выбросы сильно искажают среднее.
log_var_target = np.log(var_target)
log_var_other = np.log(var_other)

# Собираем данные в pandas DataFrame для удобной отрисовки в seaborn
data = []
# Обратите внимание, что мы сравниваем AinB и AinC (согласно вашему X_other)
condition_1 = 'AinB'
condition_2 = 'AinC'

for i in range(4):
    for val in log_var_target[:, i]:
        data.append({'Component': f'Comp {i+1}', 'Condition': condition_1, 'Log Variance': val})
    for val in log_var_other[:, i]:
        data.append({'Component': f'Comp {i+1}', 'Condition': condition_2, 'Log Variance': val})
        
df = pd.DataFrame(data)

# Создаем фигуру с двумя графиками: Barplot (Среднее + 95% ДИ) и Boxplot (Полное распределение)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# График 1: Barplot со средним и 95% доверительным интервалом (errorbar='ci')
sns.barplot(data=df, x='Component', y='Log Variance', hue='Condition', 
            capsize=0.1, errorbar=('ci', 95), ax=ax1, palette='Set1')
ax1.set_title('Средняя логарифмическая дисперсия\n(Планки ошибок = 95% CI)')
ax1.set_ylabel('Log Variance')

# График 2: Boxplot + Stripplot для оценки выбросов и формы распределения
sns.boxplot(data=df, x='Component', y='Log Variance', hue='Condition', 
            boxprops={'alpha': 0.5}, ax=ax2, palette='Set1', showfliers=False)
# Добавляем сырые точки (эпохи) поверх
sns.stripplot(data=df, x='Component', y='Log Variance', hue='Condition', 
              dodge=True, alpha=0.6, ax=ax2, palette='dark:gray', legend=False)
ax2.set_title('Полное распределение эпох (Boxplot + scatter)')
ax2.set_ylabel('Log Variance')

plt.tight_layout()
plt.show()
# %%
fpath = "C:/Users/ansbel/Downloads/Telegram Desktop/raw_final_all_rec_after_bad_ref_ICA6.fif"
raw = mne.io.read_raw_fif(fpath, preload=True)

# %%
raw.plot()

# %%
# =============================================================================
# ЧАСТОТНО-ВРЕМЕННОЙ ГРАФИК С НОРМИРОВКОЙ ДЛЯ ИСХОДНОГО КАНАЛА (FC6)
# =============================================================================
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
import numpy as np

# 1. Задаем имя нужного канала из оригинального raw
ch_name = 'FC6'

# 2. Извлекаем непрерывный сигнал этого канала прямо из raw
sfreq = raw.info['sfreq']
# get_data возвращает массив (n_channels, n_times), берем нулевой индекс для нашего 1 канала
data = raw.get_data(picks=ch_name)[0]

# 3. Считаем спектрограмму (скользящее окно 2 сек, перекрытие 1.5 сек)
nperseg = int(sfreq * 2)
noverlap = int(sfreq * 1.5)

freqs, times, Sxx = spectrogram(data, fs=sfreq, nperseg=nperseg, noverlap=noverlap)

# 4. Ограничиваем частоты до 40 Гц
freq_mask = freqs <= 40
freqs = freqs[freq_mask]
Sxx = Sxx[freq_mask, :]

# Переводим мощность в децибелы
Sxx_db = 10 * np.log10(Sxx)

# =============================================================================
# НОРМИРОВКА ВО ВРЕМЕНИ ДЛЯ КАЖДОЙ ЧАСТОТЫ (Z-SCORE)
# =============================================================================
mean_per_freq = np.mean(Sxx_db, axis=1, keepdims=True)
std_per_freq = np.std(Sxx_db, axis=1, keepdims=True)

# Z-преобразование
Sxx_norm = (Sxx_db - mean_per_freq) / (std_per_freq + 1e-8)

# 5. Отрисовка
fig, ax = plt.subplots(figsize=(16, 6))

im = ax.pcolormesh(
    times, freqs, Sxx_norm, 
    shading='gouraud', 
    cmap='RdBu_r', 
    vmin=-3, vmax=3
)

ax.set_ylabel('Частота (Гц)', fontsize=12)
ax.set_xlabel('Время записи (секунды)', fontsize=12)
ax.set_title(f'Нормированная спектрограмма (Z-score) для канала {ch_name}', fontsize=14, fontweight='bold')
ax.set_ylim(1, 40)

# Добавляем цветовую шкалу
cbar = fig.colorbar(im, ax=ax, pad=0.01)
cbar.set_label('Отклонение от среднего (Z)', rotation=270, labelpad=15)

# 6. Накладываем вертикальные линии границ условий из аннотаций оригинального raw
cond_descriptions = [
    'rest/EC/1', 'rest/EO/1',
    'ABA_1', 'ACA_1', 
    'ABA_2', 'ACA_2',
    'rest/EO/2', 'rest/EC/2'
]

for annot in raw.annotations:
    desc = annot['description']
    onset = annot['onset']
    
    if desc in cond_descriptions:
        ax.axvline(x=onset, color='black', linestyle='--', alpha=0.5, lw=1.5)
        ax.text(onset, 39, f' {desc}', color='black', fontsize=9, 
                rotation=90, va='top', ha='left', fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', boxstyle='round,pad=0.1'))
        
plt.tight_layout()
plt.show()

# %%
raw.plot_sensors()

# %%
# raw.annotations.description = np.array(['Stimulus/S  2', 'rest/EC/1', 'Stimulus/S  4', 'BAD boundary',
#        'EDGE boundary', 'Stimulus/S  6', 'rest/EO/1', 'Stimulus/S  8',
#        'BAD boundary', 'EDGE boundary', 'ABA', 'ABA_end', 'BAD boundary',
#        'EDGE boundary', 'Stimulus/S  2', 'rest/EC/2', 'Stimulus/S  4',
#        'BAD boundary', 'EDGE boundary', 'Stimulus/S  6', 'rest/EO/2',
#        'Stimulus/S  8', 'BAD boundary', 'EDGE boundary', 'ACA', 'BAD_',
#        'ACA_end', 'BAD boundary', 'EDGE boundary', 'Stimulus/S  2',
#        'rest/EC/3', 'BAD_', 'BAD_', 'Stimulus/S  4', 'BAD boundary',
#        'EDGE boundary', 'Stimulus/S  6', 'rest/EO/3', 'BAD_',
#        'Stimulus/S  8'])

cond_descriptions = ['rest/EC/1', 'rest/EO/1',
       'ABA_1', 
       'ACA_1', 
       'ABA_2',
       'ACA_2',
       'rest/EO/2',
       'rest/EC/2', 'BAD_']

# 4. Фильтрация индексов
bad_idx = [i for i, desc in enumerate(raw.annotations.description) if desc == "BAD_"]
good_idx = [i for i, desc in enumerate(raw.annotations.description) if desc in cond_descriptions]
good_ann = raw.annotations[good_idx]

# Теперь индексация сработает без ошибок
bad_ann = raw.annotations[bad_idx]
good_ann = raw.annotations[good_idx]

# Объединяем и сохраняем
new_ann = bad_ann + good_ann
raw.set_annotations(new_ann)

# %%
raw.plot()

# %%
raw_segments = raw.crop_by_annotations(good_ann)
raw_seg =  mne.concatenate_raws(raw_segments)

# %%


# %%
from mne.preprocessing import ICA
ica = ICA(n_components=0.999, random_state=97, max_iter="auto")
ica.fit(raw_seg)

ica.plot_components()  
ica.plot_sources(raw_seg)  

# %%
ica.apply(raw)

# %%
raw.interpolate_bads()

# %%
raw.plot()

# %%
raw_band = raw.copy().filter(l_freq=15,h_freq=25)

# %%
Wsize = 2
Ssize = 0.5 
overlap = Wsize - Ssize

cropped_raws = []
X_windows_unfilt = []
window_labels = []

raw_unfilt = raw_band.copy().pick_types(eeg=True)

for annot in raw.annotations:
    desc = annot['description']
    if annot['duration'] < Wsize or desc == 'BAD_':
        continue

    tmin = annot['onset'] - raw_unfilt.first_time
    tmax = min(tmin + annot['duration'], raw.times[-1])

    raw_crop_unfilt = raw_unfilt.copy().crop(tmin=tmin, tmax=tmax)
    cropped_raws.append(raw_crop_unfilt)
    
    epochs_unfilt = mne.make_fixed_length_epochs(
        raw_crop_unfilt, 
        duration=Wsize, 
        overlap=overlap, 
        preload=True, 
        reject_by_annotation=True,  
        verbose=False
    )

    X_windows_unfilt.append(epochs_unfilt.get_data(copy=False))
    window_labels.extend([desc] * len(epochs_unfilt))

X_windows_unfilt = np.concatenate(X_windows_unfilt, axis=0)
window_labels = np.array(window_labels)

covmats = Covariances(estimator='cov').fit_transform(X_windows_unfilt / np.std(X_windows_unfilt))
mean_cov = np.mean(covmats, axis=0)
eigvals, eigvecs = eigh(mean_cov)

idx_sorted = np.argsort(eigvals)[::-1]
eigvals, eigvecs = eigvals[idx_sorted], eigvecs[:, idx_sorted]
valid_mask = eigvals > 1e-6
eigvals, eigvecs = eigvals[valid_mask], eigvecs[:, valid_mask]

Ww = eigvecs / np.sqrt(eigvals)[np.newaxis, :]
n_components = Ww.shape[1]
covmats_w = Ww.T @ covmats @ Ww

# %%
dists_init = pairwise_distance(Covariances(estimator='oas').fit_transform(X_windows_unfilt / np.std(X_windows_unfilt)), metric='riemann')

# %%
reducer = umap.UMAP(n_components=2, n_neighbors=20, metric='precomputed')
coords = reducer.fit_transform(dists_init)

# %%
plt.figure(figsize=(10, 7))

unique_labels = []
for lab in window_labels:
    if lab not in unique_labels:
        unique_labels.append(lab)

cmap = plt.get_cmap('tab10')

for i, label in enumerate(unique_labels):
    mask = window_labels == label
    pts = coords[mask]
    color = cmap(i % 10)
    
    # 1. Точки с номером в легенде
    plt.scatter(
        pts[:, 0],
        pts[:, 1],
        s=15,
        alpha=0.6,
        color=color,
        label=f"[{i + 1}] {label}"
    )
    
    # 2. Центроид (робастная медиана или среднее)
    cx, cy = np.median(pts, axis=0)
    
    # 3. Текстовая метка с номером прямо на графике
    plt.text(
        cx, cy,
        str(i + 1),
        fontsize=11,
        fontweight='bold',
        color='black',
        ha='center',
        va='center',
        bbox=dict(boxstyle='circle,pad=0.25', facecolor='white', edgecolor=color, lw=2, alpha=0.9),
        zorder=5
    )

plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.title('UMAP проекция по условиям с центроидами')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True, title="Условия")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# %%
import matplotlib.pyplot as plt
import numpy as np

# 1. Порог контраста
percentile_value = 95
threshold = np.percentile(dists_init, percentile_value)

# 2. Поиск границ смены условий и их центров
# Находим индексы, где метка окна меняется на следующую
split_indices = np.where(window_labels[:-1] != window_labels[1:])[0] + 1
boundaries = np.concatenate(([0], split_indices, [len(window_labels)]))

# Центры блоков для установки подписей
tick_locs = (boundaries[:-1] + boundaries[1:]) / 2.0

# Имена условий на каждом участке
block_labels = [window_labels[b] for b in boundaries[:-1]]
unique_conditions = list(dict.fromkeys(window_labels))
label_to_id = {lab: i + 1 for i, lab in enumerate(unique_conditions)}

# Формируем подписи вида "[1] rest/EC/1"
tick_names = [f"[{label_to_id[lab]}] {lab}" for lab in block_labels]

# 3. Отрисовка
plt.figure(figsize=(9, 8))
im = plt.imshow(dists_init, vmax=threshold, cmap='viridis', origin='upper')

# 4. Сетка границ между условиями
for b in split_indices:
    plt.axhline(b - 0.5, color='black', linestyle='--', linewidth=1.2, alpha=0.8)
    plt.axvline(b - 0.5, color='black', linestyle='--', linewidth=1.2, alpha=0.8)

# 5. Разметка осей
plt.xticks(tick_locs, tick_names, rotation=45, ha='right', fontsize=9)
plt.yticks(tick_locs, tick_names, fontsize=9)

# 6. Цветовая шкала
cbar = plt.colorbar(im, fraction=0.046, pad=0.04, extend='max')
cbar.set_label(f"Риманово расстояние (ограничено {percentile_value}-м процентилем)")

plt.title("Матрица попарных расстояний с разделением условий")
plt.tight_layout()
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
        # Создаем матрицу развертки и сохраняем как внутренний НЕобучаемый тензор слоя.
        # Это избавляет от необходимости использовать глобальные переменные.
        w_init = build_unvec_matrix(self.M)
        self.W_UNVEC = self.add_weight(
            shape=w_init.shape,
            initializer=tf.keras.initializers.Constant(w_init),
            trainable=False,  # Важно: этот вес не обучается!
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
    def __init__(self, d_out, orth_weight=0.01, **kwargs):
        super().__init__(**kwargs)
        self.d_out = int(d_out)
        self.orth_weight = float(orth_weight)

    def build(self, input_shape):
        self.d_in = int(input_shape[-1])
        init_val = np.eye(self.d_out, self.d_in, dtype=np.float32)
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

        if self.orth_weight > 0.0:
            WWT = tf.matmul(W_norm, W_norm, transpose_b=True)
            I = tf.eye(self.d_out)
            orth_penalty = tf.reduce_sum(tf.square(WWT - I))
            self.add_loss(self.orth_weight * orth_penalty)

        return X_out

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_out": self.d_out,
            "orth_weight": self.orth_weight
        })
        return config


@tf.keras.utils.register_keras_serializable()
class LogDiagScaleLayer(tf.keras.layers.Layer):
    """
    z_i = softplus(s_i) · log( max( (W C W^T)_{ii}, eps ) ) + b_i
    """
    def __init__(self, n_filters, epsilon=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.n_filters = int(n_filters)
        self.epsilon = epsilon

    def build(self, input_shape):
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
        diags = tf.linalg.diag_part(inputs)
        z = tf.math.log(tf.maximum(diags, self.epsilon))
        scale = tf.nn.softplus(self.s_raw)
        return scale * z + self.bias

    def get_config(self):
        config = super().get_config()
        config.update({
            "n_filters": self.n_filters,
            "epsilon": self.epsilon
        })
        return config
    
Npatt = 10 

idx_i, idx_j = np.triu_indices(n_components)
multipliers = np.where(idx_i == idx_j, 1.0, np.sqrt(2.0)).astype(np.float32)
X_cov_flat = (covmats_w[:, idx_i, idx_j] * multipliers).astype(np.float32)
input_dim = int(X_cov_flat.shape[1])

inputs_enc = tf.keras.Input(shape=(input_dim,), dtype=tf.float32)
x = UnflattenSymmetricLayer(M=n_components)(inputs_enc)
x = BiMapLayer(d_out=Npatt, orth_weight=0.01, name="bimap_1")(x)
z_latent = LogDiagScaleLayer(n_filters=Npatt, epsilon=1e-4)(x)
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
W_bimap = embedder.encoder.get_layer("bimap_1").get_weights()[0] 
P_umap = np.zeros((len(covmats_w), Npatt))
for i, C in enumerate(covmats_w):
    C_filtered = W_bimap @ C @ W_bimap.T
    P_umap[i] = np.diag(C_filtered)
    
W_filters_w = W_bimap.T 
C_avg_w = np.mean(covmats_w, axis=0)
A_patterns_w = C_avg_w @ W_filters_w
A_patterns = np.linalg.pinv(Ww.T) @ A_patterns_w 

W_filters = Ww @ W_filters_w

# %%
# =============================================================================
# ВИЗУАЛИЗАЦИЯ (ТОЛЬКО ВАЛИДНЫЕ ОШИБКИ)
# =============================================================================
import matplotlib.pyplot as plt

history = embedder._history

# Создаем фигуру с двумя подграфиками (для Recon и для UMAP)
fig, ax1 = plt.subplots(1, 1)

ax1.plot(history['umap_loss'], color='purple', linewidth=2)
    
ax1.set_xlabel('Шаги оценки')
ax1.set_ylabel('Cross-Entropy')

# %%
# =============================================================================
# ОБУЧЕНИЕ ОТОБРАЖЕНИЯ ДЛЯ КЛИКЕРА (20D Мощности <-> 2D Экран)
# =============================================================================
print("Обучение инверсной модели визуализации (20D -> 2D -> 20D)...")
from tensorflow.keras import regularizers

# Энкодер: сжимает 20D мощности в 2D для отрисовки на экране
encoder_2d = tf.keras.Sequential([
    tf.keras.layers.InputLayer(shape=(Npatt,)),
    tf.keras.layers.Dense(100, activation="elu"),
    tf.keras.layers.Dense(100, activation="elu"),
    tf.keras.layers.Dense(100, activation="elu"),
    tf.keras.layers.Dense(2, activation="linear", name="2d_coords")
])

# Декодер: с мягкой L2-регуляризацией для страховки от выбросов
decoder_2d = tf.keras.Sequential([
    tf.keras.layers.InputLayer(shape=(2,)),
    tf.keras.layers.Dense(100, activation="elu", kernel_regularizer=regularizers.l2(1e-4)),
    tf.keras.layers.Dense(100, activation="elu", kernel_regularizer=regularizers.l2(1e-4)),
    tf.keras.layers.Dense(100, activation="elu", kernel_regularizer=regularizers.l2(1e-4)),
    tf.keras.layers.Dense(Npatt, activation="linear", name="z_reconstruction")
])

reducer_2d_nn = ParametricUMAP(
    encoder=encoder_2d,
    decoder=decoder_2d,
    n_components=2,
    parametric_reconstruction=True, 
    parametric_reconstruction_loss_fcn=tf.keras.losses.MeanSquaredError(),
    # =========================
    verbose=True
)

# Обучаем визуальное пространство
pred = encoder.predict(X_cov_flat)
umap_coords = reducer_2d_nn.fit_transform(pred)
print("Визуальное пространство обучено!")

# %%
# 2. Визуализация
plt.figure(figsize=(8, 6))
plt.scatter(umap_coords[:, 0], umap_coords[:, 1], alpha=0.6, edgecolors='w', s=30)
plt.title('Визуализация данных')
plt.xlabel('Главная компонента 1')
plt.ylabel('Главная компонента 2')
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

# %%
# =============================================================================
# 1. ИЗВЛЕЧЕНИЕ ВЕСОВ ФИЛЬТРОВ ИЗ МОДЕЛИ
# =============================================================================
powers = np.zeros((len(covmats), Npatt))
for i, C in enumerate(covmats):
    C_filtered = W_filters.T @ C @ W_filters
    powers[i] = np.diag(C_filtered)

found_filters = W_filters
found_patterns = A_patterns

# %%
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
# Предположу, что mne и raw у тебя уже импортированы/загружены
# import mne

def format_umap_axes(ax):
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(axis='both', which='both', length=0)
    ax.grid(True, linestyle='--', alpha=0.5, zorder=0)
# =============================================================================
# ВЫБОР КОМПОНЕНТЫ
# =============================================================================
# Выбираем индекс паттерна (0..N_patterns-1)
comp_idx = 0   

# Берём фильтр, который ВЫУЧИЛА НЕЙРОСЕТЬ (из матрицы W)
W_sensor = found_filters[:,comp_idx] 
# Берём соответствующий ему паттерн (из матрицы A = W^+)
A_pattern = found_patterns[:,comp_idx]

# 1. Мощность выбранного выученного источника (латентная переменная сети)
p_vals = powers[:, comp_idx]

# 2. Мощность через ручное применение пространственного фильтра (w^T * C * w)
# ВАЖНО: используем исходные ковариации (covmats), чтобы проверить честность сети
filtered_powers = np.array([W_sensor.T @ C @ W_sensor for C in covmats])

# 3. Нормализация (делим на std, без вычета среднего, чтобы сохранить масштаб)
p_vals_norm = p_vals / np.std(p_vals)
filtered_powers_norm = filtered_powers / np.std(filtered_powers)

# =============================================================================
# ПОДГОТОВКА ЦВЕТОВ И ЛЕЙБЛОВ
# =============================================================================
unique_labels_ordered = []
for lab in window_labels:
    if lab not in unique_labels_ordered:
        unique_labels_ordered.append(lab)

cmap = mpl.colormaps['viridis']
label_to_color = {lab: cmap(i / len(unique_labels_ordered)) for i, lab in enumerate(unique_labels_ordered)}

# =============================================================================
# ОТРИСОВКА
# =============================================================================
fig = plt.figure(figsize=(16, 10))
gs = GridSpec(2, 2, figure=fig, width_ratios=[1, 2], height_ratios=[1, 1])

# --- 1. ТОПОГРАФИЯ ФИЛЬТРА (W) ---
ax_filt = fig.add_subplot(gs[0, 0])
mne.viz.plot_topomap(W_sensor, raw.info, axes=ax_filt, show=False)
ax_filt.set_title('Пространственный фильтр')

# --- 2. ТОПОГРАФИЯ ПАТТЕРНА (A) ---
ax_patt = fig.add_subplot(gs[1, 0])
mne.viz.plot_topomap(A_pattern, raw.info, axes=ax_patt, show=False)
ax_patt.set_title('Паттерн источника')

import matplotlib.lines as mlines

# --- 3. UMAP ПРОЕКЦИЯ ---
ax_umap = fig.add_subplot(gs[0, 1])

# Вычисляем робастные границы (5-й и 95-й перцентили)
vmin_val = np.percentile(p_vals_norm, 5)
vmax_val = np.percentile(p_vals_norm, 95)

sc1 = ax_umap.scatter(
    umap_coords[:, 0], 
    umap_coords[:, 1], 
    c=p_vals_norm, 
    cmap='plasma', 
    s=15, 
    zorder=2,
    vmin=vmin_val,
    vmax=vmax_val   
)
plt.colorbar(sc1, ax=ax_umap, label='Мощность источника (scaled)', extend='both')
ax_umap.set_title(f'UMAP проекция\nЦвет: мощность компоненты {comp_idx+1}')
format_umap_axes(ax_umap)

# === ДОБАВЛЕНИЕ ЦЕНТРОИДОВ И ЛЕГЕНДЫ ===
legend_handles = []

for i, lab in enumerate(unique_labels_ordered):
    mask = (window_labels == lab)
    if not np.any(mask):
        continue
        
    # Находим точки текущего условия и их центр (медиану)
    pts = umap_coords[mask]
    cx, cy = np.median(pts, axis=0)
    
    # Берем цвет условия из словаря
    color = label_to_color[lab]
    
    # Отрисовываем цифру в кружочке на графике
    ax_umap.text(
        cx, cy,
        str(i + 1),
        fontsize=11,
        fontweight='bold',
        color='black',
        ha='center',
        va='center',
        bbox=dict(boxstyle='circle,pad=0.25', facecolor='white', edgecolor=color, lw=2, alpha=0.9),
        zorder=5
    )
    
    # Создаем маркер для легенды (кружочек с цветной обводкой)
    handle = mlines.Line2D(
        [], [], color='white', marker='o', markersize=9, 
        markerfacecolor='white', markeredgecolor=color, markeredgewidth=2,
        label=f'[{i + 1}] {lab}'
    )
    legend_handles.append(handle)

# Добавляем саму легенду
# Располагаем ее в левом верхнем углу (чтобы не перекрывать colorbar справа)
ax_umap.legend(handles=legend_handles, fontsize=9, framealpha=0.9)

# --- 4. ВРЕМЕННЫЕ РЯДЫ МОЩНОСТИ ---
ax_env = fig.add_subplot(gs[1, 1])
window_idx = np.arange(len(p_vals_norm))

# Отрисовка обеих мощностей для сравнения
# Если они идут "ноздря в ноздрю", значит модель выучила честный фильтр!
ax_env.plot(window_idx, filtered_powers_norm, color='gray', lw=2.5, alpha=0.5, label='Математический W-фильтр ($w^T C w$)', zorder=2)
ax_env.plot(window_idx, p_vals_norm, color='black', lw=1.2, label='Латентная мощность от сети ($z$)', zorder=3)

# Отрисовка фона (событий)
xtick_positions = []
xtick_labels = []
for lab in unique_labels_ordered:
    mask = (window_labels == lab)
    if not np.any(mask):
        continue
    changes = np.diff(np.concatenate(([0], mask.astype(int), [0])))
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]
    for s, e in zip(starts, ends):
        ax_env.axvspan(s, e-1, facecolor=label_to_color[lab], alpha=0.2, zorder=0)
        ax_env.axvline(x=s, color='gray', linestyle='--', alpha=0.7, zorder=1)
        xtick_positions.append(s)
        xtick_labels.append(lab)

ax_env.set_xticks(xtick_positions)
ax_env.set_xticklabels(xtick_labels, rotation=45, ha='center', fontsize=8)
ax_env.set_xlabel('Номер окна')
ax_env.set_ylabel('Мощность (scaled by std)')
ax_env.set_title(f'Динамика мощности компоненты {comp_idx+1}')
ax_env.grid(True, axis='y', linestyle=':', alpha=0.6, zorder=0)
ax_env.legend(loc='upper right')

plt.show()

# %%
# =============================================================================
# ВЫДЕЛЕНИЕ И ОТРИСОВКА RAW ДЛЯ НАЙДЕННОЙ КОМПОНЕНТЫ
# =============================================================================
import mne
import numpy as np

# 1. Извлекаем данные только тех каналов, на которых обучался фильтр (EEG)
raw_eeg = raw.copy().pick(picks='eeg') # Используем современный метод вместо pick_types
data_eeg = raw_eeg.get_data()  # Размерность: (n_channels, n_times)

# 2. Применяем пространственный фильтр (умножаем вектор весов на матрицу данных)
source_signal = np.dot(found_filters.T, data_eeg)

# 3. Создаем метаданные (Info) для нового Raw
ch_name = [f'Comp_{i+1}' for i in range(10)]
info_source = mne.create_info(
    ch_names=ch_name,
    sfreq=raw.info['sfreq'],
    ch_types=['eeg' for i in range(10)] 
)
# Решение ошибки: копируем абсолютное время начала записи из оригинального raw
info_source.set_meas_date(raw.info['meas_date'])

# 4. Создаем объект RawArray
# Решение возможного сдвига: передаем first_samp, если исходный файл начинался не с нуля
raw_source = mne.io.RawArray(source_signal, info_source, first_samp=raw.first_samp)

# 5. Переносим аннотации из исходного raw
raw_source.set_annotations(raw.annotations)

# %%
raw_source.plot()

# %%
# =============================================================================
# ПОСТРОЕНИЕ СПЕКТРОВ (PSD) ВСТРОЕННЫМИ МЕТОДАМИ MNE
# =============================================================================

# 1. Нарезаем сигнал компонент на эпохи, автоматически пропуская 'BAD_'
epochs_source = mne.make_fixed_length_epochs(
    raw_source, 
    duration=2.0, 
    overlap=0.0, 
    preload=True, 
    reject_by_annotation=True,
    verbose=False
)

# 2. Вычисляем спектр (PSD) методом Уэлча
spectrum = epochs_source.compute_psd(
    method='welch', 
    fmin=1, 
    fmax=40,            
    n_fft=int(raw.info['sfreq'] * 2), 
    verbose=False
)

# 3. Отрисовка встроенным методом MNE
fig = spectrum.plot(
    average=False,         # Рисовать каждую компоненту отдельно (а не среднее по всем)
    spatial_colors=False,  # Отключить раскраску по координатам (у нас их нет)
    amplitude=False        # False = рисовать в дБ (мощность), True = линейно
)

# %%
# %%
# =============================================================================
# ЧАСТОТНО-ВРЕМЕННОЙ ГРАФИК (СПЕКТРОГРАММА) ДЛЯ ВЫБРАННОЙ КОМПОНЕНТЫ
# =============================================================================
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
import numpy as np

# 1. Выбираем компоненту для отрисовки (от 0 до 9)
comp_to_plot = 7   # Соответствует Comp_10
ch_name = f'Comp_{comp_to_plot + 1}'

# 2. Извлекаем непрерывный сигнал этой компоненты
sfreq = raw_source.info['sfreq']
data = raw_source.get_data(picks=ch_name)[0]

# 3. Считаем спектрограмму (скользящее окно)
# Используем окно 2 секунды с перекрытием 1.5 секунды для плавности
nperseg = int(sfreq * 2)
noverlap = int(sfreq * 1.5)

freqs, times, Sxx = spectrogram(data, fs=sfreq, nperseg=nperseg, noverlap=noverlap)

# 4. Ограничиваем частоты (например, до 40 Гц, чтобы не смотреть на шум)
freq_mask = freqs <= 40
freqs = freqs[freq_mask]
Sxx = Sxx[freq_mask, :]

# Переводим мощность в децибелы (логарифмический масштаб)
Sxx_db = 10 * np.log10(Sxx)

# 5. Отрисовка
fig, ax = plt.subplots(figsize=(16, 6))

# Рисуем саму спектрограмму (shading='gouraud' делает её гладкой)
im = ax.pcolormesh(times, freqs, Sxx_db, shading='gouraud', cmap='viridis')

ax.set_ylabel('Частота (Гц)', fontsize=12)
ax.set_xlabel('Время записи (секунды)', fontsize=12)
ax.set_title(f'Спектрограмма (Continuous TFR) для {ch_name}', fontsize=14, fontweight='bold')
ax.set_ylim(1, 40)

# Добавляем цветовую шкалу
cbar = fig.colorbar(im, ax=ax, pad=0.01)
cbar.set_label('Мощность (дБ)', rotation=270, labelpad=15)

# 6. Накладываем вертикальные линии границ условий (аннотаций)
# Список валидных условий (берем из вашего верхнего кода)
cond_descriptions = [
    'rest/EC/1', 'rest/EO/1',
    'ABA_1', 'ACA_1', 
    'ABA_2', 'ACA_2',
    'rest/EO/2', 'rest/EC/2'
]

for annot in raw_source.annotations:
    desc = annot['description']
    onset = annot['onset']
    
    # Проверяем, есть ли текущая метка в списке разрешенных условий
    if desc in cond_descriptions:
        ax.axvline(x=onset, color='white', linestyle='--', alpha=0.6, lw=1.5)
        # Подписываем условие прямо на графике (наверху)
        ax.text(onset, 39, f' {desc}', color='white', fontsize=9, 
                rotation=90, va='top', ha='left', fontweight='bold',
                bbox=dict(facecolor='black', alpha=0.3, edgecolor='none', boxstyle='round,pad=0.1'))
        
plt.tight_layout()
plt.show()

# %%
# %%
# =============================================================================
# ЧАСТОТНО-ВРЕМЕННОЙ ГРАФИК (СПЕКТРОГРАММА) ДЛЯ ВЫБРАННОЙ КОМПОНЕНТЫ
# =============================================================================
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
import numpy as np

# 1. Выбираем компоненту для отрисовки (от 0 до 9)
comp_to_plot = 8       # Соответствует Comp_7
ch_name = f'Comp_{comp_to_plot + 1}'

# 2. Извлекаем непрерывный сигнал этой компоненты
sfreq = raw_source.info['sfreq']
data = raw_source.get_data(picks=ch_name)[0]

# 3. Считаем спектрограмму (скользящее окно)
# Используем окно 2 секунды с перекрытием 1.5 секунды для плавности
nperseg = int(sfreq * 2)
noverlap = int(sfreq * 1.5)

freqs, times, Sxx = spectrogram(data, fs=sfreq, nperseg=nperseg, noverlap=noverlap)

# 4. Ограничиваем частоты (например, до 40 Гц)
freq_mask = freqs <= 40
freqs = freqs[freq_mask]
Sxx = Sxx[freq_mask, :]

# Переводим мощность в децибелы (логарифмический масштаб)
Sxx_db = 10 * np.log10(Sxx)

# =============================================================================
# НОРМИРОВКА ВО ВРЕМЕНИ ДЛЯ КАЖДОЙ ЧАСТОТЫ (Z-SCORE)
# =============================================================================
# Считаем среднее и стандартное отклонение для каждой частоты (вдоль оси времени, axis=1)
mean_per_freq = np.mean(Sxx_db, axis=1, keepdims=True)
std_per_freq = np.std(Sxx_db, axis=1, keepdims=True)

# Вычитаем среднее и делим на стандартное отклонение
# Добавляем 1e-8, чтобы избежать деления на 0, если сигнал константный
Sxx_norm = (Sxx_db - mean_per_freq) / (std_per_freq + 1e-8)

# 5. Отрисовка
fig, ax = plt.subplots(figsize=(16, 6))

# Рисуем нормированную спектрограмму
# vmin и vmax = -3 и 3 ограничивают выбросы (±3 стандартных отклонения)
# cmap='RdBu_r' отлично подходит для отклонений (синий - падение мощности, красный - рост)
im = ax.pcolormesh(
    times, freqs, Sxx_norm, 
    shading='gouraud', 
    cmap='RdBu_r', 
    vmin=-3, vmax=3
)

ax.set_ylabel('Частота (Гц)', fontsize=12)
ax.set_xlabel('Время записи (секунды)', fontsize=12)
ax.set_title(f'Нормированная спектрограмма (Z-score) для {ch_name}', fontsize=14, fontweight='bold')
ax.set_ylim(1, 40)

# Добавляем цветовую шкалу
cbar = fig.colorbar(im, ax=ax, pad=0.01)
cbar.set_label('Отклонение от среднего (Z)', rotation=270, labelpad=15)

# 6. Накладываем вертикальные линии границ условий (аннотаций)
cond_descriptions = [
    'rest/EC/1', 'rest/EO/1',
    'ABA_1', 'ACA_1', 
    'ABA_2', 'ACA_2',
    'rest/EO/2', 'rest/EC/2'
]

for annot in raw_source.annotations:
    desc = annot['description']
    onset = annot['onset']
    
    if desc in cond_descriptions:
        ax.axvline(x=onset, color='black', linestyle='--', alpha=0.5, lw=1.5)
        ax.text(onset, 39, f' {desc}', color='black', fontsize=9, 
                rotation=90, va='top', ha='left', fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', boxstyle='round,pad=0.1'))
        
plt.tight_layout()
plt.show()

# %%


