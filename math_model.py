# -*- coding: utf-8 -*-
"""
Параметрическая модель Риманова расстояния для rank-1 возмущения.
Две панели:
1) Само расстояние d_R(P) для правой ветви (P >= P_base).
2) Аналитическая производная d_R/dP со сдвигом по оси X к 0.
"""

import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# 1. АНАЛИТИЧЕСКИЕ ФУНКЦИИ
# =============================================================================
def riemann_rank1_model(P, P_base, epsilon=0.75):
    """
    Аналитическая модель расстояния для правой ветви (P >= P_base).
    """
    return np.log(P + epsilon) - np.log(P_base + epsilon)

def riemann_rank1_deriv(P, epsilon=0.75):
    """
    Аналитическая производная: d/dP [ln(P + eps) - const] = 1 / (P + eps)
    """
    return 1.0 / (P + epsilon)

# =============================================================================
# 2. ПАРАМЕТРЫ
# =============================================================================
P_base_list = [0.0, 0.1, 1.0, 2.0, 4.0]
max_power = 200.0
epsilon = 0.05

# =============================================================================
# 3. ВИЗУАЛИЗАЦИЯ (ДВЕ ПАНЕЛИ)
# =============================================================================
fig, (ax_top, ax_bot) = plt.subplots(
    2, 1, figsize=(11, 12), sharex=False,
    gridspec_kw={'height_ratios': [1, 1]}
)

colors = plt.cm.tab10(np.linspace(0, 1, len(P_base_list)))

for idx, pb in enumerate(P_base_list):
    color = colors[idx]
    label_str = f"$P_{{base}} = {pb}$"

    # Генерируем сетку только для P >= P_base
    P_use = np.linspace(pb, max_power, 500)
    x_rel = P_use - pb  # Сдвиг по X к нулю для нижней панели

    # Расчет значений
    d_vals = riemann_rank1_model(P_use, pb, epsilon=epsilon)
    deriv_vals = riemann_rank1_deriv(P_use, epsilon=epsilon)

    # ---------- ВЕРХНЯЯ ПАНЕЛЬ: Расстояние ----------
    ax_top.plot(P_use, d_vals, color=color, linewidth=2.5, label=label_str)
    ax_top.axvline(x=pb, color=color, linestyle='--', alpha=0.5)
    ax_top.plot(P_use[0], d_vals[0], 'o', color=color, markersize=6, zorder=5)

    # ---------- НИЖНЯЯ ПАНЕЛЬ: Производная ----------
    ax_bot.plot(x_rel, deriv_vals, color=color, linewidth=2.5, label=label_str)
    ax_bot.plot(x_rel[0], deriv_vals[0], 'o', color=color, markersize=6, zorder=5)


# --- Оформление верхней панели ---
ax_top.axhline(0, color='black', linewidth=1.2, zorder=1)
ax_top.set_title("Риманово расстояние: $d_R(P) = \\ln(P + \\epsilon) - \\ln(P_{base} + \\epsilon)$, $P \\geq P_{base}$", fontsize=13)
ax_top.set_xlabel("Целевая мощность источника ($P$)", fontsize=11)
ax_top.set_ylabel("Риманово расстояние", fontsize=11)
ax_top.grid(True, linestyle=':', alpha=0.7)
ax_top.set_xlim(0, max_power)
ax_top.set_ylim(bottom=0)
ax_top.legend(fontsize=10, loc="upper left")

# --- Оформление нижней панели ---
ax_bot.axhline(0, color='black', linewidth=1.2, zorder=1)
ax_bot.axvline(0, color='black', linewidth=1.0, alpha=0.6, zorder=1)
ax_bot.set_xlabel("$P - P_{base}$", fontsize=11)
ax_bot.set_ylabel("$\\frac{d d_R}{dP} = \\frac{1}{P + \\epsilon}$", fontsize=13)
ax_bot.grid(True, linestyle=':', alpha=0.7)
ax_bot.set_xlim(0, max_power - min(P_base_list))
# Верхняя граница графика производной определяется максимальным значением при P_base = 0
ax_bot.set_ylim(0, (1.0 / epsilon) * 1.05)
ax_bot.legend(fontsize=10, loc="upper right")

plt.suptitle(f"Аналитическая модель ($\\epsilon = {epsilon}$)", fontsize=15, y=0.98)
plt.tight_layout()
plt.show()