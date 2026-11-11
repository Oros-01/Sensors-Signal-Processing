# %% [markdown]
# # Анализ гироскопа (Microstrain 3DM-GX3-45) из датасета NCLT
#
# **Студенты:** Макаров Артём, Мирзоев Сергей, Орос Тимофей (МАИ).
#
# **Конфигурация робота:** Body frame (X — вперёд, Y — вправо, Z — вниз).
# В работе анализируется гироскоп Microstrain 3DM-GX3-45 по всем трём осям:
# * **Ось X (Roll)** — крен влево/вправо.
# * **Ось Y (Pitch)** — наклон вперёд/назад (балансировка, удары, торможения).
# * **Ось Z (Yaw)** — рыскание (руление, повороты).
#
# **Дата записи:** 10 января 2013. По документации NCLT — Snow: Yes.
# Именно поэтому в сыром сигнале так много вибраций и ударов.

# %% [markdown]
# ## 1. Импорты и настройки

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import detrend

DATA = Path("data/sen/ms25.csv")
PLOTS = Path("plots")
PLOTS.mkdir(exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 150,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 10,
})

# %% [markdown]
# ## 2. Загрузка данных

# %%
COLS = [
    "t_us",
    "mag_x", "mag_y", "mag_z",
    "acc_x", "acc_y", "acc_z",
    "gx", "gy", "gz",
]
df = pd.read_csv(DATA, header=None, names=COLS)

# Время в секундах от начала записи
t = (df["t_us"].to_numpy(dtype=np.float64) - float(df["t_us"].iloc[0])) / 1e6

# Реальная частота дискретизации (медиана устойчива к пропускам)
dt = np.diff(t)
fs = 1.0 / float(np.median(dt))
duration = float(t[-1] - t[0])

print(f"Отсчётов: {len(t)} | fs ≈ {fs:.2f} Гц | Длительность: {duration:.1f} с")


# %% [markdown]
# ## 3. Функции для обработки и визуализации

# %%
def analyze_axis(raw_sig: np.ndarray, t: np.ndarray, axis_name: str, threshold: float = 3.5):
    """Очистка сигнала, поиск аномалий. Возвращает очищенный сигнал, параметры и DataFrame."""
    # 1. Убираем bias и линейный тренд
    bias = float(raw_sig.mean())
    clean_sig = detrend(raw_sig - bias, type="linear")

    # 2. Считаем СКО и Z-оценки
    sigma = float(clean_sig.std(ddof=1))
    z_scores = clean_sig / sigma

    # 3. Ищем аномалии
    mask = np.abs(z_scores) > threshold
    anomalies = pd.DataFrame({
        "t": t[mask],
        "raw": raw_sig[mask],
        "detrended": clean_sig[mask],
        "z": z_scores[mask],
        "level": np.abs(z_scores[mask])
    })

    return clean_sig, bias, sigma, mask, anomalies


def plot_time_domain(t: np.ndarray, clean_sig: np.ndarray, mask: np.ndarray,
                     sigma: float, display_name: str, color: str, filename: str):
    """Строит график очищенного сигнала и аномалий."""
    fig, ax = plt.subplots(figsize=(12, 4))

    # Только очищенный сигнал (сырой убран, чтобы не перекрывать график)
    ax.plot(t, clean_sig, lw=0.4, color=color, label="Очищенный сигнал")

    # Коридор ±3.5 сигм
    ax.axhline(+3.5 * sigma, ls="--", color="tab:orange", lw=1.2, label="Порог ±3.5σ")
    ax.axhline(-3.5 * sigma, ls="--", color="tab:orange", lw=1.2)

    # Точки аномалий
    ax.scatter(t[mask], clean_sig[mask], s=20, color="tab:red", zorder=3,
               label=f"Аномалии ({mask.sum()})")

    ax.set_xlabel("Время, с")
    ax.set_ylabel(f"{display_name}, рад/с")
    ax.set_title(f"Гироскоп ({display_name}): очищенный сигнал и аномалии")
    ax.legend(loc="upper right", fontsize=9)

    fig.tight_layout()
    fig.savefig(PLOTS / filename)
    plt.close(fig)


def plot_spectrum(clean_sig: np.ndarray, fs: float, display_name: str, color: str, filename: str):
    """Строит спектр (линейный и в дБ) с окном Ханна."""
    N = len(clean_sig)
    window = np.hanning(N)
    spectrum = np.fft.rfft(clean_sig * window)
    freqs = np.fft.rfftfreq(N, d=1.0 / fs)

    # Амплитуда с учётом когерентного усиления окна
    amplitude = np.abs(spectrum) / float(window.sum())
    amplitude_db = 20.0 * np.log10(np.maximum(amplitude, 1e-12))

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    # Линейный масштаб
    axes[0].plot(freqs, amplitude, lw=0.8, color=color)
    axes[0].set_xlim(0, 25)
    axes[0].set_xlabel("Частота, Гц")
    axes[0].set_ylabel("Амплитуда, рад/с")
    axes[0].set_title(f"Линейный масштаб ({display_name})")

    # Логарифмический масштаб (дБ)
    axes[1].plot(freqs, amplitude_db, lw=0.8, color="tab:purple")
    axes[1].set_xlim(0, 25)
    finite = amplitude_db[np.isfinite(amplitude_db)]
    axes[1].set_ylim(finite.min() - 5, finite.max() + 5)
    axes[1].set_xlabel("Частота, Гц")
    axes[1].set_ylabel("Амплитуда, дБ")
    axes[1].set_title(f"Логарифмический масштаб ({display_name})")

    fig.tight_layout()
    fig.savefig(PLOTS / filename)
    plt.close(fig)


# %% [markdown]
# ## 4. Цикл обработки по всем осям

# %%
# Список кортежей: (имя колонки, отображаемое имя, цвет, суффикс для файлов)
axes_config = [
    ("gx", "Roll (ось X)", "tab:red", "x"),
    ("gy", "Pitch (ось Y)", "tab:green", "y"),
    ("gz", "Yaw (ось Z)", "tab:blue", "z"),
]

for col_name, display_name, color, suffix in axes_config:
    print(f"\n--- Анализ {display_name} ---")
    raw_sig = df[col_name].to_numpy(dtype=np.float64)

    # Анализ
    clean_sig, bias, sigma, mask, anomalies = analyze_axis(raw_sig, t, display_name)

    # Сохранение CSV
    csv_name = f"anomalies_{suffix}.csv"
    anomalies.to_csv(csv_name, index=False)

    print(f"Смещение нуля (bias): {bias:+.4f} рад/с")
    print(f"СКО: {sigma:.4f} рад/с")
    print(f"Аномалий (|z| > 3.5): {len(anomalies)}")
    print(f"Сохранено: {csv_name}")

    # Построение графиков
    plot_time_domain(
        t, clean_sig, mask, sigma,
        display_name, color, f"time_{suffix}.png"
    )
    plot_spectrum(clean_sig, fs, display_name, color, f"spectrum_{suffix}.png")
    print(f"Сохранены графики: time_{suffix}.png, spectrum_{suffix}.png")

# %% [markdown]
# ## 5. Сравнение аномалий по осям (для отчета)

# %%
# Загружаем сохраненные CSV для удобства
anom_x = pd.read_csv("anomalies_x.csv")
anom_y = pd.read_csv("anomalies_y.csv")
anom_z = pd.read_csv("anomalies_z.csv")

fig, ax = plt.subplots(figsize=(12, 3))
ax.scatter(anom_x["t"], np.ones_like(anom_x["t"]) * 1, s=20, color="tab:red", label="Ось X (Roll)")
ax.scatter(anom_y["t"], np.ones_like(anom_y["t"]) * 2, s=20, color="tab:green", label="Ось Y (Pitch)")
ax.scatter(anom_z["t"], np.ones_like(anom_z["t"]) * 3, s=20, color="tab:blue", label="Ось Z (Yaw)")

ax.set_yticks([1, 2, 3])
ax.set_yticklabels(["X (Roll)", "Y (Pitch)", "Z (Yaw)"])
ax.set_xlabel("Время, с")
ax.set_title("Совпадение аномалий (ударов) по осям гироскопа")
ax.legend(loc="upper right", fontsize=9)
ax.set_ylim(0.5, 3.5)
ax.grid(alpha=0.5, axis="x")

fig.tight_layout()
fig.savefig(PLOTS / "anomalies_timeline.png")
plt.close(fig)
print(f"\nСохранено: {PLOTS / 'anomalies_timeline.png'}")

# %%
print("\nАнализ всех осей завершен успешно.")
