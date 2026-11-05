"""Генерация синтетических кривых блеска для 6 классов переменных звёзд."""
import numpy as np

CLASSES = ["cepheid", "eclipsing_binary", "rr_lyrae", "mira", "delta_scuti", "solar_like"]

def _time_grid(n=300, span=50.0, jitter=True):
    """Создаёт неравномерную временную сетку (имитация реальных наблюдений)."""
    t = np.sort(np.random.uniform(0, span, n))
    if jitter:
        t += np.random.normal(0, 0.01, size=n)
        t = np.sort(t)
    return t

def _cepheid(t):
    p = np.random.uniform(3, 15)       # период, дни
    a = np.random.uniform(0.3, 1.2)    # амплитуда
    ph = np.random.uniform(0, 1)       # фаза
    s = np.sin(2 * np.pi * (t / p + ph))
    asy = 0.15 * np.sin(4 * np.pi * (t / p + ph))
    return -a * (s + asy) + 10.0

def _eclipsing(t):
    p = np.random.uniform(1, 10)
    d1 = np.random.uniform(0.3, 1.0)   # глубина главного минимума
    d2 = np.random.uniform(0.1, 0.6)   # глубина вторичного
    ph0 = np.random.uniform(0, 1)
    ph = (t / p + ph0) % 1.0
    # Периодическая гауссиана: учитываем заворот через край
    e1 = -d1 * np.exp(-((np.minimum(ph, 1 - ph)) ** 2) / (2 * 0.02 ** 2))
    e2 = -d2 * np.exp(-((ph - 0.5) ** 2) / (2 * 0.02 ** 2))
    return 10.0 + e1 + e2

def _rr(t):
    p = np.random.uniform(0.3, 0.9)
    a = np.random.uniform(0.4, 1.2)
    ph = np.random.uniform(0, 1)
    ph2 = (t / p + ph) % 1.0
    return 10.0 - a * (np.sin(2 * np.pi * ph2) + 0.4 * np.sin(4 * np.pi * ph2))

def _mira(t):
    p = np.random.uniform(150, 400)
    a = np.random.uniform(1.5, 5.0)
    ph = np.random.uniform(0, 1)
    return 10.0 - a * np.sin(2 * np.pi * (t / p + ph))

def _dscuti(t):
    p = np.random.uniform(0.05, 0.3)
    a = np.random.uniform(0.01, 0.1)
    ph = np.random.uniform(0, 1)
    return 10.0 - a * np.sin(2 * np.pi * (t / p + ph))

def _solar(t):
    return 10.0 + 0.005 * np.sin(2 * np.pi * t / 13.0) + np.random.normal(0, 0.002, size=t.size)

_GEN = {
    "cepheid": _cepheid,
    "eclipsing_binary": _eclipsing,
    "rr_lyrae": _rr,
    "mira": _mira,
    "delta_scuti": _dscuti,
    "solar_like": _solar,
}

def generate_curve(cls, noise_sigma=0.02):
    """Генерирует одну кривую заданного класса."""
    if cls not in _GEN:
        raise ValueError(f"Неизвестный класс: {cls}")
    t = _time_grid()
    m = _GEN[cls](t)
    m += np.random.normal(0, noise_sigma, size=t.size)
    return t, m

def generate_dataset(n_per_class=200, noise_sigma=0.02):
    """
    Генерирует датасет: по n_per_class кривых на каждый из 6 классов.
    Возвращает список словарей {'cls': ..., 't': array, 'mag': array}.
    """
    dataset = []
    for c in CLASSES:
        for _ in range(n_per_class):
            t, mag = generate_curve(c, noise_sigma)  # генерируем ОДИН раз
            dataset.append({"cls": c, "t": t, "mag": mag})
    return dataset