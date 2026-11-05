"""Предобработка: сортировка, выбросы 3σ, интерполяция N=1024, центрирование, окно Ханна."""
import numpy as np

N_FFT = 1024

def remove_outliers(t, mag, k=3.0):
    mu, s = np.mean(mag), np.std(mag)
    m = np.abs(mag-mu) < k*s
    return t[m], mag[m]

def interpolate_uniform(t, mag, n=N_FFT):
    o = np.argsort(t); t, mag = t[o], mag[o]
    tu = np.linspace(t[0], t[-1], n); mu = np.interp(tu, t, mag)
    return tu, mu

def hann_window(n):
    return 0.5 - 0.5*np.cos(2*np.pi*np.arange(n)/(n-1))

def preprocess(t, mag, n=N_FFT):
    t, mag = remove_outliers(t, mag)
    tu, mu = interpolate_uniform(t, mag, n)
    y = mu - np.mean(mu)
    return tu, y*hann_window(n)
