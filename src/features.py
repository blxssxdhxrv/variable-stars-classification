"""23 признака из кривой блеска (БПФ, статистика, автокорреляция)."""
import numpy as np
from scipy.signal import find_peaks
from .preprocess import N_FFT, preprocess

def _spec_stats(freqs, amps):
    tot = np.sum(amps)+1e-12; p = amps/tot
    ent = -np.sum(p*np.log(p+1e-12))
    cen = np.sum(freqs*p)
    wid = np.sqrt(np.sum(((freqs-cen)**2)*p))
    thr = np.mean(amps)+2.5*np.std(amps)
    pk, _ = find_peaks(amps, height=thr)
    return ent, cen, wid, float(len(pk))

def extract_features(t, mag):
    tu, y = preprocess(t, mag)
    dt = tu[1]-tu[0]
    f = np.fft.rfftfreq(N_FFT, d=dt)
    sp = np.abs(np.fft.rfft(y)); sp[0] = 0.0
    tf = np.zeros(3); ta = np.zeros(3); tw = np.zeros(3)
    if sp.size > 1:
        idx = np.argsort(sp)[::-1]
        for i in range(3):
            if i < idx.size:
                k = idx[i]; tf[i] = f[k]; ta[i] = sp[k]
                lo, hi = max(0, k-2), min(sp.size, k+3)
                loc = sp[lo:hi]; tw[i] = np.sum(loc)/(np.max(loc)+1e-12)
    ent, cen, wid, npeaks = _spec_stats(f, sp)
    yc = y - np.mean(y)
    ac = np.correlate(yc, yc, mode="full")[len(yc)-1:]
    ac = ac/(ac[0]+1e-12)
    pk, _ = find_peaks(ac[1:])
    if pk.size:
        lag = float(pk[0]+1); val = float(ac[pk[0]+1])
    else:
        lag, val = 0.0, 0.0
    mm = float(np.mean(mag)); sm = float(np.std(mag))
    skew = float(((mag-mm)**3).mean()/(sm**3+1e-12))
    kurt = float(((mag-mm)**4).mean()/(sm**4+1e-12))
    amp = float(np.max(mag)-np.min(mag))
    features = np.array([
        tf[0], tf[1], tf[2], ta[0], ta[1], ta[2], tw[0], tw[1], tw[2],
        mm, sm, skew, kurt, amp, ent, cen, wid, npeaks, lag, val,
        float(np.mean(sp)), float(np.std(sp)), float(np.max(sp))
    ], dtype=float)
    return features
