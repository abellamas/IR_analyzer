import numpy as np
import math
from compute_tr import _lundeby
SR = 44100

def ir_sintetica(t60, dur=3.0, sr=SR, seed=0):
    n = int(dur * sr)
    t = np.arange(n) / sr
    tau = t60 / (3 * math.log(10))
    rng = np.random.default_rng(seed)
    return np.exp(-t / tau) * rng.normal(0, 1, n)

ir = ir_sintetica(1.2)
e = ir ** 2

pico = float(np.max(e))
piso_numerico = pico * 1e-12
activos = np.where(e > piso_numerico)[0]
ultimo_activo = int(activos[-1]) if len(activos) > 0 else len(e) - 1
if ultimo_activo < int(0.5 * SR):
    ultimo_activo = len(e) - 1
window = max(3, int(round(20.0 / 1000.0 * SR)))
e_smooth = np.convolve(e[:ultimo_activo+1], np.ones(window)/window, mode='same')
db = 10 * np.log10(np.maximum(e_smooth, 1e-300))
t = np.arange(len(e_smooth)) / SR
pico_idx = int(np.argmax(db))
n_tail = max(1, len(e_smooth) // 10)
noise = float(np.mean(e_smooth[-n_tail:]))
print('pico', pico, 'pico_db', db[pico_idx], 'noise', noise, 'noise_db', 10*math.log10(noise), 'range', db[pico_idx]-10*math.log10(noise))
print('pico_idx', pico_idx, 'len', len(e_smooth), 't[-1]', t[-1])
first_cand = np.where(db[pico_idx:] < 10*math.log10(noise) + 10)[0]
print('first cand after peak', first_cand[0] if len(first_cand) else None)

noise2, n_cross = _lundeby(e, SR)
print('result', noise2, n_cross)
