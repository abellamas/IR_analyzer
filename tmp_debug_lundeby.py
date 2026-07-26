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

# replicate _lundeby steps manually
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
print('db tail first', db[-10:])

# debug loop similar to _lundeby
noise_iter = noise
noise_db_iter = 10 * math.log10(max(noise, 1e-300))
for it in range(15):
    candidates = np.where(db[pico_idx:] < noise_db_iter + 10)[0]
    print('iter', it, 'noise_db_iter', noise_db_iter, 'candidates_len', len(candidates))
    if len(candidates) == 0:
        print(' no candidates -> break')
        break
    idx_cruce = pico_idx + candidates[0]
    print(' idx_cruce', idx_cruce, 'time', t[idx_cruce])
    if (idx_cruce - pico_idx) < int(0.01 * SR):
        print(' break short segment')
        break
    t_seg = t[pico_idx:idx_cruce]
    db_seg = db[pico_idx:idx_cruce]
    pendiente, intercept = np.polyfit(t_seg, db_seg, 1)
    print('  pendiente', pendiente, 'intercept', intercept)
    if pendiente >= 0:
        print(' break non-negative slope')
        break
    nuevo_t_cross_libre = (noise_db_iter - intercept) / pendiente
    print('  nuevo_t_cross_libre', nuevo_t_cross_libre)
    if not np.isfinite(nuevo_t_cross_libre) or nuevo_t_cross_libre < 0:
        print(' break invalid cross')
        break
    nuevo_t_cross = min(nuevo_t_cross_libre, t[-1])
    print('  nuevo_t_cross', nuevo_t_cross)
    convergio = abs(nuevo_t_cross - t[-1]) < 0.002
    i_desde = min(int(np.searchsorted(t, nuevo_t_cross + 0.05)), len(e) - 10)
    print('  i_desde', i_desde, 'time', t[i_desde])
    if i_desde < len(e) - 10:
        nuevo_noise = float(np.mean(e[i_desde:]))
        print('  nuevo_noise', nuevo_noise, 'nuevo_noise_db', 10 * math.log10(max(nuevo_noise, 1e-300)))
        if nuevo_noise > 0:
            noise_iter = nuevo_noise
            noise_db_iter = 10 * math.log10(max(nuevo_noise, 1e-300))
    if convergio:
        print(' converged')
        break

print('final noise', noise_iter, 'noise_db', noise_db_iter)
noise2, n_cross = _lundeby(e, SR)
print('result', noise2, n_cross)
