import numpy as np
import math
from compute_tr import _lundeby, schroeder_decay_db
SR = 44100

def ir_sintetica(t60, dur=3.0, sr=SR, seed=0):
    n = int(dur * sr)
    t = np.arange(n) / sr
    tau = t60 / (3 * math.log(10))
    rng = np.random.default_rng(seed)
    return np.exp(-t / tau) * rng.normal(0, 1, n)

ir = ir_sintetica(1.2)
e = ir ** 2
noise, n_cross = _lundeby(e, SR)
print('noise', noise, 'n_cross', n_cross)
print('noise dBFS', 10 * math.log10(noise) if noise and noise > 0 else 'nan')
t, l_db, noise_power, t_cross = schroeder_decay_db(ir, SR)
print('schroeder noise', noise_power, 't_cross', t_cross)
print('first l_db', l_db[:5] if l_db is not None else None)
