"""Self-check para compute_tr.py con IR sintetica de T60 conocido."""
import numpy as np

from compute_tr import calcular_tr, edt_t20_t30, schroeder_decay_db

SR = 44100


def _ir_sintetica(t60, dur=3.0, sr=SR, seed=0):
    n = int(dur * sr)
    t = np.arange(n) / sr
    tau = t60 / (3 * np.log(10))  # T60 = 3*ln(10)*tau
    rng = np.random.default_rng(seed)
    return np.exp(-t / tau) * rng.normal(0, 1, n)


def test_t30_y_edt_cercanos_al_t60_real():
    t60_real = 1.2
    ir = _ir_sintetica(t60_real)
    t, l_db, noise_power, t_cross = schroeder_decay_db(ir, SR)
    edt, t20, t30 = edt_t20_t30(t, l_db)
    assert abs(t30 - t60_real) < 0.1, f"T30 {t30} lejos del T60 real {t60_real}"
    assert abs(t20 - t60_real) < 0.1, f"T20 {t20} lejos del T60 real {t60_real}"
    assert abs(edt - t60_real) < 0.15, f"EDT {edt} lejos del T60 real {t60_real}"


def test_calcular_tr_por_banda_de_octava():
    ir = _ir_sintetica(0.8)
    resultados = calcular_tr(ir, SR, bands="1/1", fmin=250, fmax=2000)
    assert len(resultados) >= 3
    for f_nom, r in resultados.items():
        assert not np.isnan(r["T30"]), f"T30 invalido en banda {f_nom}"
        assert 0.5 < r["T30"] < 1.2, f"T30 {r['T30']} fuera de rango esperado en banda {f_nom}"


if __name__ == "__main__":
    test_t30_y_edt_cercanos_al_t60_real()
    test_calcular_tr_por_banda_de_octava()
    print("OK")
