"""Self-check for compute_sti.py con senales sinteticas (sin WAVs reales)."""
import numpy as np

from compute_sti import calcular_STI_desde_ir, snr_ap_de_m

SR = 44100


def test_impulso_perfecto_sin_ruido_da_sti_alto():
    ir = np.zeros(SR)
    ir[0] = 1.0  # impulso puro -> MTF(F) = 1 para toda F
    rec = np.random.default_rng(0).normal(0, 1.0, SR)     # senal fuerte
    noise = np.random.default_rng(1).normal(0, 1e-4, SR)  # ruido muy bajo
    sti, _, _ = calcular_STI_desde_ir(ir, SR, rec, SR, noise, SR)
    assert sti > 0.9, f"esperaba STI alto con impulso perfecto y SNR alto, dio {sti}"


def test_reverb_larga_y_ruido_alto_da_sti_bajo():
    t = np.arange(SR * 2) / SR
    ir = np.exp(-t / 0.5) * np.random.default_rng(2).normal(0, 1, len(t))  # T60 ~3.5s, sala muy reverberante
    rng = np.random.default_rng(3)
    rec = rng.normal(0, 0.7, SR)
    noise = rng.normal(0, 1.2, SR)  # ruido mas fuerte que la senal -> SNR negativo
    sti, _, _ = calcular_STI_desde_ir(ir, SR, rec, SR, noise, SR)
    assert sti < 0.4, f"esperaba STI bajo con reverb larga y SNR negativo, dio {sti}"


def test_snr_ap_de_m_bordes():
    assert snr_ap_de_m(0.0) == -15.0
    assert snr_ap_de_m(1.0) == 15.0
    assert snr_ap_de_m(1.5) == 15.0
    assert -15 <= snr_ap_de_m(0.5) <= 15


if __name__ == "__main__":
    test_impulso_perfecto_sin_ruido_da_sti_alto()
    test_reverb_larga_y_ruido_alto_da_sti_bajo()
    test_snr_ap_de_m_bordes()
    print("OK")
