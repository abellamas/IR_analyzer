"""
Calcula el Speech Transmission Index (STI) a partir de una respuesta al
impulso (IR) real medida por deconvolucion de Farina (ver compute_ir.py),
mas una grabacion de la senal de prueba y una grabacion del piso de ruido,
ambas en el punto receptor.

Metodo: indirecto (el que usa Aurora), IEC 60268-16. Para cada banda de
octava se filtra la IR, se calcula la MTF(F) directamente desde la
envolvente de energia h^2(t) (resultado de Schroeder) para las 14
frecuencias de modulacion, se corrige por el SNR medido (grabacion vs.
piso de ruido, en esa banda), se normaliza cada (banda, F) a un indice de
transmision TI en [0, 1] y se promedia dentro de cada banda -> MTI_k (el
"Band STI" que muestra Aurora). El STI final combina las 7 bandas con los
pesos alpha de IEC 60268-16 (voz masculina, ed. 4/5) menos un termino beta
que resta la redundancia entre bandas adyacentes.

No incluye la correccion opcional por enmascaramiento auditivo (seccion 7
del documento de referencia) — su impacto es chico salvo con mucha energia
en graves; agregar si hace falta validar contra Aurora con Masking activado.

Uso:
    python compute_sti.py IR.wav GRABACION.wav RUIDO.wav

ir: salida de compute_ir.py (o cualquier IR ya deconvolucionada).
rec: grabacion IN SITU del sweep (la misma que usaste como entrada de
     compute_ir.py) en el punto receptor -- contiene senal+ruido mezclados,
     no el sweep seco. Misma sesion/microfono/ganancia que el archivo de
     ruido, para que los niveles relativos en dBFS sean comparables.
noise: grabacion del piso de ruido (silencio) en el mismo punto receptor.
"""
import argparse

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfiltfilt

BANDAS_OCTAVA = [125, 250, 500, 1000, 2000, 4000, 8000]
FRECUENCIAS_MODULACION = [0.63, 0.8, 1.0, 1.25, 1.6, 2.0, 2.5, 3.15, 4.0, 5.0, 6.25, 8.0, 10.0, 12.5]

# IEC 60268-16 tabla A.1, voz masculina (ed. 4/5). alpha: peso por banda.
# beta: resta de redundancia con la banda siguiente (no hay beta despues de 8000 Hz).
ALPHA = {125: 0.085, 250: 0.127, 500: 0.230, 1000: 0.233, 2000: 0.309, 4000: 0.224, 8000: 0.173}
BETA = {125: 0.085, 250: 0.078, 500: 0.065, 1000: 0.011, 2000: 0.047, 4000: 0.095}


def to_mono(x):
    return x if x.ndim == 1 else x.mean(axis=1)


def octave_band_filter(x, sr, fc, order=4):
    nyq = sr / 2
    lo, hi = fc / np.sqrt(2), min(fc * np.sqrt(2), nyq * 0.999)
    if lo >= hi:
        return np.zeros_like(x, dtype=np.float64)  # banda por encima del Nyquist a este sample rate
    sos = butter(order, [lo / nyq, hi / nyq], btype="band", output="sos")
    return sosfiltfilt(sos, x)


def band_level_db(x, sr, fc):
    """Nivel RMS en dBFS de x filtrado en la banda de octava fc."""
    rms = np.sqrt(np.mean(octave_band_filter(x, sr, fc) ** 2))
    return 20 * np.log10(rms) if rms > 0 else -np.inf


def snr_medido_db(rec, noise, sr, fc):
    """SNR real por banda: la grabacion in situ contiene senal+ruido mezclados,
    asi que se resta la energia del ruido (no los dB) para aislar el nivel de
    senal pura antes de compararlo contra el piso de ruido (IEC 60268-16, nota
    de la seccion 3: Sig = (Sig+N) - Noise, ruido no correlacionado)."""
    l_sig_mas_n = band_level_db(rec, sr, fc)
    l_noise = band_level_db(noise, sr, fc)
    e_sig = 10 ** (l_sig_mas_n / 10) - 10 ** (l_noise / 10)
    if e_sig <= 0:
        return -15.0  # ruido >= grabacion completa en esta banda: SNR minimo util
    l_sig = 10 * np.log10(e_sig)
    return l_sig - l_noise


def mtf_from_ir(ir_band, sr):
    """MTF(F) (14 frecuencias de modulacion) desde la envolvente h^2(t) de una IR ya filtrada en una banda."""
    h2 = ir_band.astype(np.float64) ** 2
    t = np.arange(len(h2)) / sr
    denom = np.trapz(h2, t)
    if denom <= 0:
        return {F: 0.0 for F in FRECUENCIAS_MODULACION}
    return {
        F: float(np.abs(np.trapz(h2 * np.exp(-2j * np.pi * F * t), t)) / denom)
        for F in FRECUENCIAS_MODULACION
    }


def snr_ap_de_m(m_total):
    if m_total >= 1:
        return 15.0
    if m_total <= 0:
        return -15.0
    return float(np.clip(10 * np.log10(m_total / (1 - m_total)), -15, 15))


def calcular_STI_desde_ir(ir, sr_ir, rec, sr_rec, noise, sr_noise):
    if sr_ir != sr_rec or sr_ir != sr_noise:
        raise ValueError(f"Sample rates distintos: ir={sr_ir}, rec={sr_rec}, noise={sr_noise}")
    sr = sr_ir

    mti_banda = {}
    detalle = {}
    for fc in BANDAS_OCTAVA:
        mtf = mtf_from_ir(octave_band_filter(ir, sr, fc), sr)
        snr_medido = snr_medido_db(rec, noise, sr, fc)

        ti_values = []
        for F in FRECUENCIAS_MODULACION:
            m_corregido = mtf[F] * (1 / (1 + 10 ** (-snr_medido / 10)))
            snr_app = snr_ap_de_m(m_corregido)
            ti_values.append((snr_app + 15) / 30)

        mti_banda[fc] = float(np.mean(ti_values))  # "Band STI" (0-1), como en Aurora
        detalle[fc] = {"mtf": mtf, "snr_medido": snr_medido}

    suma_alpha = sum(ALPHA[fc] * mti_banda[fc] for fc in BANDAS_OCTAVA)
    suma_beta = sum(
        BETA[fc_lo] * np.sqrt(mti_banda[fc_lo] * mti_banda[fc_hi])
        for fc_lo, fc_hi in zip(BANDAS_OCTAVA[:-1], BANDAS_OCTAVA[1:])
    )
    sti = float(np.clip(suma_alpha - suma_beta, 0, 1))
    return sti, mti_banda, detalle


def interpretar(sti):
    if sti < 0.30:
        return "Mala"
    if sti < 0.45:
        return "Pobre"
    if sti < 0.60:
        return "Aceptable"
    if sti < 0.75:
        return "Buena"
    return "Excelente"


def main():
    ap = argparse.ArgumentParser(description="STI desde IR real (Farina) + grabacion de senal + piso de ruido")
    ap.add_argument("ir", help="WAV de la IR (salida de compute_ir.py)")
    ap.add_argument("rec", help="WAV de la grabacion de la senal de prueba en el punto receptor")
    ap.add_argument("noise", help="WAV del piso de ruido en el punto receptor")
    args = ap.parse_args()

    ir, sr_ir = sf.read(args.ir, always_2d=False)
    rec, sr_rec = sf.read(args.rec, always_2d=False)
    noise, sr_noise = sf.read(args.noise, always_2d=False)

    sti, mti_banda, detalle = calcular_STI_desde_ir(
        to_mono(ir), sr_ir, to_mono(rec), sr_rec, to_mono(noise), sr_noise
    )

    print(f"STI = {sti:.3f}  ({interpretar(sti)})\n")
    print(f"{'Banda (Hz)':>10}  {'SNR medido':>11}  {'Band STI (MTI)':>15}")
    for fc in BANDAS_OCTAVA:
        print(f"{fc:>10}  {detalle[fc]['snr_medido']:>9.1f} dB  {mti_banda[fc]:>15.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
