"""
Calcula tiempo de reverberacion (EDT, T20, T30) por banda de octava o de
tercio de octava, a partir de la integral inversa de Schroeder de la
respuesta al impulso (IR).

Metodo:
1. Filtrar la IR banda por banda con FilterBank (filterbank.py, IEC 61260-1).
2. Estimar el piso de ruido y el punto de cruce (crosspoint) entre la
   pendiente de decaimiento y el ruido con el algoritmo iterativo de
   Lundeby, Vigran, Bietz y Vorlander (1995), "Uncertainties of Measurements
   in Room Acoustics", Acustica 81 -- el mismo metodo (via ISO 3382-1 Anexo A)
   que usa Aurora, EASERA, Dirac, etc.
3. Integral de Schroeder hacia atras truncada en el crosspoint, compensando
   la contribucion lineal del ruido: E(t) = integral_t^Tc h^2(tau) dtau - (Tc-t)*N.
4. Regresion lineal de la curva de decaimiento (en dB) por tramos:
   - EDT:   0 a -10 dB   (pendiente x6 -> T60 equivalente)
   - T20:  -5 a -25 dB   (pendiente x3)
   - T30:  -5 a -35 dB   (pendiente x2)

Uso:
    python compute_tr.py IR.wav [--bands 1/1|1/3] [--fmin 100] [--fmax 5000]
"""
import argparse

import numpy as np
import soundfile as sf
from scipy.ndimage import uniform_filter1d

from filterbank import FilterBank


def to_mono(x):
    return x if x.ndim == 1 else x.mean(axis=1)


def _lundeby(e, sr, max_iter=15, min_rango_db=10.0, margen_db=10.0, ventana_suavizado_ms=20.0):
    """Estima el piso de ruido y el punto de cruce (en muestras) con el
    algoritmo iterativo de Lundeby et al. (1995).

    Sigue el esquema validado por Fernandez Ridano (TP10, IMA-UNTreF 2026):
    la busqueda del cruce corre sobre la energia suavizada con una media
    movil FIJA (~20ms, scipy.ndimage.uniform_filter1d) en vez de reblocking
    adaptativo -- evita que las fluctuaciones caoticas de alta frecuencia
    rompan la regresion y sobreestimen el ruido. El truncamiento y la resta
    de ruido se aplican despues sobre la energia CRUDA sin suavizar
    (schroeder_decay_db).

    ponytail: 20ms es una constante fija, no depende del ancho de banda de
    cada banda (se probo escalarla con el tiempo de correlacion ~1/ancho de
    banda, pero no mejoro los resultados contra Aurora en la practica --
    ver conversacion). Las bandas graves de este dataset probablemente
    fallan por comportamiento modal/no difuso de la sala, no por falta de
    promediado.

    Tambien ignora la cola de precision numerica (residuo de filtrado muy
    por debajo de peak*1e-12, no señal real) antes de buscar el cruce, igual
    que su implementacion.

    Devuelve (None, None) -- banda invalida -- si no hay al menos
    min_rango_db entre el pico y el ruido, o si nunca converge a un cruce
    real dentro del buffer. Mismo criterio que usa Aurora para mostrar "--".
    """
    n = len(e)
    pico = float(np.max(e))
    if pico <= 0:
        return None, None

    piso_numerico = pico * 1e-12
    activos = np.where(e > piso_numerico)[0]
    ultimo_activo = int(activos[-1]) if len(activos) > 0 else n - 1
    if ultimo_activo < int(0.5 * sr):
        ultimo_activo = n - 1  # cola "activa" sospechosamente corta: usar todo el buffer

    window = max(3, int(round(ventana_suavizado_ms / 1000.0 * sr)))
    e_smooth = uniform_filter1d(e[:ultimo_activo + 1], size=window, mode="nearest")

    db = 10 * np.log10(np.maximum(e_smooth, 1e-300) / pico)
    t = np.arange(len(e_smooth)) / sr
    pico_idx = int(np.argmax(db))

    n_tail = max(1, len(e_smooth) // 10)
    noise = float(np.mean(e_smooth[-n_tail:]))
    if noise <= 0 or 10 * np.log10(pico / noise) < min_rango_db:
        return None, None
    noise_db = 10 * np.log10(noise / pico)

    t_cross = t[-1]
    t_cross_libre = None  # crosspoint SIN recortar al buffer, para juzgar si el recorte es razonable
    for _ in range(max_iter):
        candidatos = np.where(db[pico_idx:] < noise_db + margen_db)[0]
        if len(candidatos) == 0:
            break
        idx_cruce = pico_idx + candidatos[0]
        if (idx_cruce - pico_idx) < int(0.01 * sr):
            break

        t_seg, db_seg = t[pico_idx:idx_cruce], db[pico_idx:idx_cruce]
        if len(t_seg) < 2:
            break
        pendiente, intercept = np.polyfit(t_seg, db_seg, 1)
        if pendiente >= 0:
            break

        nuevo_t_cross_libre = (noise_db - intercept) / pendiente
        if not np.isfinite(nuevo_t_cross_libre) or nuevo_t_cross_libre < 0:
            break
        nuevo_t_cross = min(nuevo_t_cross_libre, t[-1])

        convergio = abs(nuevo_t_cross - t_cross) < 0.002
        t_cross, t_cross_libre = nuevo_t_cross, nuevo_t_cross_libre

        i_desde = min(int(np.searchsorted(t, t_cross + 0.05)), len(e_smooth) - 10)
        if i_desde < len(e_smooth) - 10:
            nuevo_noise = float(np.mean(e_smooth[i_desde:]))
            if nuevo_noise > 0:
                noise, noise_db = nuevo_noise, 10 * np.log10(nuevo_noise / pico)

        if convergio:
            break

    # Tolerancia: un cruce extrapolado apenas mas alla del buffer disponible
    # (p.ej. exportaciones cortadas justo antes de que la cola termine de
    # morir) no es un fallo del algoritmo -- se usa el buffer completo como
    # truncamiento. Solo se rechaza si nunca convergio (t_cross_libre=None)
    # o si el cruce libre quedo groseramente mas alla del buffer (ruido
    # degenerado, ver el caso de 16-20kHz con residuo numerico de filtrado).
    margen_tolerable = max(0.05 * t[-1], 0.05)
    if t_cross_libre is None or t_cross_libre > t[-1] + margen_tolerable:
        return None, None

    return noise, int(round(t_cross * sr))


def schroeder_decay_db(h, sr, ventana_suavizado_ms=20.0):
    """Curva de decaimiento en dB: integral de Schroeder truncada en el
    crosspoint de Lundeby, compensada por el piso de ruido estimado ahi mismo.
    """
    e_full = h.astype(np.float64) ** 2
    noise_power, n_cross = _lundeby(e_full, sr, ventana_suavizado_ms=ventana_suavizado_ms)
    if noise_power is None:
        return None, None, float("nan"), float("nan")
    n_cross = int(np.clip(n_cross, 2, len(e_full)))
    e = e_full[:n_cross]

    e_raw = np.cumsum(e[::-1])[::-1]
    muestras_restantes = n_cross - np.arange(n_cross)
    e_corr = e_raw - muestras_restantes * noise_power

    if e_corr[0] <= 0:
        return None, None, noise_power, n_cross / sr
    valido = e_corr > 0
    ultimo = int(np.argmax(~valido)) if not np.all(valido) else n_cross
    if ultimo < 2:
        return None, None, noise_power, n_cross / sr

    e_corr = e_corr[:ultimo]
    t = np.arange(ultimo) / sr
    l_db = 10 * np.log10(e_corr / e_corr[0])
    return t, l_db, noise_power, n_cross / sr


def _regresion_t60(t, l_db, db_hi, db_lo):
    """Ajusta una recta a l_db en [db_lo, db_hi] (ambos <= 0, db_hi > db_lo) y devuelve el T60 equivalente."""
    mask = (l_db <= db_hi) & (l_db >= db_lo)
    if mask.sum() < 2:
        return float("nan")
    pendiente, _ = np.polyfit(t[mask], l_db[mask], 1)
    if pendiente >= 0:
        return float("nan")
    return -60.0 / pendiente


def edt_t20_t30(t, l_db):
    if t is None:
        return float("nan"), float("nan"), float("nan")
    edt = _regresion_t60(t, l_db, 0, -10)
    t20 = _regresion_t60(t, l_db, -5, -25)
    t30 = _regresion_t60(t, l_db, -5, -35)
    return edt, t20, t30


def _recortar_padding_final(ir, margen_ratio=1e-9):
    """Recorta el silencio/padding al final de la señal cruda de banda ancha
    (p.ej. exportaciones de simulacion que rellenan a una duracion fija).

    Hace falta hacerlo ACA, antes de filtrar por banda: el filtrado de octava
    hace que el ringing del filtro siga sonando un poco despues de que la
    señal real termina, asi que buscar el silencio DESPUES de filtrar no lo
    detecta de forma confiable -- ese padding queda invisible como "señal"
    y contamina la reestimacion de ruido de Lundeby en cada banda (el
    piso de ruido sale mas bajo de lo real, estirando el punto de cruce mas
    alla del buffer).
    """
    pico = np.max(np.abs(ir))
    if pico <= 0:
        return ir
    activos = np.where(np.abs(ir) > pico * margen_ratio)[0]
    if len(activos) == 0:
        return ir
    return ir[:activos[-1] + 1]


def calcular_tr(ir, sr, bands="1/1", fmin=100, fmax=5000):
    ir = _recortar_padding_final(ir)
    fb = FilterBank(sr=sr, bands=bands, fmin=fmin, fmax=fmax)
    ir_bands = fb.filter_bands(ir)

    resultados = {}
    for f_nom in fb.center_freqs_nominal:
        h, sr_band = ir_bands[f_nom]
        t, l_db, noise_power, t_cross = schroeder_decay_db(h, sr_band)
        edt, t20, t30 = edt_t20_t30(t, l_db)
        noise_db = 10 * np.log10(noise_power) if (noise_power == noise_power and noise_power > 0) else float("nan")
        resultados[f_nom] = {"EDT": edt, "T20": t20, "T30": t30, "Noise_dB": noise_db, "t_cross_s": t_cross}
    return resultados


def main():
    ap = argparse.ArgumentParser(description="TR (EDT, T20, T30) por banda via integral de Schroeder")
    ap.add_argument("ir", help="WAV de la IR (salida de compute_ir.py)")
    ap.add_argument("--bands", default="1/1", choices=["1/1", "octave", "1/3"],
                     help="Resolucion de banda (default: octava)")
    ap.add_argument("--fmin", type=float, default=100)
    ap.add_argument("--fmax", type=float, default=5000)
    args = ap.parse_args()

    ir, sr = sf.read(args.ir, always_2d=False)
    ir = to_mono(ir)

    resultados = calcular_tr(ir, sr, bands=args.bands, fmin=args.fmin, fmax=args.fmax)

    print(f"\n{'Banda (Hz)':>10}  {'EDT (s)':>8}  {'T20 (s)':>8}  {'T30 (s)':>8}  {'Ruido':>8}  {'Cross (s)':>9}")
    for f_nom in sorted(resultados):
        r = resultados[f_nom]
        print(f"{f_nom:>10}  {r['EDT']:>8.2f}  {r['T20']:>8.2f}  {r['T30']:>8.2f}  "
              f"{r['Noise_dB']:>6.1f}dB  {r['t_cross_s']:>9.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
