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
from scipy.signal import hilbert

from filterbank import FilterBank


DBFS_REF = 1.0


def to_mono(x):
    return x if x.ndim == 1 else x.mean(axis=1)


def _band_envelope(h, sr, ventana_suavizado_ms=20.0):
    """Compute the analytic envelope of a bandpassed IR and smooth it."""
    analytic = hilbert(h)
    envelope = np.abs(analytic)
    window = max(3, int(round(ventana_suavizado_ms / 1000.0 * sr)))
    return uniform_filter1d(envelope, size=window, mode="nearest")


def _band_plot_data(h, sr, smooth_lengths=(501, 2000)):
    """Return time and dBFS curves for a bandpass signal's envelope."""
    analytic = hilbert(h.astype(np.float64))
    envelope = np.abs(analytic)
    if envelope.size == 0:
        return {"t": np.array([]), "raw_db": np.array([]), "smooth_db": [], "labels": []}

    t = np.arange(envelope.size) / sr
    raw_db = 20 * np.log10(np.maximum(envelope / DBFS_REF, 1e-12))

    smooth_db = []
    labels = []
    for M in smooth_lengths:
        window = max(3, int(round(M)))
        smoothed = uniform_filter1d(envelope, size=window, mode="nearest")
        smooth_db.append(20 * np.log10(np.maximum(smoothed / DBFS_REF, 1e-12)))
        labels.append(f"E(t) M={M}")

    return {"t": t, "raw_db": raw_db, "smooth_db": smooth_db, "labels": labels}


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

    db = 10 * np.log10(np.maximum(e_smooth, 1e-300))
    t = np.arange(len(e_smooth)) / sr
    pico_idx = int(np.argmax(db))

    n_tail = max(1, len(e_smooth) // 10)
    noise = float(np.mean(e_smooth[-n_tail:]))
    if noise <= 0 or db[pico_idx] - 10 * np.log10(np.maximum(noise, 1e-300)) < min_rango_db:
        return None, None
    noise_db = 10 * np.log10(np.maximum(noise, 1e-300))

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

        i_desde = min(int(np.searchsorted(t, t_cross + 0.05)), len(e) - 10)
        if i_desde < len(e) - 10:
            nuevo_noise = float(np.mean(e[i_desde:]))
            if nuevo_noise > 0:
                noise, noise_db = nuevo_noise, 10 * np.log10(np.maximum(nuevo_noise, 1e-300))

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

    n_cross = int(round(t_cross * sr))
    if n_cross >= len(e):
        n_cross = len(e) - 1

    tail_offset = int(round(0.05 * sr))
    tail_start = min(n_cross + tail_offset, len(e))
    if tail_start < len(e):
        final_noise = float(np.mean(e[tail_start:]))
        if final_noise > 0:
            noise = final_noise

    return noise, n_cross


def schroeder_decay_db(h, sr, ventana_suavizado_ms=20.0, noise_correction=True):
    """Curva de decaimiento en dB: integral de Schroeder truncada en el
    crosspoint de Lundeby, compensada por el piso de ruido estimado ahi mismo.

    Usa la envolvente de Hilbert sobre la respuesta banda para seguir el
    flujo de envelope-based RT recomendado en MATLAB/ISO, y luego integra
    la energia de la envolvente cruda al estilo Schroeder.

    En Matlab equivalente:
        hA = abs(hilbert(h));
        E = cumsum(hA(td:-1:1).^2);
        L = 10*log10(E / E(1));

    Aquí agregamos corrección de ruido opcional:
        E_corr(t) = E(t) - (Tc - t) * N
    """
    analytic = hilbert(h.astype(np.float64))
    hA = np.abs(analytic)
    e_full = hA ** 2
    noise_power, n_cross = _lundeby(e_full, sr, ventana_suavizado_ms=ventana_suavizado_ms)
    if noise_power is None:
        return None, None, float("nan"), float("nan")
    n_cross = int(np.clip(n_cross, 2, len(e_full)))
    e = e_full[:n_cross]

    e_raw = np.cumsum(e[::-1])[::-1]
    if noise_correction:
        muestras_restantes = n_cross - np.arange(n_cross)
        e_corr = e_raw - muestras_restantes * noise_power
    else:
        e_corr = e_raw

    if e_corr[0] <= 0:
        return None, None, noise_power, n_cross / sr
    valido = e_corr > 0
    ultimo = int(np.argmax(~valido)) if not np.all(valido) else n_cross
    if ultimo < 2:
        return None, None, noise_power, n_cross / sr

    e_corr = e_corr[:ultimo]
    t = np.arange(ultimo) / sr
    l_db_abs = 10 * np.log10(np.maximum(e_corr / (DBFS_REF ** 2), 1e-300))
    l_db_rel = l_db_abs - l_db_abs[0]
    return t, l_db_rel, l_db_abs, noise_power, n_cross / sr


def _regresion_t60_line(t, l_db, db_hi, db_lo):
    """Ajusta una recta a l_db en [db_lo, db_hi] y devuelve el T60, tiempos y valores de la recta."""
    mask = (l_db <= db_hi) & (l_db >= db_lo)
    if mask.sum() < 2:
        return float("nan"), np.array([]), np.array([])
    pendiente, intercept = np.polyfit(t[mask], l_db[mask], 1)
    if pendiente >= 0:
        return float("nan"), np.array([]), np.array([])
    t_line = t[mask]
    y_line = pendiente * t_line + intercept
    return -60.0 / pendiente, t_line, y_line


def edt_t20_t30(t, l_db):
    if t is None:
        return float("nan"), float("nan"), float("nan"), np.array([]), np.array([]), np.array([]), np.array([])
    edt, _, _ = _regresion_t60_line(t, l_db, 0, -10)
    t20, t20_t, t20_y = _regresion_t60_line(t, l_db, -5, -25)
    t30, t30_t, t30_y = _regresion_t60_line(t, l_db, -5, -35)
    return edt, t20, t30, t20_t, t20_y, t30_t, t30_y


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


def calcular_tr(ir, sr, bands="1/1", fmin=100, fmax=5000, noise_correction=True):
    ir = _recortar_padding_final(ir)
    fb = FilterBank(sr=sr, bands=bands, fmin=fmin, fmax=fmax)
    ir_bands = fb.filter_bands(ir)

    resultados = {}
    for f_nom in fb.center_freqs_nominal:
        h, sr_band = ir_bands[f_nom]
        t, l_db_rel, l_db_abs, noise_power, t_cross = schroeder_decay_db(
            h, sr_band, noise_correction=noise_correction)
        edt, t20, t30, t20_t, t20_y_rel, t30_t, t30_y_rel = edt_t20_t30(t, l_db_rel)
        curve = _band_plot_data(h, sr_band)
        noise_db = 10 * np.log10(noise_power) if (noise_power == noise_power and noise_power > 0) else float("nan")
        t20_y_abs = t20_y_rel + (l_db_abs[0] if t20_y_rel.size > 0 else 0.0)
        t30_y_abs = t30_y_rel + (l_db_abs[0] if t30_y_rel.size > 0 else 0.0)
        resultados[f_nom] = {
            "EDT": edt,
            "T20": t20,
            "T30": t30,
            "Noise_dB": noise_db,
            "t_cross_s": t_cross,
            "curve": curve,
            "schroeder": {
                "t": t,
                "l_db_rel": l_db_rel,
                "l_db_abs": l_db_abs,
                "noise_power": noise_power,
                "noise_db": noise_db,
                "t_cross_s": t_cross,
                "t20_fit_t": t20_t,
                "t20_fit_y": t20_y_abs,
                "t30_fit_t": t30_t,
                "t30_fit_y": t30_y_abs,
            },
        }
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
