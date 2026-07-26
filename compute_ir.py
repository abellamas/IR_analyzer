"""
Calcula respuestas al impulso (IR) por deconvolucion de Farina (ESS).

Metodo: IR = grabacion (*) filtro_inverso   [convolucion lineal]
El filtro inverso es el sweep exponencial invertido en tiempo y compensado en
amplitud; al convolucionarlo con la grabacion, la IR lineal aparece con su pico
en un unico punto y los productos de distorsion armonica quedan ANTES del pico,
por lo que se descartan recortando desde (pico - preroll) hacia adelante.

Uso:
    python compute_ir.py FILTRO_INVERSO INPUT_DIR [-o OUTDIR] [opciones]

Ejemplo:
    python compute_ir.py audios/filtro_inverso_160_6000_48kHz_24bits.wav audios

Levanta TODOS los .wav del directorio (recursivo, salvo el propio filtro y lo que
ya este en OUTDIR), calcula la IR de cada uno y las guarda espejando la estructura
de subcarpetas (posicion_1, sala_vacia, etc.) dentro de OUTDIR.

Eficiencia: FFT real (rfft/irfft) multihilo; la FFT del filtro se calcula UNA vez
por cada longitud de bloque N y se cachea, asi todos los audios del mismo largo
reutilizan la misma transformada.
"""
import argparse
import os
import sys
import numpy as np
import soundfile as sf
from scipy.fft import rfft, irfft, next_fast_len
from scipy.ndimage import uniform_filter1d


def to_mono(x):
    return x if x.ndim == 1 else x.mean(axis=1)


class Deconvolver:
    """Deconvoluciona por FFT cacheando la transformada del filtro por longitud N."""

    def __init__(self, inv_filter, workers=-1):
        self.f = inv_filter.astype(np.float64)
        self.lf = len(self.f)
        self.workers = workers
        self._cache = {}  # N -> rfft(filtro, N)

    def _filter_fft(self, N):
        F = self._cache.get(N)
        if F is None:
            F = rfft(self.f, n=N, workers=self.workers)
            self._cache[N] = F
        return F

    def __call__(self, rec):
        L = len(rec) + self.lf - 1              # largo de la convolucion lineal
        N = next_fast_len(L)                    # padded a un largo rapido para FFT
        F = self._filter_fft(N)
        R = rfft(rec.astype(np.float64), n=N, workers=self.workers)
        y = irfft(R * F, n=N, workers=self.workers)[:L]
        return y


def _windowed_kurtosis(x, window, step):
    starts = np.arange(0, max(1, len(x) - window + 1), step, dtype=int)
    kur = np.full(len(starts), np.nan, dtype=np.float64)
    for i, s in enumerate(starts):
        w = x[s:s + window].astype(np.float64)
        m = np.mean(w)
        v = np.mean((w - m) ** 2)
        if v <= 0:
            continue
        kur[i] = np.mean((w - m) ** 4) / (v * v) - 3.0
    return starts, kur


def _detect_noise_start_by_kurtosis(full, sr, search_start, window_ms=80.0, step_ms=20.0):
    h = full[search_start:]
    if len(h) == 0:
        return None

    # Use short-term energy for kurtosis, which is more stable on decaying IR tails.
    h = h.astype(np.float64) ** 2
    window = max(1, int(round(window_ms * 1e-3 * sr)))
    step = max(1, int(round(step_ms * 1e-3 * sr)))
    starts, kur = _windowed_kurtosis(h, window, step)
    if len(kur) == 0:
        return None

    kur = np.nan_to_num(kur, nan=0.0)
    kur = uniform_filter1d(kur, size=5, mode="nearest")
    peak = np.nanmax(kur)
    if not np.isfinite(peak) or peak <= 0:
        return None

    threshold = max(1.0, peak * 0.20)
    sustained = 5
    for i in range(len(kur) - sustained + 1):
        if np.all(kur[i:i + sustained] <= threshold):
            return search_start + int(starts[i])
    return None


def _detect_ir_end_by_noise(full, sr, start, peak=None, window_ms=20.0, tail_fraction=0.1, margin_db=10.0):
    h = full[start:]
    if len(h) == 0:
        return len(full)

    e = h.astype(np.float64) ** 2
    window = max(3, int(round(window_ms * 1e-3 * sr)))
    e_smooth = uniform_filter1d(e, size=window, mode="nearest")

    pico = float(np.max(e_smooth))
    if pico <= 0:
        return len(full)

    n = len(e_smooth)
    tail_len = max(int(round(0.5 * sr)), int(round(n * tail_fraction)))
    tail_start = max(0, n - tail_len)
    tail_segment = e_smooth[tail_start:]
    if len(tail_segment) == 0:
        return len(full)

    noise = float(np.percentile(tail_segment, 20.0))
    if noise <= 0:
        return len(full)

    threshold = noise * (10.0 ** (margin_db / 10.0))
    below = e_smooth <= threshold
    sustained = max(window, int(round(0.05 * sr)))
    if peak is not None and peak >= start:
        search_offset = peak - start
    else:
        search_offset = 0

    for i in range(search_offset, n - sustained + 1):
        if np.all(below[i:i + sustained]):
            return min(start + i, len(full))

    return len(full)


def _detect_ir_end_by_kurtosis(full, sr, start, peak, noise_margin_s=3.0):
    search_start = max(start, peak)
    noise_start = _detect_noise_start_by_kurtosis(full, sr, search_start)
    if noise_start is None:
        return len(full)
    end = noise_start + int(round(noise_margin_s * sr))
    return min(len(full), end)


def extract_ir(full, sr, preroll_ms, ir_seconds, start_mode="filter", end_mode="fixed", inv_filter_len=None):
    peak = int(np.argmax(np.abs(full)))
    if start_mode == "full":
        start = 0
    elif start_mode == "filter":
        start = min(max(0, int(inv_filter_len or 0)), len(full))
    else:
        start = max(0, peak - int(round(preroll_ms * 1e-3 * sr)))

    if end_mode == "full":
        end = len(full)
    elif end_mode == "kurtosis":
        end = _detect_ir_end_by_kurtosis(full, sr, start, peak)
    else:
        if ir_seconds is None:
            end = len(full)
        else:
            if start_mode == "filter":
                end = min(len(full), peak + int(round(ir_seconds * sr)))
            else:
                end = min(len(full), start + int(round(ir_seconds * sr)))

    return full[start:end], start, peak


def find_wavs(input_dir, filter_path, outdir):
    filt_abs = os.path.abspath(filter_path)
    out_abs = os.path.abspath(outdir)
    for root, _dirs, files in os.walk(input_dir):
        if os.path.abspath(root).startswith(out_abs):
            continue
        for name in sorted(files):
            if not name.lower().endswith(".wav"):
                continue
            p = os.path.join(root, name)
            if os.path.abspath(p) == filt_abs:
                continue
            yield p


def main():
    ap = argparse.ArgumentParser(description="Deconvolucion de Farina: IR = grabacion (*) filtro_inverso")
    ap.add_argument("filtro", help="Ruta al .wav del filtro inverso (sweep invertido)")
    ap.add_argument("input_dir", help="Directorio con las grabaciones (recursivo)")
    ap.add_argument("-o", "--outdir", default=None,
                    help="Directorio de salida (default: <input_dir>/IR)")
    ap.add_argument("--ir-seconds", type=float, default=5.0,
                    help="Largo de la IR a guardar en segundos cuando --end-mode fixed (default 5; 0 = completa)")
    ap.add_argument("--end-mode", choices=["fixed", "kurtosis", "full"], default="fixed",
                    help="Cómo recortar el final de la IR: fijo por segundos, auto por kurtosis, o completa")
    ap.add_argument("--preroll-ms", type=float, default=5.0,
                    help="Milisegundos a conservar antes del pico cuando no se usa filter-length-onset (default 5). Si se usa filter-length-onset, la IR empieza exactamente al final del filtro.")
    ap.add_argument("--filter-length-onset", action="store_true",
                    help="Iniciar la IR a partir del final del filtro inverso en lugar del pico.")
    ap.add_argument("--no-normalize", action="store_true",
                    help="No normalizar; conserva la escala cruda de la deconvolucion")
    ap.add_argument("--subtype", default="FLOAT", choices=["FLOAT", "PCM_24", "PCM_16"],
                    help="Formato del .wav de salida (default FLOAT)")
    args = ap.parse_args()

    f, sr = sf.read(args.filtro, always_2d=False)
    f = to_mono(f)
    outdir = args.outdir or os.path.join(args.input_dir, "IR")
    os.makedirs(outdir, exist_ok=True)

    dec = Deconvolver(f)
    if args.end_mode == "fixed":
        ir_seconds = None if args.ir_seconds == 0 else args.ir_seconds
    else:
        ir_seconds = None

    files = list(find_wavs(args.input_dir, args.filtro, outdir))
    if not files:
        print("No se encontraron .wav para procesar.", file=sys.stderr)
        return 1

    print(f"Filtro: {os.path.basename(args.filtro)}  ({len(f)} muestras, {sr} Hz)")
    print(f"Salida: {outdir}\n{len(files)} archivos a procesar\n")

    for i, path in enumerate(files, 1):
        rec, sr_r = sf.read(path, always_2d=False)
        if sr_r != sr:
            print(f"  [SKIP] {path}: sr {sr_r} != filtro {sr}", file=sys.stderr)
            continue
        rec = to_mono(rec)
        full = dec(rec)
        ir, start, peak = extract_ir(
            full,
            sr,
            args.preroll_ms,
            ir_seconds,
            start_mode="filter" if args.filter_length_onset else "peak",
            end_mode=args.end_mode,
            inv_filter_len=dec.lf if args.filter_length_onset else None,
        )

        if not args.no_normalize:
            mx = np.max(np.abs(ir))
            if mx > 0:
                ir = ir / mx * 0.5

        rel = os.path.relpath(path, args.input_dir)
        stem, _ = os.path.splitext(rel)
        out_path = os.path.join(outdir, stem + "_IR.wav")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        sf.write(out_path, ir.astype(np.float32), sr, subtype=args.subtype)
        print(f"  [{i:>3}/{len(files)}] {rel}  ->  inicio@{start/sr:.3f}s  pico@{peak/sr:.3f}s  IR={len(ir)/sr:.2f}s")

    print("\nListo.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
