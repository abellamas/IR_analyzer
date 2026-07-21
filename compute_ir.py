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


def extract_ir(full, sr, preroll_ms, ir_seconds):
    peak = int(np.argmax(np.abs(full)))
    pre = int(round(preroll_ms * 1e-3 * sr))
    start = max(0, peak - pre)
    if ir_seconds is None:
        end = len(full)
    else:
        end = min(len(full), start + int(round(ir_seconds * sr)))
    return full[start:end], peak


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
                    help="Largo de la IR a guardar en segundos desde el onset (default 5; 0 = completa)")
    ap.add_argument("--preroll-ms", type=float, default=5.0,
                    help="Milisegundos a conservar antes del pico (default 5)")
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
    ir_seconds = None if args.ir_seconds == 0 else args.ir_seconds

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
        ir, peak = extract_ir(full, sr, args.preroll_ms, ir_seconds)

        if not args.no_normalize:
            mx = np.max(np.abs(ir))
            if mx > 0:
                ir = ir / mx * 0.5

        rel = os.path.relpath(path, args.input_dir)
        stem, _ = os.path.splitext(rel)
        out_path = os.path.join(outdir, stem + "_IR.wav")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        sf.write(out_path, ir.astype(np.float32), sr, subtype=args.subtype)
        print(f"  [{i:>3}/{len(files)}] {rel}  ->  pico@{peak/sr:.3f}s  IR={len(ir)/sr:.2f}s")

    print("\nListo.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
