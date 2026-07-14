"""
Fractional octave filter bank with IEC 61260-1 compliance.

Supports 1/1, 1/3, 1/6, 1/12 and 1/24 octave band analysis.
Uses staged decimation for numerical stability across 20 Hz – 20 kHz.
"""

import numpy as np
from scipy.signal import butter, sosfiltfilt, decimate as _scipy_decimate

# ── IEC 61260-1 nominal center frequencies ────────────────────────────────────

_NOMINAL_OCTAVE = [31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]

_NOMINAL_THIRD = [
    20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160,
    200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600,
    2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000, 12500, 16000, 20000,
]

# All known IEC nominals used for matching in sub-1/3 octave bands
_ALL_IEC_NOMINALS = set(_NOMINAL_OCTAVE + _NOMINAL_THIRD)

# ── Default filter order per band resolution N ────────────────────────────────
# Narrower bands require higher order to maintain adequate selectivity.

_DEFAULT_ORDER = {1: 4, 3: 6, 6: 8, 12: 12, 24: 16}

# ── Decimation groups ─────────────────────────────────────────────────────────
# (max_nominal_fc, [staged decimation factors])
# Group A: sr → ÷5 → ÷10  (~882 Hz,  Nyquist ~441 Hz) → covers 20–250 Hz
# Group B: sr → ÷5         (~8820 Hz, Nyquist ~4410 Hz) → covers ~280–2500 Hz
# Group C: no decimation   (sr)                          → covers ~2800–20000 Hz
# Valid for sr = 44100 Hz and 48000 Hz.

_GROUPS = [
    (250,   [5, 10]),
    (2500,  [5]),
    (20000, []),
]

# ── Helpers ───────────────────────────────────────────────────────────────────

_VALID_BANDS = ('octave', '1/1', '1/3', '1/6', '1/12', '1/24')


def _parse_N(bands):
    if bands in ('octave', '1/1'):
        return 1
    try:
        return int(bands.split('/')[1])
    except (IndexError, ValueError):
        raise ValueError(f"bands must be one of {_VALID_BANDS}")


def _round_sig(x, sig=3):
    if x == 0:
        return 0.0
    mag = 10 ** (int(np.floor(np.log10(abs(x)))) - (sig - 1))
    return round(x / mag) * mag


def _to_nominal(f_exact, tol=0.02):
    """Returns nearest IEC 61260 nominal if within tol, else rounds to 3 sig figs."""
    for f_nom in _ALL_IEC_NOMINALS:
        if abs(f_exact - f_nom) / f_nom <= tol:
            return f_nom
    return _round_sig(f_exact, 3)


# ── FilterBank ────────────────────────────────────────────────────────────────

class FilterBank:
    """
    Fractional octave filter bank with IEC 61260-1 compliance.

    - 1/1 and 1/3 octave: IEC 61260-1 nominal frequencies + base-2 exact values
    - 1/6, 1/12, 1/24:    generated via base-2 formula; IEC nominals reused where
                          they match within 2%, otherwise rounded to 3 sig figs

    Uses staged decimation for low-frequency numerical stability.

    Parameters
    ----------
    sr    : int     sample rate in Hz (default: 44100)
    bands : str     '1/1'/'octave', '1/3', '1/6', '1/12', '1/24' (default: '1/3')
    fmin  : float   lowest center frequency in Hz (default: 20)
    fmax  : float   highest center frequency in Hz (default: 20000)
    order : int     Butterworth filter order. None → auto based on band resolution.

    Examples
    --------
    fb = FilterBank(sr=44100, bands='1/3')
    freqs, levels = fb.leq(signal, method='iir')   # IEC 61260 compliant
    freqs, levels = fb.leq(signal, method='fft')   # fast, rectangular bands
    print(fb.center_freqs_nominal)                 # for reports / labels
    print(fb.center_freqs_exact)                   # used internally for filters
    """

    def __init__(self, sr=44100, bands='1/3', fmin=20, fmax=20000, order=None):
        if bands not in _VALID_BANDS:
            raise ValueError(f"bands must be one of {_VALID_BANDS}")

        self.sr    = sr
        self.bands = bands
        self.N     = _parse_N(bands)
        self.order = order if order is not None else _DEFAULT_ORDER[self.N]

        # Band edge: f_lo = f_c / edge,  f_hi = f_c × edge
        self._edge = 2 ** (1 / (2 * self.N))
        self.edge = self._edge  # alias publico (bandwidth(fc) = fc*(edge - 1/edge))

        self.center_freqs_exact, self.center_freqs_nominal = \
            self._compute_center_freqs(fmin, fmax)

        self._groups = self._build_filters()
        self._print_summary()

    # ── Frequency generation ──────────────────────────────────────────────────

    def _compute_center_freqs(self, fmin, fmax):
        N = self.N
        # Generate candidate n indices with margin
        n_min = int(np.floor(N * np.log2(fmin / 1000))) - 1
        n_max = int(np.ceil(N * np.log2(fmax / 1000)))  + 1

        exact   = []
        nominal = []
        for n in range(n_min, n_max + 1):
            fe   = 1000.0 * 2 ** (n / N)
            fn   = _to_nominal(fe)
            # Keep only bands whose nominal falls within [fmin, fmax]
            if fmin * 0.99 <= fn <= fmax * 1.01:
                exact.append(fe)
                nominal.append(fn)

        return exact, nominal

    # ── Filter design ─────────────────────────────────────────────────────────

    def _build_filters(self):
        groups   = []
        assigned = set()
        pairs    = list(zip(self.center_freqs_nominal, self.center_freqs_exact))

        for max_fc, dec_stages in _GROUPS:
            sr_work = self.sr
            for d in dec_stages:
                sr_work = sr_work // d
            nyquist = sr_work / 2

            pairs_here = [(fn, fe) for fn, fe in pairs
                          if fn <= max_fc and fn not in assigned]
            if not pairs_here:
                continue

            filters = {}
            for f_nom, f_exact in pairs_here:
                f_lo = f_exact / self._edge
                f_hi = min(f_exact * self._edge, nyquist * 0.95)
                sos  = butter(self.order, [f_lo, f_hi],
                              btype='bandpass', fs=sr_work, output='sos')
                filters[f_nom] = sos
                assigned.add(f_nom)

            groups.append({
                'dec_stages': dec_stages,
                'sr_work':    sr_work,
                'freqs':      [fn for fn, _ in pairs_here],
                'filters':    filters,
            })

        return groups

    # ── Decimation ────────────────────────────────────────────────────────────

    @staticmethod
    def _decimate(signal, stages):
        sig = signal.astype(np.float64)
        for q in stages:
            sig = _scipy_decimate(sig, q, zero_phase=True)
        return sig

    # ── Leq IIR ───────────────────────────────────────────────────────────────

    def _leq_iir(self, signal, p_ref):
        idx    = {fc: i for i, fc in enumerate(self.center_freqs_nominal)}
        levels = np.full(len(self.center_freqs_nominal), np.nan)

        for group in self._groups:
            sig_dec = self._decimate(signal, group['dec_stages'])
            for f_nom in group['freqs']:
                filtered           = sosfiltfilt(group['filters'][f_nom], sig_dec)
                rms                = np.sqrt(np.mean(filtered ** 2))
                levels[idx[f_nom]] = 20 * np.log10(rms / p_ref + 1e-12)

        return levels

    # ── Leq FFT ───────────────────────────────────────────────────────────────

    def _leq_fft(self, signal, p_ref):
        sig  = signal.astype(np.float64)
        N    = len(sig)
        X    = np.fft.rfft(sig)
        fbin = np.fft.rfftfreq(N, d=1.0 / self.sr)

        # One-sided power via Parseval
        power        = np.abs(X) ** 2 / N ** 2
        power[1:-1] *= 2

        levels = np.full(len(self.center_freqs_nominal), np.nan)
        for i, f_exact in enumerate(self.center_freqs_exact):
            f_lo      = f_exact / self._edge
            f_hi      = f_exact * self._edge
            mask      = (fbin >= f_lo) & (fbin < f_hi)
            rms       = np.sqrt(np.sum(power[mask]))
            levels[i] = 20 * np.log10(rms / p_ref + 1e-12)

        return levels

    # ── Public API ────────────────────────────────────────────────────────────

    def filter_bands(self, signal):
        """
        Filtra la senal banda por banda en el dominio temporal (no colapsa a Leq),
        reutilizando la misma decimacion escalonada y los mismos filtros que leq().

        Returns
        -------
        dict {freq_nominal: (senal_filtrada, sr_usado)}
        """
        out = {}
        for group in self._groups:
            sig_dec = self._decimate(signal, group['dec_stages'])
            for f_nom in group['freqs']:
                out[f_nom] = (sosfiltfilt(group['filters'][f_nom], sig_dec), group['sr_work'])
        return out

    def leq(self, signal, p_ref=20e-6, method='iir'):
        """
        Computes Leq per band for a 1D signal.

        Parameters
        ----------
        signal : np.ndarray   1D audio signal
                              In Pa (after MicArray.to_spl()) → dB SPL
                              In FS units → use p_ref=1.0 for dBFS
        p_ref  : float        reference value (default: 20e-6 Pa)
        method : str          'iir' — IEC 61260 compliant, exacto (default)
                              'fft' — rectangular bands, rápido

        Returns
        -------
        freqs  : np.ndarray   nominal center frequencies in Hz  (n_bands,)
        levels : np.ndarray   Leq per band in dB                (n_bands,)
        """
        if method == 'iir':
            levels = self._leq_iir(signal, p_ref)
        elif method == 'fft':
            levels = self._leq_fft(signal, p_ref)
        else:
            raise ValueError("method must be 'iir' or 'fft'")

        return np.array(self.center_freqs_nominal, dtype=float), levels

    def leq_global(self, signal, p_ref=20e-6, method='iir'):
        """Global broadband Leq by energy summation across all bands."""
        _, levels = self.leq(signal, p_ref=p_ref, method=method)
        return 10 * np.log10(np.sum(10 ** (levels / 10)))

    # ── Display ───────────────────────────────────────────────────────────────

    def _print_summary(self):
        n = len(self.center_freqs_nominal)
        print(f"FilterBank — {self.bands} octava  |  {n} bandas"
              f"  |  orden {self.order}  |  sr {self.sr} Hz\n")
        for g in self._groups:
            factor = int(np.prod(g['dec_stages'])) if g['dec_stages'] else 1
            print(f"  {g['freqs'][0]:>8} – {g['freqs'][-1]:>7} Hz  (nominal)"
                  f"  |  sr_work = {g['sr_work']:>6} Hz  (÷{factor:>3})"
                  f"  |  {len(g['freqs'])} bandas")
        print()

    def __repr__(self):
        fn = self.center_freqs_nominal
        return (f"FilterBank(bands='{self.bands}', sr={self.sr}, order={self.order}, "
                f"n_bands={len(fn)}, fmin={fn[0]}, fmax={fn[-1]})")
