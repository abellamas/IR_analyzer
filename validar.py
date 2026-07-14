"""
Compara los parametros de compute_tr.py contra un TXT exportado por Aurora
Acoustical Parameters (Angelo Farina), usando los umbrales de JND (Just
Noticeable Difference) de ISO 3382-1 -- misma metodologia de validacion que
usa Fernandez Ridano en su TP10 (scripts/validator.py, IMA-UNTreF 2026):

    EDT, T20, T30: 5% relativo

(C50, C80, D50 no se comparan: este software no los calcula todavia.)
"""
import numpy as np

JND_RELATIVO = {"EDT": 0.05, "T20": 0.05, "T30": 0.05}

_NOMINALES = [
    20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160,
    200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600,
    2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000, 12500, 16000, 20000,
]


def _freq_a_texto(f):
    """Redondea al nominal IEC 61260 mas cercano -- Aurora abrevia p.ej.
    '1,3k' (1300) y '3,2k' (3200) para los nominales reales 1250 y 3150."""
    nom = min(_NOMINALES, key=lambda n: abs(n - f))
    return str(int(nom)) if nom == int(nom) else str(nom)


def parse_aurora_txt(filepath):
    """Parsea un TXT 'Aurora 5.0 - ISO3382 Acoustical Parameter File'
    (THIRDOCTAVE/OCTAVE BAND ACOUSTICAL PARAMETERS). Devuelve
    {parametro: {freq_str: valor}} para el primer canal del archivo."""
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    data = {}
    freqs = []
    primer_canal = None
    parseando = False

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("Filename") and "Frq.band" in line:
            partes = line.split("\t")
            for rf in partes[2:]:
                rf = rf.strip().lower()
                if rf in ("a", "lin"):
                    continue
                val = float(rf.replace("k", "").replace(",", ".")) * (1000 if "k" in rf else 1)
                freqs.append(_freq_a_texto(val))
            parseando = True
            continue
        if not parseando:
            continue
        partes = line.split("\t")
        if len(partes) <= 2:
            continue
        canal = partes[0].strip()
        if primer_canal is None:
            primer_canal = canal
        if canal != primer_canal:
            continue
        param = partes[1].strip()
        if param == "strenGth":
            param = "G"
        valores = partes[2:]
        data.setdefault(param, {})
        for i, fr in enumerate(freqs):
            if i < len(valores):
                v = valores[i].strip().replace(",", ".")
                try:
                    data[param][fr] = float(v)
                except ValueError:
                    data[param][fr] = float("nan")
    return data


def comparar_jnd(resultados_propios, aurora_data):
    """
    resultados_propios: {freq_nominal: {"EDT":.., "T20":.., "T30":..}} (salida de calcular_tr)
    aurora_data: salida de parse_aurora_txt

    Devuelve {freq_nominal: {param: {"propio", "aurora", "diff_pct", "ok"}}}
    ok=True/False si hay JND definido y ambos valores son finitos; None si no se puede evaluar.
    """
    filas = {}
    for f_nom, r in resultados_propios.items():
        f_str = _freq_a_texto(f_nom)
        fila = {}
        for param, jnd in JND_RELATIVO.items():
            val_propio = r.get(param, float("nan"))
            val_aurora = aurora_data.get(param, {}).get(f_str, float("nan"))
            if not (np.isfinite(val_propio) and np.isfinite(val_aurora)) or val_aurora == 0:
                fila[param] = {"propio": val_propio, "aurora": val_aurora, "diff_pct": float("nan"), "ok": None}
                continue
            diff_pct = (val_propio - val_aurora) / abs(val_aurora) * 100
            fila[param] = {
                "propio": val_propio, "aurora": val_aurora,
                "diff_pct": diff_pct, "ok": abs(diff_pct) <= jnd * 100,
            }
        filas[f_nom] = fila
    return filas
