"""
Interfaz Tkinter para compute_ir.py, compute_sti.py y compute_tr.py: extraer
respuestas al impulso por deconvolucion de Farina, calcular el STI a partir
de una IR real + grabacion de senal + piso de ruido, y calcular tiempo de
reverberacion (EDT, T20, T30) por banda de octava o tercio de octava.
"""
import os
import threading
import tkinter as tk
from tkinter import filedialog, ttk

import numpy as np
import soundfile as sf

from compute_ir import Deconvolver, extract_ir, to_mono
from compute_sti import BANDAS_OCTAVA, calcular_STI_desde_ir, interpretar
from compute_tr import calcular_tr
from validar import comparar_jnd, parse_aurora_txt


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("IR Analyzer")
        self.geometry("640x520")

        self._ultimo_tr = None
        self._ultimo_tr_bands = None

        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True)
        tab_ir = ttk.Frame(nb)
        tab_sti = ttk.Frame(nb)
        tab_tr = ttk.Frame(nb)
        tab_val = ttk.Frame(nb)
        nb.add(tab_ir, text="Extraer IR")
        nb.add(tab_sti, text="STI")
        nb.add(tab_tr, text="TR")
        nb.add(tab_val, text="Validación")

        self._build_tab_ir(tab_ir)
        self._build_tab_sti(tab_sti)
        self._build_tab_tr(tab_tr)
        self._build_tab_val(tab_val)

    # ---------------- Tab: Extraer IR ----------------

    def _build_tab_ir(self, root):
        self.filtro_path = tk.StringVar()
        self.audio_paths = []
        self.outdir = tk.StringVar()
        self.ir_seconds = tk.StringVar(value="5.0")
        self.preroll_ms = tk.StringVar(value="5.0")
        self.normalize = tk.BooleanVar(value=True)
        self.subtype = tk.StringVar(value="FLOAT")

        pad = {"padx": 8, "pady": 4}

        f1 = ttk.Frame(root)
        f1.pack(fill="x", **pad)
        ttk.Button(f1, text="Elegir filtro inverso...", command=self.pick_filtro).pack(side="left")
        ttk.Label(f1, textvariable=self.filtro_path).pack(side="left", padx=6)

        f2 = ttk.Frame(root)
        f2.pack(fill="x", **pad)
        ttk.Button(f2, text="Elegir audios a procesar...", command=self.pick_audios).pack(side="left")
        self.audio_count_lbl = ttk.Label(f2, text="0 audios seleccionados")
        self.audio_count_lbl.pack(side="left", padx=6)

        f3 = ttk.Frame(root)
        f3.pack(fill="x", **pad)
        ttk.Button(f3, text="Carpeta de salida...", command=self.pick_outdir).pack(side="left")
        ttk.Label(f3, textvariable=self.outdir).pack(side="left", padx=6)

        f4 = ttk.Frame(root)
        f4.pack(fill="x", **pad)
        ttk.Label(f4, text="Duracion IR (s, 0=completa):").pack(side="left")
        ttk.Entry(f4, textvariable=self.ir_seconds, width=6).pack(side="left", padx=4)
        ttk.Label(f4, text="Preroll (ms):").pack(side="left", padx=(12, 0))
        ttk.Entry(f4, textvariable=self.preroll_ms, width=6).pack(side="left", padx=4)
        ttk.Checkbutton(f4, text="Normalizar", variable=self.normalize).pack(side="left", padx=(12, 0))
        ttk.Label(f4, text="Formato:").pack(side="left", padx=(12, 0))
        ttk.Combobox(f4, textvariable=self.subtype, values=["FLOAT", "PCM_24", "PCM_16"],
                     width=8, state="readonly").pack(side="left")

        self.run_btn = ttk.Button(root, text="Procesar", command=self.run)
        self.run_btn.pack(pady=8)

        self.log = tk.Text(root, height=18)
        self.log.pack(fill="both", expand=True, **pad)

    def pick_filtro(self):
        p = filedialog.askopenfilename(title="Filtro inverso", filetypes=[("WAV", "*.wav")])
        if p:
            self.filtro_path.set(p)

    def pick_audios(self):
        ps = filedialog.askopenfilenames(title="Audios a procesar", filetypes=[("WAV", "*.wav")])
        if ps:
            self.audio_paths = list(ps)
            self.audio_count_lbl.config(text=f"{len(self.audio_paths)} audios seleccionados")
            if not self.outdir.get():
                self.outdir.set(os.path.join(os.path.dirname(self.audio_paths[0]), "IR"))

    def pick_outdir(self):
        p = filedialog.askdirectory(title="Carpeta de salida")
        if p:
            self.outdir.set(p)

    def print_log(self, msg):
        self.log.insert("end", msg + "\n")
        self.log.see("end")

    def run(self):
        if not self.filtro_path.get():
            self.print_log("Falta elegir el filtro inverso.")
            return
        if not self.audio_paths:
            self.print_log("Falta elegir los audios a procesar.")
            return
        try:
            ir_seconds = float(self.ir_seconds.get())
            preroll_ms = float(self.preroll_ms.get())
        except ValueError:
            self.print_log("Duracion IR / preroll invalidos.")
            return

        self.run_btn.config(state="disabled")
        self.log.delete("1.0", "end")
        threading.Thread(target=self.process, args=(ir_seconds, preroll_ms), daemon=True).start()

    def process(self, ir_seconds, preroll_ms):
        outdir = self.outdir.get() or os.path.join(os.path.dirname(self.audio_paths[0]), "IR")
        os.makedirs(outdir, exist_ok=True)
        ir_seconds = None if ir_seconds == 0 else ir_seconds

        f, sr = sf.read(self.filtro_path.get(), always_2d=False)
        f = to_mono(f)
        dec = Deconvolver(f)
        self.after(0, self.print_log, f"Filtro: {os.path.basename(self.filtro_path.get())} ({len(f)} muestras, {sr} Hz)")

        for i, path in enumerate(self.audio_paths, 1):
            rec, sr_r = sf.read(path, always_2d=False)
            if sr_r != sr:
                self.after(0, self.print_log, f"  [SKIP] {os.path.basename(path)}: sr {sr_r} != filtro {sr}")
                continue
            rec = to_mono(rec)
            full = dec(rec)
            ir, peak = extract_ir(full, sr, preroll_ms, ir_seconds)

            if self.normalize.get():
                mx = np.max(np.abs(ir))
                if mx > 0:
                    ir = ir / mx * 0.99

            stem, _ = os.path.splitext(os.path.basename(path))
            out_path = os.path.join(outdir, stem + "_IR.wav")
            sf.write(out_path, ir.astype("float32"), sr, subtype=self.subtype.get())
            self.after(0, self.print_log,
                       f"  [{i}/{len(self.audio_paths)}] {stem}  ->  pico@{peak/sr:.3f}s  IR={len(ir)/sr:.2f}s")

        self.after(0, self.print_log, "\nListo. Salida en: " + outdir)
        self.after(0, lambda: self.run_btn.config(state="normal"))

    # ---------------- Tab: STI ----------------

    def _build_tab_sti(self, root):
        self.sti_ir_path = tk.StringVar()
        self.sti_rec_path = tk.StringVar()
        self.sti_noise_path = tk.StringVar()

        pad = {"padx": 8, "pady": 4}

        f1 = ttk.Frame(root)
        f1.pack(fill="x", **pad)
        ttk.Button(f1, text="Elegir IR...", command=lambda: self._pick_sti_file(
            self.sti_ir_path, "Respuesta al impulso (IR)")).pack(side="left")
        ttk.Label(f1, textvariable=self.sti_ir_path).pack(side="left", padx=6)

        f2 = ttk.Frame(root)
        f2.pack(fill="x", **pad)
        ttk.Button(f2, text="Elegir grabacion (senal)...", command=lambda: self._pick_sti_file(
            self.sti_rec_path, "Grabacion de la senal de prueba")).pack(side="left")
        ttk.Label(f2, textvariable=self.sti_rec_path).pack(side="left", padx=6)

        f3 = ttk.Frame(root)
        f3.pack(fill="x", **pad)
        ttk.Button(f3, text="Elegir piso de ruido...", command=lambda: self._pick_sti_file(
            self.sti_noise_path, "Grabacion del piso de ruido")).pack(side="left")
        ttk.Label(f3, textvariable=self.sti_noise_path).pack(side="left", padx=6)

        self.sti_run_btn = ttk.Button(root, text="Calcular STI", command=self.run_sti)
        self.sti_run_btn.pack(pady=8)

        self.sti_log = tk.Text(root, height=18)
        self.sti_log.pack(fill="both", expand=True, **pad)

    def _pick_sti_file(self, var, title):
        p = filedialog.askopenfilename(title=title, filetypes=[("WAV", "*.wav")])
        if p:
            var.set(p)

    def sti_print(self, msg):
        self.sti_log.insert("end", msg + "\n")
        self.sti_log.see("end")

    def run_sti(self):
        if not (self.sti_ir_path.get() and self.sti_rec_path.get() and self.sti_noise_path.get()):
            self.sti_print("Falta elegir IR, grabacion y/o piso de ruido.")
            return
        self.sti_run_btn.config(state="disabled")
        self.sti_log.delete("1.0", "end")
        threading.Thread(target=self.process_sti, daemon=True).start()

    def process_sti(self):
        try:
            ir, sr_ir = sf.read(self.sti_ir_path.get(), always_2d=False)
            rec, sr_rec = sf.read(self.sti_rec_path.get(), always_2d=False)
            noise, sr_noise = sf.read(self.sti_noise_path.get(), always_2d=False)
            sti, mti_banda, detalle = calcular_STI_desde_ir(
                to_mono(ir), sr_ir, to_mono(rec), sr_rec, to_mono(noise), sr_noise
            )
        except Exception as e:
            self.after(0, self.sti_print, f"Error: {e}")
            self.after(0, lambda: self.sti_run_btn.config(state="normal"))
            return

        self.after(0, self.sti_print, f"STI = {sti:.3f}  ({interpretar(sti)})\n")
        self.after(0, self.sti_print, f"{'Banda (Hz)':>10}  {'SNR medido':>11}  {'Band STI (MTI)':>15}")
        for fc in BANDAS_OCTAVA:
            self.after(0, self.sti_print,
                       f"{fc:>10}  {detalle[fc]['snr_medido']:>9.1f} dB  {mti_banda[fc]:>15.3f}")
        self.after(0, lambda: self.sti_run_btn.config(state="normal"))

    # ---------------- Tab: TR ----------------

    def _build_tab_tr(self, root):
        self.tr_ir_path = tk.StringVar()
        self.tr_bands = tk.StringVar(value="1/1")
        self.tr_fmin = tk.StringVar(value="100")
        self.tr_fmax = tk.StringVar(value="5000")

        pad = {"padx": 8, "pady": 4}

        f1 = ttk.Frame(root)
        f1.pack(fill="x", **pad)
        ttk.Button(f1, text="Elegir IR...", command=lambda: self._pick_sti_file(
            self.tr_ir_path, "Respuesta al impulso (IR)")).pack(side="left")
        ttk.Label(f1, textvariable=self.tr_ir_path).pack(side="left", padx=6)

        f2 = ttk.Frame(root)
        f2.pack(fill="x", **pad)
        ttk.Label(f2, text="Bandas:").pack(side="left")
        ttk.Combobox(f2, textvariable=self.tr_bands, values=["1/1", "1/3"],
                     width=6, state="readonly").pack(side="left", padx=4)
        ttk.Label(f2, text="fmin (Hz):").pack(side="left", padx=(12, 0))
        ttk.Entry(f2, textvariable=self.tr_fmin, width=6).pack(side="left", padx=4)
        ttk.Label(f2, text="fmax (Hz):").pack(side="left", padx=(12, 0))
        ttk.Entry(f2, textvariable=self.tr_fmax, width=6).pack(side="left", padx=4)

        self.tr_run_btn = ttk.Button(root, text="Calcular TR", command=self.run_tr)
        self.tr_run_btn.pack(pady=8)

        self.tr_log = tk.Text(root, height=18)
        self.tr_log.pack(fill="both", expand=True, **pad)

    def tr_print(self, msg):
        self.tr_log.insert("end", msg + "\n")
        self.tr_log.see("end")

    def run_tr(self):
        if not self.tr_ir_path.get():
            self.tr_print("Falta elegir la IR.")
            return
        try:
            fmin = float(self.tr_fmin.get())
            fmax = float(self.tr_fmax.get())
        except ValueError:
            self.tr_print("fmin / fmax invalidos.")
            return
        self.tr_run_btn.config(state="disabled")
        self.tr_log.delete("1.0", "end")
        threading.Thread(target=self.process_tr, args=(fmin, fmax), daemon=True).start()

    def process_tr(self, fmin, fmax):
        try:
            ir, sr = sf.read(self.tr_ir_path.get(), always_2d=False)
            resultados = calcular_tr(to_mono(ir), sr, bands=self.tr_bands.get(), fmin=fmin, fmax=fmax)
        except Exception as e:
            self.after(0, self.tr_print, f"Error: {e}")
            self.after(0, lambda: self.tr_run_btn.config(state="normal"))
            return

        self._ultimo_tr = resultados
        self._ultimo_tr_bands = self.tr_bands.get()

        self.after(0, self.tr_print,
                   f"{'Banda (Hz)':>10}  {'EDT (s)':>8}  {'T20 (s)':>8}  {'T30 (s)':>8}  {'Ruido':>8}  {'Cross (s)':>9}")
        for f_nom in sorted(resultados):
            r = resultados[f_nom]
            self.after(0, self.tr_print,
                       f"{f_nom:>10}  {r['EDT']:>8.2f}  {r['T20']:>8.2f}  {r['T30']:>8.2f}  "
                       f"{r['Noise_dB']:>6.1f}dB  {r['t_cross_s']:>9.2f}")
        self.after(0, lambda: self.tr_run_btn.config(state="normal"))

    # ---------------- Tab: Validación ----------------

    def _build_tab_val(self, root):
        self.val_txt_path = tk.StringVar()

        pad = {"padx": 8, "pady": 4}

        f1 = ttk.Frame(root)
        f1.pack(fill="x", **pad)
        ttk.Button(f1, text="Cargar TXT Aurora...", command=self._pick_val_txt).pack(side="left")
        ttk.Label(f1, textvariable=self.val_txt_path).pack(side="left", padx=6)

        ttk.Label(root, text="Compara contra el ultimo TR calculado en la pestaña TR (EDT/T20/T30).",
                  foreground="#666").pack(anchor="w", padx=8)
        ttk.Label(root, text="JND ISO 3382-1: EDT/T20/T30 = 5% relativo.", foreground="#666").pack(anchor="w", padx=8)

        self.val_run_btn = ttk.Button(root, text="Comparar", command=self.run_val)
        self.val_run_btn.pack(pady=8)

        self.val_log = tk.Text(root, height=20, font=("Courier New", 9))
        self.val_log.pack(fill="both", expand=True, **pad)
        self.val_log.tag_configure("ok", background="#c8e6c9")
        self.val_log.tag_configure("bad", background="#ffcdd2")
        self.val_log.tag_configure("na", background="#f0f0f0")

    def _pick_val_txt(self):
        p = filedialog.askopenfilename(title="TXT de Aurora", filetypes=[("Texto", "*.txt")])
        if p:
            self.val_txt_path.set(p)

    def run_val(self):
        self.val_log.delete("1.0", "end")
        if not self._ultimo_tr:
            self.val_log.insert("end", "Primero calcula TR en la pestana TR.\n")
            return
        if not self.val_txt_path.get():
            self.val_log.insert("end", "Falta elegir el TXT de Aurora.\n")
            return
        try:
            aurora = parse_aurora_txt(self.val_txt_path.get())
        except Exception as e:
            self.val_log.insert("end", f"Error leyendo TXT: {e}\n")
            return

        filas = comparar_jnd(self._ultimo_tr, aurora)
        self.val_log.insert("end", f"Bandas: {self._ultimo_tr_bands}\n\n")
        self.val_log.insert("end", f"{'Banda (Hz)':>10}  {'EDT':>13}  {'T20':>13}  {'T30':>13}\n")
        for f_nom in sorted(filas):
            fila = filas[f_nom]
            self.val_log.insert("end", f"{f_nom:>10}  ")
            for p in ("EDT", "T20", "T30"):
                d = fila[p]
                if d["ok"] is None:
                    self.val_log.insert("end", f"{'--':>13}  ", "na")
                else:
                    self.val_log.insert("end", f"{d['diff_pct']:>+11.1f}%  ", "ok" if d["ok"] else "bad")
            self.val_log.insert("end", "\n")


if __name__ == "__main__":
    App().mainloop()
