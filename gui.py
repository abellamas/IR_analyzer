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

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

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
        self._ultimo_tr_band_selected = tk.StringVar()

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
        self.ir_mode = tk.StringVar(value="filter")
        self.ir_end_mode = tk.StringVar(value="kurtosis")
        self.normalize = tk.BooleanVar(value=True)
        self.subtype = tk.StringVar(value="PCM_24")

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
        ttk.Label(f4, text="Duracion IR (s):").pack(side="left")
        ttk.Entry(f4, textvariable=self.ir_seconds, width=6).pack(side="left", padx=4)
        ttk.Label(f4, text="(solo si Fin = Manual)").pack(side="left", padx=(12, 0))

        f5 = ttk.Frame(root)
        f5.pack(fill="x", **pad)
        ttk.Label(f5, text="Fin de la IR:").pack(side="left")
        ttk.Radiobutton(f5, text="Auto por kurtosis", variable=self.ir_end_mode, value="kurtosis").pack(side="left", padx=4)
        ttk.Radiobutton(f5, text="Manual", variable=self.ir_end_mode, value="fixed").pack(side="left", padx=4)
        ttk.Radiobutton(f5, text="Completa", variable=self.ir_end_mode, value="full").pack(side="left", padx=4)

        f6 = ttk.Frame(root)
        f6.pack(fill="x", **pad)
        ttk.Label(f6, text="Inicio de la IR:").pack(side="left")
        ttk.Radiobutton(f6, text="Completa (N+M-1)", variable=self.ir_mode, value="full").pack(side="left", padx=4)
        ttk.Radiobutton(f6, text="Desde final del filtro", variable=self.ir_mode, value="filter").pack(side="left", padx=4)
        ttk.Checkbutton(f6, text="Normalizar", variable=self.normalize).pack(side="left", padx=(12, 0))
        ttk.Label(f6, text="Formato:").pack(side="left", padx=(12, 0))
        ttk.Combobox(f6, textvariable=self.subtype, values=["PCM_24", "FLOAT", "PCM_16"],
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
        except ValueError:
            self.print_log("Duracion IR invalidos.")
            return

        if self.ir_end_mode.get() == "fixed" and ir_seconds <= 0:
            self.print_log("Duracion IR manual debe ser mayor que 0.")
            return

        self.run_btn.config(state="disabled")
        self.log.delete("1.0", "end")
        threading.Thread(target=self.process, args=(ir_seconds,), daemon=True).start()

    def process(self, ir_seconds):
        outdir = self.outdir.get() or os.path.join(os.path.dirname(self.audio_paths[0]), "IR")
        os.makedirs(outdir, exist_ok=True)
        if self.ir_end_mode.get() != "fixed":
            ir_seconds = None

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
            ir, start, peak = extract_ir(
                full,
                sr,
                0,
                ir_seconds,
                start_mode=self.ir_mode.get(),
                end_mode=self.ir_end_mode.get(),
                inv_filter_len=dec.lf if self.ir_mode.get() == "filter" else None,
            )

            if self.normalize.get():
                mx = np.max(np.abs(ir))
                if mx > 0:
                    ir = ir / mx * 0.5

            stem, _ = os.path.splitext(os.path.basename(path))
            out_path = os.path.join(outdir, stem + "_IR.wav")
            sf.write(out_path, ir.astype("float32"), sr, subtype=self.subtype.get())
            self.after(0, self.print_log,
                       f"  [{i}/{len(self.audio_paths)}] {stem}  ->  inicio@{start/sr:.3f}s  pico@{peak/sr:.3f}s  IR={len(ir)/sr:.2f}s")

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
        self.tr_bands = tk.StringVar(value="1/3")
        self.tr_fmin = tk.StringVar(value="100")
        self.tr_fmax = tk.StringVar(value="12000")
        self.tr_noise_correction = tk.BooleanVar(value=True)

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
        ttk.Checkbutton(f2, text="Corrección de ruido", variable=self.tr_noise_correction).pack(side="left", padx=(12, 0))

        self.tr_run_btn = ttk.Button(root, text="Calcular TR", command=self.run_tr)
        self.tr_run_btn.pack(pady=8)

        f3 = ttk.Frame(root)
        f3.pack(fill="x", **pad)
        ttk.Label(f3, text="Ver banda:").pack(side="left")
        self.tr_band_select = ttk.Combobox(f3, textvariable=self._ultimo_tr_band_selected,
                                           values=[], width=10, state="readonly")
        self.tr_band_select.pack(side="left", padx=4)
        self.tr_band_select.bind("<<ComboboxSelected>>", lambda e: self.update_tr_plot())

        self.tr_log = tk.Text(root, height=10)
        self.tr_log.pack(fill="x", **pad)

        self.tr_fig = plt.Figure(figsize=(6, 3), dpi=100)
        self.tr_ax = self.tr_fig.add_subplot(111)
        self.tr_canvas = FigureCanvasTkAgg(self.tr_fig, master=root)
        self.tr_canvas.get_tk_widget().pack(fill="both", expand=True, **pad)

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
            resultados = calcular_tr(
                to_mono(ir), sr,
                bands=self.tr_bands.get(),
                fmin=fmin, fmax=fmax,
                noise_correction=self.tr_noise_correction.get(),
            )
        except Exception as e:
            self.after(0, self.tr_print, f"Error: {e}")
            self.after(0, lambda: self.tr_run_btn.config(state="normal"))
            return

        self._ultimo_tr = resultados
        self._ultimo_tr_bands = self.tr_bands.get()
        band_keys = [str(f_nom) for f_nom in sorted(resultados)]
        self.after(0, lambda: self.tr_band_select.configure(values=band_keys))
        self.after(0, lambda: self._ultimo_tr_band_selected.set(band_keys[0] if band_keys else ""))

        self.after(0, self.update_tr_plot)
        self.after(0, self.tr_print,
                   f"Corrección de ruido: {'ON' if self.tr_noise_correction.get() else 'OFF'}")
        self.after(0, self.tr_print,
                   f"{'Banda (Hz)':>10}  {'EDT (s)':>8}  {'T20 (s)':>8}  {'T30 (s)':>8}  {'Ruido':>8}  {'Cross (s)':>9}")
        for f_nom in sorted(resultados):
            r = resultados[f_nom]
            self.after(0, self.tr_print,
                       f"{f_nom:>10}  {r['EDT']:>8.2f}  {r['T20']:>8.2f}  {r['T30']:>8.2f}  "
                       f"{r['Noise_dB']:>6.1f}dB  {r['t_cross_s']:>9.2f}")
        self.after(0, lambda: self.tr_run_btn.config(state="normal"))

    def update_tr_plot(self):
        selected = self._ultimo_tr_band_selected.get()
        if not self._ultimo_tr or not selected:
            return
        try:
            band = float(selected)
        except ValueError:
            return
        if band not in self._ultimo_tr:
            return
        result = self._ultimo_tr[band]
        curve = result.get("curve", {})
        sch = result.get("schroeder", {})

        t = sch.get("t", np.array([]))
        l_db_abs = sch.get("l_db_abs", np.array([]))
        noise_db = sch.get("noise_db", float("nan"))
        t_cross = sch.get("t_cross_s", float("nan"))
        t20_t = sch.get("t20_fit_t", np.array([]))
        t20_y = sch.get("t20_fit_y", np.array([]))

        plot_t = curve.get("t", np.array([]))
        raw_db = curve.get("raw_db", np.array([]))
        smooth_db = curve.get("smooth_db", [])
        labels = curve.get("labels", [])

        self.tr_ax.clear()
        if plot_t.size > 0:
            self.tr_ax.plot(plot_t, raw_db, color="blue", linewidth=0.8, label="E(t) envelope (dBFS)")
            for y, label in zip(smooth_db, labels):
                self.tr_ax.plot(plot_t, y, linewidth=1.5, label=label)

        if t.size > 0:
            self.tr_ax.plot(t, l_db_abs, color="black", linewidth=2, label="Schroeder L(t) (dBFS)")
        if t20_t.size > 0:
            self.tr_ax.plot(t20_t, t20_y, color="orange", linestyle="--", linewidth=2, label="T20 fit")
        if np.isfinite(t_cross) and t.size > 0:
            y_cross = np.interp(t_cross, t, l_db_abs)
            self.tr_ax.axvline(t_cross, color="red", linestyle=":", linewidth=1.5, label="t_cross (ruido)")
            self.tr_ax.scatter([t_cross], [y_cross], color="red", s=40)
            self.tr_ax.text(t_cross, y_cross - 5, f"t_c={t_cross:.3f}s", color="red",
                            ha="center", va="top", fontsize=8, bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"})

        if np.isfinite(noise_db) and t.size > 0:
            self.tr_ax.axhline(noise_db, color="magenta", linestyle="-.", linewidth=1.5, label="Noise floor (dBFS)")
            self.tr_ax.text(0.02 * max(t[-1], 1.0), noise_db + 2,
                            f"Noise={noise_db:.1f} dBFS", color="magenta",
                            ha="left", va="bottom", fontsize=8, bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"})

        self.tr_ax.set_xlabel("Time [s]")
        self.tr_ax.set_ylabel("Level [dB]")
        self.tr_ax.set_title(f"Banda {selected} Hz")
        self.tr_ax.set_ylim(-120, 5)
        self.tr_ax.grid(True, linestyle="--", alpha=0.4)
        self.tr_ax.legend(loc="upper right", fontsize=7)
        self.tr_canvas.draw()

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
