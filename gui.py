"""
Interfaz Tkinter para compute_ir.py: elegir el filtro inverso y los audios
a procesar, y obtener la IR recortada a su region util.
"""
import os
import threading
import tkinter as tk
from tkinter import filedialog, ttk

import numpy as np
import soundfile as sf

from compute_ir import Deconvolver, extract_ir, to_mono


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("IR por deconvolucion (Farina)")
        self.geometry("640x480")

        self.filtro_path = tk.StringVar()
        self.audio_paths = []
        self.outdir = tk.StringVar()
        self.ir_seconds = tk.StringVar(value="5.0")
        self.preroll_ms = tk.StringVar(value="5.0")
        self.normalize = tk.BooleanVar(value=True)
        self.subtype = tk.StringVar(value="FLOAT")

        pad = {"padx": 8, "pady": 4}

        f1 = ttk.Frame(self)
        f1.pack(fill="x", **pad)
        ttk.Button(f1, text="Elegir filtro inverso...", command=self.pick_filtro).pack(side="left")
        ttk.Label(f1, textvariable=self.filtro_path).pack(side="left", padx=6)

        f2 = ttk.Frame(self)
        f2.pack(fill="x", **pad)
        ttk.Button(f2, text="Elegir audios a procesar...", command=self.pick_audios).pack(side="left")
        self.audio_count_lbl = ttk.Label(f2, text="0 audios seleccionados")
        self.audio_count_lbl.pack(side="left", padx=6)

        f3 = ttk.Frame(self)
        f3.pack(fill="x", **pad)
        ttk.Button(f3, text="Carpeta de salida...", command=self.pick_outdir).pack(side="left")
        ttk.Label(f3, textvariable=self.outdir).pack(side="left", padx=6)

        f4 = ttk.Frame(self)
        f4.pack(fill="x", **pad)
        ttk.Label(f4, text="Duracion IR (s, 0=completa):").pack(side="left")
        ttk.Entry(f4, textvariable=self.ir_seconds, width=6).pack(side="left", padx=4)
        ttk.Label(f4, text="Preroll (ms):").pack(side="left", padx=(12, 0))
        ttk.Entry(f4, textvariable=self.preroll_ms, width=6).pack(side="left", padx=4)
        ttk.Checkbutton(f4, text="Normalizar", variable=self.normalize).pack(side="left", padx=(12, 0))
        ttk.Label(f4, text="Formato:").pack(side="left", padx=(12, 0))
        ttk.Combobox(f4, textvariable=self.subtype, values=["FLOAT", "PCM_24", "PCM_16"],
                     width=8, state="readonly").pack(side="left")

        self.run_btn = ttk.Button(self, text="Procesar", command=self.run)
        self.run_btn.pack(pady=8)

        self.log = tk.Text(self, height=18)
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


if __name__ == "__main__":
    App().mainloop()
