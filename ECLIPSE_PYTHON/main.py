# main.py — ECLIPSE (UI-enhanced) — teal theme fixes
import os
import io
import sys
import time
import threading
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageTk, ImageDraw, ImageFilter, ImageEnhance

import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox

# Matplotlib
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
plt.ioff()

# PDF 
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    from reportlab.lib.utils import ImageReader
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False

# PyTorch 
TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    import torchvision.transforms as T
    import torchvision.models as models
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False
    print("PyTorch not available - will simulate predictions. Install torch & torchvision to use your .pth model.")

# Pillow resampling compatibility
try:
    RESAMPLING = Image.Resampling.LANCZOS
except Exception:
    RESAMPLING = Image.LANCZOS

# Paths safe for PyInstaller
ROOT = Path(getattr(sys, "_MEIPASS", Path.cwd()))
ASSETS = ROOT / "assets"
MODELS_DIR = ROOT / "models"
TMP_DIR = ROOT / "tmp"
TMP_DIR.mkdir(exist_ok=True)

# Model path 
MODEL_PATH = MODELS_DIR / "final.pth"
CLASS_MAP = MODELS_DIR / "class_names.json"

# UI palette -> soft teal theme
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("green")  # keep default theme base

APP_TITLE = "ECLIPSE - Skin lesion classifier"

BG = "#E8F7F3"         # MAIN BACKGROUND (soft teal)
CARD_BG = "#FFFFFF"    # card white
TEXT = "#123C36"       # dark teal text
SUBTEXT = "#4A6B66"    # softer subtitle text

# Accent/neon-like teal palette
PRIMARY_TEAL = "#2AA496"   # used for highlights, borders
ACCENT_TEAL = "#0F766E"    # used for stronger accents
PREDICT_COLOR = "#17BEBB"  # predict button color (teal)
ALT_ACCENT = "#8FE3D6"     # lighter accent for hover/gradients

# Palette for charts (teal + complementary pastels)
BAR_COLORS = ["#7ED8C9", "#A6EAD9", "#CFF6EE", "#DFF6F0", "#BFECE4", "#9FD9C9", "#E7F9F4"]
DONUT_COLORS = BAR_COLORS

# Short class names 
CLASS_NAMES_DEFAULT = [
    "akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"
]
CLASS_FULL = {
    "akiec": "akiec - Actinic Keratoses",
    "bcc": "bcc  - Basal Cell Carcinoma",
    "bkl": "bkl  - Benign Keratosis-like",
    "df":  "df   - Dermatofibroma",
    "mel": "mel  - Melanoma",
    "nv":  "nv   - Melanocytic Nevi",
    "vasc":"vasc - Vascular Lesions"
}

def load_class_names():
    import json
    if CLASS_MAP.exists():
        try:
            with open(CLASS_MAP, "r", encoding="utf-8") as f:
                arr = json.load(f)
                if isinstance(arr, list) and arr:
                    return arr
        except Exception:
            pass
    return CLASS_NAMES_DEFAULT

def simulate_prediction(np_image):
    names = load_class_names()
    probs = np.random.dirichlet([2] * len(names))
    return names, probs, None

def generate_heatmap_like_from_mask(mask_array, size):
    try:
        mask = np.array(mask_array)
        if mask.ndim == 3 and mask.shape[2] == 1:
            mask = mask[..., 0]
        mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-9)
        mask_img = (mask * 255).astype(np.uint8)
        pil = Image.fromarray(mask_img).convert("L").resize(size, RESAMPLING)
        # teal-ish overlay
        r = pil.point(lambda p: int(p * 0.15))
        g = pil.point(lambda p: int(p * 0.9))
        b = pil.point(lambda p: int(p * 0.6))
        a = pil.point(lambda p: int(p * 0.6))
        rgba = Image.merge("RGBA", (r, g, b, a))
        return rgba
    except Exception:
        return generate_heatmap_like(Image.new("RGB", size))

def generate_heatmap_like(image_pil):
    w, h = image_pil.size
    x = np.linspace(-1, 1, w)
    y = np.linspace(-1, 1, h)
    X, Y = np.meshgrid(x, y)
    Z = np.exp(-4 * (X ** 2 + Y ** 2))
    Z = (Z - Z.min()) / (Z.max() - Z.min() + 1e-9)
    cmap = np.zeros((h, w, 4), dtype=np.uint8)
    cmap[..., 0] = (Z * 90).astype(np.uint8)    # r
    cmap[..., 1] = (Z * 200).astype(np.uint8)   # g (dominant teal)
    cmap[..., 2] = (Z * 170).astype(np.uint8)   # b
    cmap[..., 3] = (Z * 160).astype(np.uint8)
    return Image.fromarray(cmap, mode="RGBA")

def generate_blur_gradient(size):
    """Soft teal blurry background"""
    w, h = size
    base = Image.new("RGB", (w, h), BG)
    overlay = Image.new("RGB", (w, h), "#FFFFFF")
    blended = Image.blend(base, overlay, alpha=0.28)
    return blended.filter(ImageFilter.GaussianBlur(45))

# Preprocessing (ImageNet)
TARGET_SZ = (224, 224)
IM_MEAN = [0.485, 0.456, 0.406]
IM_STD = [0.229, 0.224, 0.225]

def preprocess_for_torch(pil: Image.Image):
    transform = T.Compose([
        T.Resize(TARGET_SZ),
        T.ToTensor(),
        T.Normalize(IM_MEAN, IM_STD),
    ])
    t = transform(pil).unsqueeze(0)
    return t

# Softmax
def softmax_np(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

# Torch wrapper 
class TorchModelWrapper:
    def __init__(self, path, n_classes):
        if not TORCH_AVAILABLE:
            raise RuntimeError("torch not installed")
        self.device = torch.device("cpu")
        self.n_classes = n_classes
        self.model = None
        try:
            loaded = torch.load(str(path), map_location=self.device)
            # script module
            if hasattr(loaded, "forward") and isinstance(loaded, torch.nn.Module):
                self.model = loaded
            elif isinstance(loaded, dict):
                # try to load state_dict into DenseNet169
                try:
                    md = models.densenet169(weights=None)
                    in_f = md.classifier.in_features
                    md.classifier = nn.Linear(in_f, self.n_classes)
                    # direct load or nested key
                    try:
                        md.load_state_dict(loaded, strict=False)
                        self.model = md
                    except Exception:
                        # nested
                        if "model_state_dict" in loaded:
                            md.load_state_dict(loaded["model_state_dict"], strict=False)
                            self.model = md
                except Exception:
                    self.model = None
            else:
                self.model = None
        except Exception:
            self.model = None

        if self.model is None:
            raise RuntimeError("Failed to construct model from file")

        self.model.to(self.device)
        self.model.eval()
        print("LOADED TORCH MODEL:", path)

    def predict(self, x):
        with torch.no_grad():
            if isinstance(x, np.ndarray):
                t = torch.from_numpy(x).to(self.device)
            else:
                t = x.to(self.device)
            if t.ndim == 4 and t.shape[-1] == 3:
                t = t.permute(0, 3, 1, 2)
            out = self.model(t.float())
            if isinstance(out, (tuple, list)):
                out = out[0]
            out_np = out.cpu().numpy()
            if out_np.ndim == 2 and out_np.shape[0] == 1:
                out_np = out_np[0]
            if not np.all((out_np >= 0) & (out_np <= 1)):
                out_np = softmax_np(out_np)
            return out_np, None

# Try to load model
TORCH_MODEL = None
if TORCH_AVAILABLE and MODEL_PATH.exists():
    try:
        TORCH_MODEL = TorchModelWrapper(MODEL_PATH, n_classes=len(load_class_names()))
    except Exception:
        print("Failed to load torch model:", traceback.format_exc())
        TORCH_MODEL = None

# ----------------------- UI App -----------------------
class EclipseApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1366x820")
        self.minsize(1150,700)
        self.configure(fg_color=BG)

        # state
        self.class_names = load_class_names()
        self.class_full = [CLASS_FULL.get(k, k) for k in self.class_names]
        self.model = TORCH_MODEL
        self.demo_mode = (self.model is None)

        self.loaded_image = None
        self.display_image_tk = None
        self.heatmap_pil = None
        self.show_heat = False
        self.current_probs = None
        self.current_mask = None
        self.original_image = None

        # plotting state
        self.donut_angle = 0.0
        self.donut_anim_id = None
        self.bar_annotation = None

        # inference guard
        self._infer_lock = threading.Lock()
        self._inference_running = False

        # build UI
        self._make_background()
        self._build_ui()
        self._log("App started. Demo mode: " + str(self.demo_mode))

    def _make_background(self):
        # Light teal blur background image
        try:
            bg = generate_blur_gradient((1600, 900))
            from customtkinter import CTkImage
            self._bg_img = CTkImage(light_image=bg, dark_image=bg, size=(1600, 900))
            lbl = ctk.CTkLabel(self, text="", image=self._bg_img)
            lbl.place(x=0, y=0, relwidth=1, relheight=1)
        except Exception:
            self.configure(fg_color=BG)  # backup bg color

    def _add_hover(self, widget, normal_color, hover_color):
        def on_enter(event):
            try:
                widget.configure(fg_color=hover_color)
            except Exception:
                pass
        def on_leave(event):
            try:
                widget.configure(fg_color=normal_color)
            except Exception:
                pass
        widget.bind("<Enter>", on_enter)
        widget.bind("<Leave>", on_leave)

    def _build_ui(self):
        # top header
        header = ctk.CTkFrame(self, height=64, fg_color=ACCENT_TEAL, corner_radius=0)
        header.pack(fill="x", padx=12, pady=(12,6))
        ctk.CTkLabel(header, text="ECLIPSE", font=ctk.CTkFont(size=22, weight="bold"), text_color=CARD_BG).pack(side="left", padx=(12,6))
        ctk.CTkLabel(header, text="Aurora DNA Edition", font=ctk.CTkFont(size=12), text_color=CARD_BG).pack(side="left")

        content = ctk.CTkFrame(self, fg_color=None)
        content.pack(fill="both", expand=True, padx=12, pady=(6,12))

        # left column (image + controls)
        left_col = ctk.CTkFrame(content, fg_color=CARD_BG, corner_radius=14)
        left_col.pack(side="left", fill="y", padx=(0,10), pady=6)
        left_col.configure(width=360)
        left_inner = ctk.CTkFrame(left_col, fg_color=None)
        left_inner.pack(padx=12, pady=12, fill="both", expand=True)

        ctk.CTkLabel(left_inner, text="Image", text_color=PRIMARY_TEAL, font=ctk.CTkFont(size=16, weight="bold")).pack(pady=(2,10))
        self.canvas_w, self.canvas_h = 320, 240
        # use tkinter Canvas for compatibility
        self.image_canvas = tk.Canvas(left_inner, width=self.canvas_w, height=self.canvas_h, bg="#0f1413", highlightthickness=0)
        self.image_canvas.pack()
        self.image_canvas.create_text(self.canvas_w//2, self.canvas_h//2, text="No Image Loaded", fill=SUBTEXT)

        btn_row = ctk.CTkFrame(left_inner, fg_color=None)
        btn_row.pack(pady=(12,8))
        self.btn_load = ctk.CTkButton(btn_row, text="Load Image", fg_color=PRIMARY_TEAL, width=92, command=self.load_image)
        self.btn_predict = ctk.CTkButton(btn_row, text="Predict", fg_color=PREDICT_COLOR, width=92, command=self.predict_current)
        self.btn_reset = ctk.CTkButton(btn_row, text="Reset", width=92, command=self.reset_all)

        self.btn_load.pack(side="left", padx=6)
        self.btn_predict.pack(side="left", padx=6)
        self.btn_reset.pack(side="left", padx=6)

        self._add_hover(self.btn_load, PRIMARY_TEAL, ALT_ACCENT)
        self._add_hover(self.btn_predict, PREDICT_COLOR, ALT_ACCENT)
        self._add_hover(self.btn_reset, self.btn_reset.cget("fg_color"), "#cccccc")

        control_row = ctk.CTkFrame(left_inner, fg_color=None)
        control_row.pack(fill="x", pady=(6,6))
        self.btn_heat = ctk.CTkButton(control_row, text="Toggle Heatmap", width=140, command=self.toggle_heatmap)
        self.btn_heat.pack(side="left", padx=(6,8))
        ctk.CTkLabel(control_row, text="Min confidence to show", text_color=SUBTEXT, width=160).pack(side="left", padx=(6,6))
        self.threshold_var = ctk.DoubleVar(value=0.0)
        self.th_slider = ctk.CTkSlider(control_row, from_=0.0, to=100.0, number_of_steps=100, variable=self.threshold_var, width=110)
        self.th_slider.pack(side="left", padx=(6,6))

        # below left: log panel small preview & export buttons
        bottom_left = ctk.CTkFrame(left_col, fg_color=None)
        bottom_left.pack(fill="both", expand=True, padx=12, pady=(8,12))
        ctk.CTkLabel(bottom_left, text="Actions", text_color=SUBTEXT).pack(anchor="w")
        act_row = ctk.CTkFrame(bottom_left, fg_color=None)
        act_row.pack(fill="x", pady=(8,6))
        self.btn_csv = ctk.CTkButton(act_row, text="Export CSV", width=120, command=self.export_csv)
        self.btn_pdf = ctk.CTkButton(act_row, text="Export PDF", width=120, command=self.export_pdf)
        self.btn_clear = ctk.CTkButton(act_row, text="Clear Log", width=120, command=self._clear_log)

        self.btn_csv.pack(side="left", padx=6)
        self.btn_pdf.pack(side="left", padx=6)
        self.btn_clear.pack(side="left", padx=6)

        self._add_hover(self.btn_csv, self.btn_csv.cget("fg_color"), "#3b8ad9")
        self._add_hover(self.btn_pdf, self.btn_pdf.cget("fg_color"), "#3b8ad9")
        self._add_hover(self.btn_clear, self.btn_clear.cget("fg_color"), "#666666")

        # middle column (charts + log)
        mid_col = ctk.CTkFrame(content, fg_color=None)
        mid_col.pack(side="left", fill="both", expand=True, padx=(0,10), pady=6)

        chart_card = ctk.CTkFrame(mid_col, fg_color=CARD_BG, corner_radius=14)
        chart_card.pack(fill="both", expand=False, padx=6, pady=(4,6))
        chart_card.configure(height=460)

        ctk.CTkLabel(chart_card, text="Prediction", text_color=TEXT, font=ctk.CTkFont(size=14, weight="bold")).pack(anchor="nw", padx=12, pady=(8,4))

        # matplotlib figure
        self.fig = Figure(figsize=(7.2,3.2), dpi=100, facecolor=CARD_BG)
        self.ax = self.fig.add_subplot(121)
        self.ax2 = self.fig.add_subplot(122)
        self.fig.subplots_adjust(wspace=0.6, left=0.08, right=0.98, top=0.95, bottom=0.12)
        for ax in (self.ax, self.ax2):
            ax.set_facecolor(CARD_BG)
            ax.tick_params(colors=SUBTEXT)
            for spine in ax.spines.values():
                spine.set_color("#1a1b1d")

        self.canvas_fig = FigureCanvasTkAgg(self.fig, master=chart_card)
        self.canvas_widget = self.canvas_fig.get_tk_widget()
        self.canvas_widget.pack(fill="both", expand=True, padx=8, pady=(6,8))

        # connect pick event for bar clicking
        self.canvas_fig.mpl_connect("pick_event", self._on_pick)

        # log area
        log_card = ctk.CTkFrame(mid_col, fg_color=CARD_BG, corner_radius=12)
        log_card.pack(fill="both", expand=True, padx=6, pady=(6,8))
        ctk.CTkLabel(log_card, text="Log", text_color=SUBTEXT).pack(anchor="nw", padx=12, pady=(8,2))
        self.log_text = ctk.CTkTextbox(log_card, height=140, fg_color="#0b0d11", text_color=SUBTEXT, corner_radius=8)
        self.log_text.pack(fill="both", expand=True, padx=12, pady=(2,12))

        # right column 
        right_col = ctk.CTkFrame(content, fg_color=CARD_BG, corner_radius=14)
        right_col.pack(side="left", fill="y", padx=(0,0), pady=6)
        right_col.configure(width=300)
        r_inner = ctk.CTkFrame(right_col, fg_color=None)
        r_inner.pack(padx=12, pady=12, fill="both", expand=True)

        ctk.CTkLabel(r_inner, text="Summary", text_color=PRIMARY_TEAL, font=ctk.CTkFont(size=16, weight="bold")).pack()

        # legend list below
        legend_card = ctk.CTkFrame(r_inner, fg_color="#0b0d11", corner_radius=8)
        legend_card.pack(fill="both", expand=True, pady=(12,6))
        ctk.CTkLabel(legend_card, text="Legend (short)", text_color=SUBTEXT).pack(anchor="nw", padx=8, pady=(8,4))
        self.legend_box = ctk.CTkTextbox(legend_card, height=220, fg_color="#0b0d11", text_color=SUBTEXT, corner_radius=4)
        self.legend_box.pack(fill="both", expand=True, padx=8, pady=(4,12))
        self._refresh_legend()

        # seed empty plots
        self.ax.clear(); self.ax2.clear()
        self._seed_empty_chart()
        self.canvas_fig.draw()

    # helper: log
    def _log(self, s):
        ts = datetime.now().strftime("[%H:%M:%S]")
        try:
            self.log_text.insert("end", f"{ts} {s}\n")
            self.log_text.see("end")
        except Exception:
            pass
        print(s)

    def _clear_log(self):
        self.log_text.delete("0.0", "end")
        self._log("Log cleared")

    def _refresh_legend(self):
        self.legend_box.delete("0.0", "end")
        for short in self.class_names:
            full = CLASS_FULL.get(short, short)
            self.legend_box.insert("end", f"• {short:<6} {full}\n")

    def _seed_empty_chart(self):
        # visual placeholder
        self.ax.clear()
        self.ax2.clear()
        names = self.class_names
        zeros = np.zeros(len(names))
        self.ax.barh(range(len(names)), zeros, color=BAR_COLORS[:len(names)], edgecolor="#cfeee6")
        self.ax.set_yticks(range(len(names)))
        self.ax.set_yticklabels(names, color=SUBTEXT)
        self.ax.set_xlim(0, 0.5)
        self.ax.set_xlabel("Confidence Score", color=SUBTEXT)
        self.ax2.pie([1], colors=[PRIMARY_TEAL])

    # -------------- image load --------------
    def load_image(self):
        path = filedialog.askopenfilename(title="Select Image", filetypes=[("Images","*.png *.jpg *.jpeg *.bmp")])
        if not path:
            return
        try:
            pil = Image.open(path).convert("RGB")
            self.original_image = pil.copy()
            self._log(f"Image loaded: {path} size={self.original_image.size}")
        except Exception as e:
            messagebox.showerror("Image Error", str(e))
            return

        canvas_bg = Image.new("RGB", (self.canvas_w, self.canvas_h), (15,18,22))
        thumb = pil.copy()
        thumb.thumbnail((self.canvas_w-20, self.canvas_h-20), RESAMPLING)
        x = (self.canvas_w - thumb.width)//2
        y = (self.canvas_h - thumb.height)//2
        canvas_bg.paste(thumb, (x,y))
        self.loaded_image = canvas_bg
        self.display_image_tk = ImageTk.PhotoImage(canvas_bg)
        self.image_canvas.delete("all")
        self.image_canvas.create_image(self.canvas_w//2, self.canvas_h//2, image=self.display_image_tk)
        # border
        pad = 6
        x1 = max(0, x-pad); y1 = max(0, y-pad)
        x2 = min(self.canvas_w, x+thumb.width+pad); y2 = min(self.canvas_h, y+thumb.height+pad)
        # draw border rectangles using canvas create_rectangle
        self.image_canvas.create_rectangle(x1, y1, x2, y2, outline=PRIMARY_TEAL, width=2)
        self.image_canvas.create_rectangle(x1+3, y1+3, x2-3, y2-3, outline=ACCENT_TEAL, width=1)
        self.image_canvas.create_text(self.canvas_w//2, self.canvas_h-12, text="Preview", fill=SUBTEXT)

    # -------------- reset --------------
    def reset_all(self):
        self.loaded_image = None
        self.display_image_tk = None
        self.heatmap_pil = None
        self.show_heat = False
        self.current_probs = None
        self.current_mask = None
        self.original_image = None
        self.image_canvas.delete("all")
        self.image_canvas.create_text(self.canvas_w//2, self.canvas_h//2, text="No Image Loaded", fill=SUBTEXT)
        self._seed_empty_chart()
        self.canvas_fig.draw()
        self._log("UI reset")

    # -------------- predict --------------
    def predict_current(self):
        if self.original_image is None:
            messagebox.showinfo("No Image", "Please load an image first.")
            return
        if self._inference_running:
            return
        self._inference_running = True
        self._log("Running model inference...")
        self.report_thread = threading.Thread(target=self._run_prediction_thread, daemon=True)
        self.report_thread.start()

    def _run_prediction_thread(self):
        time.sleep(0.05)
        names, probs, mask = None, None, None
        try:
            img_for_model = self.original_image.copy()
            self._log(f"PREDICT INPUT SIZE (pil): {img_for_model.size}")
            if not self.demo_mode and self.model is not None and TORCH_AVAILABLE:
                with self._infer_lock:
                    try:
                        t = preprocess_for_torch(img_for_model)  # torch tensor
                        self._log(f"INPUT TENSOR SHAPE: {t.shape}")
                        probs_out, mask_out = self.model.predict(t)
                        probs = np.asarray(probs_out) if probs_out is not None else None
                        mask = np.asarray(mask_out) if mask_out is not None else None
                        if probs is not None and probs.ndim > 1:
                            probs = probs.ravel()
                        self._log(f"Model returned - probs shape: {None if probs is None else probs.shape}")
                    except Exception:
                        self._log("Exception during model.predict:\n" + traceback.format_exc())
                        probs = None
                        mask = None
            else:
                self._log("Torch model not loaded; using simulate_prediction")
                names, p = simulate_prediction(np.array(img_for_model))
                probs = p
                mask = None

            if probs is None:
                probs = np.zeros(len(self.class_names), dtype=np.float32)

            # ensure ordering 
            self.current_probs = (self.class_names, probs)
            self.current_mask = mask
        except Exception:
            self._log("Prediction thread error:\n" + traceback.format_exc())
            self.current_probs = (load_class_names(), np.zeros(len(CLASS_NAMES_DEFAULT)))
            self.current_mask = None
        finally:
            self._inference_running = False
            self.after(80, lambda: self._update_results(*self.current_probs))

    # -------------- events--------------
    def _on_pick(self, event):
        # event.artist 
        try:
            artist = event.artist
            if isinstance(artist, plt.Rectangle):
                idx = list(self.bars).index(artist)
                name = self.vis_names[idx]
                val = float(self.vis_probs[idx])
                self._show_bar_annotation(idx, name, val)
        except Exception:
            pass

    def _show_bar_annotation(self, idx, name, val):
        if self.bar_annotation:
            try:
                self.bar_annotation.remove()
            except Exception:
                pass
            self.bar_annotation = None
        # annotate on the bar area
        y = idx
        x = val
        self.bar_annotation = self.ax.annotate(
            f"{name} {val*100:.2f}%",
            xy=(x, y),
            xytext=(x + 0.05, y + 0.1),
            bbox=dict(boxstyle="round,pad=0.4", fc="#f7f9f8", ec=PRIMARY_TEAL, lw=1),
            color=ACCENT_TEAL
        )
        self.canvas_fig.draw()
        self._log(f"Clicked: {name} -> {val*100:.2f}%")

    # --------------  charts --------------
    def _update_results(self, names, probs):
        # sort desc for bars
        sorted_pairs = sorted(zip(names, probs), key=lambda x: -x[1])
        names_sorted = [p[0] for p in sorted_pairs]
        probs_sorted = np.array([p[1] for p in sorted_pairs])

        # threshold
        threshold = float(self.threshold_var.get()) / 100.0
        visible_mask = probs_sorted >= threshold
        if not visible_mask.any():
            visible_mask = np.ones_like(visible_mask, dtype=bool)
        vis_names = [n for n, m in zip(names_sorted, visible_mask) if m]
        vis_probs = np.array([p for p, m in zip(probs_sorted, visible_mask) if m])

        # store for pick events
        self.vis_names = vis_names
        self.vis_probs = vis_probs
        self.bars = []

        # bar chart styling 
        self.ax.clear()
        self.ax.set_facecolor(CARD_BG)
        self.ax.tick_params(colors=SUBTEXT)
        for spine in self.ax.spines.values():
            spine.set_color("#111214")

        inds = np.arange(len(vis_names))
        bars = self.ax.barh(inds, vis_probs,
            color=BAR_COLORS[:len(vis_probs)],
            edgecolor="#D0F0EA",
            height=0.56, linewidth=1.6, align="center"
        )

        # enable picking on bars
        for b in bars:
            try:
                b.set_picker(True)
            except Exception:
                pass

        self.bars = bars
        self.ax.set_yticks(inds)
        self.ax.set_yticklabels(vis_names, color=SUBTEXT, fontsize=11)
        self.ax.invert_yaxis()
        xlim = max(0.45, float(vis_probs.max() * 1.25))
        self.ax.set_xlim(0, xlim)
        self.ax.set_xlabel("Confidence Score", color=SUBTEXT)

        # add percent labels at end of bars
        for i, (b, v) in enumerate(zip(bars, vis_probs)):
            self.ax.text(
                v + xlim*0.02,
                b.get_y() + b.get_height()/2,
                f"{v*100:.1f}%",
                va="center",
                color=ACCENT_TEAL,
                fontweight="bold"
            )

        # donut: labels on slices and autopct inside
        self.ax2.clear()
        donut_colors = [PRIMARY_TEAL, ACCENT_TEAL, "#7CB7FF", "#A6A1FF",
                        "#8EE8E8", "#77809A", "#3F3B44"]
        wedges, texts, autotexts = self.ax2.pie(
            vis_probs,
            autopct=lambda p: f"{p:.1f}%" if p > 2 else "",
            pctdistance=0.7,
            wedgeprops=dict(width=0.52, edgecolor=BG, linewidth=1.2),
            startangle=self.donut_angle,
            colors=donut_colors[:len(vis_probs)],
            textprops={"color":TEXT, "fontsize":9}
        )
        self.ax2.axis("equal")
        for t in autotexts:
            t.set_color(ACCENT_TEAL)
            t.set_fontsize(9)
            t.set_weight("bold")

        # report in log area 
        top_i = int(np.argmax(probs_sorted))
        top_label = names_sorted[top_i]
        top_score = probs_sorted[top_i] * 100.0
        lvl = "High" if top_score >= 75 else ("Moderate" if top_score >= 45 else "Low")
        self.log_text.insert("end", "\n")
        self._log(f"Top predicted: {top_label} {top_score:.2f}% -> {lvl} Confidence")
        self._log("Full distribution:")
        for n, p in zip(names_sorted, probs_sorted):
            self._log(f"• {n:<8} {p*100:6.2f}%")

        self.canvas_fig.draw()

    def _start_donut_rotation(self):
        try:
            if self.donut_anim_id is not None:
                self.after_cancel(self.donut_anim_id)
        except Exception:
            pass

        def rotate_step():
            self.donut_angle = (self.donut_angle + 2.5) % 360.0
            # redraw 
            try:
                vis = getattr(self, "vis_probs", None)
                if vis is None or len(vis) == 0:
                    return
                self.ax2.clear()
                donut_colors = [PRIMARY_TEAL, ACCENT_TEAL, "#7CB7FF", "#A6A1FF",
                                "#8EE8E8", "#77809A", "#3F3B44"]
                wedges, texts, autotexts = self.ax2.pie(
                    vis,
                    autopct=lambda p: f"{p:.1f}%" if p > 2 else "",
                    pctdistance=0.7,
                    wedgeprops=dict(width=0.52, edgecolor=BG, linewidth=1.2),
                    startangle=self.donut_angle,
                    colors=donut_colors[:len(vis)],
                    textprops={"color":TEXT, "fontsize":9}
                )
                self.ax2.axis("equal")
                for t in autotexts:
                    t.set_color(ACCENT_TEAL)
                    t.set_fontsize(9)
                    t.set_weight("bold")
                self.canvas_fig.draw()
            except Exception:
                pass
            
        rotate_step()

    # -------------- heatmap toggle --------------
    def toggle_heatmap(self):
        if not self.loaded_image:
            return
        self.show_heat = not self.show_heat
        if self.show_heat and self.current_mask is not None:
            try:
                mask = np.asarray(self.current_mask)
                if mask.ndim == 3 and mask.shape[2] == 1:
                    mask = mask[..., 0]
                if mask.ndim == 4 and mask.shape[0] == 1:
                    mask = mask[0]
                overlay = generate_heatmap_like_from_mask(mask, (self.canvas_w, self.canvas_h))
                base = self.loaded_image.convert("RGBA")
                combo = Image.alpha_composite(base.convert("RGBA"), overlay)
                self.display_image_tk = ImageTk.PhotoImage(combo)
                self.image_canvas.delete("all")
                self.image_canvas.create_image(self.canvas_w//2, self.canvas_h//2, image=self.display_image_tk)
            except Exception:
                self._log("Error overlaying mask: " + traceback.format_exc())
        else:
            if self.loaded_image:
                self.display_image_tk = ImageTk.PhotoImage(self.loaded_image)
                self.image_canvas.delete("all")
                self.image_canvas.create_image(self.canvas_w//2, self.canvas_h//2, image=self.display_image_tk)
                # restore border
                thumb_w, thumb_h = self.loaded_image.size
                x = (self.canvas_w - thumb_w)//2
                y = (self.canvas_h - thumb_h)//2
                pad = 6
                x1 = max(0, x - pad); y1 = max(0, y - pad)
                x2 = min(self.canvas_w, x + thumb_w + pad); y2 = min(self.canvas_h, y + thumb_h + pad)
                self.image_canvas.create_rectangle(x1, y1, x2, y2, outline=PRIMARY_TEAL, width=2)
                self.image_canvas.create_rectangle(x1+3, y1+3, x2-3, y2-3, outline=ACCENT_TEAL, width=1)

    # -------------- CSV / PDF export --------------
    def export_csv(self):
        if not self.current_probs:
            return messagebox.showinfo("Error","No predictions to export.")
        names, probs = self.current_probs
        path = filedialog.asksaveasfilename(defaultextension=".csv",
                                            filetypes=[("CSV Files", "*.csv")])
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write("class,confidence\n")
                for n, p in zip(names, probs):
                    f.write(f"{n},{float(p):.6f}\n")
            self._log(f"CSV exported to: {path}")
            messagebox.showinfo("Saved", f"CSV exported to:\n{path}")
        except Exception as e:
            self._log("CSV EXPORT ERROR: " + str(e))
            messagebox.showerror("Error", "Unable to export CSV.")

    def export_pdf(self):
        if not REPORTLAB_AVAILABLE:
            return messagebox.showinfo("Missing", "reportlab not installed.")
        if (not self.current_probs) or (self.original_image is None):
            return messagebox.showinfo("Error", "Missing prediction or image.")
        names, probs = self.current_probs
        path = filedialog.asksaveasfilename(defaultextension=".pdf",
                                            filetypes=[("PDF Files", "*.pdf")])
        if not path:
            return
        try:
            c = canvas.Canvas(path, pagesize=A4)
            w, h = A4
            c.setFont("Helvetica-Bold", 16)
            c.drawString(40, h-60, "ECLIPSE Prediction Report")
            c.setFont("Helvetica", 10)
            c.drawString(40, h-80, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            buf = io.BytesIO()
            self.original_image.save(buf, format="PNG")
            buf.seek(0)
            c.drawImage(ImageReader(buf), 40, h-320, width=300, height=240)
            y = h-140
            c.setFont("Courier", 10)
            c.drawString(360, y+20, "Class".ljust(40) + "Confidence")
            for n, p in zip(names, probs):
                y -= 18
                c.drawString(360, y+20, f"{n[:32].ljust(40)}{p*100:6.2f}%")
            # mask thumbnail
            if self.current_mask is not None:
                try:
                    mask = np.asarray(self.current_mask)
                    if mask.ndim == 4 and mask.shape[0] == 1:
                        mask = mask[0]
                    if mask.ndim == 3 and mask.shape[2] == 1:
                        mask = mask[..., 0]
                    mask_img = (255 * (mask - mask.min()) / (mask.max() - mask.min() + 1e-9)).astype(np.uint8)
                    mask_pil = Image.fromarray(mask_img).convert("L").resize((200, 200), RESAMPLING)
                    buf2 = io.BytesIO(); mask_pil.save(buf2, format="PNG"); buf2.seek(0)
                    c.drawImage(ImageReader(buf2), 40, h-560, width=200, height=200)
                except Exception:
                    pass
            c.save()
            self._log(f"PDF exported to: {path}")
            messagebox.showinfo("Saved", f"PDF exported to:\n{path}")
        except Exception as e:
            self._log("PDF EXPORT ERROR: " + str(e))
            messagebox.showerror("Error", "Unable to export PDF.")

# launcher compatibility
EclipseApp = EclipseApp

if __name__ == "__main__":
    app = EclipseApp()
    app.mainloop()
