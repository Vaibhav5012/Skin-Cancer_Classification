import tkinter as tk
import threading, time
from PIL import Image, ImageTk, ImageEnhance, ImageFilter, ImageDraw

from main import EclipseApp  # DO NOT change this


class WelcomeScreen:
    def __init__(self, root):
        self.root = root
        self.root.title("ECLIPSE — Skin Lesion Analyzer")
        self.root.geometry("1200x820")
        self.root.configure(bg="#1a0736")
        self.root.attributes("-alpha", 0.0)

        self.alpha = 0.0
        self.alive = True  # <-- important

        # Background canvas
        self.canvas = tk.Canvas(self.root, highlightthickness=0, bd=0)
        self.canvas.pack(fill="both", expand=True)

        try:
            bg = Image.open("assets/background.png").convert("RGB").resize((1600, 900))
            bg = ImageEnhance.Brightness(bg).enhance(0.6)
            bg = bg.filter(ImageFilter.GaussianBlur(radius=4))

            glass = Image.new("RGBA", bg.size, (255, 255, 255, 35))
            final = Image.alpha_composite(bg.convert("RGBA"), glass)

            overlay = Image.new("RGBA", bg.size, (80, 0, 150, 70))
            final = Image.alpha_composite(final, overlay)

            self.bg_imgtk = ImageTk.PhotoImage(final)
            self.canvas.create_image(0, 0, anchor="nw", image=self.bg_imgtk)
        except:
            self.canvas.configure(bg="#1a0736")

        # Title
        self.title = tk.Label(self.root, text="ECLIPSE",
                              fg="#b69cff", bg="#1a0736",
                              font=("Segoe UI", 52, "bold"))
        self.title.place(relx=0.5, rely=0.36, anchor="center")

        # Tagline
        self.tagline = tk.Label(self.root,
                                text="Illuminating Skin Health Through AI",
                                fg="#c9d7ff", bg="#1a0736",
                                font=("Segoe UI", 18, "italic"))
        self.tagline.place(relx=0.5, rely=0.46, anchor="center")

        # Button
        self.start_btn = tk.Button(
            self.root, text="Get Started",
            fg="white", bg="#6f5aff",
            font=("Segoe UI", 14, "bold"),
            command=self.fade_out_and_start,
            relief="flat", width=16
        )
        self.start_btn.place(relx=0.5, rely=0.58, anchor="center")

        # Animations
        threading.Thread(target=lambda: playsound("assets/sounds/intro_chime.wav"), daemon=True).start()

        self.fade_in()
        self.float_text()
        self.animate_title()
        self.pulse_button()

    # ---------------------------
    def fade_in(self):
        if not self.alive: return
        self.alpha = min(self.alpha + 0.04, 1.0)
        self.root.attributes("-alpha", self.alpha)
        if self.alpha < 1.0:
            self.root.after(40, self.fade_in)

    def animate_title(self):
        if not self.alive or not self.title.winfo_exists(): return
        for c in ["#b69cff", "#a68aff", "#c2b0ff", "#8d7bff"]:
            if not self.alive: return
            self.title.config(fg=c)
            self.root.update()
            time.sleep(0.15)
        self.root.after(120, self.animate_title)

    def float_text(self):
        if not self.alive or not self.tagline.winfo_exists(): return
        y = self.tagline.winfo_y()
        self.tagline.place_configure(y= y + 1 if y < 410 else 370)
        self.root.after(80, self.float_text)

    def pulse_button(self):
        if not self.alive or not self.start_btn.winfo_exists(): return
        self.start_btn.config(bg="#8b7cff" if self.start_btn.cget("bg")=="#6f5aff" else "#6f5aff")
        self.root.after(700, self.pulse_button)

    # ---------------------------
    def fade_out_and_start(self):
        # Stop all animations
        self.alive = False

        for i in range(20):
            self.root.attributes("-alpha", 1 - i*0.05)
            self.root.update()
            time.sleep(0.03)

        self.root.destroy()

        # 🔥 START MAIN APP CORRECTLY
        app = EclipseApp()      # <-- NO ARGUMENTS
        app.mainloop()


# ---------------------------
if __name__ == "__main__":
    root = tk.Tk()
    WelcomeScreen(root)
    root.mainloop()
