"""Small Tk/XWayland fullscreen viewer; no browser or HTTP server needed."""
import argparse
from datetime import datetime
import json
import logging
from pathlib import Path
import tkinter as tk
from tkinter import font
from PIL import Image, ImageTk

BG, INK, MUTED, ACCENT = "#f3efe6", "#202620", "#657066", "#b55333"


class Display:
    def __init__(self, directory, fullscreen):
        self.directory, self.signature, self.state = directory, None, None
        self.root = tk.Tk()
        self.root.title("News Art Pi")
        self.root.configure(bg=BG)
        self.root.geometry("1100x700")
        self.root.attributes("-fullscreen", fullscreen)
        self.canvas = tk.Canvas(self.root, bg=BG, highlightthickness=0, cursor="none" if fullscreen else "")
        self.canvas.pack(fill="both", expand=True)
        self.root.bind("<Escape>", lambda _: self.root.attributes("-fullscreen", False))
        self.root.bind("<F11>", lambda _: self.root.attributes("-fullscreen", not self.root.attributes("-fullscreen")))
        self.root.bind("<Configure>", self.resized)
        self.resize_job = None
        self.root.after(200, self.poll)
        # At desktop login XWayland/the compositor can map the first window after
        # the initial fullscreen request. Reapply once after the window is mapped.
        if fullscreen:
            self.root.after(1500, lambda: self.root.attributes("-fullscreen", True))

    def resized(self, event):
        if event.widget == self.root:
            if self.resize_job:
                self.root.after_cancel(self.resize_job)
            self.resize_job = self.root.after(150, self.draw)

    def poll(self):
        try:
            raw = (self.directory / "state.json").read_text()
            if raw != self.signature:
                candidate = json.loads(raw)
                with Image.open(self.directory / candidate["image"]) as source:
                    self.picture = source.convert("RGB")
                self.state, self.signature = candidate, raw
                self.draw()
        except FileNotFoundError:
            if not self.state:
                self.draw()
        except Exception:
            logging.exception("Cannot load new poster; keeping previous display")
        self.root.after(2000, self.poll)

    def draw(self):
        self.resize_job = None
        canvas = self.canvas
        canvas.delete("all")
        w, h = max(canvas.winfo_width(), 320), max(canvas.winfo_height(), 240)
        pad = max(18, int(min(w, h) * .055))
        if not self.state:
            canvas.create_text(w/2, h/2, text="NEWS ART PI\nWaiting for the first headline…", fill=INK,
                               font=("DejaVu Sans", 22), justify="center", width=w-2*pad)
            return
        state = self.state
        header_size = max(9, int(min(w, h) * .014))
        canvas.create_text(pad, pad, anchor="nw", text=state["title"], fill=MUTED,
                           font=("DejaVu Sans", header_size, "bold"))
        canvas.create_line(pad, pad*1.7, w-pad, pad*1.7, fill="#cbc8bd")
        top = int(pad*2.3)
        landscape = w >= h * 1.15
        if landscape:
            side = max(80, min(int(w*.47), h-top-pad*2))
            x, y, text_width = pad+side+pad, top, w-side-3*pad
            text_height = h-top-3*pad
        else:
            side = max(80, min(w-2*pad, int(h*.42)))
            x, y, text_width = pad, top+side+pad, w-2*pad
            text_height = h-y-3*pad
        self.photo = ImageTk.PhotoImage(self.picture.resize((side, side), Image.Resampling.LANCZOS))
        canvas.create_image(pad if landscape else (w-side)//2, top, anchor="nw", image=self.photo)
        canvas.create_text(x, y, anchor="nw", text="LATEST DISPATCH", fill=ACCENT,
                           font=("DejaVu Sans", header_size, "bold"))
        headline_font = font.Font(family="DejaVu Serif", size=max(12, int(min(w, h)*.038)), weight="bold")
        headline = canvas.create_text(x, y+pad*.65, anchor="nw", text=state["story"]["headline"],
                                      fill=INK, font=headline_font, width=max(100, text_width))
        while headline_font.cget("size") > 10:
            box = canvas.bbox(headline)
            if box[3]-box[1] <= max(40, text_height-pad):
                break
            headline_font.configure(size=headline_font.cget("size")-1)
        self.headline_font = headline_font
        published = datetime.fromisoformat(state["story"]["published"]).astimezone().strftime("%d %b · %H:%M")
        updated = datetime.fromisoformat(state["generated_at"]).astimezone().strftime("%H:%M")
        canvas.create_text(pad, h-pad*1.45, anchor="sw", fill=MUTED, width=w-2*pad,
                           text=f'{state["story"]["source"]}  /  {published}  /  Updated {updated}',
                           font=("DejaVu Sans", header_size))
        canvas.create_text(pad, h-pad*.55, anchor="sw", fill=ACCENT, width=w-2*pad,
                           text=({"supra": "AI-generated illustration · Supra2 · Not a documentary photograph",
                                  "inko": "AI-generated illustration · Inko · Not a documentary photograph"}
                                 .get(state.get("backend"), "DEMO IMAGE · Unrelated to the headline")),
                           font=("DejaVu Sans", max(8, header_size-1)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-dir", type=Path, default=Path("state"))
    parser.add_argument("--windowed", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    Display(args.state_dir, not args.windowed).root.mainloop()


if __name__ == "__main__":
    main()
