from __future__ import annotations

import logging
import tkinter as tk
import math

class RoundedButton(tk.Canvas):

    def __init__(
        self,
        master,
        text="",
        command=None,
        width=130,
        height=38,
        radius=12,
        kind="secondary",
        font=None,
        background_key="surface",
        **kwargs,
    ):
        self._button_width = width
        self._button_height = height
        self._radius = radius
        self._kind = kind
        self._font = font or ("Segoe UI", 10)
        self._background_key = background_key

        self._text = text
        self._command = command

        self._state = "normal"
        self._hover = False
        self._pressed = False
        self._release_job = None
        self._colors = None

        initial_bg = kwargs.pop(
            "background",
            kwargs.pop("bg", "#202328"),
        )

        super().__init__(
            master,
            width=width,
            height=height,
            bg=initial_bg,
            highlightthickness=0,
            bd=0,
            relief="flat",
            cursor="hand2",
            takefocus=True,
        )

        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        self.bind("<ButtonPress-1>", self._on_press)
        self.bind("<ButtonRelease-1>", self._on_release)

    def _on_enter(self, _event=None):
        if self._state == "disabled":
            return

        self._hover = True
        self.configure(cursor="hand2")
        self._draw()

    def _on_leave(self, _event=None):
        if self._state == "disabled":
            return

        self._hover = False

        if not self._pressed:
            self._draw()

    def _on_press(self, _event=None):
        if self._state == "disabled":
            return

        self._pressed = True
        self._draw()

    def _on_release(self, event=None):
        if self._state == "disabled":
            return

        inside = True

        if event is not None:
            inside = (
                0 <= event.x <= self._button_width
                and
                0 <= event.y <= self._button_height
            )

        if not inside:
            self._pressed = False
            self._draw()
            return

        self._pressed = True
        self._draw()

        if self._release_job is not None:
            try:
                self.after_cancel(self._release_job)
            except tk.TclError:
                pass

        self._release_job = self.after(
            85,
            self._finish_click,
        )

    def _finish_click(self):
        self._release_job = None

        self._pressed = False
        self._draw()

        if self._command is not None:
            try:
                self._command()
            except Exception:
                logging.getLogger(
                    "PersianASR.UI"
                ).exception(
                    "Button callback failed."
                )

    def _rounded_points(
        self,
        x1,
        y1,
        x2,
        y2,
        radius,
    ):
        points = []

        centers = [
            (x1 + radius, y1 + radius, 180, 270),
            (x2 - radius, y1 + radius, 270, 360),
            (x2 - radius, y2 - radius, 0, 90),
            (x1 + radius, y2 - radius, 90, 180),
        ]

        for cx, cy, start, end in centers:
            for angle in range(start, end + 1, 10):
                rad = math.radians(angle)

                points.extend([
                    cx + radius * math.cos(rad),
                    cy + radius * math.sin(rad),
                ])

        return points

    def set_theme(self, colors):
        self._colors = colors

        canvas_bg = colors.get(
            self._background_key,
            colors["surface"],
        )

        self.configure(bg=canvas_bg)

        self._draw()

    def set_state(self, state):
        self._state = state

        if state == "disabled":
            self._hover = False
            self._pressed = False
            self.configure(cursor="arrow")
        else:
            self.configure(cursor="hand2")

        self._draw()

    def _draw(self):
        self.delete("all")

        if not self._colors:
            return

        colors = self._colors

        if self._state == "disabled":
            fill = colors["surface2"]
            text_color = colors["disabled"]

        elif self._pressed:
            if self._kind == "accent":
                fill = colors["accent_pressed"]
                text_color = "#FFFFFF"

            elif self._kind == "danger":
                fill = colors["border"]
                text_color = colors["error"]

            else:
                fill = colors["border"]
                text_color = colors["text"]

        elif self._hover:
            if self._kind == "accent":
                fill = colors["accent_hover"]
                text_color = "#FFFFFF"

            elif self._kind == "danger":
                fill = colors["surface"]
                text_color = colors["error"]

            else:
                fill = colors["surface"]
                text_color = colors["text"]

        else:
            if self._kind == "accent":
                fill = colors["accent"]
                text_color = "#FFFFFF"

            elif self._kind == "danger":
                fill = colors["surface2"]
                text_color = colors["error"]

            else:
                fill = colors["surface2"]
                text_color = colors["text"]

        pad = 1

        points = self._rounded_points(
            pad,
            pad,
            self._button_width - pad,
            self._button_height - pad,
            min(
                self._radius,
                self._button_height // 2,
            ),
        )

        self.create_polygon(
            points,
            fill=fill,
            outline=colors["border"],
            width=1,
            smooth=True,
        )

        self.create_text(
            self._button_width / 2,
            self._button_height / 2,
            text=self._text,
            fill=text_color,
            font=self._font,
            anchor="center",
        )

    def configure(self, cnf=None, **kwargs):
        if cnf:
            kwargs.update(cnf)

        if "text" in kwargs:
            self._text = kwargs.pop("text")

        if "command" in kwargs:
            self._command = kwargs.pop("command")

        if "state" in kwargs:
            state = kwargs.pop("state")
            self.set_state(state)

        if "button_background" in kwargs:
            self._background_key = kwargs.pop(
                "button_background"
            )

        result = super().configure(**kwargs)

        if hasattr(self, "_colors") and self._colors:
            self._draw()

        return result

    config = configure

    def cget(self, key):
        if key == "text":
            return self._text

        if key == "state":
            return self._state

        if key == "command":
            return self._command

        return super().cget(key)