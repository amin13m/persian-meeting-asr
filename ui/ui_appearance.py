from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from .transcription_text import TranscriptionText

from .ui_constants import TEXTS, THEMES, is_persian_language


class UIAppearanceMixin:
    def configure_combobox_colors(self):

        colors = THEMES[self.theme_name]

        try:
            self.root.option_add(
                "*TCombobox*Listbox.background",
                colors["input_bg"],
            )

            self.root.option_add(
                "*TCombobox*Listbox.foreground",
                colors["text"],
            )

            self.root.option_add(
                "*TCombobox*Listbox.selectBackground",
                colors["selection"],
            )

            self.root.option_add(
                "*TCombobox*Listbox.selectForeground",
                colors["text"],
            )

            self.root.option_add(
                "*TCombobox*Listbox.font",
                self.font_normal,
            )

            self.root.option_add(
                "*TCombobox*Listbox.highlightColor",
                colors["border"],
            )

            self.root.option_add(
                "*TCombobox*Listbox.highlightBackground",
                colors["border"],
            )

            self.root.option_add(
                "*TCombobox*Listbox.borderWidth",
                0,
            )

            self.root.option_add(
                "*TCombobox*Listbox.relief",
                "flat",
            )

        except tk.TclError:
            pass

    def style_combobox_popup(self, combo):

        colors = THEMES[self.theme_name]

        try:
            popdown = self.root.tk.call(
                "ttk::combobox::PopdownWindow",
                str(combo),
            )

            children = self.root.tk.call(
                "winfo",
                "children",
                popdown,
            )

            for child in self.root.tk.splitlist(
                children
            ):
                try:
                    grandchildren = self.root.tk.call(
                        "winfo",
                        "children",
                        child,
                    )

                    for grandchild in self.root.tk.splitlist(
                        grandchildren
                    ):
                        if grandchild.endswith(".l"):
                            self.root.tk.call(
                                grandchild,
                                "configure",
                                "-background",
                                colors["input_bg"],
                                "-foreground",
                                colors["text"],
                                "-selectbackground",
                                colors["selection"],
                                "-selectforeground",
                                colors["text"],
                                "-font",
                                self.font_normal,
                                "-justify",
                                (
                                    "right"
                                    if self.language == "fa"
                                    else "left"
                                ),
                                "-borderwidth",
                                0,
                                "-relief",
                                "flat",
                            )

                except tk.TclError:
                    continue

        except tk.TclError:
            pass

    def apply_theme(self):

        colors = THEMES[self.theme_name]

        self.root.configure(
            bg=colors["bg"]
        )

        self.style.configure(
            ".",
            background=colors["bg"],
            foreground=colors["text"],
            font=self.font_normal,
        )

        self.style.configure(
            "TFrame",
            background=colors["bg"],
        )

        self.style.configure(
            "Card.TFrame",
            background=colors["surface"],
            bordercolor=colors["border"],
            borderwidth=1,
            relief="solid",
        )

        self.style.configure(
            "TLabel",
            background=colors["bg"],
            foreground=colors["text"],
        )

        self.style.configure(
            "Card.TLabel",
            background=colors["surface"],
            foreground=colors["text"],
        )

        # ----------------------------------------------------
        # Entry
        # ----------------------------------------------------

        self.style.configure(
            "TEntry",
            fieldbackground=colors["input_bg"],
            foreground=colors["text"],
            bordercolor=colors["border"],
            padding=(9, 7),
        )

        self.style.map(
            "TEntry",
            fieldbackground=[
                (
                    "disabled",
                    colors["surface2"],
                ),
                (
                    "readonly",
                    colors["input_bg"],
                ),
            ],
            foreground=[
                (
                    "disabled",
                    colors["disabled"],
                ),
                (
                    "readonly",
                    colors["text"],
                ),
            ],
        )

        # ----------------------------------------------------
        # ComboBox
        # ----------------------------------------------------

        self.style.configure(
            "TCombobox",
            fieldbackground=colors["input_bg"],
            background=colors["input_bg"],
            foreground=colors["text"],
            bordercolor=colors["border"],
            arrowcolor=colors["muted"],
            padding=(8, 6),
        )

        self.style.map(
            "TCombobox",
            fieldbackground=[
                (
                    "readonly",
                    colors["input_bg"],
                ),
                (
                    "disabled",
                    colors["surface2"],
                ),
            ],
            background=[
                (
                    "readonly",
                    colors["input_bg"],
                ),
                (
                    "disabled",
                    colors["surface2"],
                ),
            ],
            foreground=[
                (
                    "readonly",
                    colors["text"],
                ),
                (
                    "disabled",
                    colors["disabled"],
                ),
            ],
            arrowcolor=[
                (
                    "readonly",
                    colors["muted"],
                ),
                (
                    "disabled",
                    colors["disabled"],
                ),
            ],
            selectbackground=[
                (
                    "readonly",
                    colors["selection"],
                ),
            ],
            selectforeground=[
                (
                    "readonly",
                    colors["text"],
                ),
            ],
        )

        self.configure_combobox_colors()

        # ----------------------------------------------------
        # Notebook
        # ----------------------------------------------------

        self.style.configure(
            "TNotebook",
            background=colors["bg"],
            borderwidth=0,
            tabmargins=(0, 0, 0, 0),
        )

        self.style.configure(
            "TNotebook.Tab",
            background=colors["surface2"],
            foreground=colors["muted"],
            padding=(16, 7),
            font=self.font_normal,
            borderwidth=0,
            relief="flat",
            focuscolor=colors["surface2"],
        )

        self.style.map(
            "TNotebook.Tab",
            background=[
                (
                    "selected",
                    colors["surface"],
                ),
                (
                    "active",
                    colors["surface2"],
                ),
            ],
            foreground=[
                (
                    "selected",
                    colors["text"],
                ),
                (
                    "active",
                    colors["text"],
                ),
            ],
        )

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        self.style.configure(
            "Horizontal.TProgressbar",
            troughcolor=colors["surface2"],
            background=colors["accent"],
            bordercolor=colors["surface2"],
            lightcolor=colors["accent"],
            darkcolor=colors["accent"],
        )

        # ----------------------------------------------------
        # Text
        # ----------------------------------------------------

        self.transcription_text.configure(
            bg=colors["input_bg"],
            fg=colors["text"],
            insertbackground=colors["text"],
            selectbackground=colors["selection"],
            selectforeground=colors["text"],
        )

        self.logs_text.configure(
            bg=colors["log_bg"],
            fg=colors["text"],
            insertbackground=colors["text"],
            selectbackground=colors["selection"],
            selectforeground=colors["text"],
        )

        # ----------------------------------------------------
        # Canvas
        # ----------------------------------------------------

        self.spinner_canvas.configure(
            bg=colors["bg"],
        )

        self.status_dot.configure(
            bg=colors["bg"],
        )

        # ----------------------------------------------------
        # Buttons
        # ----------------------------------------------------

        buttons = [
            self.load_button,
            self.unload_button,
            self.select_audio_button,
            self.start_record_button,
            self.stop_record_button,
            self.transcribe_button,
            self.copy_text_button,
            self.export_txt_button,
            self.export_json_button,
            self.open_output_button,
            self.clear_logs_button,
        ]

        for button in buttons:
            button.set_theme(colors)

        self.configure_log_tags()
        self.update_status_dot()

    def configure_log_tags(self):

        if not hasattr(
            self,
            "logs_text",
        ):
            return

        colors = THEMES[
            self.theme_name
        ]

        self.logs_text.tag_configure(
            "INFO",
            foreground=colors["info"],
        )

        self.logs_text.tag_configure(
            "WARNING",
            foreground=colors["warning"],
        )

        self.logs_text.tag_configure(
            "ERROR",
            foreground=colors["error"],
        )

        self.logs_text.tag_configure(
            "DEBUG",
            foreground=colors["muted"],
        )

        self.logs_text.tag_configure(
            "DEFAULT",
            foreground=colors["text"],
        )

    def update_language(self):

        t = TEXTS[self.language]

        self.root.title(
            t["app_title"]
        )

        self.title_label.configure(
            text=t["app_title"]
        )

        self.subtitle_label.configure(
            text=t["app_subtitle"]
        )

        self.language_label.configure(
            text=t["language"]
        )

        self.theme_label.configure(
            text=t["theme"]
        )

        self.notebook.tab(
            self.main_tab,
            text=t["main"],
        )

        self.notebook.tab(
            self.logs_tab,
            text=t["logs"],
        )

        self.model_title.configure(
            text=t["model_section"]
        )

        self.audio_title.configure(
            text=t["audio_section"]
        )

        self.processing_title.configure(
            text=t["processing_section"]
        )

        self.transcription_title.configure(
            text=t["transcription_section"]
        )

        self.model_label.configure(
            text=t["model"]
        )

        self.model_status_label.configure(
            text=t["model_status"]
        )

        self.load_button.configure(
            text=t["load_model"]
        )

        self.unload_button.configure(
            text=t["unload_model"]
        )

        self.audio_label.configure(
            text=t["audio_file"]
        )

        self.select_audio_button.configure(
            text=t["select_audio"]
        )

        self.start_record_button.configure(
            text=t["start_recording"]
        )

        self.stop_record_button.configure(
            text=t["stop_recording"]
        )

        self.transcribe_button.configure(
            text=t["transcribe"]
        )

        self.output_label.configure(
            text=t["output_section"]
        )

        self.export_txt_button.configure(
            text=t["export_txt"]
        )
        
        self.copy_text_button.configure(
            text=t["copy_text"]
        )

        self.export_json_button.configure(
            text=t["export_json"]
        )

        self.open_output_button.configure(
            text=t["open_output"]
        )

        self.logs_title_label.configure(
            text=t["logs_title"]
        )

        self.clear_logs_button.configure(
            text=t["clear_logs"]
        )

        self.refresh_status_text()

    def apply_direction(self):

        rtl = is_persian_language(
            self.language
        )

        justify = (
            "right"
            if rtl
            else "left"
        )

        anchor = (
            "e"
            if rtl
            else "w"
        )

        # ----------------------------------------------------
        # Header
        # ----------------------------------------------------

        self.header_left.pack_forget()
        self.header_right.pack_forget()

        if rtl:
            self.header_right.pack(
                side="left"
            )

            self.header_left.pack(
                side="right"
            )

        else:
            self.header_left.pack(
                side="left"
            )

            self.header_right.pack(
                side="right"
            )

        self.title_label.configure(
            anchor=anchor,
            justify=justify,
        )

        self.subtitle_label.configure(
            anchor=anchor,
            justify=justify,
        )

        # ----------------------------------------------------
        # Header Controls
        # ----------------------------------------------------

        for widget in (
            self.language_label,
            self.language_combo,
            self.theme_label,
            self.theme_combo,
        ):
            widget.pack_forget()

        if rtl:

            self.theme_combo.pack(
                side="left",
                padx=(5, 0),
            )

            self.theme_label.pack(
                side="left",
                padx=(14, 0),
            )

            self.language_combo.pack(
                side="left",
                padx=(5, 0),
            )

            self.language_label.pack(
                side="left"
            )

        else:

            self.language_label.pack(
                side="left"
            )

            self.language_combo.pack(
                side="left",
                padx=(5, 14),
            )

            self.theme_label.pack(
                side="left"
            )

            self.theme_combo.pack(
                side="left",
                padx=(5, 0),
            )

        # ----------------------------------------------------
        # Section Titles
        # ----------------------------------------------------

        for title in (
            self.model_title,
            self.audio_title,
            self.processing_title,
            self.transcription_title,
        ):
            title.configure(
                anchor=anchor,
                justify=justify,
            )

        # ----------------------------------------------------
        # Labels
        # ----------------------------------------------------

        for label in (
            self.model_label,
            self.model_status_label,
            self.audio_label,
            self.output_label,
            self.status_label,
            self.operation_label,
            self.logs_title_label,
        ):
            try:
                label.configure(
                    anchor=anchor,
                    justify=justify,
                )
            except tk.TclError:
                pass

        # ----------------------------------------------------
        # Model
        # ----------------------------------------------------

        for widget in (
            self.model_label,
            self.model_combo,
            self.load_button,
            self.unload_button,
            self.model_status_label,
        ):
            widget.grid_forget()

        if rtl:

            self.model_label.grid(
                row=1,
                column=4,
                sticky="e",
                padx=(10, 8),
            )

            self.model_combo.grid(
                row=1,
                column=3,
                sticky="ew",
                padx=8,
            )

            self.load_button.grid(
                row=1,
                column=2,
                padx=4,
            )

            self.unload_button.grid(
                row=1,
                column=1,
                padx=4,
            )

            self.model_status_label.grid(
                row=1,
                column=0,
                sticky="w",
                padx=(8, 0),
            )

        else:

            self.model_label.grid(
                row=1,
                column=0,
                sticky="w",
                padx=(0, 8),
            )

            self.model_combo.grid(
                row=1,
                column=1,
                sticky="ew",
                padx=8,
            )

            self.load_button.grid(
                row=1,
                column=2,
                padx=4,
            )

            self.unload_button.grid(
                row=1,
                column=3,
                padx=4,
            )

            self.model_status_label.grid(
                row=1,
                column=4,
                sticky="e",
                padx=(8, 0),
            )

        # ----------------------------------------------------
        # Audio
        # ----------------------------------------------------

        for widget in (
            self.audio_label,
            self.audio_entry,
            self.select_audio_button,
        ):
            widget.grid_forget()

        if rtl:

            self.audio_label.grid(
                row=1,
                column=2,
                sticky="e",
                padx=(10, 8),
            )

            self.audio_entry.grid(
                row=1,
                column=1,
                sticky="ew",
                padx=8,
            )

            self.select_audio_button.grid(
                row=1,
                column=0,
                padx=(4, 0),
            )

        else:

            self.audio_label.grid(
                row=1,
                column=0,
                sticky="w",
                padx=(0, 8),
            )

            self.audio_entry.grid(
                row=1,
                column=1,
                sticky="ew",
                padx=8,
            )

            self.select_audio_button.grid(
                row=1,
                column=2,
                padx=(4, 0),
            )

        # ----------------------------------------------------
        # Processing
        # ----------------------------------------------------

        for widget in (
            self.start_record_button,
            self.stop_record_button,
            self.transcribe_button,
        ):
            widget.pack_forget()

        self.processing_actions.pack(
            fill="x"
        )

        if rtl:

            self.transcribe_button.pack(
                side="right",
            )

            self.stop_record_button.pack(
                side="right",
                padx=8,
            )

            self.start_record_button.pack(
                side="right"
            )

        else:

            self.start_record_button.pack(
                side="left"
            )

            self.stop_record_button.pack(
                side="left",
                padx=8,
            )

            self.transcribe_button.pack(
                side="left",
                padx=(8, 0),
            )

        # ----------------------------------------------------
        # Status
        # ----------------------------------------------------

        for widget in (
            self.status_dot,
            self.status_label,
            self.operation_label,
            self.progress,
            self.progress_text_label,
            self.spinner_canvas,
        ):
            widget.grid_forget()

        for column in range(6):
            self.status_frame.columnconfigure(
                column,
                weight=0,
            )

        if rtl:

            self.status_frame.columnconfigure(
                3,
                weight=1,
            )

            self.status_dot.grid(
                row=0,
                column=5,
                padx=(8, 4),
            )

            self.status_label.grid(
                row=0,
                column=4,
                sticky="e",
                padx=(0, 10),
            )

            self.operation_label.grid(
                row=0,
                column=3,
                sticky="e",
                padx=8,
            )

            self.progress.grid(
                row=0,
                column=2,
                sticky="ew",
                padx=10,
            )

            self.progress_text_label.grid(
                row=0,
                column=1,
                padx=8,
            )

            self.spinner_canvas.grid(
                row=0,
                column=0,
                padx=(4, 0),
            )

        else:

            self.status_frame.columnconfigure(
                2,
                weight=1,
            )

            self.status_dot.grid(
                row=0,
                column=0,
                padx=(4, 8),
            )

            self.status_label.grid(
                row=0,
                column=1,
                sticky="w",
                padx=(0, 10),
            )

            self.operation_label.grid(
                row=0,
                column=2,
                sticky="w",
                padx=8,
            )

            self.progress.grid(
                row=0,
                column=3,
                sticky="ew",
                padx=10,
            )

            self.progress_text_label.grid(
                row=0,
                column=4,
                padx=8,
            )

            self.spinner_canvas.grid(
                row=0,
                column=5,
                padx=(0, 4),
            )

        # ----------------------------------------------------
        # Input Direction
        # ----------------------------------------------------

        try:
            self.language_combo.configure(
                justify=justify
            )

            self.theme_combo.configure(
                justify=justify
            )

            self.model_combo.configure(
                justify=justify
            )

            self.audio_entry.configure(
                justify=justify
            )

        except tk.TclError:
            pass

        # ----------------------------------------------------
        # Transcription
        # ----------------------------------------------------

        self.transcription_text.tag_configure(
            "rtl",
            justify="right",
            lmargin1=8,
            lmargin2=8,
            rmargin=8,
        )

        self.transcription_text.tag_configure(
            "ltr",
            justify="left",
            lmargin1=8,
            lmargin2=8,
            rmargin=8,
        )

        try:
        
            self.transcription_text.tag_remove(
                "rtl",
                "1.0",
                "end",
            )
        
            self.transcription_text.tag_remove(
                "ltr",
                "1.0",
                "end",
            )
        
            self.transcription_text.tag_add(
                "rtl" if rtl else "ltr",
                "1.0",
                "end-1c",
            )
        
        except tk.TclError:
            pass

        # ----------------------------------------------------
        # Output
        # ----------------------------------------------------

        for widget in (
            self.output_label,
            self.copy_text_button,
            self.export_txt_button,
            self.export_json_button,
            self.open_output_button,
        ):
            widget.pack_forget()

        self.output_label.pack(
            side="right" if rtl else "left"
        )

        if rtl:
        
            self.open_output_button.pack(
                side="left",
                padx=(0, 6),
            )

            self.export_json_button.pack(
                side="left",
                padx=6,
            )

            self.export_txt_button.pack(
                side="left",
                padx=6,
            )

            self.copy_text_button.pack(
                side="left",
                padx=6,
            )

        else:
        
            self.copy_text_button.pack(
                side="left",
                padx=(14, 6),
            )

            self.export_txt_button.pack(
                side="left",
                padx=6,
            )

            self.export_json_button.pack(
                side="left",
                padx=6,
            )

            self.open_output_button.pack(
                side="left",
                padx=6,
            )

        # ----------------------------------------------------
        # Logs
        # ----------------------------------------------------

        self.logs_title_label.pack_forget()
        self.clear_logs_button.pack_forget()

        if rtl:

            self.clear_logs_button.pack(
                side="left"
            )

            self.logs_title_label.pack(
                side="right"
            )

        else:

            self.logs_title_label.pack(
                side="left"
            )

            self.clear_logs_button.pack(
                side="right"
            )

    def on_theme_change(self, _event=None):

        value = self.theme_var.get()

        if value in (
            "Dark",
            "تیره",
        ):
            self.theme_name = "dark"
        else:
            self.theme_name = "light"

        self.apply_theme()

    def on_language_change(self, _event=None):

        value = self.language_var.get()

        if value == "فارسی":
            self.language = "fa"
        else:
            self.language = "en"

        self.update_language()
        self.apply_direction()

    def set_status(
        self,
        status_key,
        operation_text=None,
    ):

        self.status_var.set(
            TEXTS[self.language].get(
                status_key,
                status_key,
            )
        )

        if operation_text:
            self.operation_var.set(
                operation_text
            )
        else:
            self.operation_var.set("")

        self.update_status_dot()

    def refresh_status_text(self):

        if self.operation:

            mapping = {
                "loading": "loading",
                "unloading": "unloading",
                "processing": "processing",
                "recording": "recording",
            }

            key = mapping.get(
                self.operation,
                "ready",
            )

            self.status_var.set(
                TEXTS[self.language][key]
            )

        elif self.model_loaded:

            self.status_var.set(
                TEXTS[self.language]["loaded"]
            )

        else:

            self.status_var.set(
                TEXTS[self.language]["ready"]
            )

        if hasattr(
            self,
            "status_label",
        ):
            self.status_label.configure(
                textvariable=self.status_var
            )

        if hasattr(
            self,
            "operation_label",
        ):
            self.operation_label.configure(
                textvariable=self.operation_var
            )

        self.update_status_dot()

    def update_status_dot(self):

        if not hasattr(
            self,
            "status_dot",
        ):
            return

        colors = THEMES[
            self.theme_name
        ]

        if self.operation in (
            "loading",
            "unloading",
        ):
            color = colors["warning"]

        elif self.operation == "processing":
            color = colors["accent"]

        elif self.operation == "recording":
            color = colors["error"]

        elif self.model_loaded:
            color = colors["success"]

        else:
            color = colors["muted"]

        self.status_dot.itemconfigure(
            "dot",
            fill=color,
            outline=color,
        )

    def start_spinner(self):

        if self.spinner_running:
            return

        self.spinner_running = True
        self.spinner_angle = 0

        self.animate_spinner()

    def stop_spinner(self):

        self.spinner_running = False

        try:
            self.spinner_canvas.itemconfigure(
                "spinner",
                state="hidden",
            )
        except tk.TclError:
            pass

    def animate_spinner(self):

        if not self.spinner_running:
            return

        colors = THEMES[
            self.theme_name
        ]

        self.spinner_canvas.itemconfigure(
            "spinner",
            state="normal",
            outline=colors["accent"],
        )

        self.spinner_canvas.itemconfigure(
            "spinner",
            start=self.spinner_angle,
        )

        self.spinner_angle = (
            self.spinner_angle + 25
        ) % 360

        self.root.after(
            70,
            self.animate_spinner,
        )

    def refresh_controls(self):

        if self.processing:

            state_model = "disabled"
            state_audio = "disabled"
            state_start = "disabled"
            state_stop = "disabled"
            state_transcribe = "disabled"
            state_load = "disabled"
            state_unload = "disabled"
            state_export = "disabled"

        elif self.recording:

            state_model = "disabled"
            state_audio = "disabled"
            state_start = "disabled"
            state_stop = "normal"
            state_transcribe = "disabled"
            state_load = "disabled"
            state_unload = "disabled"
            state_export = "disabled"

        elif not self.model_loaded:

            state_model = "readonly"
            state_audio = "normal"
            state_start = "normal"
            state_stop = "disabled"
            state_transcribe = "disabled"
            state_load = "normal"
            state_unload = "disabled"

            state_export = (
                "normal"
                if self.current_result
                else "disabled"
            )

        else:

            state_model = "readonly"
            state_audio = "normal"
            state_start = "normal"
            state_stop = "disabled"

            state_transcribe = (
                "normal"
                if self.current_audio
                else "disabled"
            )

            state_load = "disabled"
            state_unload = "normal"

            state_export = (
                "normal"
                if self.current_result
                else "disabled"
            )

        self.model_combo.configure(
            state=state_model
        )

        self.load_button.configure(
            state=state_load
        )

        self.unload_button.configure(
            state=state_unload
        )

        self.select_audio_button.configure(
            state=state_audio
        )

        self.start_record_button.configure(
            state=state_start
        )

        self.stop_record_button.configure(
            state=state_stop
        )

        self.transcribe_button.configure(
            state=state_transcribe
        )

        self.export_txt_button.configure(
            state=state_export
        )
        
        self.copy_text_button.configure(
            state=state_export
        )

        self.export_json_button.configure(
            state=state_export
        )

        self.open_output_button.configure(
            state=(
                "normal"
                if self.current_result
                else "disabled"
            )
        )

        if (
            self.processing
            or self.recording
        ):
            self.start_spinner()
        else:
            self.stop_spinner()

        self.update_status_dot()

