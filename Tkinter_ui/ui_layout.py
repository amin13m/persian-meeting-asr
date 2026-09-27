from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from tkinter import font as tkfont
from .transcription_text import TranscriptionText

from .ui_components import RoundedButton
from .ui_constants import ENGINE_REGISTRY, THEMES


class UILayoutMixin:
    def _setup_fonts(self):

        available = set(
            tkfont.families(self.root)
        )

        if "Segoe UI" in available:
            family = "Segoe UI"

        elif "Tahoma" in available:
            family = "Tahoma"

        elif "Arial" in available:
            family = "Arial"

        else:
            family = (
                tkfont.nametofont(
                    "TkDefaultFont"
                ).actual(
                    "family"
                )
            )

        self.font_family = family

        self.font_normal = (
            family,
            10,
        )

        self.font_small = (
            family,
            9,
        )

        self.font_medium = (
            family,
            11,
        )

        self.font_button = (
            family,
            10,
        )

        self.font_section = (
            family,
            10,
            "bold",
        )

        self.font_title = (
            family,
            22,
            "bold",
        )

        self.font_subtitle = (
            family,
            10,
        )

        self.font_status = (
            family,
            10,
            "bold",
        )

        self.font_log = (
            "Consolas",
            9,
        ) if "Consolas" in available else (
            family,
            9,
        )

    def build_ui(self):

        colors = THEMES[self.theme_name]

        self.root.configure(
            bg=colors["bg"]
        )

        self.main_container = ttk.Frame(
            self.root,
            padding=(24, 20),
        )

        self.main_container.pack(
            fill="both",
            expand=True,
        )

        # ----------------------------------------------------
        # Header
        # ----------------------------------------------------

        self.header_frame = ttk.Frame(
            self.main_container
        )

        self.header_frame.pack(
            fill="x",
            pady=(0, 16),
        )

        self.header_left = ttk.Frame(
            self.header_frame
        )

        self.title_label = ttk.Label(
            self.header_left,
            font=self.font_title,
        )

        self.title_label.pack(
            anchor="w"
        )

        self.subtitle_label = ttk.Label(
            self.header_left,
            font=self.font_subtitle,
        )

        self.subtitle_label.pack(
            anchor="w",
            pady=(3, 0),
        )

        self.header_right = ttk.Frame(
            self.header_frame
        )

        self.language_label = ttk.Label(
            self.header_right,
            font=self.font_small,
        )

        self.language_combo = ttk.Combobox(
            self.header_right,
            textvariable=self.language_var,
            values=[
                "English",
                "فارسی",
            ],
            state="readonly",
            width=10,
            font=self.font_small,
        )

        self.theme_label = ttk.Label(
            self.header_right,
            font=self.font_small,
        )

        self.theme_combo = ttk.Combobox(
            self.header_right,
            textvariable=self.theme_var,
            values=[
                "Dark",
                "روشن",
            ],
            state="readonly",
            width=9,
            font=self.font_small,
        )

        self.language_combo.bind(
            "<<ComboboxSelected>>",
            self.on_language_change,
        )

        self.theme_combo.bind(
            "<<ComboboxSelected>>",
            self.on_theme_change,
        )

        self.language_combo.configure(
            postcommand=lambda:
            self.style_combobox_popup(
                self.language_combo
            )
        )

        self.theme_combo.configure(
            postcommand=lambda:
            self.style_combobox_popup(
                self.theme_combo
            )
        )

        # ----------------------------------------------------
        # Notebook
        # ----------------------------------------------------

        self.notebook = ttk.Notebook(
            self.main_container
        )

        self.main_tab = ttk.Frame(
            self.notebook,
            padding=(16, 16),
        )

        self.logs_tab = ttk.Frame(
            self.notebook,
            padding=(16, 16),
        )

        self.notebook.add(
            self.main_tab
        )

        self.notebook.add(
            self.logs_tab
        )

        self.notebook.pack(
            fill="both",
            expand=True,
        )

        self.build_main_tab()
        self.build_logs_tab()

    def build_main_tab(self):

        self.main_tab.columnconfigure(
            0,
            weight=1,
        )

        self.main_tab.rowconfigure(
            4,
            weight=1,
        )

        # ----------------------------------------------------
        # Model
        # ----------------------------------------------------

        self.model_frame = ttk.Frame(
            self.main_tab,
            padding=(14, 12),
            style="Card.TFrame",
        )

        self.model_frame.grid(
            row=0,
            column=0,
            sticky="ew",
            pady=(0, 10),
        )

        self.model_frame.columnconfigure(
            3,
            weight=1,
        )

        self.model_title = ttk.Label(
            self.model_frame,
            font=self.font_section,
            style="Card.TLabel",
        )

        self.model_title.grid(
            row=0,
            column=0,
            columnspan=5,
            sticky="ew",
            pady=(0, 10),
        )

        self.model_label = ttk.Label(
            self.model_frame,
            font=self.font_normal,
            style="Card.TLabel",
        )

        self.model_combo = ttk.Combobox(
            self.model_frame,
            textvariable=self.model_var,
            values=list(
                ENGINE_REGISTRY.keys()
            ),
            state="readonly",
            font=self.font_normal,
        )

        self.model_combo.configure(
            postcommand=lambda:
            self.style_combobox_popup(
                self.model_combo
            )
        )

        self.load_button = RoundedButton(
            self.model_frame,
            command=self.load_model,
            kind="accent",
            width=145,
            height=38,
            radius=12,
            font=self.font_button,
            background_key="surface",
        )

        self.unload_button = RoundedButton(
            self.model_frame,
            command=self.unload_model,
            kind="secondary",
            width=145,
            height=38,
            radius=12,
            font=self.font_button,
            background_key="surface",
        )

        self.model_status_label = ttk.Label(
            self.model_frame,
            font=self.font_status,
            style="Card.TLabel",
        )

        # ----------------------------------------------------
        # Audio
        # ----------------------------------------------------

        self.audio_frame = ttk.Frame(
            self.main_tab,
            padding=(14, 12),
            style="Card.TFrame",
        )

        self.audio_frame.grid(
            row=1,
            column=0,
            sticky="ew",
            pady=(0, 10),
        )

        self.audio_frame.columnconfigure(
            1,
            weight=1,
        )

        self.audio_title = ttk.Label(
            self.audio_frame,
            font=self.font_section,
            style="Card.TLabel",
        )

        self.audio_title.grid(
            row=0,
            column=0,
            columnspan=3,
            sticky="ew",
            pady=(0, 10),
        )

        self.audio_label = ttk.Label(
            self.audio_frame,
            font=self.font_normal,
            style="Card.TLabel",
        )

        self.audio_entry = ttk.Entry(
            self.audio_frame,
            textvariable=self.audio_var,
            state="readonly",
            font=self.font_normal,
        )

        self.select_audio_button = RoundedButton(
            self.audio_frame,
            command=self.select_audio,
            kind="secondary",
            width=145,
            height=38,
            radius=12,
            font=self.font_button,
            background_key="surface",
        )

        # ----------------------------------------------------
        # Processing
        # ----------------------------------------------------

        self.processing_frame = ttk.Frame(
            self.main_tab,
            padding=(14, 12),
            style="Card.TFrame",
        )

        self.processing_frame.grid(
            row=2,
            column=0,
            sticky="ew",
            pady=(0, 10),
        )

        self.processing_title = ttk.Label(
            self.processing_frame,
            font=self.font_section,
            style="Card.TLabel",
        )

        self.processing_title.pack(
            fill="x",
            pady=(0, 10),
        )

        self.processing_actions = ttk.Frame(
            self.processing_frame,
            style="Card.TFrame",
        )

        self.start_record_button = RoundedButton(
            self.processing_actions,
            command=self.start_recording,
            kind="secondary",
            width=145,
            height=38,
            radius=12,
            font=self.font_button,
            background_key="surface",
        )

        self.stop_record_button = RoundedButton(
            self.processing_actions,
            command=self.stop_recording,
            kind="danger",
            width=145,
            height=38,
            radius=12,
            font=self.font_button,
            background_key="surface",
        )

        self.transcribe_button = RoundedButton(
            self.processing_actions,
            command=self.transcribe,
            kind="accent",
            width=145,
            height=38,
            radius=12,
            font=self.font_button,
            background_key="surface",
        )

        # ----------------------------------------------------
        # Status
        # ----------------------------------------------------

        self.status_frame = ttk.Frame(
            self.main_tab,
            padding=(4, 4),
        )

        self.status_frame.grid(
            row=3,
            column=0,
            sticky="ew",
            pady=(0, 10),
        )

        self.status_dot = tk.Canvas(
            self.status_frame,
            width=14,
            height=14,
            highlightthickness=0,
            bd=0,
        )

        self.status_dot.create_oval(
            3,
            3,
            11,
            11,
            tags="dot",
        )

        self.status_label = ttk.Label(
            self.status_frame,
            font=self.font_status,
        )

        self.operation_label = ttk.Label(
            self.status_frame,
            font=self.font_small,
        )

        self.progress = ttk.Progressbar(
            self.status_frame,
            variable=self.progress_var,
            maximum=100,
            mode="determinate",
            length=220,
        )

        self.progress_text_label = ttk.Label(
            self.status_frame,
            textvariable=self.progress_text_var,
            font=self.font_small,
        )

        self.spinner_canvas = tk.Canvas(
            self.status_frame,
            width=24,
            height=24,
            highlightthickness=0,
            bd=0,
        )

        self.spinner_canvas.create_arc(
            4,
            4,
            20,
            20,
            start=30,
            extent=250,
            style="arc",
            width=2,
            tags="spinner",
        )

        # ----------------------------------------------------
        # Transcription
        # ----------------------------------------------------

        self.transcription_frame = ttk.Frame(
            self.main_tab,
            padding=(14, 12),
            style="Card.TFrame",
        )

        self.transcription_frame.grid(
            row=4,
            column=0,
            sticky="nsew",
            pady=(0, 10),
        )

        self.transcription_frame.columnconfigure(
            0,
            weight=1,
        )

        self.transcription_frame.rowconfigure(
            1,
            weight=1,
        )

        self.transcription_title = ttk.Label(
            self.transcription_frame,
            font=self.font_section,
            style="Card.TLabel",
        )

        self.transcription_title.grid(
            row=0,
            column=0,
            sticky="ew",
            pady=(0, 8),
        )

        text_container = ttk.Frame(
            self.transcription_frame,
            style="Card.TFrame",
        )

        text_container.grid(
            row=1,
            column=0,
            sticky="nsew",
        )

        text_container.columnconfigure(
            0,
            weight=1,
        )

        text_container.rowconfigure(
            0,
            weight=1,
        )

        self.transcription_text = TranscriptionText(
            text_container,
            wrap="word",
            undo=False,
            padx=14,
            pady=12,
            relief="flat",
            borderwidth=0,
            font=(
                self.font_family,
                11,
            ),
            spacing1=2,
            spacing2=2,
            spacing3=2,
        )

        # ----------------------------------------------------
        # Persian text selection / clipboard handling
        # ----------------------------------------------------

        self.transcription_text.bind(
            "<Control-a>",
            self._select_all_transcription,
        )

        self.transcription_text.bind(
            "<Control-A>",
            self._select_all_transcription,
        )

        self.transcription_text.bind(
            "<Control-c>",
            self._copy_transcription,
        )

        self.transcription_text.bind(
            "<Control-C>",
            self._copy_transcription,
        )

        self.transcription_scroll = ttk.Scrollbar(
            text_container,
            orient="vertical",
            command=self.transcription_text.yview,
        )

        self.transcription_text.configure(
            yscrollcommand=self.transcription_scroll.set,
        )

        self.transcription_text.grid(
            row=0,
            column=0,
            sticky="nsew",
        )

        self.transcription_scroll.grid(
            row=0,
            column=1,
            sticky="ns",
        )

        # ----------------------------------------------------
        # Output
        # ----------------------------------------------------

        self.output_frame = ttk.Frame(
            self.main_tab
        )

        self.output_frame.grid(
            row=5,
            column=0,
            sticky="ew",
        )

        self.output_label = ttk.Label(
            self.output_frame,
            font=self.font_small,
        )

        self.export_txt_button = RoundedButton(
            self.output_frame,
            command=self.export_txt,
            kind="secondary",
            width=120,
            height=36,
            radius=11,
            font=self.font_button,
            background_key="bg",
        )

        self.export_json_button = RoundedButton(
            self.output_frame,
            command=self.export_json,
            kind="secondary",
            width=125,
            height=36,
            radius=11,
            font=self.font_button,
            background_key="bg",
        )

        self.copy_text_button = RoundedButton(
            self.output_frame,
            command=self.copy_transcription_text,
            kind="secondary",
            width=120,
            height=36,
            radius=11,
            font=self.font_button,
            background_key="bg",
        )

        self.open_output_button = RoundedButton(
            self.output_frame,
            command=self.open_output,
            kind="secondary",
            width=155,
            height=36,
            radius=11,
            font=self.font_button,
            background_key="bg",
        )

    def build_logs_tab(self):

        self.logs_tab.columnconfigure(
            0,
            weight=1,
        )

        self.logs_tab.rowconfigure(
            1,
            weight=1,
        )

        self.logs_header = ttk.Frame(
            self.logs_tab
        )

        self.logs_header.grid(
            row=0,
            column=0,
            sticky="ew",
            pady=(0, 10),
        )

        self.logs_title_label = ttk.Label(
            self.logs_header,
            font=self.font_section,
        )

        self.clear_logs_button = RoundedButton(
            self.logs_header,
            command=self.clear_logs,
            kind="secondary",
            width=125,
            height=36,
            radius=11,
            font=self.font_button,
            background_key="bg",
        )

        log_container = ttk.Frame(
            self.logs_tab
        )

        log_container.grid(
            row=1,
            column=0,
            sticky="nsew",
        )

        log_container.columnconfigure(
            0,
            weight=1,
        )

        log_container.rowconfigure(
            0,
            weight=1,
        )

        self.logs_text = tk.Text(
            log_container,
            wrap="none",
            padx=12,
            pady=10,
            relief="flat",
            borderwidth=0,
            font=self.font_log,
        )

        self.logs_scroll_y = ttk.Scrollbar(
            log_container,
            orient="vertical",
            command=self.logs_text.yview,
        )

        self.logs_scroll_x = ttk.Scrollbar(
            log_container,
            orient="horizontal",
            command=self.logs_text.xview,
        )

        self.logs_text.configure(
            yscrollcommand=self.logs_scroll_y.set,
            xscrollcommand=self.logs_scroll_x.set,
            state="disabled",
        )

        self.logs_text.grid(
            row=0,
            column=0,
            sticky="nsew",
        )

        self.logs_scroll_y.grid(
            row=0,
            column=1,
            sticky="ns",
        )

        self.logs_scroll_x.grid(
            row=1,
            column=0,
            sticky="ew",
        )

        self.configure_log_tags()