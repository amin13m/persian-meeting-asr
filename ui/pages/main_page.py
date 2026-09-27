from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QLabel,
    QLineEdit,
    QComboBox,
    QProgressBar,
    QScrollArea,
    QSizePolicy,
    QFrame,
)

from PySide6.QtCore import Qt, QTimer

from ..components import AppButton, Section, StatusWidget
from ..transcription_text import TranscriptionText


class MainPage(QWidget):

    def __init__(self, app):
        super().__init__()

        self.app = app

        # ==============================================================
        # Model spinner
        # ==============================================================

        self._spinner_frames = [
            "◐",
            "◓",
            "◑",
            "◒",
        ]

        self._spinner_index = 0

        self._model_spinner_timer = QTimer(self)
        self._model_spinner_timer.setInterval(120)
        self._model_spinner_timer.timeout.connect(
            self._update_model_spinner
        )

        # ==============================================================
        # Processing progress animation
        # ==============================================================

        self._progress_timer = QTimer(self)
        self._progress_timer.setInterval(25)
        self._progress_timer.timeout.connect(
            self._animate_processing
        )

        self._processing_progress_value = 0

        self.build()

    # ==================================================================
    # Build
    # ==================================================================

    def build(self):
    
        outer = QVBoxLayout(self)
    
        outer.setContentsMargins(
            0,
            0,
            0,
            0,
        )
    
        outer.setSpacing(
            0
        )
    
        # ==============================================================
        # Scroll
        # ==============================================================
    
        self.scroll = QScrollArea()
    
        self.scroll.setWidgetResizable(
            True
        )
    
        self.scroll.setFrameShape(
            QFrame.NoFrame
        )
    
        self.scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarAlwaysOff
        )
    
        self.scroll.setVerticalScrollBarPolicy(
            Qt.ScrollBarAsNeeded
        )
    
        outer.addWidget(
            self.scroll
        )
    
        # ==============================================================
        # Content
        # ==============================================================
    
        self.content = QWidget()
    
        self.content.setObjectName(
            "MainPageContent"
        )
    
        self.scroll.setWidget(
            self.content
        )
    
        root = QVBoxLayout(
            self.content
        )
    
        root.setContentsMargins(
            24,
            24,
            24,
            24,
        )
    
        root.setSpacing(
            18
        )
    
        # ==============================================================
        # Header
        # ==============================================================
    
        header = QHBoxLayout()
    
        header.setContentsMargins(
            2,
            0,
            2,
            4,
        )
    
        header.setSpacing(
            16
        )
    
        # --------------------------------------------------------------
        # Title
        # --------------------------------------------------------------
    
        title_box = QVBoxLayout()
    
        title_box.setSpacing(
            3
        )
    
        self.title = QLabel()
    
        self.title.setObjectName(
            "Title"
        )
    
        self.title.setWordWrap(
            True
        )
    
        self.subtitle = QLabel()
    
        self.subtitle.setObjectName(
            "Subtitle"
        )
    
        self.subtitle.setWordWrap(
            True
        )
    
        title_box.addWidget(
            self.title
        )
    
        title_box.addWidget(
            self.subtitle
        )
    
        header.addLayout(
            title_box,
            1,
        )
    
        # --------------------------------------------------------------
        # Header controls
        # --------------------------------------------------------------
    
        controls = QHBoxLayout()
    
        controls.setSpacing(
            8
        )
    
        self.language = QComboBox()
    
        self.language.setObjectName(
            "HeaderCombo"
        )
    
        self.language.setMinimumWidth(
            110
        )
    
        self.language.addItem(
            "فارسی",
            "fa",
        )
    
        self.language.addItem(
            "English",
            "en",
        )
    
        self.theme = QComboBox()
    
        self.theme.setObjectName(
            "HeaderCombo"
        )
    
        self.theme.setMinimumWidth(
            110
        )
    
        self.theme.addItem(
            "Dark",
            "dark",
        )
    
        self.theme.addItem(
            "Light",
            "light",
        )
    
        controls.addWidget(
            self.language
        )
    
        controls.addWidget(
            self.theme
        )
    
        header.addLayout(
            controls
        )
    
        root.addLayout(
            header
        )
    
        # ==============================================================
        # Top cards
        # ==============================================================
    
        top_grid = QGridLayout()
    
        top_grid.setHorizontalSpacing(
            16
        )
    
        top_grid.setVerticalSpacing(
            16
        )
    
        # ==============================================================
        # MODEL
        # ==============================================================
    
        self.model_section = Section(
            ""
        )
    
        self.model_section.setObjectName(
            "ModernSection"
        )
    
        # --------------------------------------------------------------
        # Model spinner
        # --------------------------------------------------------------
    
        self.model_spinner = QLabel()
    
        self.model_spinner.setObjectName(
            "ModelLoadingSpinner"
        )
    
        self.model_spinner.setAlignment(
            Qt.AlignCenter
        )
    
        self.model_spinner.setFixedSize(
            24,
            24,
        )
    
        self.model_spinner.setVisible(
            False
        )
    
        self._model_spinner_attached = False
    
        try:
        
            title_label = (
                self.model_section.title_label
            )
    
            title_parent = (
                title_label.parentWidget()
            )
    
            if title_parent is not None:
            
                title_layout = (
                    title_parent.layout()
                )
    
                if title_layout is not None:
                
                    title_layout.addWidget(
                        self.model_spinner
                    )
    
                    self._model_spinner_attached = True
    
        except Exception:
        
            self._model_spinner_attached = False
    
        # --------------------------------------------------------------
        # Model layout
        # --------------------------------------------------------------
    
        model_layout = QGridLayout()
    
        model_layout.setContentsMargins(
            0,
            0,
            0,
            0,
        )
    
        model_layout.setHorizontalSpacing(
            10
        )
    
        model_layout.setVerticalSpacing(
            10
        )
    
        self.model_combo = QComboBox()
    
        self.model_combo.addItems(
            [
                "whisper_persian_v4"
            ]
        )
    
        self.model_combo.setMinimumHeight(
            38
        )
    
        self.load_button = AppButton()
    
        self.load_button.setMinimumHeight(
            38
        )
    
        self.unload_button = AppButton()
    
        self.unload_button.setMinimumHeight(
            38
        )
    
        if self._model_spinner_attached:
        
            model_combo_row = 0
            button_row = 1
    
        else:
        
            spinner_row = QHBoxLayout()
    
            spinner_row.setContentsMargins(
                0,
                0,
                0,
                0,
            )
    
            spinner_row.setSpacing(
                6
            )
    
            spinner_row.addWidget(
                self.model_spinner,
                0,
                Qt.AlignVCenter,
            )
    
            spinner_row.addStretch()
    
            model_layout.addLayout(
                spinner_row,
                0,
                0,
                1,
                2,
            )
    
            model_combo_row = 1
            button_row = 2
    
        model_layout.addWidget(
            self.model_combo,
            model_combo_row,
            0,
            1,
            2,
        )
    
        model_layout.addWidget(
            self.load_button,
            button_row,
            0,
        )
    
        model_layout.addWidget(
            self.unload_button,
            button_row,
            1,
        )
    
        self.model_section.layout.addLayout(
            model_layout
        )
    
        top_grid.addWidget(
            self.model_section,
            0,
            0,
        )
    
        # ==============================================================
        # AUDIO
        # ==============================================================
    
        self.audio_section = Section(
            ""
        )
    
        self.audio_section.setObjectName(
            "ModernSection"
        )
    
        audio_layout = QGridLayout()
    
        audio_layout.setContentsMargins(
            0,
            0,
            0,
            0,
        )
    
        audio_layout.setHorizontalSpacing(
            10
        )
    
        audio_layout.setVerticalSpacing(
            10
        )
    
        self.audio_path = QLineEdit()
    
        self.audio_path.setReadOnly(
            True
        )
    
        self.audio_path.setMinimumHeight(
            38
        )
    
        self.audio_path.setPlaceholderText(
            ""
        )
    
        self.select_audio_button = AppButton()
    
        self.select_audio_button.setMinimumHeight(
            38
        )
    
        self.start_record_button = AppButton()
    
        self.start_record_button.setMinimumHeight(
            38
        )
    
        self.stop_record_button = AppButton()
    
        self.stop_record_button.setMinimumHeight(
            38
        )
    
        audio_layout.addWidget(
            self.audio_path,
            0,
            0,
            1,
            3,
        )
    
        audio_layout.addWidget(
            self.select_audio_button,
            1,
            0,
        )
    
        audio_layout.addWidget(
            self.start_record_button,
            1,
            1,
        )
    
        audio_layout.addWidget(
            self.stop_record_button,
            1,
            2,
        )
    
        self.audio_section.layout.addLayout(
            audio_layout
        )
    
        top_grid.addWidget(
            self.audio_section,
            0,
            1,
        )
    
        top_grid.setColumnStretch(
            0,
            1,
        )
    
        top_grid.setColumnStretch(
            1,
            1,
        )
    
        root.addLayout(
            top_grid
        )
    
        # ==============================================================
        # PROCESSING
        # ==============================================================
    
        self.processing_section = Section(
            ""
        )
    
        self.processing_section.setObjectName(
            "ModernSection"
        )
    
        # --------------------------------------------------------------
        # One horizontal processing row
        #
        # [ Transcribe ] [ Status ] [ Progress ........ ] [ Operation ]
        # --------------------------------------------------------------
    
        processing_layout = QHBoxLayout()
    
        processing_layout.setContentsMargins(
            0,
            0,
            0,
            0,
        )
    
        processing_layout.setSpacing(
            12
        )
    
        # --------------------------------------------------------------
        # Transcribe button
        # --------------------------------------------------------------
    
        self.transcribe_button = AppButton(
            primary=True
        )
    
        self.transcribe_button.setMinimumHeight(
            42
        )
    
        processing_layout.addWidget(
            self.transcribe_button,
            0,
            Qt.AlignVCenter,
        )
    
        # --------------------------------------------------------------
        # Overall status
        # --------------------------------------------------------------
    
        self.status = StatusWidget()
    
        self.status.setMinimumHeight(
            42
        )
    
        processing_layout.addWidget(
            self.status,
            0,
            Qt.AlignVCenter,
        )
    
        # --------------------------------------------------------------
        # Progress bar
        # --------------------------------------------------------------
    
        self.progress = QProgressBar()
    
        self.progress.setObjectName(
            "ProcessingProgress"
        )
    
        self.progress.setMinimumHeight(
            8
        )
    
        self.progress.setMaximumHeight(
            8
        )
    
        self.progress.setMinimumWidth(
            180
        )
    
        self.progress.setTextVisible(
            False
        )
    
        self.progress.setRange(
            0,
            100,
        )
    
        self.progress.setValue(
            0
        )
    
        # Stretch = 1
        # Progress bar takes the remaining horizontal space.
        processing_layout.addWidget(
            self.progress,
            1,
            Qt.AlignVCenter,
        )
    
        # --------------------------------------------------------------
        # Operation label
        # --------------------------------------------------------------
    
        self.operation_label = QLabel()
    
        self.operation_label.setObjectName(
            "OperationLabel"
        )
    
        self.operation_label.setWordWrap(
            False
        )
    
        self.operation_label.setMinimumWidth(
            150
        )
    
        self.operation_label.setMaximumWidth(
            240
        )
    
        self.operation_label.setAlignment(
            Qt.AlignLeft | Qt.AlignVCenter
        )
    
        processing_layout.addWidget(
            self.operation_label,
            0,
            Qt.AlignVCenter,
        )
    
        # --------------------------------------------------------------
        # Add processing layout to section
        # --------------------------------------------------------------
    
        self.processing_section.layout.addLayout(
            processing_layout
        )
    
        root.addWidget(
            self.processing_section
        )
    
        # ==============================================================
        # TRANSCRIPTION
        # ==============================================================
    
        self.transcription_section = Section(
            ""
        )
    
        self.transcription_section.setObjectName(
            "ModernSection"
        )
    
        transcription_layout = QVBoxLayout()
    
        transcription_layout.setContentsMargins(
            0,
            0,
            0,
            0,
        )
    
        transcription_layout.setSpacing(
            10
        )
    
        self.transcription_text = (
            TranscriptionText()
        )
    
        self.transcription_text.setMinimumHeight(
            190
        )
    
        self.transcription_text.setMaximumHeight(
            240
        )
    
        self.transcription_text.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Preferred,
        )
    
        transcription_layout.addWidget(
            self.transcription_text
        )
    
        # --------------------------------------------------------------
        # Output toolbar
        # --------------------------------------------------------------
    
        output_toolbar = QHBoxLayout()
    
        output_toolbar.setSpacing(
            8
        )
    
        self.copy_text_button = AppButton()
    
        self.export_txt_button = AppButton()
    
        self.export_json_button = AppButton()
    
        self.open_output_button = AppButton()
    
        output_toolbar.addWidget(
            self.copy_text_button
        )
    
        output_toolbar.addStretch()
    
        output_toolbar.addWidget(
            self.export_txt_button
        )
    
        output_toolbar.addWidget(
            self.export_json_button
        )
    
        output_toolbar.addWidget(
            self.open_output_button
        )
    
        transcription_layout.addLayout(
            output_toolbar
        )
    
        self.transcription_section.layout.addLayout(
            transcription_layout
        )
    
        root.addWidget(
            self.transcription_section
        )
    
        # ==============================================================
        # Bottom spacing
        # ==============================================================
    
        root.addSpacing(
            4
        )
    
        # ==============================================================
        # Connections
        # ==============================================================
    
        self.language.currentIndexChanged.connect(
            self.app.on_language_change
        )
    
        self.theme.currentIndexChanged.connect(
            self.app.on_theme_change
        )
    
        self.select_audio_button.clicked.connect(
            self.app.select_audio
        )
    
        self.start_record_button.clicked.connect(
            self.app.start_recording
        )
    
        self.stop_record_button.clicked.connect(
            self.app.stop_recording
        )
    
        self.load_button.clicked.connect(
            self.app.load_model
        )
    
        self.unload_button.clicked.connect(
            self.app.unload_model
        )
    
        self.transcribe_button.clicked.connect(
            self.app.transcribe
        )
    
        self.copy_text_button.clicked.connect(
            self.app.copy_transcription_text
        )
    
        self.export_txt_button.clicked.connect(
            self.app.export_txt
        )
    
        self.export_json_button.clicked.connect(
            self.app.export_json
        )
    
        self.open_output_button.clicked.connect(
            self.app.open_output
        )
    
    
    # ==================================================================
    # MODEL SPINNER
    # ==================================================================

    def _update_model_spinner(self):

        if not self.model_spinner.isVisible():
            return

        self._spinner_index = (
            self._spinner_index + 1
        ) % len(
            self._spinner_frames
        )

        self.model_spinner.setText(
            self._spinner_frames[
                self._spinner_index
            ]
        )

    def set_model_loading(
        self,
        loading: bool,
    ):

        if loading:

            self._spinner_index = 0

            self.model_spinner.setText(
                self._spinner_frames[0]
            )

            self.model_spinner.setVisible(
                True
            )

            self._model_spinner_timer.start()

        else:

            # HARD STOP
            self._model_spinner_timer.stop()
            self._model_spinner_timer.setInterval(
                120
            )

            self.model_spinner.setVisible(
                False
            )

            self.model_spinner.clear()

            self._spinner_index = 0

    # ==================================================================
    # PROCESSING ANIMATION
    # ==================================================================

    def _animate_processing(self):

        # Safety check
        if not self._progress_timer.isActive():
            return

        self._processing_progress_value += 1

        if self._processing_progress_value > 100:
            self._processing_progress_value = 0

        self.progress.setValue(
            self._processing_progress_value
        )

    # ------------------------------------------------------------------
    # START animation
    # ------------------------------------------------------------------

    def _start_progress_animation(self):

        # --------------------------------------------------------------
        # Kill previous animation completely.
        # --------------------------------------------------------------

        self._progress_timer.stop()

        self._processing_progress_value = 0

        self.progress.setRange(
            0,
            100,
        )

        self.progress.setValue(
            0
        )

        self.progress.setVisible(
            True
        )

        self._progress_timer.start()

    # ------------------------------------------------------------------
    # STOP animation
    # ------------------------------------------------------------------

    def _stop_progress_animation(
        self,
        final_value=None,
    ):

        # ==============================================================
        # THIS IS THE IMPORTANT FIX
        # ==============================================================

        # 1. Stop our timer.
        self._progress_timer.stop()

        # 2. Reset internal value.
        self._processing_progress_value = 0

        # 3. Always normal range.
        self.progress.setRange(
            0,
            100,
        )

        # 4. Set final value.
        if final_value is None:
            final_value = 0

        self.progress.setValue(
            final_value
        )

        # 5. Force repaint.
        self.progress.update()
        self.progress.repaint()

    # ==================================================================
    # PROCESSING STATE
    # ==================================================================

    def _refresh_processing_style(
        self,
    ):

        for widget in (
            self.operation_label,
            self.progress,
        ):

            widget.style().unpolish(
                widget
            )

            widget.style().polish(
                widget
            )

            widget.update()

    # ------------------------------------------------------------------
    # Running
    # ------------------------------------------------------------------

    def set_processing_running(
        self,
        text: str,
        indeterminate: bool = False,
    ):

        self.operation_label.setText(
            text
        )

        self.operation_label.setProperty(
            "processingState",
            "running",
        )

        self.progress.setProperty(
            "processingState",
            "running",
        )

        # --------------------------------------------------------------
        # NEVER use setRange(0, 0)
        # --------------------------------------------------------------

        self._start_progress_animation()

        self._refresh_processing_style()

    # ------------------------------------------------------------------
    # Success
    # ------------------------------------------------------------------

    def set_processing_success(
        self,
        text: str,
    ):

        # --------------------------------------------------------------
        # STOP FIRST
        # --------------------------------------------------------------

        self._stop_progress_animation(
            100
        )

        self.operation_label.setText(
            text
        )

        self.operation_label.setProperty(
            "processingState",
            "success",
        )

        self.progress.setProperty(
            "processingState",
            "success",
        )

        self._refresh_processing_style()

    # ------------------------------------------------------------------
    # Error
    # ------------------------------------------------------------------

    def set_processing_error(
        self,
        text: str,
    ):

        # --------------------------------------------------------------
        # STOP FIRST
        # --------------------------------------------------------------

        self._stop_progress_animation(
            0
        )

        self.operation_label.setText(
            text
        )

        self.operation_label.setProperty(
            "processingState",
            "error",
        )

        self.progress.setProperty(
            "processingState",
            "error",
        )

        self._refresh_processing_style()

    # ------------------------------------------------------------------
    # Idle
    # ------------------------------------------------------------------

    def set_processing_idle(
        self,
        text: str = "",
    ):

        # --------------------------------------------------------------
        # STOP FIRST
        # --------------------------------------------------------------

        self._stop_progress_animation(
            0
        )

        self.operation_label.setText(
            text
        )

        self.operation_label.setProperty(
            "processingState",
            "idle",
        )

        self.progress.setProperty(
            "processingState",
            "idle",
        )

        self._refresh_processing_style()

    # ==================================================================
    # Translation
    # ==================================================================

    def retranslate(
        self,
        t,
    ):

        self.title.setText(
            t["app_title"]
        )

        self.subtitle.setText(
            t["app_subtitle"]
        )

        self.model_section.title_label.setText(
            t["model_section"]
        )

        self.audio_section.title_label.setText(
            t["audio_section"]
        )

        self.processing_section.title_label.setText(
            t["processing_section"]
        )

        self.transcription_section.title_label.setText(
            t["transcription_section"]
        )

        self.load_button.setText(
            t["load_model"]
        )

        self.unload_button.setText(
            t["unload_model"]
        )

        self.select_audio_button.setText(
            t["select_audio"]
        )

        self.start_record_button.setText(
            t["start_recording"]
        )

        self.stop_record_button.setText(
            t["stop_recording"]
        )

        self.transcribe_button.setText(
            t["transcribe"]
        )

        self.copy_text_button.setText(
            t["copy_text"]
        )

        self.export_txt_button.setText(
            t["export_txt"]
        )

        self.export_json_button.setText(
            t["export_json"]
        )

        self.open_output_button.setText(
            t["open_output"]
        )