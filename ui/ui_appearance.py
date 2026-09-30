
from __future__ import annotations

from PySide6.QtCore import Qt

from .ui_constants import TEXTS, THEMES
from .ui_theme import apply_theme


class UIAppearanceMixin:

    # ========================================================
    # Theme
    # ========================================================

    def apply_theme(self):
        """
        Apply the currently selected theme.

        self.theme_name must always be one of:
            "dark"
            "light"
        """

        # Safety normalization
        if self.theme_name not in THEMES:
            self.theme_name = "dark"

        # Apply Qt stylesheet
        colors = apply_theme(
            self.app,
            self.theme_name,
        )

        # Safety fallback
        if isinstance(colors, dict):
            self.colors = colors
        else:
            self.colors = THEMES.get(
                self.theme_name,
                THEMES["dark"],
            )


        self.update_status_dot()

    # ========================================================
    # Language
    # ========================================================

    def update_language(self):
        self.TEXT = TEXTS.get(
            self.language,
            TEXTS["fa"],
        )

        self.setWindowTitle(
            self.TEXT["app_title"]
        )

        # ----------------------------------------------------
        # Pages
        # ----------------------------------------------------

        if hasattr(self, "main_page"):
            self.main_page.retranslate(
                self.TEXT
            )

        if hasattr(self, "files_page"):
            self.files_page.retranslate(
                self.TEXT
            )

        if hasattr(self, "logs_page"):
            self.logs_page.retranslate(
                self.TEXT
            )

        # ----------------------------------------------------
        # Tabs
        # ----------------------------------------------------

        if hasattr(self, "tabs"):
            self.tabs.setTabText(
                0,
                self.TEXT["main"],
            )

            self.tabs.setTabText(
                1,
                self.TEXT["files"],
            )

            self.tabs.setTabText(
                2,
                self.TEXT["logs"],
            )

        # ----------------------------------------------------
        # Language combo
        # ----------------------------------------------------

        if (
            hasattr(self, "main_page")
            and hasattr(self.main_page, "language")
        ):
            combo = self.main_page.language

            combo.blockSignals(True)

            combo.clear()

            combo.addItems(
                [
                    self.TEXT["persian"],
                    self.TEXT["english"],
                ]
            )

            combo.setCurrentIndex(
                0 if self.language == "fa" else 1
            )

            combo.blockSignals(False)

        # ----------------------------------------------------
        # Theme combo
        # ----------------------------------------------------

        if (
            hasattr(self, "main_page")
            and hasattr(self.main_page, "theme")
        ):
            combo = self.main_page.theme

            combo.blockSignals(True)

            combo.clear()

            combo.addItems(
                [
                    self.TEXT["dark"],
                    self.TEXT["light"],
                ]
            )

            combo.setCurrentIndex(
                0 if self.theme_name == "dark" else 1
            )

            combo.blockSignals(False)

        self.refresh_status_text()

    # ========================================================
    # Direction
    # ========================================================

    def apply_direction(self):
        rtl = self.language == "fa"

        direction = (
            Qt.RightToLeft
            if rtl
            else Qt.LeftToRight
        )

        alignment = (
            Qt.AlignRight
            if rtl
            else Qt.AlignLeft
        )

        # ----------------------------------------------------
        # Main window
        # ----------------------------------------------------

        self.setLayoutDirection(
            direction
        )

        # ----------------------------------------------------
        # Transcription
        # ----------------------------------------------------

        if (
            hasattr(self, "main_page")
            and hasattr(
                self.main_page,
                "transcription_text",
            )
        ):
            self.main_page.transcription_text.setLayoutDirection(
                direction
            )

            self.main_page.transcription_text.setAlignment(
                alignment
            )

        # ----------------------------------------------------
        # Audio path
        # ----------------------------------------------------

        if (
            hasattr(self, "main_page")
            and hasattr(
                self.main_page,
                "audio_path",
            )
        ):
            # File paths should remain LTR
            self.main_page.audio_path.setLayoutDirection(
                Qt.LeftToRight
            )

    # ========================================================
    # Theme change
    # ========================================================

    def on_theme_change(self, index=None):
        """
        Called when the theme ComboBox changes.

        Index:
            0 -> dark
            1 -> light
        """

        if not hasattr(
            self,
            "main_page",
        ):
            return

        combo = self.main_page.theme

        # ----------------------------------------------------
        # Resolve theme from ComboBox
        # ----------------------------------------------------

        if index is None:
            index = combo.currentIndex()

        if index == 1:
            self.theme_name = "light"
        else:
            self.theme_name = "dark"

        # ----------------------------------------------------
        # Save setting
        # ----------------------------------------------------

        self.settings.save(
            theme=self.theme_name
        )

        # ----------------------------------------------------
        # Apply theme
        # ----------------------------------------------------

        self.apply_theme()

        # ----------------------------------------------------
        # Rebuild translated UI text
        # ----------------------------------------------------

        self.update_language()

        # ----------------------------------------------------
        # Direction
        # ----------------------------------------------------

        self.apply_direction()



    # ========================================================
    # Language change
    # ========================================================

    def on_language_change(self, index=None):
        if not hasattr(
            self,
            "main_page",
        ):
            return

        combo = self.main_page.language

        if index is None:
            index = combo.currentIndex()

        self.language = (
            "fa"
            if index == 0
            else "en"
        )

        self.settings.save(
            language=self.language
        )

        self.update_language()

        self.apply_direction()

        self.logger.info(
            "LANGUAGE | changed=%s",
            self.language,
        )

    # ========================================================
    # Status
    # ========================================================

    def set_status(
        self,
        status,
        detail="",
    ):
        self.status_key = status
        self.status_detail = detail

        self.refresh_status_text()

    def refresh_status_text(self):
        if not hasattr(
            self,
            "main_page",
        ):
            return

        if not hasattr(
            self,
            "TEXT",
        ):
            return

        text = self.TEXT.get(
            getattr(
                self,
                "status_key",
                "ready",
            ),
            self.TEXT["ready"],
        )

        detail = getattr(
            self,
            "status_detail",
            "",
        )

        if detail:
            text += f" • {detail}"

        colors = getattr(
            self,
            "colors",
            {},
        )

        if not isinstance(
            colors,
            dict,
        ):
            colors = {}

        status_color_key = {
            "error": "error",
            "completed": "success",
            "loaded": "success",
            "recording": "warning",
        }.get(
            getattr(
                self,
                "status_key",
                "ready",
            ),
            "info",
        )

        color = colors.get(
            status_color_key,
            colors.get(
                "info",
                "#6EA8FE",
            ),
        )

        self.main_page.status.set_status(
            text,
            color,
        )

    def update_status_dot(self):
        self.refresh_status_text()

    # ========================================================
    # Progress
    # ========================================================

    def set_progress(
        self,
        value,
        text="",
    ):
        if (
            hasattr(self, "main_page")
            and hasattr(
                self.main_page,
                "progress",
            )
        ):
            self.main_page.progress.setValue(
                int(value)
            )

        if (
            hasattr(self, "main_page")
            and hasattr(
                self.main_page,
                "operation_label",
            )
        ):
            self.main_page.operation_label.setText(
                text
            )

    # ========================================================
    # Controls
    # ========================================================

    def refresh_controls(self):
        processing = self.processing
        recording = self.recording
        loaded = self.model_loaded
        audio = bool(self.current_audio)

        if not hasattr(
            self,
            "main_page",
        ):
            return

        page = self.main_page

        # ----------------------------------------------------
        # Model
        # ----------------------------------------------------

        page.load_button.setEnabled(
            not processing
            and not recording
            and not loaded
        )

        page.unload_button.setEnabled(
            not processing
            and not recording
            and loaded
        )

        # ----------------------------------------------------
        # Audio
        # ----------------------------------------------------

        page.select_audio_button.setEnabled(
            not processing
            and not recording
        )

        page.start_record_button.setEnabled(
            not processing
            and not recording
        )

        page.stop_record_button.setEnabled(
            recording
        )

        # ----------------------------------------------------
        # Transcription
        # ----------------------------------------------------

        can_transcribe = (
            self.model_loaded
            and self.engine is not None
            and self.pipeline is not None
            and self.current_audio is not None
            and not self.processing
        )

        page.transcribe_button.setEnabled(
            can_transcribe
        )

        # ----------------------------------------------------
        # Export
        # ----------------------------------------------------

        has_result = bool(
            self.current_result
        )

        page.export_txt_button.setEnabled(
            has_result
            and not processing
        )

        page.export_json_button.setEnabled(
            has_result
            and not processing
        )

        # ----------------------------------------------------
        # Copy text
        # ----------------------------------------------------

        transcription_text = ""

        if hasattr(
            page,
            "transcription_text",
        ):
            transcription_text = (
                page.transcription_text.toPlainText()
            )

        page.copy_text_button.setEnabled(
            bool(
                transcription_text.strip()
            )
            and not processing
        )

        # ----------------------------------------------------
        # Audio path
        # ----------------------------------------------------

        if hasattr(
            page,
            "audio_path",
        ):
            page.audio_path.setText(
                str(
                    self.current_audio
                    or ""
                )
            )

        self.logger.info(
            "CONTROLS | processing=%s recording=%s "
            "loaded=%s audio=%s engine=%s chunker=%s "
            "| select=%s transcribe=%s",
            processing,
            recording,
            loaded,
            audio,
            self.engine is not None,
            self.chunker is not None,
            page.select_audio_button.isEnabled(),
            page.transcribe_button.isEnabled(),
        )
