
from __future__ import annotations

import logging
import queue
import sys
from pathlib import Path

import yaml

from PySide6.QtCore import QObject, Signal, QTimer , Qt
from PySide6.QtWidgets import QApplication, QMainWindow




try:
    from .ui_constants import (
        CONFIG_PATH,
        PROJECT_ROOT,
    )
    from .ui_settings import UISettings
    from .ui_layout import UILayoutMixin
    from .ui_appearance import UIAppearanceMixin
    from .ui_operations import UIOperationsMixin
    from .logger import UILogHandler

except ImportError:
    UI_DIR = Path(__file__).resolve().parent

    if str(UI_DIR.parent) not in sys.path:
        sys.path.insert(0, str(UI_DIR.parent))

    from ui.ui_constants import (
        CONFIG_PATH,
        PROJECT_ROOT,
    )
    from ui.ui_settings import UISettings
    from ui.ui_layout import UILayoutMixin
    from ui.ui_appearance import UIAppearanceMixin
    from ui.ui_operations import UIOperationsMixin
    from ui.logger import UILogHandler


class UIBridge(QObject):
    signal = Signal(object)


class PersianASRApp(
    QMainWindow,
    UILayoutMixin,
    UIAppearanceMixin,
    UIOperationsMixin,
):
    def __init__(self):
        super().__init__()

        # ----------------------------------------------------
        # Qt Application
        # ----------------------------------------------------
        self.app = QApplication.instance()

        if self.app is None:
            raise RuntimeError(
                "QApplication instance is not available."
            )

        # ----------------------------------------------------
        # Settings
        # ----------------------------------------------------
        self.settings = UISettings()

        self.language = self.settings.data.get(
            "language",
            "fa",
        )

        self.theme_name = self.settings.data.get(
            "theme",
            "dark",
        )



        # ----------------------------------------------------
        # Runtime state
        # ----------------------------------------------------
        self.model_loaded = False
        self.processing = False
        self.recording = False

        self.current_audio = None
        self.current_result = None

        self.engine = None
        self.chunker = None
        self.recorder = None
        self.operation = None

        self.model_var = "whisper_persian_v4"
        self.audio_var = ""

        self.status_key = "ready"
        self.status_detail = ""

        # ----------------------------------------------------
        # Logging
        # ----------------------------------------------------
        self.log_queue = queue.Queue()

        self.ui_bridge = UIBridge()

        self.ui_bridge.signal.connect(
            self._execute_ui_callback,
            Qt.QueuedConnection,
        )

        self.logger = logging.getLogger(
            "PersianASR.UI"
        )

        self.logger.setLevel(
            logging.DEBUG
        )

        self.ui_log_handler = UILogHandler(
            self.log_queue
        )

        self.ui_log_handler.setLevel(
            logging.DEBUG
        )

        self.ui_log_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)s | %(message)s",
                datefmt="%H:%M:%S",
            )
        )

        # Remove previous UI handlers only.
        for handler in list(self.logger.handlers):
            if isinstance(handler, UILogHandler):
                self.logger.removeHandler(handler)

        self.logger.addHandler(
            self.ui_log_handler
        )

        self.logger.propagate = False

        # ----------------------------------------------------
        # Root logger → UI
        # ----------------------------------------------------
        root_logger = logging.getLogger()

        root_logger.setLevel(
            logging.INFO
        )

        self.root_ui_log_handler = UILogHandler(
            self.log_queue
        )

        self.root_ui_log_handler.setLevel(
            logging.INFO
        )

        self.root_ui_log_handler.setFormatter(
            self.ui_log_handler.formatter
        )

        root_logger.addHandler(
            self.root_ui_log_handler
        )

        # ----------------------------------------------------
        # Window
        # ----------------------------------------------------
        self.setMinimumSize(
            1000,
            720,
        )

        self.resize(
            1220,
            860,
        )

        # ----------------------------------------------------
        # Build UI
        # ----------------------------------------------------
        self.build_ui()

        self.apply_theme()

        self.update_language()

        self.apply_direction()

        # ----------------------------------------------------
        # Restore last tab
        # ----------------------------------------------------
        try:
            last_tab = int(
                self.settings.data.get(
                    "last_tab",
                    0,
                )
            )
        except (
            TypeError,
            ValueError,
        ):
            last_tab = 0

        if hasattr(self, "tabs"):
            self.tabs.setCurrentIndex(
                max(
                    0,
                    last_tab,
                )
            )

        # ----------------------------------------------------
        # UI timer
        # ----------------------------------------------------
        self.timer = QTimer(self)

        self.timer.timeout.connect(
            self.process_log_queue
        )

        self.timer.start(80)

        # ----------------------------------------------------
        # Startup logs
        # ----------------------------------------------------
        self.logger.info(
            "=" * 60
        )

        self.logger.info(
            "Persian ASR PySide6 UI started"
        )

        self.logger.info(
            "Project root: %s",
            PROJECT_ROOT,
        )

        self.logger.info(
            "Configuration: %s",
            CONFIG_PATH,
        )

        # ----------------------------------------------------
        # Initial control state
        # ----------------------------------------------------
        self.refresh_controls()

    # ========================================================
    # Configuration
    # ========================================================

    def load_config(self):
        if not CONFIG_PATH.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {CONFIG_PATH}"
            )

        config = yaml.safe_load(
            CONFIG_PATH.read_text(
                encoding="utf-8"
            )
        )

        if not isinstance(config, dict):
            raise ValueError(
                "Configuration file is empty or invalid."
            )

        return config
    
    
    
    

    # ========================================================
    # Directories
    # ========================================================

    def get_input_dir(
        self,
        config,
    ):
        value = (
            config
            .get("audio", {})
            .get(
                "input_dir",
                "input",
            )
        )

        path = Path(value)

        if path.is_absolute():
            result = path
        else:
            result = PROJECT_ROOT / path

        result.mkdir(
            parents=True,
            exist_ok=True,
        )

        return result

    def get_output_dir(
        self,
        config,
    ):
        value = (
            config
            .get("audio", {})
            .get(
                "output_dir",
                "output",
            )
        )

        path = Path(value)

        if path.is_absolute():
            result = path
        else:
            result = PROJECT_ROOT / path

        result.mkdir(
            parents=True,
            exist_ok=True,
        )

        return result

    # ========================================================
    # First show
    # ========================================================

    def showEvent(
        self,
        event,
    ):
        super().showEvent(event)

        if hasattr(
            self,
            "main_page",
        ) and hasattr(
            self.main_page,
            "transcription_text",
        ):
            self.main_page.transcription_text.set_text(
                self.TEXT["no_transcription"],
                self.language == "fa",
            )
            
            
    def _execute_ui_callback(self, callback):
        try:
            callback()
        except Exception:
            self.logger.exception(
                "UI callback failed."
            )


# ============================================================
# Main
# ============================================================

def main():
    app = QApplication.instance()

    if app is None:
        app = QApplication(sys.argv)

    window = PersianASRApp()

    window.show()

    return app.exec()


if __name__ == "__main__":
    main()

