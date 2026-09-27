from pathlib import Path
import os
import sys
import subprocess
from datetime import datetime

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QAbstractItemView,
)

from PySide6.QtCore import Qt

from ..components import AppButton, Card
from ..ui_constants import RECORDINGS_DIR


class FilesPage(QWidget):
    def __init__(self, app):
        self.app = app
        super().__init__()
        self.build()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def build(self):
        layout = QVBoxLayout(self)

        # --------------------------------------------------------------
        # Toolbar
        # --------------------------------------------------------------

        toolbar = QHBoxLayout()

        self.recordings_btn = AppButton()
        self.input_btn = AppButton()
        self.output_btn = AppButton()

        self.refresh = AppButton()
        self.open = AppButton()
        self.delete = AppButton(danger=True)

        toolbar.addWidget(self.recordings_btn)
        toolbar.addWidget(self.input_btn)
        toolbar.addWidget(self.output_btn)

        toolbar.addStretch()

        toolbar.addWidget(self.refresh)
        toolbar.addWidget(self.open)
        toolbar.addWidget(self.delete)

        layout.addLayout(toolbar)

        # --------------------------------------------------------------
        # Files table
        # --------------------------------------------------------------

        self.table = QTableWidget(0, 3)

        self.table.setSelectionBehavior(
            QAbstractItemView.SelectRows
        )

        self.table.setSelectionMode(
            QAbstractItemView.SingleSelection
        )

        self.table.setEditTriggers(
            QAbstractItemView.NoEditTriggers
        )

        self.table.setAlternatingRowColors(True)

        self.table.doubleClicked.connect(
            self.open_selected
        )

        layout.addWidget(self.table, 1)

        # --------------------------------------------------------------
        # State
        # --------------------------------------------------------------

        # Keep previous behavior:
        # Input is the default page.
        self.mode = "input"

        # --------------------------------------------------------------
        # Signals
        # --------------------------------------------------------------

        self.recordings_btn.clicked.connect(
            lambda: self.set_mode("recordings")
        )

        self.input_btn.clicked.connect(
            lambda: self.set_mode("input")
        )

        self.output_btn.clicked.connect(
            lambda: self.set_mode("output")
        )

        self.refresh.clicked.connect(
            self.reload
        )

        self.open.clicked.connect(
            self.open_selected
        )

        self.delete.clicked.connect(
            self.delete_selected
        )

        self.reload()

    # ------------------------------------------------------------------
    # Translation
    # ------------------------------------------------------------------

    def retranslate(self, t):
        """
        Update all visible texts.

        files_recordings is optional so this page remains compatible
        with the existing ui_constants.py.
        """

        if self.app.language == "fa":
            recordings_text = t.get(
                "files_recordings",
                "ضبط‌ها",
            )
        else:
            recordings_text = t.get(
                "files_recordings",
                "Recordings",
            )

        self.recordings_btn.setText(
            recordings_text
        )

        self.input_btn.setText(
            t["files_input"]
        )

        self.output_btn.setText(
            t["files_output"]
        )

        self.refresh.setText(
            t["refresh"]
        )

        self.open.setText(
            t["open"]
        )

        self.delete.setText(
            t["delete"]
        )

        self.table.setHorizontalHeaderLabels(
            [
                t["name"],
                t["size"],
                t["modified"],
            ]
        )

    # ------------------------------------------------------------------
    # Directory
    # ------------------------------------------------------------------

    def directory(self):
        """
        Return the directory associated with the current mode.
        """

        if self.mode == "recordings":
            return Path(RECORDINGS_DIR)

        config = self.app.load_config()

        if self.mode == "input":
            return Path(
                self.app.get_input_dir(config)
            )

        return Path(
            self.app.get_output_dir(config)
        )

    # ------------------------------------------------------------------
    # Mode
    # ------------------------------------------------------------------

    def set_mode(self, mode):
        """
        Change displayed directory.
        """

        if mode not in (
            "recordings",
            "input",
            "output",
        ):
            return

        self.mode = mode

        self.reload()

    # ------------------------------------------------------------------
    # Reload
    # ------------------------------------------------------------------

    def reload(self):
        """
        Reload files from the currently selected directory.
        """

        try:
            path = self.directory()

        except Exception as exc:
            self.app.logger.exception(
                "Failed to resolve files directory: %s",
                exc,
            )

            path = Path.cwd()

        try:
            if not path.exists():
                self.table.setRowCount(0)
                return

            if not path.is_dir():
                self.table.setRowCount(0)
                return

            files = [
                p
                for p in path.iterdir()
                if p.is_file()
            ]

            files.sort(
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )

            self.table.setRowCount(
                len(files)
            )

            for row, file_path in enumerate(files):
                # ------------------------------------------------------
                # Name
                # ------------------------------------------------------

                name_item = QTableWidgetItem(
                    file_path.name
                )

                # ------------------------------------------------------
                # Size
                # ------------------------------------------------------

                try:
                    size = file_path.stat().st_size
                except OSError:
                    size = 0

                size_item = QTableWidgetItem(
                    self.human(size)
                )

                # ------------------------------------------------------
                # Modified
                # ------------------------------------------------------

                try:
                    modified = datetime.fromtimestamp(
                        file_path.stat().st_mtime
                    ).strftime(
                        "%Y-%m-%d %H:%M"
                    )
                except OSError:
                    modified = ""

                modified_item = QTableWidgetItem(
                    modified
                )

                # ------------------------------------------------------
                # Insert
                # ------------------------------------------------------

                self.table.setItem(
                    row,
                    0,
                    name_item,
                )

                self.table.setItem(
                    row,
                    1,
                    size_item,
                )

                self.table.setItem(
                    row,
                    2,
                    modified_item,
                )

            self.table.resizeColumnsToContents()

            # Keep filename column reasonably wide.
            if self.table.columnWidth(0) < 250:
                self.table.setColumnWidth(
                    0,
                    250,
                )

        except Exception as exc:
            self.app.logger.exception(
                "Failed to reload files: %s",
                exc,
            )

            self.table.setRowCount(0)

    # ------------------------------------------------------------------
    # Human readable file size
    # ------------------------------------------------------------------

    def human(self, n):
        """
        Convert byte count to a human-readable size.
        """

        for unit in (
            "B",
            "KB",
            "MB",
            "GB",
        ):
            if n < 1024:
                return f"{n:.0f} {unit}"

            n /= 1024

        return f"{n:.1f} TB"

    # ------------------------------------------------------------------
    # Selected file
    # ------------------------------------------------------------------

    def selected_path(self):
        """
        Return the full path of the selected file.
        """

        row = self.table.currentRow()

        if row < 0:
            return None

        item = self.table.item(
            row,
            0,
        )

        if item is None:
            return None

        try:
            return self.directory() / item.text()

        except Exception as exc:
            self.app.logger.exception(
                "Failed to resolve selected file: %s",
                exc,
            )

            return None

    # ------------------------------------------------------------------
    # Open
    # ------------------------------------------------------------------

    def open_selected(self):
        """
        Open the selected file using the operating system.
        """

        path = self.selected_path()

        if path is None:
            return

        if not path.exists():
            self.reload()
            return

        try:
            self.app.open_path(path)

        except Exception as exc:
            self.app.logger.exception(
                "Failed to open file: %s",
                exc,
            )

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    def delete_selected(self):
        """
        Delete the selected file after confirmation.
        """

        path = self.selected_path()

        if path is None:
            return

        if not path.exists():
            self.reload()
            return

        from ..components.dialogs import confirm

        try:
            title = self.app.TEXT["warning"]

            message = (
                f'{self.app.TEXT["delete"]}: '
                f"{path.name}?"
            )

            confirmed = confirm(
                self.app,
                title,
                message,
            )

            if not confirmed:
                return

            path.unlink()

            self.reload()

            self.app.logger.info(
                "File deleted: %s",
                path,
            )

        except Exception as exc:
            self.app.logger.exception(
                "File delete failed: %s",
                exc,
            )