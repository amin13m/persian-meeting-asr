
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QScrollArea,
    QTabWidget,
    QWidget,
    QVBoxLayout,
)

from .pages import (
    MainPage,
    FilesPage,
    LogsPage,
)


class UILayoutMixin:

    # ========================================================
    # Helpers
    # ========================================================

    def _make_scroll_area(self, widget):
        """
        Put a page inside a vertical scroll area.

        The page keeps its natural minimum size while allowing
        the user to scroll when the window is too small.
        """

        container = QWidget()

        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(widget)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarAlwaysOff
        )
        scroll.setVerticalScrollBarPolicy(
            Qt.ScrollBarAsNeeded
        )
        scroll.setWidget(container)

        return scroll

    # ========================================================
    # Build UI
    # ========================================================

    def build_ui(self):

        self.tabs = QTabWidget()

        # ----------------------------------------------------
        # Pages
        # ----------------------------------------------------

        self.main_page = MainPage(self)
        self.files_page = FilesPage(self)
        self.logs_page = LogsPage(self)

        # ----------------------------------------------------
        # Scroll areas
        # ----------------------------------------------------

        self.main_scroll = self._make_scroll_area(
            self.main_page
        )

        self.files_scroll = self._make_scroll_area(
            self.files_page
        )

        # ----------------------------------------------------
        # Tabs
        # ----------------------------------------------------

        self.tabs.addTab(
            self.main_scroll,
            "",
        )

        self.tabs.addTab(
            self.files_scroll,
            "",
        )

        self.tabs.addTab(
            self.logs_page,
            "",
        )

        # ----------------------------------------------------
        # Central widget
        # ----------------------------------------------------

        self.setCentralWidget(
            self.tabs
        )

        # ----------------------------------------------------
        # Remember last tab
        # ----------------------------------------------------

        self.tabs.currentChanged.connect(
            lambda index: self.settings.save(
                last_tab=index
            )
        )

