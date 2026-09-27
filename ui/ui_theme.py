from __future__ import annotations

from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication, QStyleFactory

from .ui_constants import THEMES


def apply_theme(
    app: QApplication,
    name: str = "dark",
):
    """
    Apply the selected application theme.

    Parameters
    ----------
    app:
        QApplication instance.

    name:
        Theme name. Supported values are defined in THEMES,
        normally "dark" and "light".

    Returns
    -------
    dict
        Active theme color dictionary.
    """

    # --------------------------------------------------------
    # Resolve theme
    # --------------------------------------------------------

    if name not in THEMES:
        name = "dark"

    colors = THEMES[name]

    # --------------------------------------------------------
    # Qt base style
    # --------------------------------------------------------

    fusion_style = QStyleFactory.create("Fusion")

    if fusion_style is not None:
        app.setStyle(fusion_style)

    # --------------------------------------------------------
    # Global font
    # --------------------------------------------------------

    app.setFont(
        QFont(
            "Segoe UI",
            10,
        )
    )

    # --------------------------------------------------------
    # Global stylesheet
    # --------------------------------------------------------

    stylesheet = f"""
/* =========================================================
   GLOBAL
   ========================================================= */

QWidget {{
    background-color: {colors["bg"]};
    color: {colors["text"]};
}}

QMainWindow {{
    background-color: {colors["bg"]};
    color: {colors["text"]};
}}


/* =========================================================
   MAIN PAGE
   ========================================================= */

QWidget#MainPageContent {{
    background-color: {colors["bg"]};
    color: {colors["text"]};
}}


/* =========================================================
   CARDS / SECTIONS
   ========================================================= */

QFrame#Card,
QFrame#ModernSection {{
    background-color: {colors["surface"]};
    color: {colors["text"]};
    border: 1px solid {colors["border"]};
    border-radius: 14px;
}}


/* =========================================================
   LABELS
   ========================================================= */

QLabel {{
    background: transparent;
    color: {colors["text"]};
}}

QLabel#Title {{
    background: transparent;
    color: {colors["text"]};
    font-size: 24px;
    font-weight: 700;
}}

QLabel#Subtitle {{
    background: transparent;
    color: {colors["muted"]};
}}

QLabel#SectionTitle {{
    background: transparent;
    color: {colors["text"]};
    font-size: 15px;
    font-weight: 700;
}}

QLabel#OperationLabel {{
    background: transparent;
    color: {colors["muted"]};
}}


/* =========================================================
   INPUTS
   ========================================================= */

QLineEdit,
QComboBox,
QPlainTextEdit,
QTextEdit,
QTableWidget {{
    background-color: {colors["input_bg"]};
    color: {colors["text"]};
    border: 1px solid {colors["border"]};
    border-radius: 9px;
    padding: 7px;
    selection-background-color: {colors["selection"]};
    selection-color: {colors["text"]};
}}

QLineEdit:focus,
QComboBox:focus,
QPlainTextEdit:focus,
QTextEdit:focus {{
    border: 1px solid {colors["accent"]};
}}


/* =========================================================
   HEADER COMBOBOX
   ========================================================= */

QComboBox#HeaderCombo {{
    background-color: {colors["surface"]};
    color: {colors["text"]};
    border: 1px solid {colors["border"]};
    border-radius: 8px;
    padding: 6px 10px;
    min-height: 20px;
}}

QComboBox#HeaderCombo:hover {{
    border: 1px solid {colors["accent"]};
}}

QComboBox#HeaderCombo:focus {{
    border: 1px solid {colors["accent"]};
}}


/* =========================================================
   COMBOBOX POPUP
   ========================================================= */

QComboBox QAbstractItemView {{
    background-color: {colors["surface"]};
    color: {colors["text"]};
    border: 1px solid {colors["border"]};
    selection-background-color: {colors["selection"]};
    selection-color: {colors["text"]};
    outline: none;
}}


/* =========================================================
   BUTTONS
   ========================================================= */

QPushButton {{
    background-color: {colors["surface2"]};
    color: {colors["text"]};
    border: 1px solid {colors["border"]};
    border-radius: 9px;
    padding: 8px 14px;
    font-weight: 600;
}}

QPushButton:hover {{
    background-color: {colors["accent_hover"]};
    color: white;
    border-color: {colors["accent_hover"]};
}}

QPushButton:pressed {{
    background-color: {colors["accent_pressed"]};
    color: white;
}}

QPushButton:disabled {{
    background-color: {colors["surface2"]};
    color: {colors["disabled"]};
    border-color: {colors["border"]};
}}


/* =========================================================
   PRIMARY BUTTON
   ========================================================= */

QPushButton#Primary {{
    background-color: {colors["accent"]};
    color: white;
    border: 0;
}}

QPushButton#Primary:hover {{
    background-color: {colors["accent_hover"]};
    color: white;
}}

QPushButton#Primary:pressed {{
    background-color: {colors["accent_pressed"]};
    color: white;
}}


/* =========================================================
   DANGER BUTTON
   ========================================================= */

QPushButton#Danger {{
    background-color: {colors["error"]};
    color: white;
    border: 0;
}}

QPushButton#Danger:hover {{
    background-color: {colors["error"]};
    color: white;
}}

QPushButton#Danger:pressed {{
    background-color: {colors["error"]};
    color: white;
}}


/* =========================================================
   PROGRESS BAR
   ========================================================= */

QProgressBar {{
    background-color: {colors["surface2"]};
    color: {colors["text"]};
    border: 0;
    border-radius: 6px;
    text-align: center;
}}

QProgressBar::chunk {{
    background-color: {colors["accent"]};
    border-radius: 6px;
}}


/* =========================================================
   TABS
   ========================================================= */

QTabWidget::pane {{
    background-color: {colors["surface"]};
    border: 1px solid {colors["border"]};
    border-radius: 10px;
}}

QTabBar::tab {{
    background-color: {colors["surface2"]};
    color: {colors["muted"]};
    padding: 10px 18px;
    border: 0;
}}

QTabBar::tab:selected {{
    background-color: {colors["surface"]};
    color: {colors["accent"]};
    font-weight: 700;
}}

QTabBar::tab:hover {{
    background-color: {colors["surface"]};
    color: {colors["text"]};
}}


/* =========================================================
   TABLES
   ========================================================= */

QTableWidget {{
    background-color: {colors["input_bg"]};
    alternate-background-color: {colors["surface2"]};
    color: {colors["text"]};
    gridline-color: {colors["border"]};
}}

QTableWidget::item {{
    color: {colors["text"]};
    background-color: transparent;
    padding: 4px;
}}

QTableWidget::item:selected {{
    background-color: {colors["selection"]};
    color: {colors["text"]};
}}

QHeaderView::section {{
    background-color: {colors["surface2"]};
    color: {colors["text"]};
    border: 0;
    border-bottom: 1px solid {colors["border"]};
    padding: 8px;
}}


/* =========================================================
   SCROLL AREA
   ========================================================= */

QScrollArea {{
    background-color: {colors["bg"]};
    border: none;
}}

QScrollArea > QWidget > QWidget {{
    background-color: {colors["bg"]};
}}


/* =========================================================
   VERTICAL SCROLLBAR
   ========================================================= */

QScrollBar:vertical {{
    background-color: {colors["surface"]};
    width: 10px;
    margin: 0;
    border: none;
}}

QScrollBar::handle:vertical {{
    background-color: {colors["border"]};
    border-radius: 5px;
    min-height: 25px;
}}

QScrollBar::handle:vertical:hover {{
    background-color: {colors["muted"]};
}}

QScrollBar::add-line:vertical,
QScrollBar::sub-line:vertical {{
    height: 0;
    border: none;
}}

QScrollBar::add-page:vertical,
QScrollBar::sub-page:vertical {{
    background: transparent;
}}


/* =========================================================
   HORIZONTAL SCROLLBAR
   ========================================================= */

QScrollBar:horizontal {{
    background-color: {colors["surface"]};
    height: 10px;
    margin: 0;
    border: none;
}}

QScrollBar::handle:horizontal {{
    background-color: {colors["border"]};
    border-radius: 5px;
    min-width: 25px;
}}

QScrollBar::handle:horizontal:hover {{
    background-color: {colors["muted"]};
}}

QScrollBar::add-line:horizontal,
QScrollBar::sub-line:horizontal {{
    width: 0;
    border: none;
}}

QScrollBar::add-page:horizontal,
QScrollBar::sub-page:horizontal {{
    background: transparent;
}}


/* =========================================================
   TOOLTIP
   ========================================================= */

QToolTip {{
    background-color: {colors["surface"]};
    color: {colors["text"]};
    border: 1px solid {colors["border"]};
    padding: 5px;
}}


/* =========================================================
   CHECKBOX / RADIO
   ========================================================= */

QCheckBox,
QRadioButton {{
    background: transparent;
    color: {colors["text"]};
}}


/* =========================================================
   GROUP BOX
   ========================================================= */

QGroupBox {{
    background-color: {colors["surface"]};
    color: {colors["text"]};
    border: 1px solid {colors["border"]};
    border-radius: 10px;
    margin-top: 10px;
    padding: 10px;
}}

QGroupBox::title {{
    background-color: transparent;
    color: {colors["text"]};
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 5px;
}}



QProgressBar[processingState="running"] {{
    background-color: {colors["surface2"]};
    border: 0;
    border-radius: 5px;
}}

QProgressBar[processingState="running"]::chunk {{
    background-color: {colors["accent"]};
    border-radius: 5px;
}}

QProgressBar[processingState="success"] {{
    background-color: {colors["surface2"]};
}}

QProgressBar[processingState="success"]::chunk {{
    background-color: {colors["success"]};
    border-radius: 5px;
}}

QProgressBar[processingState="error"] {{
    background-color: {colors["surface2"]};
}}

QProgressBar[processingState="error"]::chunk {{
    background-color: {colors["error"]};
    border-radius: 5px;
}}
"""

    # --------------------------------------------------------
    # THIS IS THE IMPORTANT PART
    # --------------------------------------------------------
    # The stylesheet must actually be installed on the
    # QApplication. Merely creating the string does nothing.

    app.setStyleSheet(stylesheet)

    # Make sure the new style is processed immediately.
    app.processEvents()

    # Return the active colors to UIAppearanceMixin.
    return colors


