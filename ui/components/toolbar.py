from PySide6.QtWidgets import QToolBar
from PySide6.QtCore import Qt
class AppToolBar(QToolBar):
    def __init__(self,parent=None): super().__init__(parent); self.setMovable(False); self.setFloatable(False); self.setToolButtonStyle(Qt.ToolButtonTextOnly)
