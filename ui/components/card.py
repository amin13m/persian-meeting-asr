from PySide6.QtWidgets import QFrame, QVBoxLayout
class Card(QFrame):
    def __init__(self,parent=None):
        super().__init__(parent); self.setObjectName('Card'); self.layout=QVBoxLayout(self); self.layout.setContentsMargins(18,16,18,16); self.layout.setSpacing(12)
