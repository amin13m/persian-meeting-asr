from PySide6.QtWidgets import QWidget,QVBoxLayout,QLabel
from .card import Card
class Section(Card):
    def __init__(self,title,parent=None):
        super().__init__(parent); self.title_label=QLabel(title); self.title_label.setObjectName('SectionTitle'); self.layout.addWidget(self.title_label)
