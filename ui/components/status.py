from PySide6.QtWidgets import QWidget,QHBoxLayout,QLabel
class StatusWidget(QWidget):
    def __init__(self,parent=None):
        super().__init__(parent); l=QHBoxLayout(self); l.setContentsMargins(0,0,0,0); self.dot=QLabel('●'); self.text=QLabel(); l.addWidget(self.dot); l.addWidget(self.text); l.addStretch()
    def set_status(self,text,color): self.text.setText(text); self.dot.setStyleSheet(f'color:{color}; font-size:16px;')
