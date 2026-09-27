from PySide6.QtWidgets import QWidget,QVBoxLayout,QHBoxLayout,QPlainTextEdit
from ..components import AppButton
class LogsPage(QWidget):
    def __init__(self,app): self.app=app; super().__init__(); l=QVBoxLayout(self); bar=QHBoxLayout(); self.clear=AppButton(); bar.addStretch(); bar.addWidget(self.clear); l.addLayout(bar); self.text=QPlainTextEdit(); self.text.setReadOnly(True); l.addWidget(self.text,1); self.clear.clicked.connect(app.clear_logs)
    def retranslate(self,t): self.clear.setText(t['clear_logs'])
