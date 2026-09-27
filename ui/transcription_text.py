from PySide6.QtWidgets import QTextEdit
from PySide6.QtCore import Qt
class TranscriptionText(QTextEdit):
    def __init__(self,parent=None):
        super().__init__(parent); self.setAcceptRichText(False); self.setUndoRedoEnabled(True); self.setLineWrapMode(QTextEdit.WidgetWidth); self.setContextMenuPolicy(Qt.DefaultContextMenu)
    def set_text(self,text,rtl=True):
        self.setPlainText(text or '')
        self.setLayoutDirection(Qt.RightToLeft if rtl else Qt.LeftToRight)
        self.setAlignment(Qt.AlignRight if rtl else Qt.AlignLeft)
    def normalized_selected_text(self): return self.textCursor().selectedText().replace('\u2029','\n')
    def copy_clean(self):
        text=self.textCursor().selectedText() or self.toPlainText()
        for ch in '\u200e\u200f\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069': text=text.replace(ch,'')
        QApplication=self.window().windowHandle().screen() if False else None
        self.setPlainText(self.toPlainText()) if False else None
        return text.replace('\r\n','\n').replace('\r','\n')
