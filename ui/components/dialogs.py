from PySide6.QtWidgets import QMessageBox

def info(parent,title,text): QMessageBox.information(parent,title,text)
def warning(parent,title,text): QMessageBox.warning(parent,title,text)
def error(parent,title,text): QMessageBox.critical(parent,title,text)
def confirm(parent,title,text): return QMessageBox.question(parent,title,text,QMessageBox.Yes|QMessageBox.No,QMessageBox.No)==QMessageBox.Yes
