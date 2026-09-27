from PySide6.QtWidgets import QPushButton
class AppButton(QPushButton):
    def __init__(self,text='',primary=False,danger=False,parent=None):
        super().__init__(text,parent)
        if primary:self.setObjectName('Primary')
        if danger:self.setObjectName('Danger')
        self.setCursor(__import__('PySide6').QtCore.Qt.PointingHandCursor)
