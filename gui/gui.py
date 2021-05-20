from PySide2.QtWidgets import QMainWindow

from gui import Ui_Form


class MainWindow(QMainWindow, Ui_Form):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
