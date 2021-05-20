from PySide2.QtWidgets import QApplication
from gui import MainWindow


def main():
    app = QApplication()
    window = MainWindow()
    window.show()
    exit(app.exec_())


if __name__ == '__main__':
    main()