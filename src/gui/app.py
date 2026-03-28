import sys
import traceback

try:
    from .main_window import MainWindow
    from .qt import QtGui, QtWidgets, exec_app
except Exception:
    try:
        from gui.main_window import MainWindow
        from gui.qt import QtGui, QtWidgets, exec_app
    except Exception:
        from main_window import MainWindow
        from qt import QtGui, QtWidgets, exec_app


def create_app():
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("交通违章识别系统")
    app.setFont(QtGui.QFont("Microsoft YaHei UI", 10))
    app.setStyleSheet(
        "QMainWindow{background:#0b1220;color:#d1d5db;}"
        "QWidget{color:#d1d5db;font-size:12px;}"
    )
    return app


def run():
    app = create_app()

    global _main_window_ref
    try:
        _main_window_ref = MainWindow()
        _main_window_ref.show()
    except Exception:
        traceback.print_exc()
        raise

    return exec_app(app)


_main_window_ref = None


if __name__ == "__main__":
    raise SystemExit(run())
