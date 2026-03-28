from pathlib import Path
import sys

try:
    from gui.app import create_app
    from gui.main_window import MainWindow
    from gui.qt import QtCore, exec_app
except Exception:
    current_dir = Path(__file__).resolve().parent
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))
    from gui.app import create_app
    from gui.main_window import MainWindow
    from gui.qt import QtCore, exec_app


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_IMPORT_FOLDERS = [
    PROJECT_ROOT / "images" / "test_images" / "highway",
    PROJECT_ROOT / "images" / "test_images" / "illegal",
]


def _bootstrap(window: MainWindow):
    folders = [str(path) for path in DEFAULT_IMPORT_FOLDERS if path.exists()]
    missing = [str(path) for path in DEFAULT_IMPORT_FOLDERS if not path.exists()]

    if missing:
        window._log("warning", f"Missing import folders: {', '.join(missing)}")

    if not folders:
        window._log("error", "No valid import folders found; auto-run cancelled")
        return

    added = window.workspace.add_paths(folders)
    window._log("info", f"Auto-import complete: {added} images added from {len(folders)} folders")

    if window.workspace.list.count() == 0:
        window._log("warning", "Workspace is empty after auto-import")
        return

    if window.workspace.list.currentRow() < 0 and window.workspace.list.count() > 0:
        window.workspace.list.setCurrentRow(0)

    QtCore.QTimer.singleShot(150, window._run_all)


def run():
    app = create_app()
    window = MainWindow()
    window.show()

    QtCore.QTimer.singleShot(0, lambda: _bootstrap(window))
    return exec_app(app)


if __name__ == "__main__":
    raise SystemExit(run())
