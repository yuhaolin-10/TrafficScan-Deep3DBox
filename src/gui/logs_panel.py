try:
    from gui.qt import QtCore, QtGui, QtWidgets
except Exception:
    from qt import QtCore, QtGui, QtWidgets


class LogsPanel(QtWidgets.QListWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(180)
        self.setStyleSheet("background:#0a0a0a;border-top:1px solid #2a2a2a;")
        self.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._open_menu)

    def log(self, level: str, message: str):
        color = {
            "info": "#60a5fa",
            "warning": "#f59e0b",
            "success": "#22c55e",
            "error": "#ef4444",
        }.get(level, "#9ca3af")
        item = QtWidgets.QListWidgetItem(message)
        item.setForeground(QtGui.QColor(color))
        item.setFont(QtGui.QFont("Consolas"))
        self.addItem(item)
        self.scrollToBottom()

    def _open_menu(self, pos):
        menu = QtWidgets.QMenu(self)
        menu.setStyleSheet(
            "QMenu{background:#111827;border:1px solid #2a2a2a;min-width:220px;}"
            "QMenu::item{padding:6px 18px;color:#d1d5db;font-family:Consolas;}"
            "QMenu::item:selected{background:#1f2937;}"
        )
        act_copy_selected = menu.addAction("Copy Selected")
        act_copy_all = menu.addAction("Copy All")
        act_clear = menu.addAction("Clear")

        exec_menu = getattr(menu, "exec", None) or getattr(menu, "exec_", None)
        chosen = exec_menu(self.mapToGlobal(pos))
        if chosen is None:
            return

        if chosen == act_copy_selected:
            lines = [it.text() for it in self.selectedItems()]
            if lines:
                QtWidgets.QApplication.clipboard().setText("\n".join(lines))
            return

        if chosen == act_copy_all:
            lines = [self.item(i).text() for i in range(self.count())]
            if lines:
                QtWidgets.QApplication.clipboard().setText("\n".join(lines))
            return

        if chosen == act_clear:
            self.clear()
            return
