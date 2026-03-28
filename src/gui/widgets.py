"""
可复用 Widgets（M1 UI 骨架）

本文件放“跨页面/跨窗口可复用”的基础控件：
- PreviewLabel：中央预览区（发出 resized 信号，便于外部做重渲染节流）
- DropArea：支持从资源管理器拖拽文件/文件夹的投放区域
- WorkspacePanel：左侧工作区面板（DropArea + 列表 + 路径去重/过滤逻辑）

约束：
- 这些控件不依赖 MainWindow，避免循环依赖
- 信号只传“必要的数据”（例如 file_selected 只传路径 str）
"""

try:
    from .image_viewer import ImageViewer
except Exception:
    try:
        from gui.image_viewer import ImageViewer
    except Exception:
        from image_viewer import ImageViewer

try:
    from gui.qt import QtCore, QtWidgets
except Exception:
    from qt import QtCore, QtWidgets


class PreviewLabel(QtWidgets.QLabel):
    """
    用于中央预览区的 QLabel。

    为什么要继承 QLabel：
    - M1 阶段只验证“按窗口尺寸等比缩放显示图片”的链路
    - QLabel.setPixmap 是最小闭环方案（无需自绘/场景图/缩放控件）

    resized 信号的作用：
    - 主窗口/分割器拖动会触发 resizeEvent
    - 预览渲染（pixmap.scaled）应当节流/合并，避免高频卡顿
    """
    resized = QtCore.pyqtSignal() if hasattr(QtCore, "pyqtSignal") else QtCore.Signal()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.resized.emit()

try:
    from .workspace_panel import DropArea, WorkspacePanel
except Exception:
    try:
        from gui.workspace_panel import DropArea, WorkspacePanel
    except Exception:
        from workspace_panel import DropArea, WorkspacePanel
