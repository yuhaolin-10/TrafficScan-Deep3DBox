"""
Qt 绑定兼容层（PySide6 优先，PyQt5 回退）

这个模块的目标是“让项目其他地方不再写条件导入”：
- 统一导出 QtCore / QtGui / QtWidgets 命名空间
- 统一导出 exec_app：屏蔽 PySide6(app.exec) 与 PyQt5(app.exec_) 的差异
- 通过缓存保证“只初始化一次 Qt 绑定”，避免混用两套 Qt 导致的崩溃/异常

注意：
- M1 阶段我们不引入 Qt Designer 的 .ui 动态加载逻辑；后续需要再加
"""

_qt_cache = None


def load_qt():
    """
    实际执行导入（不缓存）。

    返回：
    - QtCore, QtGui, QtWidgets：Qt 三大命名空间
    - exec_app：事件循环入口函数（兼容 PySide6 / PyQt5）
    """
    try:
        from PySide6 import QtCore, QtGui, QtWidgets

        def exec_app(app):
            return app.exec()

        return QtCore, QtGui, QtWidgets, exec_app
    except Exception:
        from PyQt5 import QtCore, QtGui, QtWidgets

        def exec_app(app):
            return app.exec_()

        return QtCore, QtGui, QtWidgets, exec_app


def get_qt():
    """
    获取（并缓存）Qt 命名空间。

    为什么要缓存：
    - Python 里多处“回退导入”容易出现某处 PySide6、某处 PyQt5 的混用
    - 一旦混用，同类型对象的 isinstance 判断、信号槽连接、甚至 Qt 内部会崩
    """
    global _qt_cache
    if _qt_cache is None:
        _qt_cache = load_qt()
    return _qt_cache


# 模块级导出：其他模块直接 from gui.qt import QtCore, QtGui, QtWidgets, exec_app
# 约束：不要在其他地方再次调用 load_qt()，统一从这里拿同一套绑定
QtCore, QtGui, QtWidgets, exec_app = get_qt()
