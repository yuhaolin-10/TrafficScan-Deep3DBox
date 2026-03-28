# TrafficScan Lab — UI 计划（精简版）

本文档只保留 3 个部分：
1) 组件的功能  
2) 文件结构  
3) 开发里程碑

参考实现/现状入口：
- 当前 Qt UI 入口：[app.py](file:///d:/college/GraduationProject/testProject/TrafficScan_lab/src/gui/app.py)
- 当前主窗口组装：[main_window.py](file:///d:/college/GraduationProject/testProject/TrafficScan_lab/src/gui/main_window.py)
- 当前 Qt 兼容层：[qt.py](file:///d:/college/GraduationProject/testProject/TrafficScan_lab/src/gui/qt.py)
- 当前通用/兼容聚合：[widgets.py](file:///d:/college/GraduationProject/testProject/TrafficScan_lab/src/gui/widgets.py)
- Viewer 面板：[viewer_panel.py](file:///d:/college/GraduationProject/testProject/TrafficScan_lab/src/gui/viewer_panel.py)
- Workspace 面板：[workspace_panel.py](file:///d:/college/GraduationProject/testProject/TrafficScan_lab/src/gui/workspace_panel.py)
- ImageViewer 控件：[image_viewer.py](file:///d:/college/GraduationProject/testProject/TrafficScan_lab/src/gui/image_viewer.py)
- Violations 表：[violations_table.py](file:///d:/college/GraduationProject/testProject/TrafficScan_lab/src/gui/violations_table.py)
- Logs 面板：[logs_panel.py](file:///d:/college/GraduationProject/testProject/TrafficScan_lab/src/gui/logs_panel.py)

---

## 1. 组件的功能

### 1.1 顶部工具栏（Toolbar）
- 放置全局动作：打开图片、运行（单张/批量）、进入历史页、设置（模型路径/阈值/输出目录）
- 展示应用信息：系统名/版本/当前工作状态（空闲/运行中/错误）

### 1.2 工作区（Workspace）
- 导入：支持文件/文件夹拖拽导入；支持打开文件对话框导入
- 队列：显示任务列表（文件名、可选缩略图、状态：PENDING/RUNNING/DONE/FAILED）
- 选择：选中任务后驱动中央预览刷新
- 批量：支持清空、移除选中、仅运行选中（后续）

### 1.3 中央预览（Image Viewer）
- 展示：显示当前选中图片
- 缩放与视图：适配窗口（默认）/ 滚轮缩放 / 双击回到适配窗口（可选：补充 1:1 模式与显式缩放控件）
- 图层开关：车道 mask、底面 footprint、3D 框、标签、违章高亮
- 图例：固定语义颜色（正常/违章/应急车道）

### 1.4 右侧结果表（Violations / Vehicles Table）
- 列表：展示车辆检测结果与关键字段（idx、type、conf、ratio、is_violating）
- 联动：点击行 → 中央预览高亮对应车辆（只重绘，不重跑模型）
- 确认：确认勾选/仅显示未确认（后续）

### 1.5 底部日志与进度（Logs / Progress）
- 日志：分级颜色（info/warning/success/error），自动滚动到最新行
- 进度：批量运行时显示进度与耗时（后续）

### 1.6 后台处理（Processing Worker）
- 线程化：图片读取 → 模型推理 → 违章判定 → 返回结构化结果，避免 UI 假死
- 返回：只回传“纯数据包”（mask/vehicles/any_violation/错误信息），UI 决定如何展示

### 1.7 渲染服务（Renderer）
- 输入：原图 + mask + vehicles + layers + selected_idx
- 输出：渲染后的叠加图（图层开关只触发重绘，不触发推理）

### 1.8 存证与历史（Archive / History）
- 存证：any_violation 为 True 时保存结果图并写入 SQLite（复用 DatabaseManager）
- 历史：查询 SQLite 并展示记录列表、原图/结果图对比（后续）

---

## 2. 文件结构

### 2.1 当前已落地（M1.2 现状）

```text
src/gui/
  app.py              # 应用入口：创建 QApplication + 全局样式 + 进入事件循环
  qt.py               # Qt 兼容层：PySide6 优先、PyQt5 回退、导出 exec_app
  main_window.py      # 主窗口编排：组装布局、连接信号、驱动 M1.2 的 UI-only 流程
  workspace_panel.py  # Workspace 面板：导入、列表、多选、去重、任务状态与基础批量操作
  viewer_panel.py     # Viewer 面板：组合 ImageViewer + Layers UI + 状态栏（文件/缩放/高亮）
  image_viewer.py     # ImageViewer 控件：滚轮缩放、双击 fit、拖拽平移、placeholder
  violations_table.py # Violations 表：mock 填充入口、Confirmed 勾选交互、行选中信号
  logs_panel.py       # Logs 面板：分级彩色日志 + 右键 Copy/Clear
  widgets.py          # 兼容/聚合：转导出常用控件，避免拆分阶段频繁改 import
```

### 2.2 目标结构（后续按组件逐步拆分，不要求一次到位）

```text
src/
  gui/
    app.py
    qt.py
    main_window.py
    theme/
      dark.qss
    dialogs/
      settings_dialog.py
    widgets/
      toolbar.py
      workspace_panel.py
      image_viewer.py
      violations_table.py
      history_view.py
      logs_panel.py
    models/
      workspace_model.py
      violations_model.py
      history_model.py
    workers/
      processing_worker.py
  services/
    pipeline.py      # 处理单张图：跑模型→返回结构化结果
    renderer.py      # 叠加渲染：按图层开关画图（不跑模型）
    history_repo.py  # 查询 sqlite 历史记录（只读）
```

约束：
- 拆分原则：单文件职责单一，优先拆“可复用控件”与“主窗口编排”
- 拆分节奏：一次只拆一个面板/模块，保证 review 成本可控
- 目录化节奏：目标结构保留，但允许先在 src/gui/ 平铺落地；稳定后再迁入 gui/widgets/ 等子目录

---

## 3. 开发里程碑

### M1：UI 骨架（不接模型）
- 骨架布局：Toolbar / Workspace / Tabs(Live/History) / Logs / 右侧表占位
- 导入闭环：打开/拖拽 → 加入列表 → 选中 → 中央预览显示 → 日志反馈
- 样式基线：深色主题 + 基本选中态 + 边框分割

验收：不依赖任何模型文件也能稳定运行。

### M1.1：可维护结构（已完成）
- 单文件骨架拆分为 app/qt/widgets/main_window
- 启动链路稳定：进入 Qt 事件循环，窗口不再“闪退”

验收：结构清晰，便于后续按组件继续拆分与迭代。

### M1.2：预览与组件体验补齐（已完成）
- Viewer：滚轮缩放 + 双击回 fit + 拖拽平移；图层开关 UI；状态提示（英文）
- Workspace：任务状态字段占位（PENDING/RUNNING/DONE/FAILED）；多选；Clear/Remove Selected
- Violations 表：mock 数据填充；Confirmed 勾选交互；行选中联动 Viewer 高亮占位
- Logs：分级颜色；右键菜单 Copy Selected / Copy All / Clear

验收：交互体验贴近设计稿，仍不依赖模型与数据库。

### M2：接入 pipeline（单张）
- `services/pipeline.py`：复用现有后端流程（参考 [main_system.py](file:///d:/college/GraduationProject/testProject/TrafficScan_lab/src/main_system.py)）
- Worker：单张运行不阻塞 UI；回传结构化结果并填充右侧表

验收：单张运行稳定，UI 无卡死，结果可展示。

### M3：图层渲染与联动高亮
- `services/renderer.py`：叠加渲染（lane/footprint/3D box/label/highlight）
- 表格选中行 → 预览高亮对应车辆（只重绘）

验收：图层开关即时响应，联动高亮稳定。

### M4：批量运行 + 存证入库 + 历史查询
- 批量队列调度 + 进度显示
- any_violation 才存证 + 写入 SQLite（复用 DatabaseManager）
- History：从 sqlite 查询并展示记录与对比视图

验收：批量可跑、存证可查、历史可回放。
