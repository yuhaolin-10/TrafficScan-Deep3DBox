# Deep3DBox 整合迁移说明

## 1. 项目定位

`TrafficScan_Deep3DBox_UI` 是一个新的整合工程。

它的目标不是继续在原目录上混改，而是明确分出一个新项目，用来做下面这件事：

- 保留 `TrafficScan_UI` 的界面、任务流、数据库和可视化结构
- 逐步把当前项目里的 3D 车辆几何恢复替换为 Deep3DBox 路线

---

## 2. 当前来源

### UI 来源

- `D:\college\GraduationProject\testProject\TrafficScan_UI`

### Deep3DBox 验证来源

- `D:\college\GraduationProject\testProject\deep3dbox_independent_test`

### 新项目目录

- `D:\college\GraduationProject\testProject\TrafficScan_Deep3DBox_UI`

---

## 3. 已完成的初始化

目前已经完成：

1. 复制 `TrafficScan_UI` 作为主工程基础
2. 将 Deep3DBox demo checkpoint 放入新项目：
   - `external/deep3dbox_demo_model/`
3. 将独立验证脚本迁入新项目：
   - `src/tools/run_deep3dbox_illegal_test.py`
4. 调整独立测试脚本默认路径，使其指向新项目内部资源

---

## 4. 推荐迁移路线

### 第一阶段：保持独立验证

先不要直接改 GUI 主链路，先保证下面这条命令能稳定输出结果：

- `src/tools/run_deep3dbox_illegal_test.py`

目标：

- 确认 Deep3DBox 在当前违法测试图片上效果可接受
- 明确输出字段能否满足 GUI 渲染和违章判定需要

### 第二阶段：封装算法类

建议新增：

- `src/core/vehicle_detector_deep3dbox.py`

职责：

- 对外保持和当前 `VehicleDetector3D` 尽量接近的接口
- 对内复用 Deep3DBox 的检测、alpha 解码、位置求解逻辑

### 第三阶段：接入服务层

重点对接：

- `src/services/pipeline.py`
- `src/gui/processing_worker.py`

建议方式：

- 先做“切换式接入”
- 允许选择 `legacy` / `deep3dbox` 两种 3D 检测实现

### 第四阶段：接入 GUI

在 GUI 中补一项配置：

- 当前 3D 算法模式

这样可以避免一开始就把旧流程完全替掉，便于对比效果。

---

## 5. 当前最重要的接口约束

如果后续要顺利替换，Deep3DBox 输出最好逐步兼容这些字段：

- `bbox`
- `type`
- `conf`
- `corners_2d`
- `footprint_2d`
- `yaw`

原因很简单：

当前 UI 渲染、违章判定和数据库入库流程，都是围绕这些字段工作的。

---

## 6. 结论

现在已经可以把 `TrafficScan_Deep3DBox_UI` 作为新的正式整合目录继续开发。
接下来最合理的动作是：先把 Deep3DBox 封装成 `src/core/` 下的新检测器，再逐步接到 GUI 处理链路里。
