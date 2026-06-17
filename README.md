# TrafficScan Deep3DBox UI

TrafficScan Deep3DBox UI 是一个面向固定机位交通场景的桌面端车辆违章检测原型系统。项目以 Python + Qt 为主界面，整合车道分割、车辆检测、Deep3DBox 单目 3D 估计、区域规则判定、结果渲染和 SQLite 持久化，用于毕业设计、算法验证和交通视频处理流程演示。

> 当前仓库默认不包含模型权重、测试视频、运行结果数据库等大文件。本地复现时需要按本文档准备对应资产。

## 功能特性

- 桌面端批处理界面：支持图片/视频导入、批量处理、结果预览、日志和违章记录展示。
- 车道与区域理解：基于 YOLO 分割模型进行车道区域识别，并支持固定机位下的区域规则配置。
- 车辆检测与单目 3D 估计：使用 YOLO 进行 2D 车辆检测，并接入 Deep3DBox checkpoint 估计 3D 姿态与空间关系。
- 违章规则判定：支持逆行、禁停等规则的检测流程扩展。
- 结果持久化：处理记录、违章信息和输出路径写入 SQLite，便于后续检索。
- 独立验证脚本：提供图像批处理、系统流程和 Deep3DBox 单独验证入口。
- 前端原型：`Figma/Traffic Violation Detection App` 保留 React/Vite 原型，用于 UI 方案参考，不影响 Python 桌面端运行。

## 项目结构

```text
TrafficScan_Deep3DBox_UI/
  src/
    core/           # 车道分割、车辆检测、深度/3D估计、违章检测等核心模块
    gui/            # Qt 桌面端界面
    services/       # 处理流水线、数据库、场景配置、视频读写等服务层
    tests/          # 单元测试
    tools/          # Deep3DBox、OBB 预览等独立工具脚本
    run_gui_auto_batch.py
    main_image_test.py
    main_system.py
  Docs/             # 设计、部署和阶段性文档
  experiments/      # 车牌识别等实验性功能
  Figma/            # React/Vite UI 原型
  requirements.txt
```

## 环境要求

- Windows 10/11 64 位。
- Python 3.10。Deep3DBox 依赖的 TensorFlow 2.10.x 在 Windows 原生环境下对 Python 3.10 更稳定。
- 推荐使用 Anaconda/Miniconda。
- CPU 可以运行完整流程；若要较快推理，建议使用 NVIDIA GPU，并准备匹配 PyTorch/TensorFlow 的 CUDA/cuDNN 环境。
- 详细部署说明见 [Docs/部署与安装要求.md](Docs/部署与安装要求.md)。

## 快速开始

创建环境并安装依赖：

```powershell
conda create -n trafficscan-deep3dbox python=3.10 -y
conda activate trafficscan-deep3dbox
python -m pip install --upgrade pip
pip install -r requirements.txt
```

准备本地模型文件：

```text
src/models/best.pt
src/models/yolo11l.pt
external/deep3dbox_demo_model/demo_model.data-00000-of-00001
external/deep3dbox_demo_model/demo_model.index
external/deep3dbox_demo_model/demo_model.meta
```

运行前可检查关键文件：

```powershell
Test-Path src\models\best.pt
Test-Path src\models\yolo11l.pt
Test-Path external\deep3dbox_demo_model\demo_model.index
```

启动桌面 GUI：

```powershell
python src\run_gui_auto_batch.py
```

也可以执行项目根目录下的批处理脚本：

```powershell
.\run_gui_auto_batch.bat
```

## 常用命令

运行图片批处理验证：

```powershell
python src\main_image_test.py
```

运行带数据库持久化的系统流程：

```powershell
python src\main_system.py
```

运行 Deep3DBox 独立验证：

```powershell
python src\tools\run_deep3dbox_illegal_test.py
```

运行单元测试：

```powershell
pytest src\tests
```

## 数据、模型与输出

以下内容通常较大或属于本地运行产物，默认不纳入 Git：

- `data/`：SQLite 数据库、处理后的图片/视频、场景配置等运行输出。
- `images/`：本地测试图片、视频和演示素材。
- `src/models/`：YOLO/车道分割等模型权重。
- `external/deep3dbox_demo_model/`：Deep3DBox checkpoint。
- `*.pt`、`*.pth`、`*.ckpt`、`*.onnx`、`*.engine` 等模型文件。

如果你希望发布可复现实验，需要在 README 或 Release 中说明模型、样例数据的获取方式，并注意对应数据集和模型许可证。

## 可选前端原型

React/Vite 原型位于：

```text
Figma/Traffic Violation Detection App
```

运行方式：

```powershell
cd "Figma\Traffic Violation Detection App"
npm install
npm run dev
```

该目录只是 UI 原型，不是 Python 桌面端的运行入口。

## 开源前检查

- 确认未提交真实车牌、人脸、隐私视频、数据库和本地模型权重。
- 确认第三方模型、数据集、Figma 资源和文档引用允许公开分发。
- 在发布前选择并添加开源许可证，例如 MIT、Apache-2.0 或 GPL 系列。未添加许可证时，外部用户默认没有明确的复制和再分发授权。

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
