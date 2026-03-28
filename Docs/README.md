# TrafficScan - 车辆违章识别系统

基于 YOLO 的高速公路应急车道占用检测系统。本系统利用 3D 目标检测技术，不仅能识别应急车道占用，还具备扩展检测逆行、压线、超限等多种违章行为的能力。

## 项目结构

- `src/models/`: 存放模型相关代码（车道分割、车辆检测）。
- `src/core/`: 存放核心业务逻辑（几何计算、违章判定）。
- `src/ui/`: 存放 PyQt5 界面代码。
- `DESIGN.md`: 软件设计文档。

## 快速开始

1. 安装依赖: `pip install -r requirements.txt`
2. 运行主程序: `python main.py`
