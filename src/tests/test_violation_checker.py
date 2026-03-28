import sys
import os
import numpy as np
import cv2

# 将 src 目录添加到路径中，以便导入 core 模块
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
sys.path.append(src_dir)

from core.violation_checker import ViolationChecker

def run_test():
    print("=== 开始测试 ViolationChecker ===")
    
    # 1. 初始化裁判员 (阈值设为 0.3)
    checker = ViolationChecker(threshold=0.3)
    
    # 2. 创建一个 100x100 的画布作为"车道图"
    # 假设右半边 (x > 50) 是应急车道 (白色 255)
    lane_mask = np.zeros((100, 100), dtype=np.uint8)
    lane_mask[:, 50:] = 255 
    
    print(f"测试环境: 画布大小 100x100, 右半边(x>50)为应急车道")

    # === 测试用例 1: 乖宝宝 (完全在左边，不压线) ===
    # 车辆坐标: x=10~40, y=10~40 (完全在非车道区域)
    vehicle_1 = np.array([[10, 10], [40, 10], [40, 40], [10, 40]])
    is_violating, ratio = checker.check(vehicle_1, lane_mask)
    print(f"\n[测试 1] 正常车辆:")
    print(f"  - 预期: False (0.0%)")
    print(f"  - 实际: {is_violating} ({ratio:.1%})")
    assert is_violating == False
    assert ratio == 0.0

    # === 测试用例 2: 违章大王 (完全在右边，全压线) ===
    # 车辆坐标: x=60~90, y=10~40 (完全在车道区域)
    vehicle_2 = np.array([[60, 10], [90, 10], [90, 40], [60, 40]])
    is_violating, ratio = checker.check(vehicle_2, lane_mask)
    print(f"\n[测试 2] 严重违章车辆:")
    print(f"  - 预期: True (100.0%)")
    print(f"  - 实际: {is_violating} ({ratio:.1%})")
    assert is_violating == True
    assert ratio > 0.99

    # === 测试用例 3: 边缘试探 (正好一半压线) ===
    # 车辆坐标: x=30~70, y=10~40 (跨越 x=50 分界线)
    # 宽度 40，其中 20 在左边，20 在右边 -> 理论比例 50%
    vehicle_3 = np.array([[30, 10], [70, 10], [70, 40], [30, 40]])
    is_violating, ratio = checker.check(vehicle_3, lane_mask)
    print(f"\n[测试 3] 压线车辆 (跨越中线):")
    print(f"  - 预期: True (约 50.0%)")
    print(f"  - 实际: {is_violating} ({ratio:.1%})")
    assert is_violating == True
    assert 0.45 < ratio < 0.55

    print("\n=== 所有测试通过! 裁判员逻辑正常 ===")

if __name__ == "__main__":
    run_test()