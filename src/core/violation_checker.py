import cv2
import numpy as np

class ViolationChecker:
    def __init__(self, threshold=0.3):
        """
        初始化违章判定器。
        
        设计思路:
        我们将判定阈值作为参数传入，而不是写死在代码里。
        这样以后如果需要调整灵敏度（比如从 30% 改成 50%），
        只需要在初始化时修改配置，而不需要改动核心代码。
        
        Args:
            threshold (float): 违章判定的面积占比阈值 (0.0 ~ 1.0)。
                               默认 0.3 表示车身 30% 进入应急车道即算违章。
        """
        self.threshold = threshold

    def check(self, vehicle_footprint, lane_mask):
        """
        执行具体的违章检查逻辑。
        
        Args:
            vehicle_footprint (np.array): 车辆底面接触地面的多边形点集。
                                          格式: [[x1, y1], [x2, y2], ...]
            lane_mask (np.array): 应急车道的二值掩码图。
                                  0 表示背景，255 (或非0) 表示车道区域。
            
        Returns:
            tuple: (is_violating, ratio)
                - is_violating (bool): 是否违章
                - ratio (float): 具体的侵占比例 (0.0 ~ 1.0)，便于后续显示或统计
        """
        # 1. 安全检查: 如果没有车道信息，直接返回正常
        if lane_mask is None:
            return False, 0.0

        # 2. 准备画布: 创建一个和车道图一样大的全黑图片
        # 这一步是为了把车辆的"矢量坐标"转换成"位图区域"，方便后续做像素级运算
        vehicle_mask = np.zeros_like(lane_mask)
        
        # 3. 绘制车辆: 将车辆底面多边形填充为白色 (255)
        # cv2.fillPoly 是 OpenCV 填充多边形的函数
        cv2.fillPoly(vehicle_mask, [vehicle_footprint.astype(np.int32)], 255)
        
        # 4. 计算交集: 找出"既在车道上，又是车辆"的区域
        # bitwise_and: 只有当两个图在同一个位置都是白色(255)时，结果才是白色
        intersection = cv2.bitwise_and(lane_mask, vehicle_mask)
        
        # 5. 统计面积: 数一数白色像素点的个数
        intersection_area = np.count_nonzero(intersection) # 重叠面积
        vehicle_area = np.count_nonzero(vehicle_mask)      # 车辆总底面积
        
        # 6. 防御性编程: 防止车辆太小或绘制失败导致除以零错误
        if vehicle_area == 0:
            return False, 0.0
        
        # 7. 计算比例
        ratio = intersection_area / vehicle_area
        
        # 8. 最终裁决
        is_violating = ratio > self.threshold
        
        return is_violating, ratio
