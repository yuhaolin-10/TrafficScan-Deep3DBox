import cv2
import numpy as np
from ultralytics import YOLO

class LaneSegmenter:
    def __init__(self, model_path):
        """
        初始化车道分割器。
        Args:
            model_path: 训练好的 YOLO 分割模型路径 (best.pt)
        """
        print(f"正在加载车道分割模型: {model_path}")
        self.model = YOLO(model_path)
        
    def detect(self, image):
        """
        检测图像中的应急车道。
        
        Args:
            image: 输入图像 (numpy array)
            
        Returns:
            mask: 应急车道的二值掩码 (0 或 255)，尺寸与输入图像相同。
            polygons: 车道的多边形轮廓列表 (numpy arrays)。
        """
        results = self.model.predict(
            source=image,
            imgsz=640,
            conf=0.25,
            save=False,
            verbose=False
        )
        
        h, w = image.shape[:2]
        combined_mask = np.zeros((h, w), dtype=np.uint8)
        polygons = []
        
        if results[0].masks is not None:
            # 遍历检测到的掩码
            if results[0].masks.xy is not None:
                for seg in results[0].masks.xy:
                    # 确保分割点存在
                    if len(seg) > 0:
                        # 转换为整数格式以便绘制
                        poly = np.array(seg, dtype=np.int32)
                        polygons.append(poly)
                        
                        # 在掩码上填充多边形 (255表示白色，即车道区域)
                        cv2.fillPoly(combined_mask, [poly], 255)
                
        return combined_mask, polygons