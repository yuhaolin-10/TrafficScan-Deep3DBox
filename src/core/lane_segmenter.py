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
        
        # 安全地检查结果，防止数组比较错误
        if len(results) > 0:
            result = results[0]
            if result.masks is not None:
                # 使用 .cpu().numpy() 确保是标准的numpy数组，而不是tensor
                try:
                    masks_xy = result.masks.xy
                    # 使用 .any() 来安全地检查数组
                    if masks_xy is not None and len(masks_xy) > 0:
                        for seg in masks_xy:
                            if len(seg) > 0:
                                poly = np.array(seg, dtype=np.int32)
                                polygons.append(poly)
                                cv2.fillPoly(combined_mask, [poly], 255)
                except AttributeError:
                    # 如果masks.xy不存在，跳过
                    pass
                except ValueError:
                    # 如果存在数组比较错误，安全处理
                    pass
                
        return combined_mask, polygons