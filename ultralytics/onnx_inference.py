import numpy as np
import cv2
import onnxruntime as ort
from typing import Tuple, List, Optional
import os


class ONNXDetector:
    """基于ONNX Runtime的目标检测推理器"""
    
    def __init__(self):
        self.session = None
        self.input_name = None
        self.output_names = None
        self.input_shape = None
        self.conf_threshold = 0.25
        self.nms_threshold = 0.45
        self.max_det = 300
        
    def init(self, model_path: str, conf_threshold: float = 0.25, nms_threshold: float = 0.45) -> bool:
        """
        初始化ONNX模型
        
        Args:
            model_path (str): ONNX模型文件路径
            conf_threshold (float): 置信度阈值
            nms_threshold (float): NMS阈值
            
        Returns:
            bool: 初始化成功返回True，失败返回False
        """
        try:
            # 检查模型文件是否存在
            if not os.path.exists(model_path):
                print(f"错误: 模型文件不存在: {model_path}")
                return False
            
            # 设置ONNX Runtime参数
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            
            # 创建推理会话
            self.session = ort.InferenceSession(model_path, providers=providers)
            
            # 获取输入输出信息
            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [output.name for output in self.session.get_outputs()]
            self.input_shape = self.session.get_inputs()[0].shape
            
            # 设置阈值参数
            self.conf_threshold = conf_threshold
            self.nms_threshold = nms_threshold
            
            print(f"ONNX模型加载成功: {model_path}")
            print(f"输入名称: {self.input_name}")
            print(f"输入形状: {self.input_shape}")
            print(f"输出名称: {self.output_names}")
            print(f"置信度阈值: {self.conf_threshold}")
            print(f"NMS阈值: {self.nms_threshold}")
            
            return True
            
        except Exception as e:
            print(f"模型初始化失败: {str(e)}")
            return False
    
    def preprocess(self, img: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """
        图像预处理
        
        Args:
            img (np.ndarray): 输入图像 (H, W, C) BGR格式
            
        Returns:
            Tuple[np.ndarray, float, float]: 预处理后的图像, x缩放比例, y缩放比例
        """
        # 获取原始图像尺寸
        h, w = img.shape[:2]
        
        # 获取模型输入尺寸 (假设为正方形输入，如640x640)
        if len(self.input_shape) == 4:  # (batch, channel, height, width)
            target_h, target_w = self.input_shape[2], self.input_shape[3]
        else:
            target_h, target_w = 640, 640  # 默认尺寸
        
        # 计算缩放比例
        scale_x = target_w / w
        scale_y = target_h / h
        
        # 等比例缩放 (letterbox)
        scale = min(scale_x, scale_y)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # 缩放图像
        resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # 创建目标尺寸的图像并填充
        padded_img = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
        
        # 计算填充位置
        pad_x = (target_w - new_w) // 2
        pad_y = (target_h - new_h) // 2
        
        # 将缩放后的图像放到中心位置
        padded_img[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized_img
        
        # 转换为RGB格式并归一化
        padded_img = cv2.cvtColor(padded_img, cv2.COLOR_BGR2RGB)
        padded_img = padded_img.astype(np.float32) / 255.0
        
        # 转换为CHW格式并增加batch维度
        padded_img = np.transpose(padded_img, (2, 0, 1))  # HWC -> CHW
        padded_img = np.expand_dims(padded_img, axis=0)   # CHW -> BCHW
        
        return padded_img, scale, scale
    
    def postprocess(self, outputs: List[np.ndarray], scale_x: float, scale_y: float, 
                   orig_h: int, orig_w: int) -> np.ndarray:
        """
        后处理：解析模型输出，进行NMS等操作
        
        Args:
            outputs (List[np.ndarray]): 模型输出
            scale_x (float): x方向缩放比例
            scale_y (float): y方向缩放比例
            orig_h (int): 原始图像高度
            orig_w (int): 原始图像宽度
            
        Returns:
            np.ndarray: 检测结果 (N, 7) [cls, leftx, topy, width, height, conf, reserve]
        """
        # 获取模型输出 (假设输出格式为 [batch, num_detections, 85] 或类似)
        predictions = outputs[0]  # 取第一个输出
        
        if len(predictions.shape) == 3:
            predictions = predictions[0]  # 去掉batch维度
        
        # 如果输出格式是 (num_detections, 85)，其中85 = 4(bbox) + 1(conf) + 80(classes)
        # 或者 (num_detections, 6) 格式: [x1, y1, x2, y2, conf, class]
        
        detections = []
        
        # 处理每个检测结果
        for detection in predictions:
            if len(detection) >= 6:  # 至少包含坐标、置信度和类别
                # 格式1: [x1, y1, x2, y2, conf, class] (YOLO格式)
                if len(detection) == 6:
                    x1, y1, x2, y2, conf, cls = detection
                else:
                    # 格式2: [x_center, y_center, w, h, conf, class0, class1, ...] (YOLO v5/v8格式)
                    x_center, y_center, w, h, obj_conf = detection[:5]
                    class_probs = detection[5:]
                    
                    # 找到最大类别概率
                    cls = np.argmax(class_probs)
                    class_conf = class_probs[cls]
                    conf = obj_conf * class_conf
                    
                    # 转换为x1, y1, x2, y2格式
                    x1 = x_center - w / 2
                    y1 = y_center - h / 2
                    x2 = x_center + w / 2
                    y2 = y_center + h / 2
                
                # 置信度过滤
                if conf < self.conf_threshold:
                    continue
                
                # 坐标转换回原始图像尺寸
                x1 = x1 / scale_x
                y1 = y1 / scale_y
                x2 = x2 / scale_x
                y2 = y2 / scale_y
                
                # 确保坐标在图像范围内
                x1 = max(0, min(x1, orig_w - 1))
                y1 = max(0, min(y1, orig_h - 1))
                x2 = max(0, min(x2, orig_w - 1))
                y2 = max(0, min(y2, orig_h - 1))
                
                # 计算宽高
                width = x2 - x1
                height = y2 - y1
                
                # 过滤无效框
                if width <= 0 or height <= 0:
                    continue
                
                # 转换置信度为整数 (0.5 -> 32767)
                conf_int = int(conf * 65535.0)
                
                detections.append([int(cls), x1, y1, width, height, conf_int, 0])
        
        if len(detections) == 0:
            return np.array([]).reshape(0, 7)
        
        detections = np.array(detections)
        
        # 应用NMS (Non-Maximum Suppression)
        if len(detections) > 1:
            detections = self._apply_nms(detections)
        
        return detections
    
    def _apply_nms(self, detections: np.ndarray) -> np.ndarray:
        """
        应用非极大值抑制
        
        Args:
            detections (np.ndarray): 检测结果 (N, 7)
            
        Returns:
            np.ndarray: NMS后的检测结果
        """
        if len(detections) == 0:
            return detections
        
        # 提取坐标和置信度
        x1 = detections[:, 1]
        y1 = detections[:, 2]
        x2 = x1 + detections[:, 3]  # x1 + width
        y2 = y1 + detections[:, 4]  # y1 + height
        scores = detections[:, 5] / 65535.0  # 转换回浮点置信度
        
        # 计算面积
        areas = (x2 - x1) * (y2 - y1)
        
        # 按置信度排序
        order = scores.argsort()[::-1]
        
        keep = []
        while len(order) > 0:
            i = order[0]
            keep.append(i)
            
            if len(order) == 1:
                break
            
            # 计算IoU
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            
            # 保留IoU小于阈值的检测框
            inds = np.where(iou <= self.nms_threshold)[0]
            order = order[inds + 1]
        
        return detections[keep]
    
    def predict(self, img: np.ndarray) -> np.ndarray:
        """
        对输入图像进行目标检测
        
        Args:
            img (np.ndarray): 输入图像 (H, W, C) BGR格式
            
        Returns:
            np.ndarray: 检测结果 (N, 7) [cls, leftx, topy, width, height, conf, reserve]
        """
        if self.session is None:
            print("错误: 模型未初始化，请先调用init()方法")
            return np.array([]).reshape(0, 7)
        
        try:
            # 获取原始图像尺寸
            orig_h, orig_w = img.shape[:2]
            
            # 预处理
            input_data, scale_x, scale_y = self.preprocess(img)
            
            # 推理
            outputs = self.session.run(self.output_names, {self.input_name: input_data})
            
            # 后处理
            detections = self.postprocess(outputs, scale_x, scale_y, orig_h, orig_w)
            
            return detections
            
        except Exception as e:
            print(f"推理过程中发生错误: {str(e)}")
            return np.array([]).reshape(0, 7)


# 全局检测器实例
detector = ONNXDetector()


def init(config: str) -> bool:
    """
    从配置文件中读取模型初始化的配置文件，并完成算法的初始化
    
    Args:
        config (str): 配置文件的路径或ONNX模型路径
        
    Returns:
        bool: 成功返回 True，失败返回 False
    """
    global detector
    
    # 如果config是ONNX文件，直接使用
    if config.endswith('.onnx'):
        model_path = config
    else:
        # 否则假设是配置文件，这里简化处理，实际可以解析配置文件
        model_path = config  # 可以根据需要修改为从配置文件读取模型路径
    
    return detector.init(model_path)


def getPredict(img: np.ndarray) -> np.ndarray:
    """
    输入一张图片，输出检测结果 dets
    
    Args:
        img (np.ndarray): 三维数组（高度、宽度、通道数），数据类型为 uint8，通道顺序为 BGR（OpenCV 格式）
        
    Returns:
        np.ndarray: 形状为 (目标数量, 7) 的数组，每一行为一个目标的信息：
                   [cls, leftx, topy, width, height, conf, reserve]
    
    字段说明：
        cls: 类别 ID（例如0, 1, 2 如果是检测请输出0）
        leftx: 检测/识别框左上顶点 x 坐标（像素）
        topy: 检测/识别框左上顶点 y 坐标（像素）
        width: 检测/识别框宽度（像素）
        height: 检测/识别框高度（像素）
        conf: 置信度，转化为整数（例如 0.5 对应转化为int(0.5*65535.0)=32767）
        reserve: 保留字，始终设为 0
    """
    global detector
    return detector.predict(img)


def test():
    """简单测试：初始化 + 推理"""
    # 1. 假装有一张 480×640 的 BGR 图片
    fake_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # 2. 初始化模型 (需要提供实际的ONNX模型路径)
    model_path = "best.onnx"  # 请替换为实际的模型路径
    
    print(f"尝试初始化模型: {model_path}")
    ok = init(model_path)
    print("init 返回:", ok)
    
    if ok:
        # 3. 推理
        dets = getPredict(fake_img)
        print("检测结果 (shape={}):".format(dets.shape))
        print(dets)
        
        if len(dets) > 0:
            print("\n检测结果详情:")
            for i, det in enumerate(dets):
                cls, leftx, topy, width, height, conf_int, reserve = det
                conf_float = conf_int / 65535.0
                print(f"目标 {i+1}: 类别={int(cls)}, 位置=({leftx:.1f}, {topy:.1f}), "
                      f"尺寸=({width:.1f}x{height:.1f}), 置信度={conf_float:.3f}")
    else:
        print("模型初始化失败，无法进行推理测试")


if __name__ == "__main__":
    test()
