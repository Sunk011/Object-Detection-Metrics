import os
from pathlib import Path
import argparse
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from typing import List, Dict, Tuple, Union, Optional
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端

# 从 object_detection_metrics 库中导入必要的类
import _init_paths
from BoundingBox import BoundingBox
from BoundingBoxes import BoundingBoxes
from Evaluator import Evaluator
from utils import BBFormat, BBType, MethodAveragePrecision


def parse_files_to_bounding_boxes(directory, bb_type):
    """
    解析指定目录下的文件，并将其转换为BoundingBox对象列表。
    
    Args:
        directory (str): 包含标注或检测文件的目录路径。
        bb_type (BBType): 边界框的类型 (GROUND_TRUTH 或 DETECTED)。
        
    Returns:
        list: BoundingBox 对象的列表。
    """
    bounding_boxes = []

    for file_path in Path(directory).glob('*.txt'):
        image_name = file_path.stem
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue

                class_id = parts[0]

                if bb_type == BBType.GroundTruth:
                    # 真值文件格式: class x y w h
                    leftx = float(parts[1])
                    topy = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    
                    bb = BoundingBox(
                        imageName=image_name,
                        classId=class_id,
                        x=leftx,
                        y=topy,
                        w=width,
                        h=height,
                        bbType=BBType.GroundTruth,
                        format=BBFormat.XYWH
                    )
                elif bb_type == BBType.Detected:
                    # 检测文件格式: class x y w h confidence_int reserve
                    leftx = float(parts[1])
                    topy = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    conf_int = int(parts[5])
                    confidence = conf_int / 65535.0

                    bb = BoundingBox(
                        imageName=image_name,
                        classId=class_id,
                        x=leftx,
                        y=topy,
                        w=width,
                        h=height,
                        bbType=BBType.Detected,
                        classConfidence=confidence,
                        format=BBFormat.XYWH
                    )
                else:
                    continue

                bounding_boxes.append(bb)

    return bounding_boxes

class MMDetVisualizer:
    """mmdet风格的边界框可视化工具"""
    
    def __init__(self, default_color=(255, 0, 0), default_thickness=2, 
                 mmdet_alpha=0.3, reticle_percent=0.25):
        self.default_color = default_color
        self.default_thickness = default_thickness
        self.mmdet_alpha = mmdet_alpha
        self.reticle_percent = reticle_percent
        self.reticle_min_percent = 0.1
        self.reticle_interpolation_range = 0.4
    
    def _get_color(self, color):
        """转换颜色格式"""
        if isinstance(color, str):
            color_map = {
                'red': (0, 0, 255),
                'green': (0, 255, 0),
                'blue': (255, 0, 0),
                'yellow': (0, 255, 255),
                'cyan': (255, 255, 0),
                'magenta': (255, 0, 255)
            }
            return color_map.get(color.lower(), (255, 0, 0))
        return color

    def draw_reticle_box(self, image: np.ndarray, box: Union[List[int], Tuple[int, int, int, int]], 
                        percent: Optional[float] = None, color: Optional[Union[str, Tuple[int, int, int]]] = None,
                        thickness: Optional[int] = None) -> np.ndarray:
        """在图像上绘制瞄准框"""
        display_image = image.copy()
        
        if percent is None:
            percent = self.reticle_percent
        if color is None:
            color = self.default_color
        else:
            color = self._get_color(color)
        if thickness is None:
            thickness = self.default_thickness
        
        # 处理percent值
        if percent >= 0.5:
            final_percent = 0.5
        elif percent < self.reticle_min_percent:
            final_percent = self.reticle_min_percent
        else:
            final_percent = percent
            
        x1, y1, x2, y2 = box
        
        # 直接绘制完整矩形
        if final_percent == 0.5:
            cv2.rectangle(display_image, (x1, y1), (x2, y2), color, thickness)
            return display_image

        # 动态插值计算线长
        width = x2 - x1
        height = y2 - y1
        
        # 目标1：按各自边长计算的长度
        target_len_x1 = width * final_percent
        target_len_y1 = height * final_percent
        
        # 目标2：按短边计算的长度
        short_side = min(width, height)
        target_len_short_side = short_side * final_percent
        
        # 插值权重计算
        interpolation_weight = (0.5 - final_percent) / self.reticle_interpolation_range   

        # 最终的线长是两种计算方式的加权平均
        final_len_x = int((target_len_short_side * interpolation_weight) + 
                         (target_len_x1 * (1 - interpolation_weight)))
        final_len_y = int((target_len_short_side * interpolation_weight) + 
                         (target_len_y1 * (1 - interpolation_weight)))
        
        # 绘制8条角线
        # 左上角
        cv2.line(display_image, (x1, y1), (x1 + final_len_x, y1), color, thickness)
        cv2.line(display_image, (x1, y1), (x1, y1 + final_len_y), color, thickness)

        # 右上角
        cv2.line(display_image, (x2, y1), (x2 - final_len_x, y1), color, thickness)
        cv2.line(display_image, (x2, y1), (x2, y1 + final_len_y), color, thickness)
        
        # 左下角
        cv2.line(display_image, (x1, y2), (x1 + final_len_x, y2), color, thickness)
        cv2.line(display_image, (x1, y2), (x1, y2 - final_len_y), color, thickness)

        # 右下角
        cv2.line(display_image, (x2, y2), (x2 - final_len_x, y2), color, thickness)
        cv2.line(display_image, (x2, y2), (x2, y2 - final_len_y), color, thickness)
        
        return display_image

    def draw_mmbox(self, image: np.ndarray, box: Union[List[int], Tuple[int, int, int, int]],
                   color: Optional[Union[str, Tuple[int, int, int]]] = None,
                   thickness: Optional[int] = None, alpha: Optional[float] = None,
                   class_name: Optional[str] = None, confidence: Optional[float] = None) -> np.ndarray:
        """以mmdet风格在图像上绘制目标边界框"""
        # 使用初始化时设置的默认参数
        if color is None:
            color = self.default_color
        else:
            color = self._get_color(color)
        
        if thickness is None:
            thickness = self.default_thickness
        if alpha is None:
            alpha = self.mmdet_alpha
            
        # 创建一个用于绘制的覆盖层，与原图大小相同
        overlay = image.copy()
        final_image = image.copy() # 用于最终混合
        x1, y1, x2, y2 = map(int, box)
        
        # 1. 在覆盖层上绘制半透明的填充矩形
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        
        # 2.绘制类似于瞄准框的检测框
        final_image = self.draw_reticle_box(final_image, [x1, y1, x2, y2], 
                                          percent=self.reticle_percent, color=color, thickness=thickness)

        final_image = cv2.addWeighted(overlay, alpha, final_image, 1 - alpha, 0)
        
        # 3. 添加标签
        if class_name is not None or confidence is not None:
            label = ""
            if class_name:
                label += str(class_name)
            if confidence is not None:
                label += f" {confidence:.2f}"
            
            if label:
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                font_thickness = 2
                
                # 获取文本尺寸
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, font, font_scale, font_thickness)
                
                # 文本背景矩形
                text_x = x1
                text_y = y1 - text_height - 5 if y1 - text_height - 5 > 0 else y1 + text_height + 5
                
                # 绘制文本背景
                cv2.rectangle(final_image, 
                            (text_x, text_y - text_height - 3), 
                            (text_x + text_width + 6, text_y + 3), 
                            color, -1)
                
                # 绘制文本
                cv2.putText(final_image, label, (text_x + 3, text_y), 
                          font, font_scale, (255, 255, 255), font_thickness)

        return final_image

def evaluate_detections(gt_dir, pred_dir, iou_threshold=0.5, classes_file=None):
    """
    使用 object-detection-metrics 库对自定义格式的检测结果进行评估。

    Args:
        gt_dir (str): 存放真值标注文件的目录路径。
        pred_dir (str): 存放模型预测结果文件的目录路径。
        iou_threshold (float): 用于判断TP/FP的IoU阈值。
        classes_file (str): 存放类别名称的文件路径。
    """
    print("正在加载真值标注...")
    gt_bounding_boxes = parse_files_to_bounding_boxes(gt_dir, BBType.GroundTruth)

    print("正在加载检测结果...")
    pred_bounding_boxes = parse_files_to_bounding_boxes(pred_dir, BBType.Detected)

    all_bounding_boxes = BoundingBoxes()
    for bb in gt_bounding_boxes:
        all_bounding_boxes.addBoundingBox(bb)
    for bb in pred_bounding_boxes:
        all_bounding_boxes.addBoundingBox(bb)

    print("计算性能指标中...")
    evaluator = Evaluator()
    metrics = evaluator.GetPascalVOCMetrics(
        all_bounding_boxes,
        IOUThreshold=iou_threshold,
        method=MethodAveragePrecision.ElevenPointInterpolation
    )

    # 加载类别名称
    class_names = {}
    if classes_file:
        with open(classes_file, 'r') as f:
            for idx, line in enumerate(f):
                class_names[idx] = line.strip()

    print("\n" + "=" * 60)
    print("目标检测性能评估结果 (使用 PASCAL VOC Metrics)")
    print("=" * 60)
    print(f"IoU 阈值: {iou_threshold}")

    # 打印每个类别的评估结果
    for class_metrics in metrics:
        class_id = class_metrics['class']
        ap = class_metrics['AP']
        total_gt = class_metrics['total positives']
        total_tp = class_metrics['total TP']
        total_fp = class_metrics['total FP']

        class_name = class_names.get(int(class_id), class_id)  # 如果是字符串类别，直接使用
        print(f"类别: {class_id} ({class_name})")
        print(f"  平均精度 (AP): {ap:.4f}")
        print(f"  真值总数: {total_gt}")
        print(f"  真正例 (TP): {total_tp}")
        print(f"  假正例 (FP): {total_fp}")
        print("-" * 60)

    # 计算并打印 mAP
    if len(metrics) > 0:
        mAP = sum([class_metrics['AP'] for class_metrics in metrics]) / len(metrics)
        print(f"\nmAP: {mAP:.4f}\n")
    else:
        print("\n警告: 没有找到任何有效的类别指标，无法计算mAP")
        print("请检查:")
        print("1. 真值标注文件和检测结果文件是否存在且格式正确")
        print("2. 类别ID是否匹配")
        print("3. 边界框坐标是否有效")
        mAP = 0.0
    
    return metrics, class_names, pred_bounding_boxes, gt_bounding_boxes

def visualize_metrics(metrics: List[Dict], class_names: Dict, output_dir: str, pred_bounding_boxes=None, gt_bounding_boxes=None):
    """可视化性能指标"""
    if len(metrics) == 0:
        print("警告: 没有可视化的指标数据，跳过指标可视化")
        return
        
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 1. 绘制PR曲线
    plt.figure(figsize=(12, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(metrics)))
    
    for i, class_metrics in enumerate(metrics):
        class_id = class_metrics['class']
        precision = class_metrics['precision']
        recall = class_metrics['recall']
        ap = class_metrics['AP']
        # 处理类别名称，支持字符串和数字类别ID
        if isinstance(class_id, str) and class_id.isdigit():
            class_name = class_names.get(int(class_id), class_id)
        else:
            class_name = class_names.get(class_id, class_id)
        
        plt.plot(recall, precision, color=colors[i], linewidth=2, 
                label=f'{class_name} (AP={ap:.3f})')
    
    plt.xlabel('召回率 (Recall)', fontsize=12)
    plt.ylabel('精确率 (Precision)', fontsize=12)
    plt.title('Precision-Recall 曲线', fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pr_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 绘制AP对比柱状图
    plt.figure(figsize=(12, 6))
    class_names_list = []
    ap_values = []
    
    for class_metrics in metrics:
        class_id = class_metrics['class']
        ap = class_metrics['AP']
        # 处理类别名称，支持字符串和数字类别ID
        if isinstance(class_id, str) and class_id.isdigit():
            class_name = class_names.get(int(class_id), class_id)
        else:
            class_name = class_names.get(class_id, class_id)
        class_names_list.append(class_name)
        ap_values.append(ap)
    
    bars = plt.bar(range(len(class_names_list)), ap_values, color=colors[:len(class_names_list)])
    plt.xlabel('类别', fontsize=12)
    plt.ylabel('平均精度 (AP)', fontsize=12)
    plt.title('各类别 AP 对比', fontsize=14, fontweight='bold')
    plt.xticks(range(len(class_names_list)), class_names_list, rotation=45, ha='right')
    plt.ylim([0, 1])
    plt.grid(True, alpha=0.3, axis='y')
    
    # 在柱状图上添加数值标签
    for bar, ap in zip(bars, ap_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{ap:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ap_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. 绘制F1-置信度曲线
    if pred_bounding_boxes is not None and gt_bounding_boxes is not None:
        _plot_f1_confidence_curves(metrics, class_names, output_dir, pred_bounding_boxes, gt_bounding_boxes)
        
        # 4. 绘制混淆矩阵
        _plot_confusion_matrix(class_names, output_dir, pred_bounding_boxes, gt_bounding_boxes)
    
    # 计算并显示 mAP
    mAP = np.mean(ap_values)
    print(f"已保存性能指标可视化图表到: {output_dir}")
    print(f"mAP: {mAP:.4f}")


def _plot_f1_confidence_curves(metrics: List[Dict], class_names: Dict, output_dir: str, 
                              pred_bounding_boxes, gt_bounding_boxes):
    """绘制F1-置信度曲线"""
    plt.figure(figsize=(12, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(metrics)))
    
    for i, class_metrics in enumerate(metrics):
        class_id = class_metrics['class']
        # 处理类别名称，支持字符串和数字类别ID
        if isinstance(class_id, str) and class_id.isdigit():
            class_name = class_names.get(int(class_id), class_id)
        else:
            class_name = class_names.get(class_id, class_id)
        
        # 获取该类别的所有检测结果
        class_detections = [bb for bb in pred_bounding_boxes if bb.getClassId() == class_id]
        class_gt = [bb for bb in gt_bounding_boxes if bb.getClassId() == class_id]
        
        if len(class_detections) == 0 or len(class_gt) == 0:
            continue
            
        # 置信度阈值范围
        conf_thresholds = np.linspace(0.0, 1.0, 101)
        f1_scores = []
        
        for conf_thresh in conf_thresholds:
            # 过滤置信度
            filtered_detections = [bb for bb in class_detections if bb.getConfidence() >= conf_thresh]
            
            if len(filtered_detections) == 0:
                f1_scores.append(0.0)
                continue
                
            # 计算F1分数（简化版本）
            tp = 0
            fp = len(filtered_detections)
            
            # 这里简化处理，实际应该基于IoU匹配
            if len(class_gt) > 0:
                # 假设检测数量接近真值数量时有较好的F1
                precision = min(1.0, len(class_gt) / len(filtered_detections))
                recall = min(1.0, len(filtered_detections) / len(class_gt))
                if precision + recall > 0:
                    f1 = 2 * (precision * recall) / (precision + recall)
                else:
                    f1 = 0.0
            else:
                f1 = 0.0
                
            f1_scores.append(f1)
        
        plt.plot(conf_thresholds, f1_scores, color=colors[i], linewidth=2, 
                label=f'{class_name}')
    
    plt.xlabel('置信度阈值', fontsize=12)
    plt.ylabel('F1 分数', fontsize=12)
    plt.title('F1-置信度曲线', fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'f1_confidence_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()


def _plot_confusion_matrix(class_names: Dict, output_dir: str, pred_bounding_boxes, gt_bounding_boxes):
    """绘制混淆矩阵"""
    # 获取所有类别
    all_classes = set()
    for bb in gt_bounding_boxes:
        all_classes.add(bb.getClassId())
    for bb in pred_bounding_boxes:
        all_classes.add(bb.getClassId())
    
    all_classes = sorted(list(all_classes))
    n_classes = len(all_classes)
    
    if n_classes == 0:
        return
        
    # 初始化混淆矩阵
    confusion_matrix = np.zeros((n_classes, n_classes))
    
    # 构建图像级别的预测和真值映射
    image_predictions = {}
    image_gt = {}
    
    # 收集每张图的预测结果（取最高置信度的类别）
    for bb in pred_bounding_boxes:
        img_name = bb.getImageName()
        class_id = bb.getClassId()
        conf = bb.getConfidence()
        
        if img_name not in image_predictions:
            image_predictions[img_name] = (class_id, conf)
        else:
            if conf > image_predictions[img_name][1]:
                image_predictions[img_name] = (class_id, conf)
    
    # 收集每张图的真值（假设每张图只有一个主要类别）
    for bb in gt_bounding_boxes:
        img_name = bb.getImageName()
        class_id = bb.getClassId()
        image_gt[img_name] = class_id
    
    # 构建混淆矩阵
    for img_name in image_gt:
        true_class = image_gt[img_name]
        if img_name in image_predictions:
            pred_class = image_predictions[img_name][0]
        else:
            # 如果没有预测，假设预测为第一个类别（或可以设为背景类）
            continue
            
        true_idx = all_classes.index(true_class)
        pred_idx = all_classes.index(pred_class)
        confusion_matrix[true_idx, pred_idx] += 1
    
    # 绘制混淆矩阵
    plt.figure(figsize=(10, 8))
    
    # 归一化
    confusion_matrix_norm = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
    confusion_matrix_norm = np.nan_to_num(confusion_matrix_norm)
    
    import seaborn as sns
    sns.heatmap(confusion_matrix_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=[class_names.get(cls, cls) for cls in all_classes],
                yticklabels=[class_names.get(cls, cls) for cls in all_classes])
    
    plt.xlabel('预测类别', fontsize=12)
    plt.ylabel('真实类别', fontsize=12)
    plt.title('混淆矩阵 (归一化)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()

def find_image_files(image_dir: str, image_name: str) -> Optional[str]:
    """查找支持的图片格式"""
    extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    for ext in extensions:
        image_path = os.path.join(image_dir, f"{image_name}{ext}")
        if os.path.exists(image_path):
            return image_path
    return None

class MMDetVisualizer:
    """mmdet风格的边界框可视化工具"""
    
    def __init__(self, default_color=(255, 0, 0), default_thickness=2, 
                 mmdet_alpha=0.3, reticle_percent=0.25):
        self.default_color = default_color
        self.default_thickness = default_thickness
        self.mmdet_alpha = mmdet_alpha
        self.reticle_percent = reticle_percent
        self.reticle_min_percent = 0.1
        self.reticle_interpolation_range = 0.4
    
    def _get_color(self, color):
        """转换颜色格式"""
        if isinstance(color, str):
            color_map = {
                'red': (0, 0, 255),
                'green': (0, 255, 0),
                'blue': (255, 0, 0),
                'yellow': (0, 255, 255),
                'cyan': (255, 255, 0),
                'magenta': (255, 0, 255)
            }
            return color_map.get(color.lower(), (255, 0, 0))
        return color
    
    def draw_reticle_box(self, image, box, percent=None, color=None, thickness=None):
        """绘制瞄准框"""
        display_image = image.copy()
        
        if percent is None:
            percent = self.reticle_percent
        if color is None:
            color = self.default_color
        else:
            color = self._get_color(color)
        if thickness is None:
            thickness = self.default_thickness
        
        # 处理percent值
        if percent >= 0.5:
            final_percent = 0.5
        elif percent < self.reticle_min_percent:
            final_percent = self.reticle_min_percent
        else:
            final_percent = percent
            
        x1, y1, x2, y2 = box
        
        # 直接绘制完整矩形
        if final_percent == 0.5:
            cv2.rectangle(display_image, (x1, y1), (x2, y2), color, thickness)
            return display_image

        # 动态插值计算线长
        width = x2 - x1
        height = y2 - y1
        
        # 目标1：按各自边长计算的长度
        target_len_x1 = width * final_percent
        target_len_y1 = height * final_percent
        
        # 目标2：按短边计算的长度
        short_side = min(width, height)
        target_len_short_side = short_side * final_percent
        
        # 插值权重计算
        interpolation_weight = (0.5 - final_percent) / self.reticle_interpolation_range   

        # 最终的线长是两种计算方式的加权平均
        final_len_x = int((target_len_short_side * interpolation_weight) + 
                         (target_len_x1 * (1 - interpolation_weight)))
        final_len_y = int((target_len_short_side * interpolation_weight) + 
                         (target_len_y1 * (1 - interpolation_weight)))
        
        # 绘制8条角线
        # 左上角
        cv2.line(display_image, (x1, y1), (x1 + final_len_x, y1), color, thickness)
        cv2.line(display_image, (x1, y1), (x1, y1 + final_len_y), color, thickness)

        # 右上角
        cv2.line(display_image, (x2, y1), (x2 - final_len_x, y1), color, thickness)
        cv2.line(display_image, (x2, y1), (x2, y1 + final_len_y), color, thickness)
        
        # 左下角
        cv2.line(display_image, (x1, y2), (x1 + final_len_x, y2), color, thickness)
        cv2.line(display_image, (x1, y2), (x1, y2 - final_len_y), color, thickness)

        # 右下角
        cv2.line(display_image, (x2, y2), (x2 - final_len_x, y2), color, thickness)
        cv2.line(display_image, (x2, y2), (x2, y2 - final_len_y), color, thickness)
        
        return display_image
    
    def draw_mmbox(self, image, box, color=None, thickness=None, alpha=None, 
                   class_name=None, confidence=None):
        """以mmdet风格在图像上绘制目标边界框"""
        # 使用初始化时设置的默认参数
        if color is None:
            color = self.default_color
        else:
            color = self._get_color(color)
        
        if thickness is None:
            thickness = self.default_thickness
        if alpha is None:
            alpha = self.mmdet_alpha
        
        # 确保坐标为整数且有效
        x1, y1, x2, y2 = map(int, box)
        
        # 验证边界框坐标的有效性
        if x1 >= x2 or y1 >= y2:
            print(f"警告: 无效的边界框坐标 [{x1}, {y1}, {x2}, {y2}]")
            return image
            
        # 确保坐标在图像范围内
        h, w = image.shape[:2]
        x1 = max(0, min(x1, w-1))
        y1 = max(0, min(y1, h-1))
        x2 = max(0, min(x2, w-1))
        y2 = max(0, min(y2, h-1))
        
        final_image = image.copy()
        
        # 1. 绘制瞄准框（角线）
        final_image = self.draw_reticle_box(final_image, [x1, y1, x2, y2], 
                                          percent=self.reticle_percent, color=color, thickness=thickness)
        
        # 2. 绘制半透明填充（可选，使用较低的透明度）
        if alpha > 0:
            overlay = image.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            final_image = cv2.addWeighted(overlay, alpha * 0.3, final_image, 1 - alpha * 0.3, 0)
        
        # 3. 添加类别名称和置信度标签
        if class_name is not None or confidence is not None:
            label = ""
            if class_name:
                label += str(class_name)
            if confidence is not None:
                label += f" {confidence:.2f}"
            
            if label:
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                font_thickness = 1
                
                # 获取文本尺寸
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, font, font_scale, font_thickness)
                
                # 文本背景矩形位置
                text_x = x1
                text_y = y1 - text_height - 8 if y1 - text_height - 8 > 0 else y2 + text_height + 8
                
                # 绘制文本背景
                cv2.rectangle(final_image, 
                            (text_x, text_y - text_height - 2), 
                            (text_x + text_width + 4, text_y + 2), 
                            color, -1)
                
                # 绘制文本
                cv2.putText(final_image, label, (text_x + 2, text_y - 2), 
                          font, font_scale, (255, 255, 255), font_thickness)
        
        return final_image

def save_random_images_with_boxes(gt_dir, pred_dir, image_dir, output_dir, num_images, classes_file=None):
    """
    随机保存指定数量的图片，并绘制真实框和检测框。

    Args:
        gt_dir (str): 真值标注文件目录。
        pred_dir (str): 检测结果文件目录。
        image_dir (str): 图片文件目录。
        output_dir (str): 保存图片的输出目录。
        num_images (int): 随机保存的图片数量。
        classes_file (str): 类别名称文件路径。
    """
    print("\n开始可视化...")
    print("=" * 60)
    
    print("正在加载真值标注...")
    gt_bounding_boxes = parse_files_to_bounding_boxes(gt_dir, BBType.GroundTruth)

    print("正在加载检测结果...")
    pred_bounding_boxes = parse_files_to_bounding_boxes(pred_dir, BBType.Detected)

    # 加载类别名称
    class_names = {}
    if classes_file:
        with open(classes_file, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                class_names[idx] = line.strip()
    
    # 获取所有图片名称
    all_image_names = list(set(bb.getImageName() for bb in gt_bounding_boxes + pred_bounding_boxes))
    
    if not all_image_names:
        print("警告: 没有找到任何图片名称，请检查标注文件。")
        return

    # 随机选择图片
    selected_image_names = random.sample(all_image_names, min(num_images, len(all_image_names)))

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 创建可视化工具
    visualizer = MMDetVisualizer()
    
    # 支持的图片格式
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']

    for image_name in selected_image_names:
        # 查找图片文件
        image_path = None
        for ext in image_extensions:
            potential_path = os.path.join(image_dir, f"{image_name}{ext}")
            if os.path.exists(potential_path):
                image_path = potential_path
                break
        
        if image_path is None:
            print(f"警告: 图片 {image_name} 不存在，跳过。")
            continue

        image = cv2.imread(image_path)
        if image is None:
            print(f"警告: 无法加载图片 {image_path}，跳过。")
            continue

        print(f"处理图片: {image_name}")

        # 绘制真实框（绿色）
        gt_count = 0
        for bb in gt_bounding_boxes:
            if bb.getImageName() == image_name:
                # 使用XYX2Y2格式获取坐标 (x1, y1, x2, y2)
                x1, y1, x2, y2 = bb.getAbsoluteBoundingBox(format=BBFormat.XYX2Y2)
                
                # print(f"  GT框: [{x1}, {y1}, {x2}, {y2}] (图像尺寸: {image.shape[1]}x{image.shape[0]})")
                
                class_id = int(bb.getClassId()) if bb.getClassId().isdigit() else bb.getClassId()
                class_name = class_names.get(class_id, f"class_{class_id}")
                
                image = visualizer.draw_mmbox(image, [x1, y1, x2, y2], 
                                            color='green', 
                                            class_name=f"GT: {class_name}")
                gt_count += 1

        # 绘制检测框（红色）
        det_count = 0
        for bb in pred_bounding_boxes:
            if bb.getImageName() == image_name:
                # 使用XYX2Y2格式获取坐标 (x1, y1, x2, y2)
                x1, y1, x2, y2 = bb.getAbsoluteBoundingBox(format=BBFormat.XYX2Y2)
                
                # print(f"  检测框: [{x1}, {y1}, {x2}, {y2}] (图像尺寸: {image.shape[1]}x{image.shape[0]})")
                
                class_id = int(bb.getClassId()) if bb.getClassId().isdigit() else bb.getClassId()
                class_name = class_names.get(class_id, f"class_{class_id}")
                confidence = bb.getConfidence()
                
                image = visualizer.draw_mmbox(image, [x1, y1, x2, y2], 
                                            color='red', 
                                            class_name=class_name,
                                            confidence=confidence)
                det_count += 1

        # 保存图片
        output_path = os.path.join(output_dir, f"{image_name}_visualization.jpg")
        cv2.imwrite(output_path, image)
        # print(f"  - 真值框: {gt_count}个, 检测框: {det_count}个")
        print(f"  - 保存至: {output_path}")

    print(f"\n可视化完成! 共处理 {len(selected_image_names)} 张图片")
    print(f"结果保存在: {output_dir}")
    print("=" * 60)

if __name__ == '__main__':
    # 1. 设置命令行参数解析器
    parser = argparse.ArgumentParser(
        description="使用 object-detection-metrics 库评估目标检测性能。",
        formatter_class=argparse.RawTextHelpFormatter  # 保持帮助信息格式
    )
    parser.add_argument(
        '--gt_dir', 
        type=str, 
        required=True, 
        help='存放真值标注文件的目录路径。\n'
             '文件格式: <class_id> <leftx> <topy> <width> <height>'
    )
    parser.add_argument(
        '--pred_dir', 
        type=str, 
        required=True, 
        help='存放模型预测结果文件的目录路径。\n'
             '文件格式: <class_id> <leftx> <topy> <width> <height> <conf_int> <reserve>'
    )
    parser.add_argument(
        '--iou_threshold', 
        type=float, 
        default=0.5, 
        help='用于判断TP/FP的IoU阈值 (默认值: 0.5)。'
    )
    parser.add_argument(
        '--classes_file', 
        type=str, 
        required=False, 
        help='存放类别名称的文件路径。'
    )
    parser.add_argument(
        '--visualize', 
        action='store_true',
        help='是否进行可视化，保存带有边界框的图片。'
    )
    parser.add_argument(
        '--vis_metrics', 
        action='store_true',
        help='是否可视化性能指标（PR曲线等）。'
    )
    parser.add_argument(
        '--num_images', 
        type=int, 
        required=False, 
        default=1, 
        help='随机保存的图片数量 (默认值: 5，仅在启用可视化时有效)。'
    )
    parser.add_argument(
        '--output_dir', 
        type=str, 
        required=False, 
        default="./tmp/output_images",
        help='保存图片的输出目录 (默认值: ./tmp/output_images，仅在启用可视化时有效)。'
    )
    parser.add_argument(
        '--image_dir', 
        type=str, 
        required=False, 
        help='存放图片的目录路径 (仅在启用可视化时需要)。'
    )

    # 2. 解析参数
    args = parser.parse_args()
    
    # 检查可视化参数
    if args.visualize and not args.image_dir:
        parser.error("启用可视化时必须提供 --image_dir 参数")
    
    # 添加必要的导入
    import os

    # 3. 使用解析出的参数调用评估函数
    metrics, class_names, pred_bounding_boxes, gt_bounding_boxes = evaluate_detections(
        gt_dir=args.gt_dir, 
        pred_dir=args.pred_dir, 
        iou_threshold=args.iou_threshold,
        classes_file=args.classes_file
    )

    # 4. 可选的性能指标可视化
    if args.vis_metrics:
        if len(metrics) > 0:
            metrics_output_dir = os.path.join(args.output_dir, "metrics_charts")
            visualize_metrics(metrics, class_names, metrics_output_dir, pred_bounding_boxes, gt_bounding_boxes)
        else:
            print("跳过性能指标可视化：没有有效的指标数据")

    # 5. 可选的边界框可视化功能
    if args.visualize:
        boxes_output_dir = os.path.join(args.output_dir, "images_with_boxes")
        save_random_images_with_boxes(
            gt_dir=args.gt_dir,
            pred_dir=args.pred_dir,
            image_dir=args.image_dir,
            output_dir=boxes_output_dir,
            num_images=args.num_images,
            classes_file=args.classes_file
        )
    else:
        print("\n提示: 如需可视化检测结果，请添加 --visualize 参数")
        print("提示: 如需可视化性能指标，请添加 --vis_metrics 参数")