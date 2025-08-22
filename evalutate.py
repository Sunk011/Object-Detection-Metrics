import os
from pathlib import Path
import argparse


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
                leftx = float(parts[1])
                topy = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])

                if bb_type == BBType.GroundTruth:
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
    
    print(classes_file)
    print(class_names)

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

        class_name = class_names.get(class_id, "未知类别")
        print(f"类别: {class_id} ({class_name})")
        print(f"  平均精度 (AP): {ap:.4f}")
        print(f"  真值总数: {total_gt}")
        print(f"  真正例 (TP): {total_tp}")
        print(f"  假正例 (FP): {total_fp}")
        print("-" * 60)

    # 计算并打印 mAP
    mAP = sum([class_metrics['AP'] for class_metrics in metrics]) / len(metrics)
    print(f"\nmAP: {mAP:.4f}\n")

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
        help='存放类别名称的文件路径 (格式: <class_id>: <class_name>)。'
    )

    # 2. 解析参数
    args = parser.parse_args()

    # 3. 使用解析出的参数调用评估函数
    evaluate_detections(
        gt_dir=args.gt_dir, 
        pred_dir=args.pred_dir, 
        iou_threshold=args.iou_threshold,
        classes_file=args.classes_file
    )