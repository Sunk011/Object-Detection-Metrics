import os
from pathlib import Path
import argparse
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm
import torch

def convert_yolo_results(yolo_results):
    """
    将 YOLO 模型的推理结果转换为您指定的 7 字段格式。

    Args:
        yolo_results (np.ndarray): YOLO 模型输出的检测结果。
            期望的输入格式是一个 NumPy 数组，其中每一行代表一个检测框，
            格式为 [x1, y1, x2, y2, conf, cls]。
            - x1, y1: 检测框左上顶点的像素坐标
            - x2, y2: 检测框右下顶点的像素坐标
            - conf: 置信度 (浮点数, 范围 0.0 到 1.0)
            - cls: 类别 ID (整数或浮点数)

    Returns:
        np.ndarray: 一个 Nx7 的 Numpy 数组，其格式为：
                    [cls, leftx, topy, width, height, conf_int, reserve]。
    """
    custom_detections = []

    # 如果没有任何检测结果，则返回一个空的数组
    if yolo_results is None or len(yolo_results) == 0:
        return np.array(custom_detections, dtype=np.int32)

    for det in yolo_results:
        # 1. 解析 YOLO 检测结果
        x1, y1, x2, y2, conf, cls = det

        # 2. 计算坐标和尺寸 (leftx, topy, width, height)
        leftx = x1
        topy = y1
        width = x2 - x1
        height = y2 - y1

        # 3. 转换置信度为整数格式
        conf_int = int(conf * 65535.0)

        # 4. 准备其他字段
        class_id = int(cls)
        reserve = 0  # 保留字段，始终为 0

        # 5. 组装成最终的自定义格式
        custom_det = [
            class_id,
            int(round(leftx)),
            int(round(topy)),
            int(round(width)),
            int(round(height)),
            conf_int,
            reserve
        ]
        custom_detections.append(custom_det)

    # 将检测结果列表转换为整数类型的 NumPy 数组
    return np.array(custom_detections, dtype=np.int32)

def run_batch_inference(model_weights, image_dir, output_dir):
    """
    对一个目录中的所有图像运行推理，并以自定义格式保存结果。

    Args:
        model_weights (str): YOLO模型权重文件的路径 (例如 'yolov8n.pt')。
        image_dir (str): 包含输入图像的目录路径。
        output_dir (str): 用于保存推理结果的目录路径。
    """
    # --- 1. 初始化设置 ---
    # 加载 YOLO 模型
    print(f"正在从 {model_weights} 加载模型...")
    model = YOLO(model_weights)
    
    # 根据实际情况判断是否使用 GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    print("模型加载成功。")

    # 确保输出目录存在
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 查找输入目录中所有支持的图像文件
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    image_files = [p for p in Path(image_dir).iterdir() if p.suffix.lower() in image_extensions]

    if not image_files:
        print(f"错误：在目录 '{image_dir}' 中未找到任何图像文件。")
        return

    print(f"共找到 {len(image_files)} 张图像待处理。")

    # --- 2. 循环推理 ---
    for image_file in tqdm(image_files, desc="正在处理图像"):
        try:
            # 执行推理
            results = model(image_file, verbose=True, imgsz=1280) # verbose=False 让日志保持干净

            # results 是一个列表，我们处理第一个元素
            # .boxes.data 包含了 [x1, y1, x2, y2, conf, cls] 格式的张量(Tensor)
            detections_tensor = results[0].boxes.data
            
            # 将张量移动到CPU并转换为NumPy数组
            detections_numpy = detections_tensor.cpu().numpy()

            # 将 NumPy 结果转换为我们的自定义格式
            custom_formatted_dets = convert_yolo_results(detections_numpy)

            # --- 3. 保存结果 ---
            # 定义输出文件的路径，例如 'image1.jpg' -> 'image1.txt'
            output_txt_path = output_path / f"{image_file.stem}.txt"

            # 保存结果。如果图中没有检测到任何物体，这将创建一个空文件。
            np.savetxt(output_txt_path, custom_formatted_dets, fmt='%d', delimiter=' ')

        except Exception as e:
            print(f"\n处理文件 {image_file.name} 时发生错误: {e}")

    print("\n批量推理完成。")
    print(f"所有结果已保存至: {output_dir}")


if __name__ == "__main__":
    # --- 命令行参数解析 ---
    parser = argparse.ArgumentParser(description="YOLO 批量推理脚本")
    parser.add_argument(
        "--model_weights", 
        type=str, 
        required=True, 
        help="YOLO 模型权重文件的路径 (例如: yolov8n.pt)。"
    )
    parser.add_argument(
        "--image_dir", 
        type=str, 
        required=True, 
        help="包含待推理图像的目录路径。"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        required=True, 
        help="用于保存检测结果(.txt文件)的目录路径。"
    )
    
    args = parser.parse_args()

    # 运行主函数
    run_batch_inference(args.model_weights, args.image_dir, args.output_dir)