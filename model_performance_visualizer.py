#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
目标检测模型性能可视化分析工具
主要用于对比不同模型类型和分辨率下的查准率和查全率
"""

import os
import re
import argparse
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
from pathlib import Path
import seaborn as sns
from typing import Dict, List, Tuple, Optional

# 设置字体 - 使用英文，无需中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class ModelPerformanceAnalyzer:
    def __init__(self, input_dir: str, output_dir: str):
        """
        初始化性能分析器
        
        Args:
            input_dir: 包含结果txt文件的目录
            output_dir: 保存可视化图表的目录
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 存储解析后的数据
        self.data = {}
        
    def parse_filename(self, filename: str) -> Optional[Dict[str, str]]:
        """
        解析文件名，提取模型类型、分辨率、IoU阈值等信息
        
        支持的格式：
        - engine_{imgsz}_{iou}_{precision}.txt
        - pt_{imgsz}_{iou}.txt  
        - onnx_{imgsz}_{iou}.txt
        """
        patterns = [
            # Engine模型格式
            r'(engine)_(\d+-\d+)_(iou\d+)_(\w+)\.txt',
            # PT和ONNX模型格式
            r'(pt|onnx)_(\d+-\d+)_(iou\d+)\.txt'
        ]
        
        for pattern in patterns:
            match = re.match(pattern, filename)
            if match:
                groups = match.groups()
                result = {
                    'model_type': groups[0],
                    'resolution': groups[1],
                    'iou': groups[2]
                }
                # Engine模型有精度信息
                if len(groups) > 3:
                    result['precision'] = groups[3]
                
                return result
        
        return None
    
    def parse_result_file(self, filepath: Path) -> Dict:
        """
        解析单个结果文件，提取性能指标
        """
        results = {
            'categories': {},
            'overall': {}
        }
        
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 提取总体性能指标
        overall_pattern = r'总体性能指标:\s*mAP \(平均精度均值\): ([\d.]+)\s*平均查准率: ([\d.]+)\s*平均查全率: ([\d.]+)\s*平均F1分数: ([\d.]+)'
        overall_match = re.search(overall_pattern, content)
        
        if overall_match:
            results['overall'] = {
                'mAP': float(overall_match.group(1)),
                'avg_precision': float(overall_match.group(2)),
                'avg_recall': float(overall_match.group(3)),
                'avg_f1': float(overall_match.group(4))
            }
        
        # 提取各类别详细指标
        category_pattern = r'类别: (\d+) \(([^)]+)\)\s*平均精度 \(AP\): ([\d.]+)\s*查准率 \(Precision\): ([\d.]+)\s*查全率 \(Recall\): ([\d.]+)\s*F1分数: ([\d.]+)'
        category_matches = re.findall(category_pattern, content)
        
        for match in category_matches:
            cat_id, cat_name, ap, precision, recall, f1 = match
            results['categories'][cat_name] = {
                'id': int(cat_id),
                'ap': float(ap),
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1)
            }
        
        return results
    
    def load_data(self):
        """
        加载所有结果文件的数据
        """
        for txt_file in self.input_dir.glob('*.txt'):
            file_info = self.parse_filename(txt_file.name)
            if file_info:
                try:
                    results = self.parse_result_file(txt_file)
                    
                    # 构建唯一键
                    key = f"{file_info['model_type']}_{file_info['resolution']}"
                    if 'precision' in file_info:
                        key += f"_{file_info['precision']}"
                    
                    self.data[key] = {
                        'file_info': file_info,
                        'results': results,
                        'filename': txt_file.name
                    }
                    
                    print(f"Loaded: {txt_file.name}")
                    
                except Exception as e:
                    print(f"Error parsing file {txt_file.name}: {e}")
    
    def get_resolution_pixels(self, resolution: str) -> int:
        """
        将分辨率字符串转换为像素总数，用于排序
        """
        try:
            w, h = map(int, resolution.split('-'))
            return w * h
        except:
            return 0
    
    def create_engine_comparison_chart(self):
        """
        创建Engine模型在不同分辨率下的查准率和查全率对比图
        """
        # 筛选Engine模型数据
        engine_data = {k: v for k, v in self.data.items() 
                      if v['file_info']['model_type'] == 'engine'}
        
        if not engine_data:
            print("No Engine model data found")
            return
        
        # 按分辨率排序
        sorted_data = sorted(engine_data.items(), 
                           key=lambda x: self.get_resolution_pixels(x[1]['file_info']['resolution']))
        
        # 准备数据
        labels = []
        precisions = []
        recalls = []
        colors = plt.cm.Set1(np.linspace(0, 1, len(sorted_data)))
        
        for key, data in sorted_data:
            resolution = data['file_info']['resolution']
            precision_val = data['results']['overall']['avg_precision']
            recall_val = data['results']['overall']['avg_recall']
            
            labels.append(f"{resolution}")
            precisions.append(precision_val)
            recalls.append(recall_val)
        
        # 创建图表
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Engine Model Performance Comparison', fontsize=16, fontweight='bold')
        
        # 1. 查准率柱状图
        bars1 = ax1.bar(labels, precisions, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        ax1.set_title('Average Precision by Resolution', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Precision', fontsize=12)
        ax1.set_xlabel('Resolution', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)
        
        # 添加数值标签
        for bar, val in zip(bars1, precisions):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. 查全率柱状图
        bars2 = ax2.bar(labels, recalls, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        ax2.set_title('Average Recall by Resolution', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Recall', fontsize=12)
        ax2.set_xlabel('Resolution', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)
        
        # 添加数值标签
        for bar, val in zip(bars2, recalls):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 3. 查准率和查全率折线图对比
        x_pos = np.arange(len(labels))
        ax3.plot(x_pos, precisions, 'o-', linewidth=3, markersize=8, 
                label='Precision', color='#2E8B57', markerfacecolor='white', 
                markeredgewidth=2, markeredgecolor='#2E8B57')
        ax3.plot(x_pos, recalls, 's-', linewidth=3, markersize=8, 
                label='Recall', color='#FF6347', markerfacecolor='white',
                markeredgewidth=2, markeredgecolor='#FF6347')
        
        ax3.set_title('Precision vs Recall Trend', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Performance Metrics', fontsize=12)
        ax3.set_xlabel('Resolution', fontsize=12)
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(labels)
        ax3.legend(fontsize=11)
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 1)
        
        # 添加数值标签
        for i, (p, r) in enumerate(zip(precisions, recalls)):
            ax3.text(i, p + 0.02, f'{p:.3f}', ha='center', va='bottom', 
                    color='#2E8B57', fontweight='bold')
            ax3.text(i, r + 0.02, f'{r:.3f}', ha='center', va='bottom', 
                    color='#FF6347', fontweight='bold')
        
        # 4. 散点图：查准率 vs 查全率
        scatter = ax4.scatter(precisions, recalls, c=colors, s=200, alpha=0.8, 
                            edgecolors='black', linewidth=2)
        
        # 添加标签
        for i, (p, r, label) in enumerate(zip(precisions, recalls, labels)):
            ax4.annotate(label, (p, r), xytext=(5, 5), textcoords='offset points',
                        fontsize=10, fontweight='bold')
        
        ax4.set_title('Precision vs Recall Distribution', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Precision', fontsize=12)
        ax4.set_ylabel('Recall', fontsize=12)
        ax4.grid(True, alpha=0.3)
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        
        # 添加对角线参考线
        ax4.plot([0, 1], [0, 1], '--', color='gray', alpha=0.5, linewidth=1)
        
        plt.tight_layout()
        
        # 保存图表
        output_path = self.output_dir / 'engine_models_performance_comparison.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Chart saved to: {output_path}")
        
        plt.show()
    
    def create_all_models_comparison(self):
        """
        创建所有模型类型的对比图表（如果有多种类型的话）
        """
        if not self.data:
            print("没有找到任何数据")
            return
        
        # 按模型类型分组
        model_types = {}
        for key, data in self.data.items():
            model_type = data['file_info']['model_type']
            if model_type not in model_types:
                model_types[model_type] = []
            model_types[model_type].append((key, data))
        
        if len(model_types) == 1:
            print("Only one model type found, creating detailed analysis...")
            self.create_engine_comparison_chart()
            return
        
        # 如果有多种模型类型，创建对比图表
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle('Performance Comparison Across Model Types', fontsize=16, fontweight='bold')
        
        # 准备数据
        all_precisions = []
        all_recalls = []
        all_labels = []
        all_colors = []
        
        color_maps = {'engine': '#FF6B6B', 'pt': '#4ECDC4', 'onnx': '#45B7D1'}
        
        for model_type, models in model_types.items():
            for key, data in models:
                resolution = data['file_info']['resolution']
                precision_val = data['results']['overall']['avg_precision']
                recall_val = data['results']['overall']['avg_recall']
                
                all_precisions.append(precision_val)
                all_recalls.append(recall_val)
                all_labels.append(f"{model_type}_{resolution}")
                all_colors.append(color_maps.get(model_type, '#95A5A6'))
        
        # 查准率对比
        bars1 = ax1.bar(range(len(all_labels)), all_precisions, color=all_colors, 
                       alpha=0.8, edgecolor='black', linewidth=1)
        ax1.set_title('Precision Comparison', fontsize=14)
        ax1.set_ylabel('Precision', fontsize=12)
        ax1.set_xticks(range(len(all_labels)))
        ax1.set_xticklabels(all_labels, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        
        # 查全率对比
        bars2 = ax2.bar(range(len(all_labels)), all_recalls, color=all_colors, 
                       alpha=0.8, edgecolor='black', linewidth=1)
        ax2.set_title('Recall Comparison', fontsize=14)
        ax2.set_ylabel('Recall', fontsize=12)
        ax2.set_xticks(range(len(all_labels)))
        ax2.set_xticklabels(all_labels, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图表
        output_path = self.output_dir / 'all_models_performance_comparison.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Chart saved to: {output_path}")
        
        plt.show()
    
    def print_summary(self):
        """
        打印数据摘要信息
        """
        print("\n" + "="*60)
        print("Data Summary")
        print("="*60)
        
        if not self.data:
            print("No valid data found")
            return
        
        for key, data in self.data.items():
            info = data['file_info']
            results = data['results']['overall']
            
            print(f"\nFile: {data['filename']}")
            print(f"  Model Type: {info['model_type']}")
            print(f"  Resolution: {info['resolution']}")
            print(f"  IoU Threshold: {info['iou']}")
            if 'precision' in info:
                print(f"  Precision Type: {info['precision']}")
            print(f"  Average Precision: {results['avg_precision']:.4f}")
            print(f"  Average Recall: {results['avg_recall']:.4f}")
            print(f"  mAP: {results['mAP']:.4f}")
    
    def run_analysis(self):
        """
        运行完整的分析流程
        """
        print("Loading data...")
        self.load_data()
        
        if not self.data:
            print("No valid data files found")
            return
        
        self.print_summary()
        
        print(f"\nGenerating visualization charts...")
        
        # 检查是否只有Engine模型
        engine_count = sum(1 for data in self.data.values() 
                          if data['file_info']['model_type'] == 'engine')
        
        if engine_count > 0:
            self.create_engine_comparison_chart()
        
        # 如果有多种模型类型，也创建全模型对比图
        model_types = set(data['file_info']['model_type'] for data in self.data.values())
        if len(model_types) > 1:
            self.create_all_models_comparison()
        
        print("\nAnalysis completed!")


def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description='Object Detection Model Performance Visualization Tool')
    parser.add_argument('input_dir', '', help='Directory path containing result txt files')
    parser.add_argument('output_dir', help='Directory path to save visualization charts')
    
    args = parser.parse_args()
    
    # 检查输入目录是否存在
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' does not exist")
        return
    
    # 创建分析器并运行分析
    analyzer = ModelPerformanceAnalyzer(args.input_dir, args.output_dir)
    analyzer.run_analysis()


if __name__ == "__main__":
    main()
