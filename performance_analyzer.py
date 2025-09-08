#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
性能分析可视化脚本
用于分析和可视化目标检测模型在不同配置下的性能对比
"""

import os
import re
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


class PerformanceAnalyzer:
    """性能分析器"""
    
    def __init__(self, font_path: str = "/home/sk/project/Object-Detection-Metrics/assets/simhei.ttf"):
        """
        初始化性能分析器
        
        Args:
            font_path (str): 中文字体文件路径
        """
        self.font_path = font_path
        self.setup_matplotlib()
        
        # 类别名称映射（英文到中文）
        self.class_names = {
            'pedestrian': 'pedestrian',
            'people': 'people', 
            'bicycle': 'bicycle',
            'car': 'car',
            'van': 'van',
            'truck': 'truck',
            'tricycle': 'tricycle',
            'awning-tricycle': 'awning-tricycle',
            'bus': 'bus',
            'motor': 'motor'
        }
        
        # 数据存储
        self.data = []
        self.category_data = defaultdict(list)
        
    def setup_matplotlib(self):
        """设置matplotlib的样式和字体"""
        plt.style.use('default')
        # 设置字体为系统可用的英文字体
        plt.rcParams['font.family'] = ['DejaVu Sans', 'Liberation Sans', 'sans-serif']
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.unicode_minus'] = False
    
    def parse_filename(self, filename: str) -> Dict[str, Any]:
        """
        解析文件名，提取模型信息
        
        Args:
            filename (str): 文件名
            
        Returns:
            Dict: 解析出的信息
        """
        # 移除文件扩展名
        name = Path(filename).stem
        
        # 正则表达式匹配不同格式
        # 格式1: model_width-height_iou_threshold.txt (pt, onnx)
        # 格式2: model_width-height_iou_threshold_precision.txt (engine)
        
        pattern1 = r'^(\w+)_(\d+)-(\d+)_iou(\d+)$'
        pattern2 = r'^(\w+)_(\d+)-(\d+)_iou(\d+)_(\w+)$'
        
        match2 = re.match(pattern2, name)
        if match2:
            model, width, height, iou, precision = match2.groups()
            return {
                'model': model,
                'width': int(width),
                'height': int(height),
                'iou_threshold': float(f"0.{iou}"),
                'precision': precision,
                'resolution': f"{width}×{height}",
                'pixels': int(width) * int(height)
            }
        
        match1 = re.match(pattern1, name)
        if match1:
            model, width, height, iou = match1.groups()
            return {
                'model': model,
                'width': int(width),
                'height': int(height),
                'iou_threshold': float(f"0.{iou}"),
                'precision': None,
                'resolution': f"{width}×{height}",
                'pixels': int(width) * int(height)
            }
        
        raise ValueError(f"无法解析文件名格式: {filename}")
    
    def parse_txt_file(self, filepath: str) -> Dict[str, Any]:
        """
        解析txt文件内容
        
        Args:
            filepath (str): 文件路径
            
        Returns:
            Dict: 解析出的性能数据
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 解析总体性能指标
        metrics = {}
        
        # 解析mAP
        map_match = re.search(r'mAP \(平均精度均值\):\s*([\d.]+)', content)
        if map_match:
            metrics['mAP'] = float(map_match.group(1))
        
        # 解析平均指标
        avg_precision_match = re.search(r'平均查准率:\s*([\d.]+)', content)
        if avg_precision_match:
            metrics['avg_precision'] = float(avg_precision_match.group(1))
            
        avg_recall_match = re.search(r'平均查全率:\s*([\d.]+)', content)
        if avg_recall_match:
            metrics['avg_recall'] = float(avg_recall_match.group(1))
            
        avg_f1_match = re.search(r'平均F1分数:\s*([\d.]+)', content)
        if avg_f1_match:
            metrics['avg_f1'] = float(avg_f1_match.group(1))
        
        # 解析IoU阈值
        iou_match = re.search(r'IoU 阈值:\s*([\d.]+)', content)
        if iou_match:
            metrics['iou_threshold'] = float(iou_match.group(1))
        
        # 解析各类别性能
        category_metrics = {}
        
        # 匹配类别信息的正则表达式
        category_pattern = r'类别:\s*(\d+)\s*\((\w+[\w-]*)\)\s*\n\s*平均精度\s*\(AP\):\s*([\d.]+)\s*\n\s*查准率\s*\(Precision\):\s*([\d.]+)\s*\n\s*查全率\s*\(Recall\):\s*([\d.]+)\s*\n\s*F1分数:\s*([\d.]+)'
        
        for match in re.finditer(category_pattern, content):
            class_id, class_name, ap, precision, recall, f1 = match.groups()
            category_metrics[class_name] = {
                'class_id': int(class_id),
                'AP': float(ap),
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1)
            }
        
        return {
            'overall_metrics': metrics,
            'category_metrics': category_metrics
        }
    
    def load_data(self, input_dir: str):
        """
        加载所有txt文件的数据
        
        Args:
            input_dir (str): 输入目录路径
        """
        input_path = Path(input_dir)
        txt_files = list(input_path.glob('*.txt'))
        
        if not txt_files:
            raise ValueError(f"在目录 {input_dir} 中未找到txt文件")
        
        print(f"找到 {len(txt_files)} 个txt文件")
        
        for txt_file in txt_files:
            try:
                # 解析文件名
                file_info = self.parse_filename(txt_file.name)
                
                # 解析文件内容
                performance_data = self.parse_txt_file(str(txt_file))
                
                # 合并数据
                combined_data = {**file_info, **performance_data['overall_metrics']}
                combined_data['filename'] = txt_file.name
                
                self.data.append(combined_data)
                
                # 存储类别数据
                for class_name, class_metrics in performance_data['category_metrics'].items():
                    category_record = {**file_info, **class_metrics}
                    category_record['class_name'] = class_name
                    self.category_data[class_name].append(category_record)
                
                print(f"成功解析: {txt_file.name}")
                
            except Exception as e:
                print(f"解析文件 {txt_file.name} 失败: {e}")
        
        if not self.data:
            raise ValueError("没有成功解析任何文件")
        
        print(f"总共成功加载 {len(self.data)} 个配置的数据")
    
    def create_professional_comparison(self, output_dir: str):
        """创建专业风格的性能对比图表"""
        df = pd.DataFrame(self.data)
        
        # 创建专业风格的多子图布局
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        
        # 颜色方案
        colors = {
            'onnx': '#5B9BD5',  # 蓝色
            'pt': '#A5677E',    # 紫红色 
            'engine': '#70AD47' # 绿色
        }
        
        # 1. mAP Comparison (左上)
        ax1 = fig.add_subplot(gs[0, 0])
        models = df['model'].unique()
        model_map = df.groupby('model')['mAP'].mean()
        
        bars1 = ax1.bar(range(len(models)), [model_map[m] for m in models], 
                       color=[colors.get(m, '#cccccc') for m in models],
                       alpha=0.8, edgecolor='white', linewidth=1)
        
        # 添加数值标签
        for i, model in enumerate(models):
            value = model_map[model]
            ax1.text(i, value + 0.01, f'{value:.3f}', ha='center', va='bottom', 
                    fontweight='bold', fontsize=11)
        
        ax1.set_title('Model Performance Comparison\nmAP Values', fontsize=14, fontweight='bold', pad=20)
        ax1.set_xlabel('Model Type', fontsize=12)
        ax1.set_ylabel('mAP', fontsize=12)
        ax1.set_xticks(range(len(models)))
        ax1.set_xticklabels([m.upper() for m in models])
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.set_ylim(0, model_map.max() * 1.15)
        
        # 2. Resolution vs Performance (右上)
        ax2 = fig.add_subplot(gs[0, 1])
        
        for model in models:
            model_data = df[df['model'] == model]
            ax2.scatter(model_data['pixels']/1e6, model_data['mAP'], 
                       color=colors.get(model, '#cccccc'), alpha=0.7, s=120, 
                       label=model.upper(), edgecolors='white', linewidth=1)
            
            # 添加分辨率标签
            for _, row in model_data.iterrows():
                ax2.annotate(row['resolution'], 
                           (row['pixels']/1e6, row['mAP']),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=9, alpha=0.8)
        
        ax2.set_xlabel('Resolution (Megapixels)', fontsize=12)
        ax2.set_ylabel('mAP', fontsize=12)
        ax2.set_title('Resolution Impact on Performance', fontsize=14, fontweight='bold', pad=20)
        ax2.legend(framealpha=0.9)
        ax2.grid(True, alpha=0.3)
        
        # 3. Multi-metric Comparison (右下)
        ax3 = fig.add_subplot(gs[0, 2])
        
        metrics = ['mAP', 'avg_precision', 'avg_recall', 'avg_f1']
        metric_labels = ['mAP', 'Precision', 'Recall', 'F1-Score']
        
        x = np.arange(len(metric_labels))
        width = 0.35
        
        for i, model in enumerate(models):
            model_data = df[df['model'] == model]
            values = [model_data[metric].mean() for metric in metrics]
            
            offset = (i - len(models)/2 + 0.5) * width
            bars = ax3.bar(x + offset, values, width, 
                          label=model.upper(), 
                          color=colors.get(model, '#cccccc'),
                          alpha=0.8, edgecolor='white', linewidth=1)
            
            # 添加数值标签
            for j, v in enumerate(values):
                ax3.text(x[j] + offset, v + 0.01, f'{v:.3f}', 
                        ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax3.set_xlabel('Metrics', fontsize=12)
        ax3.set_ylabel('Score', fontsize=12)
        ax3.set_title('Comprehensive Performance Metrics', fontsize=14, fontweight='bold', pad=20)
        ax3.set_xticks(x)
        ax3.set_xticklabels(metric_labels)
        ax3.legend(framealpha=0.9)
        ax3.grid(True, alpha=0.3, axis='y')
        ax3.set_ylim(0, 1)
        
        # 4. Processing Efficiency (左下，跨两列)
        ax4 = fig.add_subplot(gs[1, :2])
        
        # 创建效率对比（假设基于像素数计算）
        efficiency_data = []
        for _, row in df.iterrows():
            efficiency = row['mAP'] / (row['pixels'] / 1e6)  # mAP per megapixel
            efficiency_data.append({
                'model': row['model'],
                'resolution': row['resolution'],
                'efficiency': efficiency,
                'mAP': row['mAP'],
                'pixels': row['pixels']
            })
        
        eff_df = pd.DataFrame(efficiency_data)
        
        for model in models:
            model_eff = eff_df[eff_df['model'] == model]
            ax4.scatter(model_eff['pixels']/1e6, model_eff['efficiency'], 
                       color=colors.get(model, '#cccccc'), alpha=0.7, s=120,
                       label=f'{model.upper()} Efficiency', 
                       edgecolors='white', linewidth=1)
            
            # 添加趋势线
            if len(model_eff) > 1:
                z = np.polyfit(model_eff['pixels']/1e6, model_eff['efficiency'], 1)
                p = np.poly1d(z)
                x_trend = np.linspace(model_eff['pixels'].min()/1e6, model_eff['pixels'].max()/1e6, 100)
                ax4.plot(x_trend, p(x_trend), '--', color=colors.get(model, '#cccccc'), alpha=0.5)
        
        ax4.set_xlabel('Resolution (Megapixels)', fontsize=12)
        ax4.set_ylabel('Efficiency (mAP/MP)', fontsize=12)
        ax4.set_title('Processing Efficiency Analysis', fontsize=14, fontweight='bold', pad=20)
        ax4.legend(framealpha=0.9)
        ax4.grid(True, alpha=0.3)
        
        # 5. Performance Distribution (右下)
        ax5 = fig.add_subplot(gs[1, 2])
        
        # 创建箱线图显示性能分布
        model_mAPs = [df[df['model'] == model]['mAP'].values for model in models]
        
        bp = ax5.boxplot(model_mAPs, labels=[m.upper() for m in models],
                        patch_artist=True, notch=True)
        
        # 设置箱线图颜色
        for patch, model in zip(bp['boxes'], models):
            patch.set_facecolor(colors.get(model, '#cccccc'))
            patch.set_alpha(0.7)
        
        ax5.set_ylabel('mAP Distribution', fontsize=12)
        ax5.set_xlabel('Model Type', fontsize=12)
        ax5.set_title('Performance Distribution', fontsize=14, fontweight='bold', pad=20)
        ax5.grid(True, alpha=0.3, axis='y')
        
        # 添加总标题
        fig.suptitle('Object Detection Model Performance Analysis', 
                    fontsize=18, fontweight='bold', y=0.95)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'professional_performance_analysis.png'), 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print("Generated: professional_performance_analysis.png")
    
    def create_speedup_comparison(self, output_dir: str):
        """创建类似PopSift风格的性能对比图"""
        df = pd.DataFrame(self.data)
        
        # 创建6个子图的布局 (2x3)
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Model Performance Comparison Analysis', fontsize=16, fontweight='bold', y=0.95)
        
        # 颜色方案：蓝色用于基准，紫红色用于对比
        color_base = '#5B9BD5'    # 蓝色
        color_compare = '#A5677E'  # 紫红色
        
        # 假设PT是基准，ONNX是对比对象
        models = df['model'].unique()
        if 'pt' in models and 'onnx' in models:
            base_model = 'pt'
            compare_model = 'onnx'
        else:
            base_model = models[0]
            compare_model = models[1] if len(models) > 1 else models[0]
        
        base_data = df[df['model'] == base_model]
        compare_data = df[df['model'] == compare_model]
        
        # 计算平均性能
        base_mAP = base_data['mAP'].mean()
        compare_mAP = compare_data['mAP'].mean()
        base_precision = base_data['avg_precision'].mean()
        compare_precision = compare_data['avg_precision'].mean()
        base_recall = base_data['avg_recall'].mean()
        compare_recall = compare_data['avg_recall'].mean()
        
        # 1. mAP Performance (左上)
        ax1 = axes[0, 0]
        values1 = [base_mAP, compare_mAP]
        bars1 = ax1.bar([base_model.upper(), compare_model.upper()], values1,
                       color=[color_base, color_compare], alpha=0.8,
                       edgecolor='white', linewidth=1)
        
        # 添加性能提升标签
        if compare_mAP > base_mAP:
            speedup = compare_mAP / base_mAP
            ax1.text(0.5, max(values1) * 0.8, f'Improvement: {speedup:.2f}x', 
                    ha='center', va='center', fontweight='bold', fontsize=11,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7))
        
        for i, v in enumerate(values1):
            ax1.text(i, v + max(values1) * 0.02, f'{v:.3f}', 
                    ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        ax1.set_title('Average mAP Performance', fontsize=12, fontweight='bold')
        ax1.set_ylabel('mAP Score', fontsize=10)
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.set_ylim(0, max(values1) * 1.2)
        
        # 2. Precision Comparison (中上)
        ax2 = axes[0, 1]
        values2 = [base_precision, compare_precision]
        bars2 = ax2.bar([base_model.upper(), compare_model.upper()], values2,
                       color=[color_base, color_compare], alpha=0.8,
                       edgecolor='white', linewidth=1)
        
        if compare_precision > base_precision:
            speedup = compare_precision / base_precision
            ax2.text(0.5, max(values2) * 0.8, f'Improvement: {speedup:.2f}x', 
                    ha='center', va='center', fontweight='bold', fontsize=11,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7))
        
        for i, v in enumerate(values2):
            ax2.text(i, v + max(values2) * 0.02, f'{v:.3f}', 
                    ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        ax2.set_title('Average Precision', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Precision Score', fontsize=10)
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.set_ylim(0, max(values2) * 1.2)
        
        # 3. Number of Configurations (右上)
        ax3 = axes[0, 2]
        config_counts = [len(base_data), len(compare_data)]
        bars3 = ax3.bar([base_model.upper(), compare_model.upper()], config_counts,
                       color=[color_base, color_compare], alpha=0.8,
                       edgecolor='white', linewidth=1)
        
        for i, v in enumerate(config_counts):
            ax3.text(i, v + max(config_counts) * 0.02, f'{v}', 
                    ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        ax3.set_title('Number of Test Configurations', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Count', fontsize=10)
        ax3.grid(True, alpha=0.3, axis='y')
        ax3.set_ylim(0, max(config_counts) * 1.2)
        
        # 4. Performance Efficiency (左下)
        ax4 = axes[1, 0]
        # 计算效率：mAP per megapixel
        base_efficiency = [row['mAP'] / (row['pixels'] / 1e6) for _, row in base_data.iterrows()]
        compare_efficiency = [row['mAP'] / (row['pixels'] / 1e6) for _, row in compare_data.iterrows()]
        
        avg_base_eff = np.mean(base_efficiency)
        avg_compare_eff = np.mean(compare_efficiency)
        
        values4 = [avg_base_eff, avg_compare_eff]
        bars4 = ax4.bar([base_model.upper(), compare_model.upper()], values4,
                       color=[color_base, color_compare], alpha=0.8,
                       edgecolor='white', linewidth=1)
        
        if avg_compare_eff > avg_base_eff:
            speedup = avg_compare_eff / avg_base_eff
            ax4.text(0.5, max(values4) * 0.8, f'Efficiency: {speedup:.2f}x', 
                    ha='center', va='center', fontweight='bold', fontsize=11,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7))
        
        for i, v in enumerate(values4):
            ax4.text(i, v + max(values4) * 0.02, f'{v:.2f}', 
                    ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        ax4.set_title('Processing Efficiency', fontsize=12, fontweight='bold')
        ax4.set_ylabel('mAP/Megapixel', fontsize=10)
        ax4.grid(True, alpha=0.3, axis='y')
        ax4.set_ylim(0, max(values4) * 1.2)
        
        # 5. Recall Distribution (中下)
        ax5 = axes[1, 1]
        
        # 使用箱线图展示召回率分布
        recall_data = [base_data['avg_recall'].values, compare_data['avg_recall'].values]
        bp = ax5.boxplot(recall_data, labels=[base_model.upper(), compare_model.upper()],
                        patch_artist=True, notch=True)
        
        bp['boxes'][0].set_facecolor(color_base)
        bp['boxes'][1].set_facecolor(color_compare)
        bp['boxes'][0].set_alpha(0.8)
        bp['boxes'][1].set_alpha(0.8)
        
        ax5.set_title('Recall Distribution', fontsize=12, fontweight='bold')
        ax5.set_ylabel('Recall Score', fontsize=10)
        ax5.grid(True, alpha=0.3, axis='y')
        
        # 6. Performance vs Resolution (右下)
        ax6 = axes[1, 2]
        
        # 绘制性能随分辨率变化的趋势
        for model, color in zip([base_model, compare_model], [color_base, color_compare]):
            model_data = df[df['model'] == model]
            sorted_data = model_data.sort_values('pixels')
            
            ax6.plot(sorted_data['pixels']/1e6, sorted_data['mAP'], 
                    marker='o', linewidth=2, markersize=6, color=color, 
                    label=f'{model.upper()}', alpha=0.8)
            
            # 添加数据点标签
            for _, row in sorted_data.iterrows():
                ax6.annotate(f"{row['resolution']}", 
                           (row['pixels']/1e6, row['mAP']),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8, alpha=0.7)
        
        # 添加FPS参考线
        ax6.axhline(y=0.3, color='red', linestyle='--', alpha=0.5, label='30 mAP Target')
        ax6.axhline(y=0.6, color='red', linestyle=':', alpha=0.5, label='60 mAP Target')
        
        ax6.set_xlabel('Resolution (Megapixels)', fontsize=10)
        ax6.set_ylabel('mAP Score', fontsize=10)
        ax6.set_title('Performance vs Resolution', fontsize=12, fontweight='bold')
        ax6.legend(fontsize=9)
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'speedup_style_comparison.png'), 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print("Generated: speedup_style_comparison.png")
    
    def create_resolution_analysis(self, output_dir: str):
        """创建分辨率影响分析图"""
        df = pd.DataFrame(self.data)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # 左图：分辨率趋势分析
        resolutions = sorted(df['resolution'].unique(), 
                           key=lambda x: int(x.split('×')[0]) * int(x.split('×')[1]))
        
        model_colors = {'onnx': '#5B9BD5', 'pt': '#A5677E', 'engine': '#70AD47'}
        
        for model in df['model'].unique():
            model_data = df[df['model'] == model]
            res_mAP = []
            res_labels = []
            
            for res in resolutions:
                res_data = model_data[model_data['resolution'] == res]
                if not res_data.empty:
                    res_mAP.append(res_data['mAP'].mean())
                    res_labels.append(res)
            
            if res_mAP:
                ax1.plot(range(len(res_labels)), res_mAP, 
                        marker='o', linewidth=2, markersize=8,
                        color=model_colors.get(model, '#cccccc'),
                        label=model.upper())
        
        ax1.set_xlabel('Resolution', fontsize=12)
        ax1.set_ylabel('mAP', fontsize=12)
        ax1.set_title('mAP Performance Across Different Resolutions', fontsize=14, fontweight='bold')
        ax1.set_xticks(range(len(resolutions)))
        ax1.set_xticklabels(resolutions, rotation=45)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 右图：宽高比影响分析
        df['aspect_ratio'] = df['width'] / df['height']
        
        for model in df['model'].unique():
            model_data = df[df['model'] == model]
            ax2.scatter(model_data['aspect_ratio'], model_data['mAP'],
                       color=model_colors.get(model, '#cccccc'),
                       alpha=0.7, s=100, label=model.upper())
            
            # 添加分辨率标签
            for _, row in model_data.iterrows():
                ax2.annotate(row['resolution'], 
                           (row['aspect_ratio'], row['mAP']),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8, alpha=0.8)
        
        ax2.set_xlabel('Aspect Ratio (width/height)', fontsize=12)
        ax2.set_ylabel('mAP', fontsize=12)
        ax2.set_title('Impact of Aspect Ratio on mAP Performance', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'resolution_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print("Generated: resolution_analysis.png")
    
    def create_category_heatmap(self, output_dir: str):
        """创建类别性能热力图"""
        # 准备数据
        categories = list(self.category_data.keys())
        configs = []
        ap_matrix = []
        
        # 获取所有配置
        for record in self.data:
            config_name = f"{record['model']}\n{record['resolution']}"
            if record['precision']:
                config_name += f"\n{record['precision']}"
            configs.append(config_name)
        
        # 构建AP矩阵
        for category in categories:
            category_aps = []
            for record in self.data:
                # 找到对应的类别数据
                matching_cat_data = None
                for cat_record in self.category_data[category]:
                    if (cat_record['model'] == record['model'] and 
                        cat_record['width'] == record['width'] and 
                        cat_record['height'] == record['height']):
                        matching_cat_data = cat_record
                        break
                
                if matching_cat_data:
                    category_aps.append(matching_cat_data['AP'])
                else:
                    category_aps.append(0)  # 如果没有数据，设为0
            
            ap_matrix.append(category_aps)
        
        # 创建热力图
        fig, ax = plt.subplots(figsize=(14, 10))
        
        im = ax.imshow(ap_matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
        
        # 设置坐标轴
        ax.set_xticks(range(len(configs)))
        ax.set_xticklabels(configs, rotation=45, ha='right')
        ax.set_yticks(range(len(categories)))
        ax.set_yticklabels(categories)
        
        # 添加数值标签
        for i in range(len(categories)):
            for j in range(len(configs)):
                text = ax.text(j, i, f'{ap_matrix[i][j]:.3f}',
                              ha="center", va="center", color="black", fontsize=8)
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('AP值', rotation=270, labelpad=15)
        
        ax.set_title('Category Performance Heatmap Across Different Configurations', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'category_heatmap.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print("Generated: category_heatmap.png")
    
    def create_radar_chart_by_model_type(self, output_dir: str):
        """根据模型类型分别创建雷达图"""
        df = pd.DataFrame(self.data)
        
        # 获取所有模型类型
        model_types = sorted(df['model'].unique())
        
        # 准备雷达图数据
        metrics = ['mAP', 'avg_precision', 'avg_recall', 'avg_f1']
        metric_labels = ['mAP', 'Precision', 'Recall', 'F1-Score']
        
        # 角度设置
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # 闭合雷达图
        
        for model_type in model_types:
            # 筛选当前模型类型的数据
            model_data = df[df['model'] == model_type]
            
            if model_data.empty:
                continue
                
            fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(projection='polar'))
            
            # 为当前模型类型的每个配置生成不同颜色
            import matplotlib.cm as cm
            colors = cm.tab10(np.linspace(0, 1, len(model_data)))
            
            # 为每个配置绘制一条线
            for idx, (_, row) in enumerate(model_data.iterrows()):
                # 获取当前配置的值
                values = []
                for metric in metrics:
                    values.append(row[metric])
                
                values += values[:1]  # 闭合雷达图
                
                # 创建标签（分辨率_IoU）
                label = f"{row['resolution']}_IoU{row['iou_threshold']:.1f}"
                
                # 绘制雷达图
                ax.plot(angles, values, 'o-', linewidth=2.5, 
                       color=colors[idx], label=label, markersize=8, alpha=0.8)
                ax.fill(angles, values, alpha=0.15, color=colors[idx])
            
            # 设置标签
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(metric_labels, fontsize=14, fontweight='bold')
            ax.set_ylim(0, 1)
            
            # 添加网格线标签
            ax.set_yticks([0.2, 0.4, 0.6, 0.8])
            ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8'], fontsize=12)
            ax.grid(True, alpha=0.3)
            
            # 调整图例位置
            plt.legend(loc='center left', bbox_to_anchor=(1.2, 0.5), 
                      ncol=1, fontsize=11, framealpha=0.9, 
                      title=f'{model_type.upper()} Model Configurations',
                      title_fontsize=12)
            
            plt.title(f'{model_type.upper()} Model Performance Radar Chart', 
                     size=18, fontweight='bold', pad=30)
            
            plt.tight_layout()
            
            # 保存文件
            filename = f'radar_chart_{model_type}.png'
            plt.savefig(os.path.join(output_dir, filename), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Generated: {filename}")

    def create_radar_chart(self, output_dir: str, individual_mode: bool = True):
        """创建雷达图对比
        
        Args:
            output_dir: 输出目录
            individual_mode: True=每个txt文件一个实例, False=按模型类型分组
        """
        df = pd.DataFrame(self.data)
        
        # 准备雷达图数据
        metrics = ['mAP', 'avg_precision', 'avg_recall', 'avg_f1']
        metric_labels = ['mAP', 'Precision', 'Recall', 'F1-Score']
        
        fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(projection='polar'))
        
        # 角度设置
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # 闭合雷达图
        
        if individual_mode:
            # 模式1：每个txt文件（配置）一个实例
            import matplotlib.cm as cm
            colors = cm.tab20(np.linspace(0, 1, len(df)))
            
            # 为每个txt文件（配置）绘制一条线
            for idx, (_, row) in enumerate(df.iterrows()):
                # 获取当前配置的值
                values = []
                for metric in metrics:
                    values.append(row[metric])
                
                values += values[:1]  # 闭合雷达图
                
                # 创建标签（模型类型_分辨率_IoU）
                label = f"{row['model'].upper()}_{row['resolution']}_IoU{row['iou_threshold']:.1f}"
                
                # 绘制雷达图
                ax.plot(angles, values, 'o-', linewidth=2, 
                       color=colors[idx], label=label, markersize=6)
                ax.fill(angles, values, alpha=0.1, color=colors[idx])
            
            title = 'Individual Configuration Performance Radar Chart'
            legend_cols = 1
            
        else:
            # 模式2：按模型类型分组（原来的模式）
            model_colors = {'onnx': '#5B9BD5', 'pt': '#A5677E', 'engine': '#70AD47'}
            
            for model in df['model'].unique():
                model_data = df[df['model'] == model]
                
                # 计算平均值
                values = []
                for metric in metrics:
                    values.append(model_data[metric].mean())
                
                values += values[:1]  # 闭合雷达图
                
                # 绘制雷达图
                ax.plot(angles, values, 'o-', linewidth=3, 
                       color=model_colors.get(model, '#cccccc'),
                       label=model.upper(), markersize=8)
                ax.fill(angles, values, alpha=0.25, 
                       color=model_colors.get(model, '#cccccc'))
            
            title = 'Model Type Performance Comparison Radar Chart'
            legend_cols = 1
        
        # 设置标签
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_labels, fontsize=12)
        ax.set_ylim(0, 1)
        
        # 添加网格线标签
        ax.set_yticks([0.2, 0.4, 0.6, 0.8])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8'], fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # 调整图例位置
        plt.legend(loc='center left', bbox_to_anchor=(1.2, 0.5), 
                  ncol=legend_cols, fontsize=9, framealpha=0.9)
        plt.title(title, size=16, fontweight='bold', pad=30)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'radar_chart.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print("Generated: radar_chart.png")
    
    def create_performance_matrix(self, output_dir: str):
        """创建性能矩阵图"""
        df = pd.DataFrame(self.data)
        
        # 创建模型×分辨率的性能矩阵
        models = sorted(df['model'].unique())
        resolutions = sorted(df['resolution'].unique(), 
                           key=lambda x: int(x.split('×')[0]) * int(x.split('×')[1]))
        
        # 创建mAP矩阵
        mAP_matrix = np.zeros((len(models), len(resolutions)))
        
        for i, model in enumerate(models):
            for j, resolution in enumerate(resolutions):
                subset = df[(df['model'] == model) & (df['resolution'] == resolution)]
                if not subset.empty:
                    mAP_matrix[i, j] = subset['mAP'].mean()
                else:
                    mAP_matrix[i, j] = np.nan
        
        # 创建热力图
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # 使用mask处理nan值
        mask = np.isnan(mAP_matrix)
        
        im = sns.heatmap(mAP_matrix, 
                        xticklabels=resolutions,
                        yticklabels=[m.upper() for m in models],
                        annot=True, fmt='.3f', 
                        cmap='RdYlBu_r',
                        mask=mask,
                        cbar_kws={'label': 'mAP值'},
                        ax=ax)
        
        ax.set_title('Model Type × Resolution Performance Matrix (mAP)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Resolution', fontsize=12)
        ax.set_ylabel('Model Type', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'performance_matrix.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print("Generated: performance_matrix.png")
    
    def generate_summary_report(self, output_dir: str):
        """生成总结报告"""
        df = pd.DataFrame(self.data)
        
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("性能分析总结报告")
        report_lines.append("=" * 60)
        report_lines.append("")
        
        # 基本统计
        report_lines.append("1. 数据概览")
        report_lines.append(f"   - 总配置数: {len(df)}")
        report_lines.append(f"   - 模型类型: {', '.join(df['model'].unique())}")
        report_lines.append(f"   - 分辨率范围: {len(df['resolution'].unique())} 种")
        report_lines.append(f"   - IoU阈值范围: {df['iou_threshold'].min():.1f} - {df['iou_threshold'].max():.1f}")
        report_lines.append("")
        
        # 最佳性能
        best_mAP = df.loc[df['mAP'].idxmax()]
        report_lines.append("2. 最佳性能配置")
        report_lines.append(f"   - 模型: {best_mAP['model'].upper()}")
        report_lines.append(f"   - 分辨率: {best_mAP['resolution']}")
        report_lines.append(f"   - mAP: {best_mAP['mAP']:.4f}")
        report_lines.append(f"   - 平均查准率: {best_mAP['avg_precision']:.4f}")
        report_lines.append(f"   - 平均查全率: {best_mAP['avg_recall']:.4f}")
        report_lines.append("")
        
        # 模型对比
        report_lines.append("3. 模型性能对比 (平均值)")
        for model in df['model'].unique():
            model_data = df[df['model'] == model]
            report_lines.append(f"   {model.upper()}:")
            report_lines.append(f"     - 平均mAP: {model_data['mAP'].mean():.4f}")
            report_lines.append(f"     - 平均查准率: {model_data['avg_precision'].mean():.4f}")
            report_lines.append(f"     - 平均查全率: {model_data['avg_recall'].mean():.4f}")
        report_lines.append("")
        
        # 分辨率影响
        report_lines.append("4. 分辨率影响分析")
        resolution_performance = df.groupby('resolution')['mAP'].mean().sort_values(ascending=False)
        report_lines.append("   按mAP排序的分辨率:")
        for res, mAP in resolution_performance.head(5).items():
            report_lines.append(f"     - {res}: {mAP:.4f}")
        report_lines.append("")
        
        # 保存报告
        with open(os.path.join(output_dir, 'summary_report.txt'), 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        print("已生成: summary_report.txt")
    
    def analyze(self, input_dir: str, output_dir: str, radar_mode: str = 'by_model'):
        """
        执行完整的性能分析
        
        Args:
            input_dir (str): 输入目录路径
            output_dir (str): 输出目录路径
            radar_mode (str): 雷达图模式
        """
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"开始分析目录: {input_dir}")
        print(f"输出目录: {output_dir}")
        print(f"雷达图模式: {radar_mode}")
        
        # 加载数据
        self.load_data(input_dir)
        
        # 生成各种图表
        print("\nGenerating charts...")
        
        self.create_professional_comparison(output_dir)
        self.create_speedup_comparison(output_dir)
        self.create_resolution_analysis(output_dir)
        self.create_category_heatmap(output_dir)
        
        # 根据模式生成雷达图
        if radar_mode == 'by_model':
            self.create_radar_chart_by_model_type(output_dir)
        elif radar_mode == 'combined':
            self.create_radar_chart(output_dir, individual_mode=True)
        elif radar_mode == 'both':
            self.create_radar_chart_by_model_type(output_dir)
            self.create_radar_chart(output_dir, individual_mode=True)
            
        self.create_performance_matrix(output_dir)
        
        # 生成总结报告
        self.generate_summary_report(output_dir)
        
        print(f"\nAnalysis completed! All charts saved to: {output_dir}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="目标检测性能分析可视化工具")
    parser.add_argument('--input_dir', type=str, required=True,
                       help='包含性能评估txt文件的输入目录')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='保存分析图表的输出目录')
    parser.add_argument('--font_path', type=str, 
                       default='/home/sk/project/Object-Detection-Metrics/assets/simhei.ttf',
                       help='中文字体文件路径')
    parser.add_argument('--radar_mode', type=str, 
                       choices=['by_model', 'combined', 'both'], 
                       default='by_model',
                       help='雷达图模式: by_model=按模型类型分组, combined=所有配置在一张图, both=生成两种模式')
    
    args = parser.parse_args()
    
    # 创建分析器并执行分析
    analyzer = PerformanceAnalyzer(font_path=args.font_path)
    analyzer.analyze(args.input_dir, args.output_dir, radar_mode=args.radar_mode)


if __name__ == '__main__':
    main()
