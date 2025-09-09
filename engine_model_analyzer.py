#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Engine Model Performance Visualization Tool
Focused on comparing Engine models across different resolutions
"""

import os
import re
import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap

# 设置字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class EngineModelAnalyzer:
    def __init__(self, input_dir: str, output_dir: str):
        """
        Initialize the Engine model analyzer
        
        Args:
            input_dir: Directory containing result txt files
            output_dir: Directory to save visualization charts
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Store parsed data
        self.engine_data = {}
        
        # VisDrone类别ID与名称的映射
        self.class_mapping = {
            '0': 'pedestrian',
            '1': 'people', 
            '2': 'bicycle',
            '3': 'car',
            '4': 'van',
            '5': 'truck',
            '6': 'tricycle',
            '7': 'awning-tricycle',
            '8': 'bus',
            '9': 'motor'
        }
        
    def parse_filename(self, filename: str) -> Optional[Dict[str, str]]:
        """
        Parse filename to extract model info
        Focus on Engine models: engine_{imgsz}_{iou}_{precision}.txt
        """
        pattern = r'(engine)_(\d+-\d+)_(iou\d+)_(\w+)\.txt'
        match = re.match(pattern, filename)
        
        if match:
            return {
                'model_type': match.group(1),
                'resolution': match.group(2),
                'iou': match.group(3),
                'precision': match.group(4)
            }
        return None
    
    def parse_result_file(self, filepath: Path) -> Dict:
        """
        Parse a single result file to extract performance metrics
        """
        results = {'overall': {}, 'per_class': {}}
        
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract overall performance metrics
        overall_pattern = r'总体性能指标:\s*mAP \(平均精度均值\): ([\d.]+)\s*平均查准率: ([\d.]+)\s*平均查全率: ([\d.]+)\s*平均F1分数: ([\d.]+)'
        overall_match = re.search(overall_pattern, content)
        
        if overall_match:
            results['overall'] = {
                'mAP': float(overall_match.group(1)),
                'avg_precision': float(overall_match.group(2)),
                'avg_recall': float(overall_match.group(3)),
                'avg_f1': float(overall_match.group(4))
            }
        
        # Extract per-class metrics
        class_pattern = r'类别: (\d+) \(([^)]+)\)\s+平均精度 \(AP\): ([\d.]+)\s+查准率 \(Precision\): ([\d.]+)\s+查全率 \(Recall\): ([\d.]+)\s+F1分数: ([\d.]+)'
        class_matches = re.findall(class_pattern, content)
        
        for match in class_matches:
            class_id = match[0]
            class_name_in_file = match[1]
            
            # 使用映射关系获取标准类别名称，如果映射中没有则使用文件中的名称
            class_name = self.class_mapping.get(class_id, class_name_in_file)
            
            results['per_class'][class_name] = {
                'class_id': class_id,
                'AP': float(match[2]),
                'precision': float(match[3]),
                'recall': float(match[4]),
                'f1': float(match[5])
            }
        
        return results
    
    def load_engine_data(self):
        """
        Load all Engine model data files
        """
        for txt_file in self.input_dir.glob('*.txt'):
            file_info = self.parse_filename(txt_file.name)
            if file_info and file_info['model_type'] == 'engine':
                try:
                    results = self.parse_result_file(txt_file)
                    
                    # Build unique key
                    key = f"{file_info['resolution']}_{file_info['precision']}"
                    
                    self.engine_data[key] = {
                        'file_info': file_info,
                        'results': results,
                        'filename': txt_file.name
                    }
                    
                    print(f"Loaded Engine model: {txt_file.name}")
                    
                except Exception as e:
                    print(f"Error parsing file {txt_file.name}: {e}")
    
    def get_distinct_colors(self, n_colors: int) -> List[str]:
        """
        Generate distinct colors with high contrast for better visibility
        """
        # 使用高对比度的颜色集合
        base_colors = [
            '#FF0000',  # 红色
            '#0000FF',  # 蓝色
            '#00FF00',  # 绿色
            '#FF8000',  # 橙色
            '#800080',  # 紫色
            '#008080',  # 青色
            '#FF1493',  # 深粉色
            '#32CD32',  # 酸橙绿
            '#FF4500',  # 橙红色
            '#4169E1',  # 皇家蓝
            '#DC143C',  # 深红色
            '#00CED1',  # 深青色
            '#FFD700',  # 金色
            '#9932CC',  # 深兰花紫
            '#FF6347',  # 番茄色
        ]
        
        if n_colors <= len(base_colors):
            return base_colors[:n_colors]
        else:
            # 如果需要更多颜色，则使用色彩映射补充
            additional_colors = plt.cm.tab20(np.linspace(0, 1, n_colors - len(base_colors)))
            additional_hex = ['#%02x%02x%02x' % tuple((255 * np.array(color[:3])).astype(int)) 
                            for color in additional_colors]
            return base_colors + additional_hex
    
    def get_resolution_pixels(self, resolution: str) -> int:
        """
        Convert resolution string to total pixels for sorting
        """
        try:
            w, h = map(int, resolution.split('-'))
            return w * h
        except:
            return 0
    
    def create_radar_chart(self, sorted_data, colors):
        """
        Create radar chart showing precision, recall, and mAP for each model
        """
        # Create a separate figure for radar chart
        fig_radar = plt.figure(figsize=(12, 10))
        ax = fig_radar.add_subplot(111, projection='polar')
        
        # 设置雷达图的角度和标签
        angles = np.linspace(0, 2 * np.pi, 3, endpoint=False).tolist()
        angles += angles[:1]  # 闭合圆圈
        
        labels_radar = ['Precision', 'Recall', 'mAP']
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels_radar, fontsize=12, fontweight='bold')
        
        # Calculate dynamic range for better visualization
        all_precision = [data['results']['overall']['avg_precision'] for _, data in sorted_data]
        all_recall = [data['results']['overall']['avg_recall'] for _, data in sorted_data]
        all_map = [data['results']['overall']['mAP'] for _, data in sorted_data]
        
        # Find the range for better scaling
        min_val = min(min(all_precision), min(all_recall), min(all_map))
        max_val = max(max(all_precision), max(all_recall), max(all_map))
        
        # Add some padding
        padding = (max_val - min_val) * 0.1
        y_min = max(0, min_val - padding)
        y_max = min(1, max_val + padding)
        
        # Set focused range instead of 0-1
        ax.set_ylim(y_min, y_max)
        
        # Create more granular ticks
        n_ticks = 6
        tick_values = np.linspace(y_min, y_max, n_ticks)
        ax.set_yticks(tick_values)
        ax.set_yticklabels([f'{val:.3f}' for val in tick_values], fontsize=10)
        ax.grid(True)
        
        # 为每个模型绘制雷达图
        for i, (key, data) in enumerate(sorted_data):
            resolution = data['file_info']['resolution']
            overall = data['results']['overall']
            
            # 数据点
            values = [
                overall['avg_precision'],
                overall['avg_recall'],
                overall['mAP']
            ]
            values += values[:1]  # 闭合线条
            
            # 使用更粗的线条和更大的标记点来增强可见性
            ax.plot(angles, values, 'o-', linewidth=3, 
                   label=f"{resolution}", color=colors[i], markersize=8, 
                   markerfacecolor='white', markeredgewidth=2, markeredgecolor=colors[i])
            ax.fill(angles, values, alpha=0.15, color=colors[i])
            
            # 添加数值标签
            for angle, value in zip(angles[:-1], values[:-1]):
                ax.annotate(f'{value:.3f}', (angle, value), 
                           xytext=(10, 0), textcoords='offset points',
                           fontsize=8, fontweight='bold', color=colors[i],
                           ha='center')
        
        # 设置标题和图例
        ax.set_title('Engine Models Performance Radar Chart\n(Precision, Recall, mAP - Focused View)', 
                    fontsize=14, fontweight='bold', pad=30)
        ax.legend(loc='upper right', bbox_to_anchor=(1.4, 1.0), fontsize=9)
        
        # 保存雷达图
        radar_path = self.output_dir / 'engine_models_radar_chart.png'
        plt.savefig(radar_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Radar chart saved to: {radar_path}")
        plt.show()
    
    def create_per_class_analysis(self, sorted_data, colors):
        """
        Create per-class analysis charts
        """
        # 收集所有类别
        all_classes = set()
        for key, data in sorted_data:
            all_classes.update(data['results']['per_class'].keys())
        
        if not all_classes:
            print("No per-class data found")
            return
        
        all_classes = sorted(list(all_classes))
        n_classes = len(all_classes)
        
        # 创建分类别分析图
        fig_class = plt.figure(figsize=(20, 6 * ((n_classes + 2) // 3)))
        
        fig_class.suptitle('Per-Class Performance Analysis Across Engine Models', 
                          fontsize=16, fontweight='bold', y=0.98)
        
        # 为每个类别创建子图
        for class_idx, class_name in enumerate(all_classes):
            ax = fig_class.add_subplot((n_classes + 2) // 3, 3, class_idx + 1)
            
            # 收集该类别的数据
            model_names = []
            precisions = []
            recalls = []
            aps = []
            
            for key, data in sorted_data:
                resolution = data['file_info']['resolution']
                model_names.append(resolution)
                
                if class_name in data['results']['per_class']:
                    class_data = data['results']['per_class'][class_name]
                    precisions.append(class_data['precision'])
                    recalls.append(class_data['recall'])
                    aps.append(class_data['AP'])
                else:
                    precisions.append(0)
                    recalls.append(0)
                    aps.append(0)
            
            # 创建分组柱状图
            x = np.arange(len(model_names))
            width = 0.25
            
            bars1 = ax.bar(x - width, precisions, width, label='Precision', 
                          color=[colors[i] for i in range(len(model_names))], alpha=0.8)
            bars2 = ax.bar(x, recalls, width, label='Recall',
                          color=[colors[i] for i in range(len(model_names))], alpha=0.6)
            bars3 = ax.bar(x + width, aps, width, label='AP',
                          color=[colors[i] for i in range(len(model_names))], alpha=0.4)
            
            # 设置标签和标题
            ax.set_xlabel('Model Resolution', fontsize=10)
            ax.set_ylabel('Performance', fontsize=10)
            ax.set_title(f'Class: {class_name}', fontsize=12, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(model_names, rotation=45, ha='right', fontsize=8)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1)
            
            # 添加数值标签
            def add_value_labels(bars, values):
                for bar, val in zip(bars, values):
                    if val > 0:
                        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                               f'{val:.2f}', ha='center', va='bottom', fontsize=7)
            
            add_value_labels(bars1, precisions)
            add_value_labels(bars2, recalls)
            add_value_labels(bars3, aps)
        
        plt.tight_layout()
        
        # 保存分类别分析图
        class_path = self.output_dir / 'engine_models_per_class_analysis.png'
        plt.savefig(class_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Per-class analysis saved to: {class_path}")
        plt.show()
    
    def create_engine_analysis_chart(self):
        """
        Create comprehensive Engine model comparison chart
        """
        if not self.engine_data:
            print("No Engine model data found")
            return
        
        # Sort by resolution
        sorted_data = sorted(self.engine_data.items(), 
                           key=lambda x: self.get_resolution_pixels(x[1]['file_info']['resolution']))
        
        # Prepare data
        labels = []
        precisions = []
        recalls = []
        f1_scores = []
        maps = []
        
        # Generate distinct colors for each model
        colors = self.get_distinct_colors(len(sorted_data))
        
        for key, data in sorted_data:
            resolution = data['file_info']['resolution']
            precision_type = data['file_info']['precision']
            overall = data['results']['overall']
            
            label = f"{resolution}\n({precision_type})"
            labels.append(label)
            precisions.append(overall['avg_precision'])
            recalls.append(overall['avg_recall'])
            f1_scores.append(overall['avg_f1'])
            maps.append(overall['mAP'])
        
        # Create comprehensive chart
        fig = plt.figure(figsize=(20, 12))
        
        # Main title
        fig.suptitle('Engine Model Performance Analysis Across Different Resolutions', 
                    fontsize=16, fontweight='bold', y=0.95)
        
        # Create subplots
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Precision Bar Chart
        ax1 = fig.add_subplot(gs[0, 0])
        bars1 = ax1.bar(range(len(labels)), precisions, color=colors, alpha=0.8, 
                       edgecolor='black', linewidth=1.5)
        ax1.set_title('Average Precision by Resolution', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Precision', fontsize=10)
        ax1.set_xticks(range(len(labels)))
        ax1.set_xticklabels(labels, rotation=0, ha='center', fontsize=5)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars1, precisions)):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=8)
        
        # 2. Recall Bar Chart
        ax2 = fig.add_subplot(gs[0, 1])
        bars2 = ax2.bar(range(len(labels)), recalls, color=colors, alpha=0.8, 
                       edgecolor='black', linewidth=1.5)
        ax2.set_title('Average Recall by Resolution', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Recall', fontsize=10)
        ax2.set_xticks(range(len(labels)))
        ax2.set_xticklabels(labels, rotation=0, ha='center', fontsize=5)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars2, recalls)):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=8)
        
        # 3. F1 Score Bar Chart
        ax3 = fig.add_subplot(gs[0, 2])
        bars3 = ax3.bar(range(len(labels)), f1_scores, color=colors, alpha=0.8, 
                       edgecolor='black', linewidth=1.5)
        ax3.set_title('Average F1 Score by Resolution', fontsize=12, fontweight='bold')
        ax3.set_ylabel('F1 Score', fontsize=10)
        ax3.set_xticks(range(len(labels)))
        ax3.set_xticklabels(labels, rotation=0, ha='center', fontsize=5)
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 1)
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars3, f1_scores)):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=8)
        
        # 4. Precision vs Recall Line Chart
        ax4 = fig.add_subplot(gs[1, :2])
        x_pos = np.arange(len(labels))
        
        line1 = ax4.plot(x_pos, precisions, 'o-', linewidth=3, markersize=10, 
                        label='Precision', color='#2E8B57', markerfacecolor='white', 
                        markeredgewidth=3, markeredgecolor='#2E8B57')
        line2 = ax4.plot(x_pos, recalls, 's-', linewidth=3, markersize=10, 
                        label='Recall', color='#FF6347', markerfacecolor='white',
                        markeredgewidth=3, markeredgecolor='#FF6347')
        
        ax4.set_title('Precision vs Recall Trend Comparison', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Performance Metrics', fontsize=10)
        ax4.set_xlabel('Model Configuration (Resolution + Precision)', fontsize=10)
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(labels, rotation=0, ha='center', fontsize=8)
        ax4.legend(fontsize=10, loc='best')
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(0, 1)
        
        # Add value annotations
        for i, (p, r) in enumerate(zip(precisions, recalls)):
            ax4.annotate(f'{p:.3f}', (i, p), xytext=(0, 10), textcoords='offset points',
                        ha='center', va='bottom', color='#2E8B57', fontweight='bold', fontsize=7)
            ax4.annotate(f'{r:.3f}', (i, r), xytext=(0, -15), textcoords='offset points',
                        ha='center', va='top', color='#FF6347', fontweight='bold', fontsize=7)
        
        # 5. Precision vs Recall Scatter Plot
        ax5 = fig.add_subplot(gs[1, 2])
        
        # Calculate dynamic axis limits based on data range
        precision_range = max(precisions) - min(precisions)
        recall_range = max(recalls) - min(recalls)
        
        # Set more focused limits to show differences better
        x_margin = max(0.02, precision_range * 0.2)
        y_margin = max(0.02, recall_range * 0.2)
        
        x_min = max(0, min(precisions) - x_margin)
        x_max = min(1, max(precisions) + x_margin)
        y_min = max(0, min(recalls) - y_margin)
        y_max = min(1, max(recalls) + y_margin)
        
        # Calculate plot dimensions for jitter
        plot_width = x_max - x_min
        plot_height = y_max - y_min
        
        # Use 'x' markers for data points with slight position jitter to avoid overlap
        jitter_amount = min(0.002, min(plot_width, plot_height) * 0.01)
        for i, (p, r) in enumerate(zip(precisions, recalls)):
            # Add small jitter to avoid complete overlap
            jitter_x = (i % 5 - 2) * jitter_amount * 0.5
            jitter_y = (i // 5 - 1) * jitter_amount * 0.3
            ax5.scatter(p + jitter_x, r + jitter_y, marker='x', s=100, 
                       color=colors[i], linewidth=1)
        
        # Add color legend in the most empty area with better spacing        
        # Prioritize left side for legend to avoid data cluster
        potential_areas = [
            (x_min + 0.02 * plot_width, y_max - 0.08 * plot_height),  # top-left (preferred)
            (x_min + 0.02 * plot_width, y_min + 0.08 * plot_height),  # bottom-left
            (x_min + 0.02 * plot_width, y_min + 0.5 * plot_height),   # middle-left
        ]
        
        # Choose the leftmost area that has good spacing
        best_area_idx = 0
        for area_idx, (area_x, area_y) in enumerate(potential_areas):
            distances = [((area_x - p)**2 + (area_y - r)**2)**0.5 
                        for p, r in zip(precisions, recalls)]
            min_dist = min(distances)
            # Prefer left areas and sufficient distance
            if min_dist > 0.03:  # Good separation
                best_area_idx = area_idx
                break
        
        legend_start_x, legend_start_y = potential_areas[best_area_idx]
        
        # Create a more compact legend layout
        n_models = len(sorted_data)
        max_models_per_col = 5
        n_cols = (n_models + max_models_per_col - 1) // max_models_per_col
        
        spacing_y = min(0.02, plot_height * 0.6 / max_models_per_col)
        spacing_x = 0.15 * plot_width
        
        for i, (key, data) in enumerate(sorted_data):
            resolution = data['file_info']['resolution']
            col = i // max_models_per_col
            row = i % max_models_per_col
            
            legend_x = legend_start_x + col * spacing_x
            legend_y = legend_start_y - row * spacing_y
            
            # Add smaller 'x' marker for legend
            ax5.scatter(legend_x, legend_y, marker='x', s=120, 
                       color=colors[i], linewidth=2)
            
            # Add text label with better positioning
            ax5.text(legend_x + 0.012 * plot_width, legend_y, resolution, 
                    fontsize=6, fontweight='bold', color=colors[i],
                    verticalalignment='center', horizontalalignment='left')
        
        ax5.set_title('Precision vs Recall Distribution (Zoomed)', fontsize=12, fontweight='bold')
        ax5.set_xlabel('Precision', fontsize=10)
        ax5.set_ylabel('Recall', fontsize=10)
        ax5.grid(True, alpha=0.3)
        ax5.set_xlim(x_min, x_max)
        ax5.set_ylim(y_min, y_max)
        
        # Add diagonal reference line only if it's in the visible range
        diag_start = max(x_min, y_min)
        diag_end = min(x_max, y_max)
        if diag_start < diag_end:
            ax5.plot([diag_start, diag_end], [diag_start, diag_end], '--', 
                    color='gray', alpha=0.5, linewidth=1)
        
        # 6. mAP Comparison
        ax6 = fig.add_subplot(gs[2, :])
        bars6 = ax6.bar(range(len(labels)), maps, color=colors, alpha=0.8, 
                       edgecolor='black', linewidth=1.5)
        ax6.set_title('Mean Average Precision (mAP) Comparison', fontsize=12, fontweight='bold')
        ax6.set_ylabel('mAP', fontsize=10)
        ax6.set_xlabel('Model Configuration (Resolution + Precision)', fontsize=10)
        ax6.set_xticks(range(len(labels)))
        ax6.set_xticklabels(labels, rotation=0, ha='center', fontsize=8)
        ax6.grid(True, alpha=0.3)
        ax6.set_ylim(0, max(maps) * 1.2)
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars6, maps)):
            ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(maps) * 0.02,
                    f'{val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=8)
        
        # Save chart
        output_path = self.output_dir / 'engine_models_detailed_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Detailed chart saved to: {output_path}")
        
        plt.show()
    
    def print_summary(self):
        """
        Print data summary
        """
        print("\n" + "="*60)
        print("Engine Models Data Summary")
        print("="*60)
        
        if not self.engine_data:
            print("No Engine model data found")
            return
        
        for key, data in self.engine_data.items():
            info = data['file_info']
            results = data['results']['overall']
            
            print(f"\nFile: {data['filename']}")
            print(f"  Resolution: {info['resolution']}")
            print(f"  Precision Type: {info['precision']}")
            print(f"  IoU Threshold: {info['iou']}")
            print(f"  Average Precision: {results['avg_precision']:.4f}")
            print(f"  Average Recall: {results['avg_recall']:.4f}")
            print(f"  Average F1 Score: {results['avg_f1']:.4f}")
            print(f"  mAP: {results['mAP']:.4f}")
            
            # 显示分类别信息
            if data['results']['per_class']:
                print(f"  Per-class data available for {len(data['results']['per_class'])} classes:")
                for class_name, class_data in data['results']['per_class'].items():
                    class_id = class_data.get('class_id', 'N/A')
                    print(f"    Class {class_id} ({class_name}): AP={class_data['AP']:.3f}, "
                          f"P={class_data['precision']:.3f}, R={class_data['recall']:.3f}")
        
        # 显示类别映射信息
        print(f"\n" + "="*60)
        print("VisDrone Classes Mapping:")
        print("="*60)
        for class_id, class_name in self.class_mapping.items():
            print(f"  {class_id}: {class_name}")
    
    def run_analysis(self):
        """
        Run complete analysis
        """
        print("Loading Engine model data...")
        self.load_engine_data()
        
        if not self.engine_data:
            print("No Engine model data files found")
            return
        
        # Sort data for consistent ordering across all charts
        sorted_data = sorted(self.engine_data.items(), 
                           key=lambda x: self.get_resolution_pixels(x[1]['file_info']['resolution']))
        
        # Generate distinct colors
        colors = self.get_distinct_colors(len(sorted_data))
        
        self.print_summary()
        
        print(f"\nGenerating detailed visualization chart...")
        self.create_engine_analysis_chart()
        
        print(f"\nGenerating radar chart...")
        self.create_radar_chart(sorted_data, colors)
        
        print(f"\nGenerating per-class analysis...")
        self.create_per_class_analysis(sorted_data, colors)
        
        print("\nEngine model analysis completed!")


def main():
    """
    Main function
    """
    parser = argparse.ArgumentParser(description='Engine Model Performance Visualization Tool')
    parser.add_argument('--input_dir',  '-i', help='Directory path containing result txt files')
    parser.add_argument('--output_dir', '-o', help='Directory path to save visualization charts')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' does not exist")
        return
    
    analyzer = EngineModelAnalyzer(args.input_dir, args.output_dir)
    analyzer.run_analysis()


if __name__ == "__main__":
    main()
