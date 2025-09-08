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
        results = {'overall': {}}
        
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
    
    def get_resolution_pixels(self, resolution: str) -> int:
        """
        Convert resolution string to total pixels for sorting
        """
        try:
            w, h = map(int, resolution.split('-'))
            return w * h
        except:
            return 0
    
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
        colors = plt.cm.Set1(np.linspace(0, 1, len(sorted_data)))
        
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
                    fontsize=18, fontweight='bold', y=0.95)
        
        # Create subplots
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Precision Bar Chart
        ax1 = fig.add_subplot(gs[0, 0])
        bars1 = ax1.bar(range(len(labels)), precisions, color=colors, alpha=0.8, 
                       edgecolor='black', linewidth=1.5)
        ax1.set_title('Average Precision by Resolution', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Precision', fontsize=12)
        ax1.set_xticks(range(len(labels)))
        ax1.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars1, precisions)):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        # 2. Recall Bar Chart
        ax2 = fig.add_subplot(gs[0, 1])
        bars2 = ax2.bar(range(len(labels)), recalls, color=colors, alpha=0.8, 
                       edgecolor='black', linewidth=1.5)
        ax2.set_title('Average Recall by Resolution', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Recall', fontsize=12)
        ax2.set_xticks(range(len(labels)))
        ax2.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars2, recalls)):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        # 3. F1 Score Bar Chart
        ax3 = fig.add_subplot(gs[0, 2])
        bars3 = ax3.bar(range(len(labels)), f1_scores, color=colors, alpha=0.8, 
                       edgecolor='black', linewidth=1.5)
        ax3.set_title('Average F1 Score by Resolution', fontsize=14, fontweight='bold')
        ax3.set_ylabel('F1 Score', fontsize=12)
        ax3.set_xticks(range(len(labels)))
        ax3.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 1)
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars3, f1_scores)):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        # 4. Precision vs Recall Line Chart
        ax4 = fig.add_subplot(gs[1, :2])
        x_pos = np.arange(len(labels))
        
        line1 = ax4.plot(x_pos, precisions, 'o-', linewidth=3, markersize=10, 
                        label='Precision', color='#2E8B57', markerfacecolor='white', 
                        markeredgewidth=3, markeredgecolor='#2E8B57')
        line2 = ax4.plot(x_pos, recalls, 's-', linewidth=3, markersize=10, 
                        label='Recall', color='#FF6347', markerfacecolor='white',
                        markeredgewidth=3, markeredgecolor='#FF6347')
        
        ax4.set_title('Precision vs Recall Trend Comparison', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Performance Metrics', fontsize=12)
        ax4.set_xlabel('Model Configuration (Resolution + Precision)', fontsize=12)
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
        ax4.legend(fontsize=12, loc='best')
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(0, 1)
        
        # Add value annotations
        for i, (p, r) in enumerate(zip(precisions, recalls)):
            ax4.annotate(f'{p:.3f}', (i, p), xytext=(0, 10), textcoords='offset points',
                        ha='center', va='bottom', color='#2E8B57', fontweight='bold')
            ax4.annotate(f'{r:.3f}', (i, r), xytext=(0, -15), textcoords='offset points',
                        ha='center', va='top', color='#FF6347', fontweight='bold')
        
        # 5. Precision vs Recall Scatter Plot
        ax5 = fig.add_subplot(gs[1, 2])
        scatter = ax5.scatter(precisions, recalls, c=colors, s=300, alpha=0.8, 
                            edgecolors='black', linewidth=2)
        
        # Add labels for each point
        for i, (p, r, label) in enumerate(zip(precisions, recalls, labels)):
            ax5.annotate(label.split('\n')[0], (p, r), xytext=(5, 5), 
                        textcoords='offset points', fontsize=9, fontweight='bold')
        
        ax5.set_title('Precision vs Recall Distribution', fontsize=14, fontweight='bold')
        ax5.set_xlabel('Precision', fontsize=12)
        ax5.set_ylabel('Recall', fontsize=12)
        ax5.grid(True, alpha=0.3)
        ax5.set_xlim(0, 1)
        ax5.set_ylim(0, 1)
        
        # Add diagonal reference line
        ax5.plot([0, 1], [0, 1], '--', color='gray', alpha=0.5, linewidth=1)
        
        # 6. mAP Comparison
        ax6 = fig.add_subplot(gs[2, :])
        bars6 = ax6.bar(range(len(labels)), maps, color=colors, alpha=0.8, 
                       edgecolor='black', linewidth=1.5)
        ax6.set_title('Mean Average Precision (mAP) Comparison', fontsize=14, fontweight='bold')
        ax6.set_ylabel('mAP', fontsize=12)
        ax6.set_xlabel('Model Configuration (Resolution + Precision)', fontsize=12)
        ax6.set_xticks(range(len(labels)))
        ax6.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
        ax6.grid(True, alpha=0.3)
        ax6.set_ylim(0, max(maps) * 1.2)
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars6, maps)):
            ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(maps) * 0.02,
                    f'{val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
        
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
    
    def run_analysis(self):
        """
        Run complete analysis
        """
        print("Loading Engine model data...")
        self.load_engine_data()
        
        if not self.engine_data:
            print("No Engine model data files found")
            return
        
        self.print_summary()
        print(f"\nGenerating detailed visualization chart...")
        self.create_engine_analysis_chart()
        print("\nEngine model analysis completed!")


def main():
    """
    Main function
    """
    parser = argparse.ArgumentParser(description='Engine Model Performance Visualization Tool')
    parser.add_argument('input_dir', help='Directory path containing result txt files')
    parser.add_argument('output_dir', help='Directory path to save visualization charts')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' does not exist")
        return
    
    analyzer = EngineModelAnalyzer(args.input_dir, args.output_dir)
    analyzer.run_analysis()


if __name__ == "__main__":
    main()
