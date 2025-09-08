#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量分析示例脚本
演示如何批量处理多个实验目录的结果文件
"""

import os
import subprocess
from pathlib import Path

def batch_analysis_example():
    """
    批量分析示例
    """
    # 示例：假设你有多个实验目录
    experiment_dirs = [
        "assets/visdrone/result",
        # "experiment_2/results",
        # "experiment_3/results",
    ]
    
    base_output_dir = "tmp/batch_analysis"
    
    print("🚀 开始批量分析...")
    
    for i, exp_dir in enumerate(experiment_dirs, 1):
        if not os.path.exists(exp_dir):
            print(f"⚠️  跳过不存在的目录: {exp_dir}")
            continue
            
        print(f"\n📁 处理实验 {i}: {exp_dir}")
        
        # 为每个实验创建单独的输出目录
        output_dir = f"{base_output_dir}/experiment_{i}"
        
        try:
            # 运行Engine模型专用分析
            cmd = [
                "python", "engine_model_analyzer.py",
                exp_dir, output_dir
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"✅ 实验 {i} 分析完成")
                print(f"   输出目录: {output_dir}")
            else:
                print(f"❌ 实验 {i} 分析失败")
                print(f"   错误信息: {result.stderr}")
                
        except Exception as e:
            print(f"❌ 处理实验 {i} 时出错: {e}")
    
    print(f"\n🎉 批量分析完成! 结果保存在: {base_output_dir}")

def create_comparison_script():
    """
    创建用于对比不同实验的脚本
    """
    script_content = '''#!/usr/bin/env python3
"""
实验对比脚本 - 对比多个实验的Engine模型性能
"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def compare_experiments():
    # 这里可以添加跨实验对比的代码
    # 例如：加载多个实验的结果，进行横向对比
    pass

if __name__ == "__main__":
    compare_experiments()
'''
    
    with open("experiment_comparison.py", "w", encoding="utf-8") as f:
        f.write(script_content)
    
    print("📝 已创建实验对比脚本: experiment_comparison.py")

def main():
    """
    主函数 - 演示各种使用场景
    """
    print("=" * 60)
    print("目标检测模型性能可视化工具 - 使用示例")
    print("=" * 60)
    
    print("\n1️⃣ 单次分析示例:")
    print("python engine_model_analyzer.py assets/visdrone/result tmp/single_analysis")
    
    print("\n2️⃣ 批量分析示例:")
    batch_analysis_example()
    
    print("\n3️⃣ 创建对比脚本:")
    create_comparison_script()
    
    print("\n📋 使用建议:")
    print("- 对于单个实验: 使用 engine_model_analyzer.py")
    print("- 对于多个实验: 使用本脚本的批量处理功能")
    print("- 对于深度对比: 自定义开发对比脚本")
    
    print("\n📚 更多信息请查看:")
    print("- README_visualization.md (详细说明)")
    print("- QUICK_START.md (快速入门)")

if __name__ == "__main__":
    main()
