# 快速入门指南

## 目标检测模型性能可视化工具

### 🚀 快速开始

```bash
# 1. 安装依赖
pip install matplotlib seaborn numpy

# 2. 运行分析（选择其中一种）
# 通用工具 - 分析所有模型类型
python model_performance_visualizer.py input_dir output_dir

# 专用工具 - 专门分析Engine模型
python engine_model_analyzer.py input_dir output_dir
```

### 📁 文件命名格式

确保你的结果文件符合以下命名格式：

```
✅ 支持的格式：
engine_640-640_iou04_fp16.txt    # Engine模型
pt_1024-576_iou04.txt           # PT模型  
onnx_960-544_iou04.txt          # ONNX模型

❌ 不支持的格式：
result_640x640.txt              # 错误格式
engine-640-640.txt              # 缺少IoU信息
```

### 🎯 主要功能

| 工具 | 用途 | 输出图表数量 |
|------|------|-------------|
| `model_performance_visualizer.py` | 对比多种模型类型 | 2张图 |
| `engine_model_analyzer.py` | 深度分析Engine模型 | 1张详细图(6个子图) |

### 📊 输出示例

运行成功后，你将看到：

```bash
Loading data...
Loaded: engine_640-640_iou04_fp16.txt

============================================================
Data Summary
============================================================

File: engine_640-640_iou04_fp16.txt
  Resolution: 640-640
  Precision Type: fp16
  Average Precision: 0.6647
  Average Recall: 0.3906
  mAP: 0.3458

Chart saved to: output_dir/engine_models_detailed_analysis.png
Analysis completed!
```

### 🔧 常见问题

**Q: 没有找到数据文件？**
A: 检查文件命名格式和目录路径

**Q: 图表显示不正常？**  
A: 确保txt文件包含中文格式的性能指标

**Q: 想要分析特定指标？**
A: 使用`engine_model_analyzer.py`获得更详细的分析

### 📈 图表说明

生成的图表包含：
- **查准率对比** - 不同配置下的Precision
- **查全率对比** - 不同配置下的Recall  
- **趋势分析** - 性能随分辨率的变化
- **散点分布** - Precision vs Recall关系
- **综合指标** - mAP和F1分数对比

---
💡 **提示：** 详细使用说明请参考 `README_visualization.md`
