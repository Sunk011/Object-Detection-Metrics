# 目标检测模型性能可视化工具使用说明

## 概述

本项目提供了两个Python脚本，用于分析和可视化目标检测模型的性能指标。主要关注查准率（Precision）和查全率（Recall）的对比分析。

## 工具介绍

### 1. `model_performance_visualizer.py` - 通用模型性能分析工具

**适用场景：** 需要对比多种模型类型（pt、onnx、engine）的性能
**主要功能：** 
- 支持所有模型类型的数据解析
- 自动识别不同的文件命名格式
- 生成综合性能对比图表

### 2. `engine_model_analyzer.py` - Engine模型专用分析工具

**适用场景：** 专门分析Engine模型在不同配置下的性能
**主要功能：**
- 专注于Engine模型的深度分析
- 提供更详细的可视化布局
- 包含更多性能指标（mAP、F1分数等）

## 支持的文件格式

工具支持以下命名格式的txt结果文件：

```
# Engine模型
engine_{分辨率}_{IoU阈值}_{精度类型}.txt
例如: engine_640-640_iou04_fp16.txt

# PT模型
pt_{分辨率}_{IoU阈值}.txt
例如: pt_1024-576_iou04.txt

# ONNX模型
onnx_{分辨率}_{IoU阈值}.txt
例如: onnx_960-544_iou04.txt
```

## 安装依赖

```bash
pip install matplotlib seaborn numpy
```

## 使用方法

### 基本用法

```bash
# 通用分析工具
python model_performance_visualizer.py <输入目录> <输出目录>

# Engine模型专用工具
python engine_model_analyzer.py <输入目录> <输出目录>
```

### 实际示例

```bash
# 分析所有模型类型
python model_performance_visualizer.py assets/visdrone/result tmp/performance_analysis

# 专门分析Engine模型
python engine_model_analyzer.py assets/visdrone/result tmp/engine_analysis
```

## 输出文件说明

### 通用分析工具输出

1. **`engine_models_performance_comparison.png`**
   - Engine模型详细对比图（2×2布局）
   - 包含查准率、查全率柱状图和趋势分析

2. **`all_models_performance_comparison.png`**
   - 所有模型类型对比图（当存在多种模型时）
   - 横向对比不同模型类型的性能

### Engine专用工具输出

1. **`engine_models_detailed_analysis.png`**
   - 6个子图的综合分析图表
   - 详细的性能指标对比和趋势分析

## 图表内容详解

### 通用分析工具图表

#### Engine模型性能对比图（2×2布局）
- **左上：** 不同分辨率下的平均查准率柱状图
- **右上：** 不同分辨率下的平均查全率柱状图
- **左下：** 查准率vs查全率趋势折线图
- **右下：** 查准率vs查全率散点分布图

#### 全模型对比图（1×2布局）
- **左图：** 各模型查准率对比
- **右图：** 各模型查全率对比

### Engine专用工具图表

#### 详细分析图（3×3网格布局）
- **第一行：** 
  - 查准率柱状图
  - 查全率柱状图
  - F1分数柱状图
- **第二行：**
  - 查准率vs查全率趋势图（占2列）
  - 查准率vs查全率散点图
- **第三行：**
  - mAP对比图（占全行）

## 数据解析说明

### 支持的性能指标

工具会自动从结果文件中提取以下指标：
- **mAP（平均精度均值）**
- **平均查准率**
- **平均查全率**
- **平均F1分数**

### 文件解析格式

工具能够解析包含以下格式的中文结果文件：

```
总体性能指标:
  mAP (平均精度均值): 0.3458
  平均查准率: 0.6647
  平均查全率: 0.3906
  平均F1分数: 0.4920
```

## 功能特点

### 自动化特性
- **智能文件识别：** 自动识别和分类不同类型的模型文件
- **动态颜色分配：** 为每个模型配置分配独特的颜色
- **自动排序：** 按分辨率像素总数自动排序显示

### 可视化特性
- **高质量输出：** 300 DPI高分辨率图片
- **专业图表：** 包含网格线、图例、数值标签
- **英文标签：** 所有图表使用英文标签，便于国际化展示

### 数据验证
- **错误处理：** 智能处理文件解析错误
- **数据验证：** 自动验证数据完整性
- **详细日志：** 提供详细的加载和处理日志

## 错误排查

### 常见问题

1. **找不到数据文件**
   ```
   Error: Input directory 'xxx' does not exist
   ```
   **解决方案：** 检查输入目录路径是否正确

2. **没有Engine模型数据**
   ```
   No Engine model data found
   ```
   **解决方案：** 确认目录中包含engine_开头的txt文件

3. **文件解析失败**
   ```
   Error parsing file xxx: xxx
   ```
   **解决方案：** 检查txt文件格式是否符合预期

### 调试建议

1. **验证文件格式：** 确保txt文件包含预期的中文性能指标
2. **检查权限：** 确保对输入和输出目录有读写权限
3. **查看日志：** 运行时会显示详细的加载信息

## 扩展功能

### 自定义配置

如需自定义分析功能，可以修改以下参数：

```python
# 在脚本中调整图表尺寸
fig = plt.figure(figsize=(20, 12))

# 修改颜色方案
colors = plt.cm.Set1(np.linspace(0, 1, len(sorted_data)))

# 调整图表DPI
plt.savefig(output_path, dpi=300, bbox_inches='tight')
```

### 添加新指标

要添加新的性能指标，可以在`parse_result_file`方法中添加相应的正则表达式模式。

## 最佳实践

1. **数据准备：** 确保所有结果文件格式一致
2. **目录组织：** 将同一批实验的结果文件放在同一目录
3. **命名规范：** 严格按照支持的命名格式命名文件
4. **结果备份：** 及时保存生成的可视化图表

## 示例工作流程

```bash
# 1. 准备数据目录
mkdir -p experiment_results
cp *.txt experiment_results/

# 2. 创建输出目录
mkdir -p visualization_output

# 3. 运行分析
python engine_model_analyzer.py experiment_results visualization_output

# 4. 查看结果
ls visualization_output/
```

## 技术支持

如遇到问题，请检查：
1. Python版本（建议3.7+）
2. 依赖包版本
3. 文件路径和权限
4. 数据文件格式

---

**版本信息**
- 工具版本：1.0
- 支持的Python版本：3.7+
- 最后更新：2025年9月8日
