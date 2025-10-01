# WSI数据处理脚本使用说明

这个脚本用于将SVS格式的WSI文件和PNG格式的mask文件处理成1024x1024的训练图片。

## 功能特点

- 支持SVS格式的WSI文件
- 支持PNG格式的mask文件
- 自动调整mask尺寸以匹配WSI尺寸
- 支持tissue区域检测，避免裁剪空白区域
- 可配置的patch大小和stride
- 自动分割训练集和验证集
- 生成训练所需的文件列表
- **支持TSV文件定义类别映射关系**
- **自动生成训练配置文件**
- **自动设置正确的类别数量**

## 使用方法

### 基本用法

```bash
python process_training_data.py \
    --wsi_dir /path/to/wsi/files \
    --mask_dir /path/to/mask/files \
    --output_dir /path/to/output \
    --tsv_file /path/to/class_mapping.tsv \
    --tissue_detection
```

### 参数说明

- `--wsi_dir`: WSI文件目录（包含.svs文件）
- `--mask_dir`: PNG mask文件目录（包含.png文件）
- `--output_dir`: 输出目录
- `--tsv_file`: TSV文件路径，包含mask_id到class_id的映射（可选）
- `--base_config`: 基础配置文件路径（可选）
- `--patch_size`: patch大小（默认1024）
- `--stride`: 步长（默认256）
- `--tissue_detection`: 启用tissue检测（推荐）
- `--min_tissue_ratio`: 最小tissue比例阈值（默认0.1）
- `--train_ratio`: 训练集比例（默认0.8）

### 完整示例

```bash
python process_training_data.py \
    --wsi_dir ./data/wsi \
    --mask_dir ./data/masks \
    --output_dir ./processed_data \
    --tsv_file ./gtruth_codes.tsv \
    --base_config ./configs/DMMN-OST.yml \
    --patch_size 1024 \
    --stride 256 \
    --tissue_detection \
    --min_tissue_ratio 0.1 \
    --train_ratio 0.8
```

## TSV文件格式

TSV文件用于定义mask中的像素值与训练类别的映射关系，格式如下：

```tsv
label	GT_code
unlabeled	0
Non-Bone Active Tumor	1
Bone Active Tumor	2
Non-Bone Necrosis	3
Bone Necrosis	4
Normal Tissue	5
Background	6
Sparse Cellular Area	7
```

- `label`: 类别的名称（用于配置文件）
- `GT_code`: PNG mask文件中的像素值（训练时使用的类别ID）

## 输出结构

处理完成后，输出目录将包含以下结构：

```
output_dir/
├── slide_tiles/          # 裁剪的图片patches
│   ├── slide1_0_0.jpg
│   ├── slide1_256_0.jpg
│   └── ...
├── label_tiles/          # 对应的mask patches
│   ├── slide1_0_0.png
│   ├── slide1_256_0.png
│   └── ...
├── configs/              # 配置文件目录（重要！）
│   ├── training_config.yml  # 完整的训练配置文件
│   └── class_mapping.json   # 类别映射JSON文件
├── train_tiles.txt       # 训练集文件列表
└── val_tiles.txt         # 验证集文件列表
```

## 生成的配置文件说明

### `configs/training_config.yml`
这是**最重要的文件**，包含：
- ✅ 正确的类别数量（从TSV文件自动计算）
- ✅ 正确的类别映射关系
- ✅ 正确的数据路径设置
- ✅ 完整的训练参数配置
- ✅ Wandb监控配置
- ✅ 模型保存路径配置

### `configs/class_mapping.json`
类别映射文件，包含：
- 类别ID到类别名称的映射
- 用于推理时的类别标签显示

## 文件命名规则

- WSI文件: `slide_name.svs`
- Mask文件: `slide_name.png`
- 输出图片: `slide_name_x_y.jpg`
- 输出mask: `slide_name_x_y.png`

其中`x`和`y`是patch在WSI中的坐标。

## 注意事项

1. **文件对应关系**: WSI文件名和mask文件名必须对应（除了扩展名）
2. **内存使用**: 处理大型WSI文件时可能需要较多内存
3. **Tissue检测**: 建议启用tissue检测以过滤空白区域
4. **Stride设置**: 确保stride与训练脚本中的设置一致

## 训练脚本集成

处理完成后，可以直接使用生成的文件进行训练：

### 方法1：使用生成的完整配置文件（推荐）

```bash
# 使用数据处理生成的完整配置文件
python training.py --config processed_data/configs/training_config.yml
```

**优势：**
- 包含正确的类别数量（从TSV文件自动计算）
- 包含正确的类别映射
- 数据路径已正确设置
- 包含所有必要的训练参数

### 方法2：使用预定义配置文件

```bash
# 使用预定义的OST配置文件
python training.py --config configs/DMMN-OST.yml
```

**注意：** 需要手动确保数据路径和类别数量正确

### Wandb监控

训练脚本现在支持Wandb监控，可以在配置文件中启用：

```yaml
training:
  wandb:
    enabled: true
    project: 'OST-DMMN'
    name: 'DMMN-OST-Experiment'
    tags: ['OST', 'DMMN', 'segmentation']
    notes: 'Osteosarcoma tissue segmentation using DMMN'
```

### 模型保存路径配置

可以在配置文件中自定义模型保存路径：

```yaml
model:
  arch: DMMN
  save_dir: 'runs/DMMN-OST'  # 模型保存目录
  save_name: 'DMMN_ost_seg'  # 模型保存名称前缀
```

### 数据路径配置

可以在配置文件中指定训练和验证数据文件：

```yaml
data:
  dataset: ost_seg
  train_file: 'train_tiles.txt'      # 训练数据文件列表
  val_file: 'val_tiles.txt'          # 验证数据文件列表
  train_split: train_aug
  val_split: val
  img_rows: 'same'
  img_cols: 'same'
  path: ''
```

**数据文件格式：**
- 每行包含一个patch的路径信息
- 格式：`{数据目录},{patch文件名}`
- 例如：`processed_data,slide1_0_0`

## 故障排除

### 常见问题

1. **找不到mask文件**: 确保mask文件名与WSI文件名对应
2. **内存不足**: 尝试减小patch_size或使用更小的stride
3. **Tissue检测失败**: 检查WSI文件是否包含有效的tissue区域

### 调试模式

如果遇到问题，可以添加调试信息：

```python
# 在process_training_data.py中添加
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 性能优化

- 使用SSD存储以提高I/O性能
- 确保有足够的内存处理大型WSI文件
- 可以考虑并行处理多个WSI文件（需要修改脚本）
