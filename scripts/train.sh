#!/bin/bash

# 训练脚本示例
# 使用方法: bash scripts/train.sh

# 使用数据处理生成的完整配置文件（推荐）
python training.py --config processed_data/configs/training_config.yml

# 或者使用默认配置文件
# python training.py --config configs/DMMN-OST.yml

# 或者指定其他配置文件
# python training.py --config /path/to/your/config.yml
