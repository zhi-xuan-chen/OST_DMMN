#!/bin/bash

# 训练脚本示例
# 使用方法: bash scripts/train.sh

# 使用数据处理生成的完整配置文件（推荐）
CUDA_VISIBLE_DEVICES=1,2 python /home/chenzhixuan/Workspace/OST_DMMN/training.py --config /home/chenzhixuan/Workspace/OST_DMMN/configs/DMMN-OST.yml
# 或者使用默认配置文件
# python training.py --config configs/DMMN-OST.yml

# 或者指定其他配置文件
# python training.py --config /path/to/your/config.yml
