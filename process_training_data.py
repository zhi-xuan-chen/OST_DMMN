#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WSI和PNG mask数据处理脚本
将SVS格式的WSI和PNG格式的mask处理成1024x1024的训练图片
"""

import os
import sys
import cv2
import numpy as np
import openslide
import argparse
from PIL import Image
import shutil
from tqdm import tqdm
import pandas as pd
from pathlib import Path
import extract_tissue
import yaml
import json

# 解决PIL DecompressionBombError
Image.MAX_IMAGE_PIXELS = None

# 设置环境变量避免线程冲突
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

def load_class_mapping(tsv_path):
    """
    从TSV文件加载类别映射关系
    
    Args:
        tsv_path: TSV文件路径，格式应为: label, GT_code
        
    Returns:
        dict: GT_code到class_id的映射字典
        int: 类别总数
        dict: class_id到class_name的映射字典
    """
    try:
        df = pd.read_csv(tsv_path, sep='\t', header=0)
        
        # 检查必要的列
        required_columns = ['label', 'GT_code']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"TSV文件缺少必要的列: {col}")
        
        # 创建映射字典
        gt_code_to_class = {}
        class_id_to_name = {}
        
        # 按GT_code排序，确保类别ID连续
        df_sorted = df.sort_values('GT_code')
        
        for idx, (_, row) in enumerate(df_sorted.iterrows()):
            gt_code = int(row['GT_code'])
            label = str(row['label']).strip()
            
            # 使用GT_code作为类别ID，或者使用索引
            class_id = gt_code
            gt_code_to_class[gt_code] = class_id
            class_id_to_name[class_id] = label
        
        # 计算类别总数
        num_classes = max(class_id_to_name.keys()) + 1
        
        print(f"加载类别映射:")
        print(f"  类别总数: {num_classes}")
        print(f"  映射关系: {gt_code_to_class}")
        print(f"  类别名称: {class_id_to_name}")
        
        return gt_code_to_class, num_classes, class_id_to_name
        
    except Exception as e:
        print(f"加载TSV文件失败: {e}")
        raise

def apply_class_mapping(mask_array, gt_code_to_class):
    """
    将mask中的像素值映射到训练类别
    
    Args:
        mask_array: 原始mask数组
        gt_code_to_class: GT_code到class_id的映射字典
        
    Returns:
        np.ndarray: 映射后的mask数组
    """
    mapped_mask = np.zeros_like(mask_array)
    
    for gt_code, class_id in gt_code_to_class.items():
        mapped_mask[mask_array == gt_code] = class_id
    
    return mapped_mask

def create_directories(output_dir):
    """创建输出目录结构"""
    dirs = [
        os.path.join(output_dir, 'slide_tiles'),
        os.path.join(output_dir, 'label_tiles'),
        os.path.join(output_dir, 'train'),
        os.path.join(output_dir, 'val'),
        os.path.join(output_dir, 'configs')
    ]
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
    return dirs

def resize_mask_to_slide(mask_path, slide_dimensions, gt_code_to_class=None):
    """将mask调整到与slide相同的尺寸并应用类别映射"""
    try:
        # 使用cv2读取大图像，避免PIL的限制
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"无法读取mask文件: {mask_path}")
        
        # 调整到slide尺寸
        mask_resized = cv2.resize(mask, slide_dimensions, interpolation=cv2.INTER_NEAREST)
        mask_array = mask_resized
        
    except Exception as e:
        print(f"使用cv2读取失败，尝试使用PIL: {e}")
        try:
            # 如果cv2失败，使用PIL（已设置MAX_IMAGE_PIXELS = None）
            mask = Image.open(mask_path)
            mask_resized = mask.resize(slide_dimensions, Image.NEAREST)
            mask_array = np.array(mask_resized)
        except Exception as e2:
            print(f"PIL读取也失败: {e2}")
            raise ValueError(f"无法读取mask文件 {mask_path}: {e2}")
    
    # 如果提供了类别映射，则应用映射
    if gt_code_to_class is not None:
        mask_array = apply_class_mapping(mask_array, gt_code_to_class)
    
    return mask_array

def extract_patches_from_slide_and_mask(slide_path, mask_path, output_dir, 
                                      patch_size=1024, stride=256, 
                                      tissue_detection=True, min_tissue_ratio=0.1,
                                      gt_code_to_class=None):
    """
    从WSI和mask中提取patches
    
    Args:
        slide_path: SVS文件路径
        mask_path: PNG mask文件路径
        output_dir: 输出目录
        patch_size: patch大小
        stride: 步长
        tissue_detection: 是否使用tissue检测
        min_tissue_ratio: 最小tissue比例阈值
        gt_code_to_class: GT_code到class_id的映射字典
    """
    print(f"处理文件: {slide_path}")
    
    # 打开slide
    slide = openslide.OpenSlide(slide_path)
    slide_id = os.path.splitext(os.path.basename(slide_path))[0]
    
    # 加载并调整mask尺寸，应用类别映射
    mask = resize_mask_to_slide(mask_path, slide.dimensions, gt_code_to_class)
    
    # 获取tissue区域坐标
    if tissue_detection:
        print("使用tissue检测...")
        print(f"使用stride: {stride}, patch_size: {patch_size}")
        print(f"Mask尺寸: {mask.shape}")
        print(f"Slide尺寸: {slide.dimensions}")
        
        # 直接使用mask来判断tissue区域
        # mask已经调整到与slide相同的尺寸
        grid = []
        for x in range(0, slide.dimensions[0] - patch_size + 1, stride):
            for y in range(0, slide.dimensions[1] - patch_size + 1, stride):
                # 检查这个位置是否有足够的tissue（非背景）
                # 假设背景类别为0，其他类别都是tissue
                mask_region = mask[y:y+patch_size, x:x+patch_size]
                tissue_pixels = np.sum(mask_region > 0)  # 非背景像素
                tissue_ratio = tissue_pixels / (patch_size * patch_size)
                
                if tissue_ratio >= min_tissue_ratio:
                    grid.append((x, y))
        
        print(f"检测到 {len(grid)} 个tissue区域")
    else:
        print("使用规则网格...")
        # 使用规则网格
        grid = []
        for x in range(0, slide.dimensions[0] - patch_size + 1, stride):
            for y in range(0, slide.dimensions[1] - patch_size + 1, stride):
                grid.append((x, y))
        print(f"生成 {len(grid)} 个网格点")
    
    # 提取patches
    valid_patches = []
    patch_count = 0
    
    for i, (x, y) in enumerate(tqdm(grid, desc="提取patches")):
        try:
            # 确保坐标在slide范围内
            if x + patch_size > slide.dimensions[0] or y + patch_size > slide.dimensions[1]:
                continue
                
            # 从slide中读取patch
            patch = slide.read_region((x, y), 0, (patch_size, patch_size))
            patch = patch.convert('RGB')
            patch_array = np.array(patch)
            
            # 从mask中提取对应区域
            mask_patch = mask[y:y+patch_size, x:x+patch_size]
            
            # 检查tissue比例（如果启用tissue检测）
            if tissue_detection:
                # 计算非零像素比例
                tissue_ratio = np.count_nonzero(mask_patch) / (patch_size * patch_size)
                if tissue_ratio < min_tissue_ratio:
                    continue
            
            # 保存patch图片
            patch_filename = f"{slide_id}_{x}_{y}.jpg"
            patch_path = os.path.join(output_dir, 'slide_tiles', patch_filename)
            cv2.imwrite(patch_path, cv2.cvtColor(patch_array, cv2.COLOR_RGB2BGR))
            
            # 保存mask
            mask_filename = f"{slide_id}_{x}_{y}.png"
            mask_patch_path = os.path.join(output_dir, 'label_tiles', mask_filename)
            cv2.imwrite(mask_patch_path, mask_patch)
            
            valid_patches.append({
                'slide_id': slide_id,
                'x': x,
                'y': y,
                'patch_file': patch_filename,
                'mask_file': mask_filename
            })
            
            patch_count += 1
            
        except Exception as e:
            print(f"处理patch ({x}, {y}) 时出错: {e}")
            continue
    
    print(f"成功提取 {patch_count} 个有效patches")
    return valid_patches

def create_file_lists(patches, output_dir, train_ratio=0.8):
    """创建训练和验证文件列表"""
    # 按slide_id分组
    slide_patches = {}
    for patch in patches:
        slide_id = patch['slide_id']
        if slide_id not in slide_patches:
            slide_patches[slide_id] = []
        slide_patches[slide_id].append(patch)
    
    # 分割训练和验证集
    train_patches = []
    val_patches = []
    
    for slide_id, slide_patch_list in slide_patches.items():
        # 随机打乱
        np.random.shuffle(slide_patch_list)
        
        # 分割
        split_idx = int(len(slide_patch_list) * train_ratio)
        train_patches.extend(slide_patch_list[:split_idx])
        val_patches.extend(slide_patch_list[split_idx:])
    
    # 保存文件列表
    train_file = os.path.join(output_dir, 'train_tiles.txt')
    val_file = os.path.join(output_dir, 'val_tiles.txt')
    
    with open(train_file, 'w') as f:
        for patch in train_patches:
            f.write(f"{output_dir},{patch['patch_file'].replace('.jpg', '')}\n")
    
    with open(val_file, 'w') as f:
        for patch in val_patches:
            f.write(f"{output_dir},{patch['patch_file'].replace('.jpg', '')}\n")
    
    print(f"训练集: {len(train_patches)} patches")
    print(f"验证集: {len(val_patches)} patches")
    print(f"文件列表已保存到: {train_file}, {val_file}")
    
    return train_patches, val_patches

def generate_training_config(output_dir, num_classes, class_id_to_name, 
                           base_config_path=None):
    """
    生成训练配置文件
    
    Args:
        output_dir: 输出目录
        num_classes: 类别总数
        class_id_to_name: 类别ID到名称的映射
        base_config_path: 基础配置文件路径（可选）
    """
    config = {
        'model': {
            'arch': 'DMMN',
            'n_classes': num_classes,
            'save_dir': 'runs/DMMN-OST',
            'save_name': 'DMMN_ost_seg'
        },
        'data': {
            'dataset': 'custom',
            'train_file': 'train_tiles.txt',
            'val_file': 'val_tiles.txt',
            'n_classes': num_classes,
            'train_split': 'train_aug',
            'val_split': 'val',
            'img_rows': 'same',
            'img_cols': 'same',
            'path': ''
        },
        'training': {
            'batch_size': 4,
            'train_iters': 10000,
            'val_interval': 100,
            'print_interval': 10,
            'resume': None,
            'optimizer': {
                'name': 'SGD',
                'lr': 0.01,
                'momentum': 0.9,
                'weight_decay': 0.0005
            },
            'lr_schedule': {
                'name': 'poly',
                'power': 0.9
            },
            'wandb': {
                'enabled': True,
                'project': 'OST-DMMN',
                'name': 'DMMN-OST-Experiment',
                'tags': ['OST', 'DMMN', 'segmentation'],
                'notes': 'Osteosarcoma tissue segmentation using DMMN'
            }
        },
        'loss': {
            'name': 'cross_entropy'
        },
        'classes': class_id_to_name
    }
    
    # 如果提供了基础配置文件，则加载并更新
    if base_config_path and os.path.exists(base_config_path):
        try:
            with open(base_config_path, 'r') as f:
                base_config = yaml.safe_load(f)
            # 更新类别相关配置
            base_config['model']['n_classes'] = num_classes
            base_config['data']['n_classes'] = num_classes
            base_config['classes'] = class_id_to_name
            config = base_config
            print(f"基于 {base_config_path} 生成配置文件")
        except Exception as e:
            print(f"加载基础配置文件失败: {e}")
            print("使用默认配置")
    
    # 保存配置文件
    config_path = os.path.join(output_dir, 'configs', 'training_config.yml')
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    # 保存类别映射JSON文件
    mapping_path = os.path.join(output_dir, 'configs', 'class_mapping.json')
    with open(mapping_path, 'w', encoding='utf-8') as f:
        json.dump(class_id_to_name, f, ensure_ascii=False, indent=2)
    
    print(f"配置文件已保存到: {config_path}")
    print(f"类别映射已保存到: {mapping_path}")
    
    return config_path

def main():
    parser = argparse.ArgumentParser(description='处理WSI和PNG mask数据')
    parser.add_argument('--wsi_dir', type=str, required=True, help='WSI文件目录')
    parser.add_argument('--mask_dir', type=str, required=True, help='PNG mask文件目录')
    parser.add_argument('--output_dir', type=str, required=True, help='输出目录')
    parser.add_argument('--tsv_file', type=str, help='TSV文件路径，包含mask_id到class_id的映射')
    parser.add_argument('--base_config', type=str, help='基础配置文件路径（可选）')
    parser.add_argument('--patch_size', type=int, default=1024, help='patch大小')
    parser.add_argument('--stride', type=int, default=256, help='步长')
    parser.add_argument('--tissue_detection', action='store_true', help='使用tissue检测')
    parser.add_argument('--min_tissue_ratio', type=float, default=0.1, help='最小tissue比例')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='训练集比例')
    
    args = parser.parse_args()
    
    # 创建输出目录
    create_directories(args.output_dir)
    
    # 加载类别映射（如果提供了TSV文件）
    gt_code_to_class = None
    num_classes = 7  # 默认类别数
    class_id_to_name = {}
    
    if args.tsv_file:
        if not os.path.exists(args.tsv_file):
            print(f"错误: TSV文件不存在: {args.tsv_file}")
            return
        
        try:
            gt_code_to_class, num_classes, class_id_to_name = load_class_mapping(args.tsv_file)
            print(f"成功加载类别映射，类别总数: {num_classes}")
        except Exception as e:
            print(f"加载TSV文件失败: {e}")
            return
    else:
        print("未提供TSV文件，使用默认类别设置")
    
    # 获取所有WSI文件
    wsi_files = []
    for ext in ['*.svs', '*.SVS']:
        wsi_files.extend(Path(args.wsi_dir).glob(ext))
    
    if not wsi_files:
        print(f"在 {args.wsi_dir} 中未找到SVS文件")
        return
    
    print(f"找到 {len(wsi_files)} 个WSI文件")
    
    all_patches = []
    
    # 处理每个WSI文件
    for wsi_path in wsi_files:
        wsi_name = wsi_path.stem
        mask_path = Path(args.mask_dir) / f"{wsi_name}.png"
        
        if not mask_path.exists():
            print(f"警告: 未找到对应的mask文件 {mask_path}")
            continue
        
        patches = extract_patches_from_slide_and_mask(
            str(wsi_path), str(mask_path), args.output_dir,
            patch_size=args.patch_size,
            stride=args.stride,
            tissue_detection=args.tissue_detection,
            min_tissue_ratio=args.min_tissue_ratio,
            gt_code_to_class=gt_code_to_class
        )
        
        all_patches.extend(patches)
    
    if not all_patches:
        print("未提取到任何有效patches")
        return
    
    # 创建文件列表
    create_file_lists(all_patches, args.output_dir, args.train_ratio)
    
    # 生成训练配置文件
    if gt_code_to_class is not None:
        generate_training_config(
            args.output_dir, 
            num_classes, 
            class_id_to_name, 
            args.base_config
        )
    
    print("数据处理完成！")
    print(f"总共提取了 {len(all_patches)} 个patches")
    print(f"输出目录: {args.output_dir}")
    if gt_code_to_class is not None:
        print(f"类别总数: {num_classes}")
        print(f"配置文件已生成，可直接用于训练")

if __name__ == "__main__":
    main()
