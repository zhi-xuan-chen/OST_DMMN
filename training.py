import os
import sys
import yaml
import time
import shutil
import torch
import random
import argparse
import datetime
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
from torch.utils import data
from tqdm import tqdm
from ptsemseg.models import get_model
from ptsemseg.loss import get_loss_function
from ptsemseg.loader import get_loader 
from ptsemseg.utils import get_logger
from ptsemseg.metrics import runningScore, averageMeter
from ptsemseg.augmentations import get_composed_augmentations
from ptsemseg.schedulers import get_scheduler
from ptsemseg.optimizers import get_optimizer
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from pdb import set_trace
from dataset import RoboticsDataset
import wandb
from albumentations import (
    HorizontalFlip,
    VerticalFlip,
    RandomRotate90,
    Normalize,
    Compose,
    PadIfNeeded,
    RandomCrop,
    CenterCrop,
    HueSaturationValue,
    RandomBrightnessContrast,
    ElasticTransform,
)
import pandas as pd
import seaborn as sns
import time
from datetime import datetime
import re
from itertools import chain
from torch.nn import Module, Conv2d, ConvTranspose2d
import torch.nn.init as init
from torch.autograd import Variable
import cv2

def count_conv2d(module: Module):
    """
    Counts the number of convolutions and transposed convolutions in a Module
    """
    return len([m for m in module.modules() if isinstance(m, Conv2d) or isinstance(m, ConvTranspose2d)])

def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        # init.xavier_uniform(m.weight, gain=np.sqrt(2.0))
        init.kaiming_normal_(m.weight, nonlinearity='relu')
        init.constant(m.bias,0.0)

# 采用用户提供的可视化类别与 BGR 颜色定义
ANNOTATION_CLASSES = {
    1: {
        'cn': 'ID1 红色 无骨活肿瘤',
        'en': 'Non-Bone Active Tumor',
        'color': (0, 0, 255)  # BGR 红
    },
    2: {
        'cn': 'ID2 橙色 有骨活肿瘤',
        'en': 'Bone Active Tumor',
        'color': (0, 165, 255)  # BGR 橙
    },
    3: {
        'cn': 'ID3 绿色 无骨坏死',
        'en': 'Non-Bone Necrosis',
        'color': (0, 255, 0)  # BGR 绿
    },
    4: {
        'cn': 'ID4 深绿 有骨坏死',
        'en': 'Bone Necrosis',
        'color': (0, 100, 0)  # BGR 深绿
    },
    5: {
        'cn': 'ID5 蓝色 正常组织',
        'en': 'Normal Tissue',
        'color': (255, 0, 0)  # BGR 蓝
    },
    6: {
        'cn': 'ID6 黄色 背景',
        'en': 'Background',
        'color': (0, 255, 255)  # BGR 黄
    },
    7: {
        'cn': 'ID7 紫色 细胞稀疏区',
        'en': 'Sparse Cellular Area',
        'color': (255, 0, 255)  # BGR 紫
    }
}

def build_palette_from_annotation_classes(num_classes: int):
    """
    构造可视化调色板：索引即类别 id。
    输入为 BGR，输出转换为 RGB；0 类保留为黑色（未标注）。
    """
    palette = [(0, 0, 0)] * max(1, num_classes)
    for cls_id, meta in ANNOTATION_CLASSES.items():
        if 0 <= cls_id < num_classes:
            b, g, r = meta.get('color', (0, 0, 0))
            palette[cls_id] = (r, g, b)  # 转成 RGB 供可视化
    return palette

def colorize_mask(mask: np.ndarray, palette: list):
    """
    Convert a HxW integer mask to an RGB image using a given palette.
    palette: list of (R, G, B) for class indices [0..n_classes-1]
    """
    h, w = mask.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    max_cls = min(len(palette) - 1, int(mask.max()))
    for cls_id in range(max_cls + 1):
        color[mask == cls_id] = palette[cls_id]
    return color

def to_uint8_rgb_image(chw_tensor: torch.Tensor):
    """
    Convert CHW float tensor (0..1 or 0..255, likely BGR from cv2) to HWC uint8 RGB image.
    """
    arr = chw_tensor.detach().cpu().numpy()
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    # scale to 0..1
    maxv = float(arr.max()) if arr.size else 1.0
    arr = arr / 255.0 if maxv > 1.5 else arr
    arr = np.clip(arr, 0.0, 1.0)
    arr = (arr * 255.0).astype(np.uint8)
    # BGR -> RGB (cv2 read)
    if arr.shape[-1] == 3:
        arr = arr[..., ::-1]
    return arr

def make_sample_panel(img_rgb: np.ndarray, gt_mask_rgb: np.ndarray, pred_mask_rgb: np.ndarray):
    """
    Concatenate original image, GT mask (RGB), and prediction mask (RGB) horizontally.
    Ensures same H,W by center-cropping to the smallest shape if needed.
    """
    h = min(img_rgb.shape[0], gt_mask_rgb.shape[0], pred_mask_rgb.shape[0])
    w = min(img_rgb.shape[1], gt_mask_rgb.shape[1], pred_mask_rgb.shape[1])
    def cc(x):
        hh, ww = x.shape[:2]
        sy = max(0, (hh - h) // 2)
        sx = max(0, (ww - w) // 2)
        return x[sy:sy+h, sx:sx+w]
    img_c = cc(img_rgb)
    gt_c = cc(gt_mask_rgb)
    pred_c = cc(pred_mask_rgb)
    panel = np.concatenate([img_c, gt_c, pred_c], axis=1)
    return panel

def train(cfg, writer, logger, config_path=None):
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Setup seeds
    torch.manual_seed(cfg.get('seed', 999))
    np.random.seed(999)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(999)

    # 从配置文件读取类别数量
    n_classes = cfg['data'].get('n_classes', 7)
    logger.info("n_classes: {}".format(n_classes))
    
    # 初始化wandb
    if cfg['training'].get('wandb', {}).get('enabled', False):
        wandb_config = cfg['training']['wandb']
        wandb.init(
            project=wandb_config.get('project', 'OST-DMMN'),
            name=wandb_config.get('name', 'DMMN-OST-Experiment'),
            tags=wandb_config.get('tags', ['OST', 'DMMN', 'segmentation']),
            notes=wandb_config.get('notes', 'Osteosarcoma tissue segmentation using DMMN'),
            entity=wandb_config.get('entity', None),
            config=cfg
        )
        logger.info("Wandb initialized successfully")
    
    # 获取模型保存路径配置
    model_save_dir = cfg['model'].get('save_dir', 'runs/DMMN-OST')
    model_save_name = cfg['model'].get('save_name', 'DMMN_ost_seg')
    
    # 创建模型保存目录
    os.makedirs(model_save_dir, exist_ok=True)
    logger.info(f"Model will be saved to: {model_save_dir}")
    
    tile_size = 1024
    problem_type = 'tissue'
    workers = 12
    batch_size = cfg['training']['batch_size']
    device_ids='0'
    
    fileEpochLoss = open(os.path.join(writer.file_writer.get_logdir(),'epoch_loss_train_seg.txt'),'w')
    fileEpochLossVal = open(os.path.join(writer.file_writer.get_logdir(),'epoch_loss_val_seg.txt'),'w')
    fileLR = open(os.path.join(writer.file_writer.get_logdir(),'lr.txt'),'w')
    fileMiou = open(os.path.join(writer.file_writer.get_logdir(),'miou.txt'),'w')

    
    def make_loader(file_names, shuffle=False, transform=None, problem_type='tissue', batch_size=batch_size):
        return DataLoader(
            dataset=RoboticsDataset(file_names, transform=transform, problem_type=problem_type),
            shuffle=shuffle,
            num_workers=workers,
            batch_size=batch_size,
            pin_memory=torch.cuda.is_available())
    
    def train_transform(p=1):
        return Compose([
            RandomCrop(height=tile_size, width=tile_size, p=1),
            RandomRotate90(p=0.5),
            VerticalFlip(p=0.5),
            HorizontalFlip(p=0.5),
            HueSaturationValue(hue_shift_limit=20,sat_shift_limit=30,val_shift_limit=0,p=0.5),
            RandomBrightnessContrast(brightness_limit=(-0.25,0.25),contrast_limit=(0.25,1.75),p=0.5),
            ElasticTransform(alpha=1,sigma=4),
            # Normalize(p=1)
        ], p=p)

    def val_transform(p=1):
        return Compose([
            CenterCrop(height=tile_size, width=tile_size, p=1),
            # Normalize(p=1)
        ], p=p)
    
    # 从配置文件读取数据路径
    train_file = cfg['data'].get('train_file', 'train_tiles.txt')
    val_file = cfg['data'].get('val_file', 'val_tiles.txt')
    
    logger.info(f"Loading training data from: {train_file}")
    logger.info(f"Loading validation data from: {val_file}")
    
    with open(train_file) as f:
        train_file_names = [line.rstrip('\n') for line in f]

    with open(val_file) as f:
        val_file_names = [line.rstrip('\n') for line in f]
        
    trainloader = make_loader(train_file_names, shuffle=True, transform=train_transform(p=1), problem_type=problem_type,
                           batch_size=batch_size)
    valloader = make_loader(val_file_names, transform=val_transform(p=1), problem_type=problem_type,
                               batch_size=len(device_ids))

    num_train_files = len(train_file_names)
    num_val_files = len(val_file_names)
    logger.info('num_train = {}, num_val = {}'.format(num_train_files, num_val_files))
    
    # 计算每个epoch的iteration数
    # 更精确：每个 epoch 的迭代数取决于 DataLoader 长度
    iterations_per_epoch = max(1, len(trainloader))
    logger.info('iterations_per_epoch = {}'.format(iterations_per_epoch))
    
    # 支持epoch和iteration两种设置方式
    if 'epochs' in cfg['training']:
        # 使用epoch设置
        total_epochs = cfg['training']['epochs']
        train_iters = total_epochs * iterations_per_epoch
        logger.info('Training for {} epochs ({} iterations)'.format(total_epochs, train_iters))
    else:
        # 使用iteration设置（向后兼容）
        train_iters = cfg['training']['train_iters']
        total_epochs = train_iters // iterations_per_epoch
        logger.info('Training for {} iterations ({} epochs)'.format(train_iters, total_epochs))
    
    # 更新配置中的train_iters用于后续使用
    cfg['training']['train_iters'] = train_iters

    # Setup Metrics
    running_metrics_val = runningScore(n_classes)

    # Setup Model
    # 仅将 arch 传递给模型工厂，避免把 save_dir/save_name 等无关键传入构造函数
    model_cfg = { 'arch': cfg['model'].get('arch', 'DMMN') }
    model = get_model(model_cfg, n_classes).to(device)
    model.apply(init_weights)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    logger.info("pytorch_total_params {}".format(pytorch_total_params))
    pytorch_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("pytorch_trainable_params {}".format(pytorch_trainable_params))
    logger.info("Model Layers {}".format(count_conv2d(model)))
    

    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))

    # Setup optimizer, lr_scheduler and loss function
    # 兼容大小写：将优化器名称统一为小写
    try:
        if 'training' in cfg and 'optimizer' in cfg['training'] and 'name' in cfg['training']['optimizer']:
            cfg['training']['optimizer']['name'] = str(cfg['training']['optimizer']['name']).lower()
    except Exception:
        pass
    optimizer_cls = get_optimizer(cfg)
    optimizer_params = {k:v for k, v in cfg['training']['optimizer'].items() 
                        if k != 'name'}

    optimizer = optimizer_cls(model.parameters(), **optimizer_params)
    logger.info("Using optimizer {}".format(optimizer))

    # 规范化与兼容 lr_schedule；若为 poly 则内置实现，避免 KeyError
    try:
        if 'training' in cfg and 'lr_schedule' in cfg['training'] and 'name' in cfg['training']['lr_schedule']:
            cfg['training']['lr_schedule']['name'] = str(cfg['training']['lr_schedule']['name']).lower()
    except Exception:
        pass

    scheduler = None
    lr_sched_cfg = cfg['training'].get('lr_schedule', {})
    sched_name = lr_sched_cfg.get('name', '')
    if sched_name == 'poly':
        power = float(lr_sched_cfg.get('power', 0.9))
        total_iters = int(cfg['training'].get('train_iters', 10000))
        def poly_lambda(step_idx):
            # 防止负值
            return max(0.0, (1.0 - float(step_idx) / float(total_iters)) ** power)
        from torch.optim.lr_scheduler import LambdaLR
        scheduler = LambdaLR(optimizer, lr_lambda=poly_lambda)
        logger.info(f"Using built-in poly LR scheduler: power={power}, total_iters={total_iters}")
    else:
        scheduler = get_scheduler(optimizer, cfg['training']['lr_schedule'])

    loss_fn = get_loss_function(cfg)
    logger.info("Using loss {}".format(loss_fn))

    # weighted cross entropy
    # Note class 0 is unannotated regions and will not contribute to the loss function.
    # 从配置文件读取类别权重；若未提供，则从训练集mask中统计计算
    if 'class_weights' in cfg['data']:
        # 确保为纯Python float，避免YAML中出现numpy标记
        class_weights = [float(w) for w in cfg['data']['class_weights']]
        logger.info("Using class weights from config: {}".format(class_weights))
    else:
        logger.info("Computing class weights from training masks ...")
        counts = np.zeros(n_classes, dtype=np.int64)
        for name in tqdm(train_file_names, desc="Counting class pixels"):
            try:
                parts = name.split(",")
                # 构造对应的mask路径：{output_dir}/label_tiles/{tile_name}.png
                mask_path = os.path.join(parts[0], "label_tiles", parts[1] + ".png")
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    raise ValueError("cv2.imread returned None")
                binc = np.bincount(mask.reshape(-1), minlength=n_classes)
                # 限制到前n_classes个类别
                counts += binc[:n_classes]
            except Exception as e:
                logger.info(f"Failed to read mask for {name}: {e}")
                continue

        sum_foreground = int(counts[1:].sum())
        if sum_foreground == 0:
            raise ValueError("No foreground pixels found when computing class weights. Please check your masks or configuration.")

        class_weights = [0.0] * n_classes
        for k in range(1, n_classes):
            class_weights[k] = 1.0 - (counts[k] / sum_foreground)
        # 转为纯Python float，避免yaml保存numpy scalar
        class_weights = [float(w) for w in class_weights]
        logger.info("Computed class weights from data: {}".format(class_weights))
        
        # 将计算好的class_weights保存到配置文件，避免重复计算
        try:
            if config_path is None:
                config_path = 'configs/DMMN-OST.yml'
            if os.path.exists(config_path):
                # 读取现有配置
                with open(config_path, 'r', encoding='utf-8') as f:
                    config_data = yaml.safe_load(f)
                
                # 更新class_weights
                if 'data' not in config_data:
                    config_data['data'] = {}
                config_data['data']['class_weights'] = class_weights
                
                # 保存更新后的配置
                with open(config_path, 'w', encoding='utf-8') as f:
                    yaml.safe_dump(config_data, f, default_flow_style=False, allow_unicode=True)
                
                logger.info(f"Updated class_weights in config file: {config_path}")
            else:
                logger.info(f"Config file not found: {config_path}, skipping class_weights update")
        except Exception as e:
            logger.warning(f"Failed to update config file with class_weights: {e}")

    start_time=datetime.now()
    start_iter = 0
    if cfg['training']['resume'] is not None:
        if os.path.isfile(cfg['training']['resume']):
            logger.info(
                "Loading model and optimizer from checkpoint '{}'".format(cfg['training']['resume'])
            )
            checkpoint = torch.load(cfg['training']['resume'])
            model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            start_iter = checkpoint["epoch"]
            logger.info(
                "Loaded checkpoint '{}' (iter {})".format(
                    cfg['training']['resume'], checkpoint["epoch"]
                )
            )
        else:
            logger.info("No checkpoint found at '{}'".format(cfg['training']['resume']))

    train_loss_meter = averageMeter()
    val_loss_meter = averageMeter()
    time_meter = averageMeter()

    best_iou = -100.0
    i = start_iter
    flag = True
    
    # 美化训练进度显示：全局 tqdm 进度（按 iteration 计）
    pbar = tqdm(total=cfg['training']['train_iters'], initial=i, dynamic_ncols=True, desc='Train', leave=True)
    while i <= cfg['training']['train_iters'] and flag:
        for (images, labels) in trainloader:
            i += 1
            
            images_20x = images[:,:,384:640,384:640]
            images_10x = images[:,:,::2,::2]
            images_10x = images_10x[:,:,128:384,128:384]
            images_5x = images[:,:,::4,::4]
            labels_20x = labels[:,384:640,384:640]

            start_ts = time.time()
            scheduler.step()
            model.train()
            images_20x = images_20x.to(device)
            images_10x = images_10x.to(device)
            images_5x = images_5x.to(device)
            labels_20x = labels_20x.to(device)

            images_20x, images_10x, images_5x, labels_20x = Variable(images_20x), Variable(images_10x), Variable(images_5x), Variable(labels_20x)

            outputs = model(images_20x, images_10x, images_5x)
            
            loss = loss_fn(input=outputs, target=labels_20x, class_weights=class_weights)

            optimizer.zero_grad()

            loss.backward()
            optimizer.step()
                
            time_meter.update(time.time() - start_ts)
            train_loss_meter.update(loss.item())

            # tqdm 进度更新与后缀展示（每步）
            epoch_float = (i + 1) / float(iterations_per_epoch)
            pbar.update(1)
            pbar.set_postfix({
                'epoch': f"{epoch_float:.3f}",
                'iter': f"{i+1}/{cfg['training']['train_iters']}",
                'loss': f"{loss.item():.4f}",
                'lr': f"{optimizer.param_groups[0]['lr']:.2e}",
                't/img(s)': f"{time_meter.avg / cfg['training']['batch_size']:.4f}"
            })

            # 控制打印/记录频率
            if (i + 1) % cfg['training']['print_interval'] == 0:
                writer.add_scalar('loss/train_loss', loss.item(), i+1)

                # 记录到 wandb（仅必要信息，epoch 为随 step 变化的曲线）
                if cfg['training'].get('wandb', {}).get('enabled', False):
                    wandb.log({
                        'train/loss': loss.item(),
                        'train/avg_loss': train_loss_meter.avg,
                        'train/lr': optimizer.param_groups[0]['lr'],
                        'epoch': epoch_float,
                        'iteration': i + 1,
                        'time/per_batch_s': time_meter.avg / cfg['training']['batch_size'],
                        'gpu/mem_allocated_gb': torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0,
                        'gpu/mem_reserved_gb': torch.cuda.memory_reserved() / 1024**3 if torch.cuda.is_available() else 0
                    }, step=i+1)

                time_meter.reset()
                logger.info("avg train loss: " + str(train_loss_meter.avg))
                fileEpochLoss.write(str(train_loss_meter.avg))
                fileEpochLoss.write('\n')
                train_loss_meter.reset()
                for param_group in optimizer.param_groups:
                    logger.info('current_lr  {}'.format(param_group['lr']))
                    fileLR.write(str(param_group['lr']))
                    fileLR.write('\n')

            if (i + 1) % cfg['training']['val_interval'] == 0 or \
               (i + 1) == cfg['training']['train_iters']:
                model.eval()
                sample_panels = []
                max_samples_to_log = cfg['training'].get('wandb', {}).get('num_val_samples', 4)
                num_val_total = 0
                num_val_used = 0
                num_val_bg_only = 0
                with torch.no_grad():
                    for i_val, (images_val, labels_val) in tqdm(enumerate(valloader), total=len(valloader), dynamic_ncols=True, desc='Val', leave=False):

                        images_20x_val = images_val[:,:,384:640,384:640]
                        images_10x_val = images_val[:,:,::2,::2]
                        images_10x_val = images_10x_val[:,:,128:384,128:384]
                        images_5x_val = images_val[:,:,::4,::4]
                        labels_20x_val = labels_val[:,384:640,384:640]

                        images_20x_val = images_20x_val.to(device)
                        images_10x_val = images_10x_val.to(device)
                        images_5x_val = images_5x_val.to(device)
                        labels_20x_val = labels_20x_val.to(device)

                        num_val_total += 1
                        # 若该 batch 全部为未标注（class 0），则跳过计算损失和指标
                        has_fg = (labels_20x_val > 0).any().item()
                        if not has_fg:
                            num_val_bg_only += 1
                            continue

                        outputs = model(images_20x_val,images_10x_val,images_5x_val)
                        val_loss = loss_fn(input=outputs, target=labels_20x_val, class_weights=class_weights)

                        pred = outputs.data.max(1)[1].cpu().numpy()
                        gt = labels_20x_val.data.cpu().numpy()

                        running_metrics_val.update(gt, pred)
                        val_loss_meter.update(val_loss.item())
                        num_val_used += 1

                        # 收集样例：仅前 max_samples_to_log 个 batch
                        if cfg['training'].get('wandb', {}).get('enabled', False) and len(sample_panels) < max_samples_to_log:
                            try:
                                # 还原原图（注意训练时做了裁剪与缩放，这里取与 labels 对齐的中心 256 区域）
                                # images_val: N,C,H,W；提取与 labels_20x_val 对齐区域
                                img_20x = images_val[:,:,384:640,384:640][0]
                                img_rgb = to_uint8_rgb_image(img_20x)

                                # 取第一个样本的 GT 与预测
                                gt_mask = gt[0].astype(np.uint8)
                                pred_mask = pred[0].astype(np.uint8)

                                # 使用用户提供的 BGR 配色并转换为 RGB
                                palette = build_palette_from_annotation_classes(n_classes)
                                gt_rgb = colorize_mask(gt_mask, palette)
                                pred_rgb = colorize_mask(pred_mask, palette)

                                panel = make_sample_panel(img_rgb, gt_rgb, pred_rgb)
                                sample_panels.append(panel)
                            except Exception as _:
                                pass

                writer.add_scalar('loss/val_loss', val_loss_meter.avg, i+1)
                logger.info("Iter %d Loss: %.4f" % (i + 1, val_loss_meter.avg))
                fileEpochLossVal.write(str(val_loss_meter.avg))
                fileEpochLossVal.write('\n')

                score, class_iou, hist, mean_iu, recalls, precisions, average_recall, average_precision = running_metrics_val.get_scores()
                
                # 记录验证指标到 wandb（带连续 epoch 曲线）
                if cfg['training'].get('wandb', {}).get('enabled', False):
                    epoch_float = (i + 1) / float(iterations_per_epoch)
                    wandb_metrics = {
                        'val/loss': val_loss_meter.avg,
                        'val/mean_iou': mean_iu,
                        'val/average_recall': average_recall,
                        'val/average_precision': average_precision,
                        'iteration': i + 1,
                        'best_iou': best_iou,
                        'epoch': epoch_float,
                        'data/val_samples': num_val_files,
                        'data/train_samples': num_train_files,
                        'val/num_batches_total': num_val_total,
                        'val/num_batches_used': num_val_used,
                        'val/num_batches_bg_only': num_val_bg_only
                    }

                    # 添加每个类别的 IoU
                    for k, v in class_iou.items():
                        wandb_metrics[f'val/class_{k}_iou'] = v

                    # 添加其他指标
                    for k, v in score.items():
                        wandb_metrics[f"val/{k.replace(' ', '_').replace(':', '').lower()}"] = v

                    # 添加类别权重信息
                    if 'class_weights' in cfg['data']:
                        for idx, weight in enumerate(cfg['data']['class_weights']):
                            wandb_metrics[f'data/class_weight_{idx}'] = weight

                    # 追加样例图
                    if len(sample_panels) > 0:
                        # 将样例拼为网格或逐张上传（逐张更清晰）
                        wandb_images = [wandb.Image(p, caption=f"val_sample_{idx}") for idx, p in enumerate(sample_panels)]
                        wandb_metrics['val/samples'] = wandb_images

                    wandb.log(wandb_metrics, step=i+1)
                
                for k, v in score.items():
                    print(k, v)
                    logger.info('{}: {}'.format(k, v))
                    writer.add_scalar('val_metrics/{}'.format(k), v, i+1)

                for k, v in class_iou.items():
                    logger.info('{}: {}'.format(k, v))
                    writer.add_scalar('val_metrics/cls_{}'.format(k), v, i+1)

                val_loss_meter.reset()
                running_metrics_val.reset()
                fileMiou.write(str(mean_iu))
                fileMiou.write('\n')
                np.savetxt(os.path.join(writer.file_writer.get_logdir(), "hist.csv"), hist, delimiter=",")
                logger.info('recalls  {}'.format(recalls))
                logger.info('average_recall  {}'.format(average_recall))
                logger.info('precisions  {}'.format(precisions))
                logger.info('average_precision  {}'.format(average_precision))
                
                # 计算训练时间统计
                elapsed_time = datetime.now() - start_time
                total_hours = elapsed_time.total_seconds() / 3600
                
                logger.info('time since start = {}'.format(elapsed_time))
                logger.info('total training time = {:.2f} hours'.format(total_hours))
                
                # 打印验证结果摘要
                current_epoch = (i + 1) // iterations_per_epoch + 1
                print(f"\n{'='*60}")
                print(f"Validation Results (Epoch {current_epoch}/{total_epochs}, Iter {i+1})")
                print(f"{'='*60}")
                print(f"Val Loss: {val_loss_meter.avg:.4f}")
                print(f"Mean IoU: {mean_iu:.4f}")
                print(f"Best IoU: {best_iou:.4f}")
                print(f"Training Time: {total_hours:.2f}h")
                print(f"{'='*60}\n")

                if score["Mean IoU : \t"] >= best_iou:
                    best_iou = score["Mean IoU : \t"]
                    state = {
                        "epoch": i + 1,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "scheduler_state": scheduler.state_dict(),
                        "best_iou": best_iou,
                    }
                    # 使用配置文件中的保存路径
                    save_path = os.path.join(model_save_dir,
                                             f"{model_save_name}_best_model.pkl")
                    torch.save(state, save_path)
                    logger.info(f'Model saved to: {save_path}')
                    logger.info('current_best_iou_value  {}'.format(best_iou))
                    
                    # 记录到 wandb（epoch 为连续小数）
                    if cfg['training'].get('wandb', {}).get('enabled', False):
                        epoch_float = (i + 1) / float(iterations_per_epoch)
                        wandb.log({
                            'best_iou': best_iou,
                            'epoch': epoch_float,
                            'artifact/model_save_path': save_path
                        }, step=i+1)

            if (i + 1) == cfg['training']['train_iters']:
                flag = False
                break
    
    # 训练结束，关闭wandb
    if cfg['training'].get('wandb', {}).get('enabled', False):
        wandb.finish()
        logger.info("Wandb session finished")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="configs/DMMN-OST.yml",
        help="Configuration file to use"
    )

    args = parser.parse_args()

    with open(args.config) as fp:
        cfg = yaml.safe_load(fp)

    timestr = time.strftime("%Y%m%d_%H%M%S")
    run_id = random.randint(1,100000)
    folder_name = str(timestr) + "_" +str(cfg['model']['arch'])
    print(folder_name)
    logdir = os.path.join('runs', os.path.basename(args.config)[:-4] , str(folder_name))
    print(logdir)
    writer = SummaryWriter(log_dir=logdir)

    print('RUNDIR: {}'.format(logdir))
    shutil.copy(args.config, logdir)

    logger = get_logger(logdir)
    logger.info('Start training:')

    train(cfg, writer, logger, args.config)
