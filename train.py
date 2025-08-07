#!/usr/bin/env python3
"""
SCL-YOLO Training Script
========================

Training script for SCL-YOLO blood cell detection model.

Authors: Yang Fu, Yong Hong Wu
Department of Statistics, Wuhan University of Technology
"""

import argparse
import os
import sys
from pathlib import Path
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from models.scl_yolo import SCL_YOLO
from utils.datasets import BloodCellDataset, create_dataloader
from utils.general import (
    check_img_size, check_requirements, colorstr, increment_path,
    init_seeds, one_cycle, labels_to_class_weights, strip_optimizer
)
from utils.metrics import fitness, ap_per_class
from utils.plots import plot_lr_scheduler, plot_results
from utils.torch_utils import ModelEMA, select_device, torch_distributed_zero_first


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train SCL-YOLO')
    
    # Model parameters
    parser.add_argument('--config', type=str, default='configs/scl_yolo.yaml',
                       help='model configuration file')
    parser.add_argument('--data', type=str, default='datasets/TXL_PBC.yaml',
                       help='dataset configuration file')
    parser.add_argument('--weights', type=str, default='',
                       help='initial weights path')
    parser.add_argument('--resume', type=str, default='',
                       help='resume from checkpoint')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100,
                       help='number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='batch size')
    parser.add_argument('--img-size', type=int, default=640,
                       help='input image size')
    parser.add_argument('--device', default='',
                       help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=8,
                       help='number of dataloader workers')
    
    # Optimizer parameters
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam', 'AdamW'],
                       default='AdamW', help='optimizer type')
    parser.add_argument('--lr', type=float, default=0.01,
                       help='initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.937,
                       help='SGD momentum')
    parser.add_argument('--weight-decay', type=float, default=0.0005,
                       help='weight decay')
    
    # Training options
    parser.add_argument('--cache', type=str, nargs='?', const='ram',
                       help='image cache mode (ram/disk)')
    parser.add_argument('--quad', action='store_true',
                       help='quad dataloader')
    parser.add_argument('--cos-lr', action='store_true',
                       help='cosine LR scheduler')
    parser.add_argument('--label-smoothing', type=float, default=0.0,
                       help='label smoothing epsilon')
    parser.add_argument('--patience', type=int, default=100,
                       help='epochs to wait for no improvement')
    parser.add_argument('--freeze', nargs='+', type=int, default=[0],
                       help='freeze layers: backbone=10, first3=0 1 2')
    parser.add_argument('--save-period', type=int, default=-1,
                       help='save checkpoint every x epochs')
    
    # Logging
    parser.add_argument('--project', default='runs/train',
                       help='save results to project/name')
    parser.add_argument('--name', default='exp',
                       help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true',
                       help='existing project/name ok, do not increment')
    parser.add_argument('--verbose', action='store_true',
                       help='verbose output')
    
    # Mixed precision
    parser.add_argument('--amp', action='store_true',
                       help='automatic mixed precision training')
    
    # Multi-GPU
    parser.add_argument('--sync-bn', action='store_true',
                       help='use SyncBatchNorm')
    parser.add_argument('--local_rank', type=int, default=-1,
                       help='DDP parameter')
    
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def train(args, config):
    """Main training function."""
    
    # Initialize
    init_seeds(1)
    device = select_device(args.device, batch_size=args.batch_size)
    
    # Directories
    save_dir = Path(increment_path(Path(args.project) / args.name, exist_ok=args.exist_ok))
    wdir = save_dir / 'weights'
    wdir.mkdir(parents=True, exist_ok=True)
    last = wdir / 'last.pt'
    best = wdir / 'best.pt'
    
    # Save config
    with open(save_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)
    
    # Model
    model = SCL_YOLO(config['model'])
    model = model.to(device)
    
    # Optimizer
    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / args.batch_size), 1)
    config['train']['weight_decay'] *= args.batch_size * accumulate / nbs
    
    optimizer = torch.optim.AdamW(model.parameters(), 
                                  lr=args.lr, 
                                  weight_decay=args.weight_decay)
    
    # Scheduler
    if args.cos_lr:
        lf = one_cycle(1, config['train']['lr_scheduler']['eta_min'] / args.lr, args.epochs)
    else:
        lf = lambda x: (1 - x / args.epochs) * (1.0 - config['train']['lr_scheduler']['eta_min'] / args.lr) + config['train']['lr_scheduler']['eta_min'] / args.lr
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    
    # EMA
    ema = ModelEMA(model)
    
    # Resume
    start_epoch, best_fitness = 0, 0.0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch'] + 1
        best_fitness = ckpt['best_fitness']
        if ema and ckpt.get('ema'):
            ema.ema.load_state_dict(ckpt['ema'])
            ema.updates = ckpt['updates']
        del ckpt
    
    # Image size
    gs = max(int(model.stride.max()), 32)
    imgsz = check_img_size(args.img_size, gs)
    
    # Trainloader
    train_loader, dataset = create_dataloader(
        config['data']['train'], imgsz, args.batch_size // WORLD_SIZE, gs,
        single_cls=False, hyp=config['train']['augmentation'], 
        augment=True, cache=args.cache, rect=False, rank=LOCAL_RANK,
        workers=args.workers, image_weights=False, quad=args.quad,
        prefix=colorstr('train: '), shuffle=True
    )
    
    # Validation loader
    val_loader = create_dataloader(
        config['data']['val'], imgsz, args.batch_size // WORLD_SIZE * 2, gs,
        single_cls=False, hyp=config['train']['augmentation'],
        augment=False, cache=None if args.noval else args.cache,
        rect=True, rank=-1, workers=args.workers * 2, pad=0.5,
        prefix=colorstr('val: ')
    )[0]
    
    # Model parameters
    model.nc = config['data']['nc']
    model.hyp = config['train']['augmentation']
    model.class_weights = labels_to_class_weights(dataset.labels, config['data']['nc']).to(device) * config['data']['nc']
    model.names = config['data']['names']
    
    # Start training
    t0 = time.time()
    nw = max(round(config['train']['warmup_epochs'] * len(train_loader)), 1000)
    maps = np.zeros(config['data']['nc'])
    results = (0, 0, 0, 0, 0, 0, 0)
    scheduler.last_epoch = start_epoch - 1
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    stopper = EarlyStopping(patience=args.patience)
    
    print(f'Image sizes {imgsz} train, {imgsz} val\n'
          f'Using {train_loader.num_workers * WORLD_SIZE} dataloader workers\n'
          f"Logging results to {colorstr('bold', save_dir)}\n"
          f'Starting training for {args.epochs} epochs...')
    
    for epoch in range(start_epoch, args.epochs):
        model.train()
        
        mloss = torch.zeros(3, device=device)
        if RANK != -1:
            train_loader.sampler.set_epoch(epoch)
        
        pbar = enumerate(train_loader)
        print(('\n' + '%10s' * 7) % ('Epoch', 'gpu_mem', 'box', 'obj', 'cls', 'labels', 'img_size'))
        if RANK in (-1, 0):
            pbar = tqdm(pbar, total=len(train_loader), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
        
        optimizer.zero_grad()
        
        for i, (imgs, targets, paths, _) in pbar:
            ni = i + len(train_loader) * epoch
            imgs = imgs.to(device, non_blocking=True).float() / 255
            
            # Warmup
            if ni <= nw:
                xi = [0, nw]
                accumulate = max(1, np.interp(ni, xi, [1, nbs / args.batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    x['lr'] = np.interp(ni, xi, [config['train']['warmup_bias_lr'] if j == 0 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [config['train']['warmup_momentum'], config['train']['momentum']])
            
            # Multi-scale
            if config['train']['multi_scale']:
                sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs
                sf = sz / max(imgs.shape[2:])
                if sf != 1:
                    ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]
                    imgs = nn.functional.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)
            
            # Forward
            with torch.cuda.amp.autocast(enabled=args.amp):
                pred = model(imgs)
                loss, loss_items = compute_loss(pred, targets.to(device))
                if RANK != -1:
                    loss *= WORLD_SIZE
            
            # Backward
            scaler.scale(loss).backward()
            
            # Optimize
            if ni - last_opt_step >= accumulate:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                if ema:
                    ema.update(model)
                last_opt_step = ni
            
            # Log
            if RANK in (-1, 0):
                mloss = (mloss * i + loss_items) / (i + 1)
                mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'
                pbar.set_description(('%10s' * 2 + '%10.4g' * 5) %
                                   (f'{epoch}/{args.epochs - 1}', mem, *mloss, targets.shape[0], imgs.shape[-1]))
        
        # Scheduler
        lr = [x['lr'] for x in optimizer.param_groups]
        scheduler.step()
        
        if RANK in (-1, 0):
            # mAP
            ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'names', 'stride', 'class_weights'])
            final_epoch = (epoch + 1 == args.epochs) or stopper.possible_stop
            if not args.noval or final_epoch:
                results, maps, _ = val.run(config['data'],
                                         batch_size=args.batch_size // WORLD_SIZE * 2,
                                         imgsz=imgsz,
                                         model=ema.ema,
                                         single_cls=False,
                                         dataloader=val_loader,
                                         save_dir=save_dir,
                                         plots=False,
                                         callbacks=None,
                                         compute_loss=compute_loss)
            
            # Update best mAP
            fi = fitness(np.array(results).reshape(1, -1))
            if fi > best_fitness:
                best_fitness = fi
            log_vals = list(mloss) + list(results) + lr
            
            # Save model
            if (not args.nosave) or (final_epoch and not args.evolve):
                ckpt = {
                    'epoch': epoch,
                    'best_fitness': best_fitness,
                    'model': deepcopy(de_parallel(model)).half(),
                    'ema': deepcopy(ema.ema).half(),
                    'updates': ema.updates,
                    'optimizer': optimizer.state_dict(),
                    'wandb_id': None
                }
                
                # Save last, best and delete
                torch.save(ckpt, last)
                if best_fitness == fi:
                    torch.save(ckpt, best)
                if args.save_period > 0 and epoch % args.save_period == 0:
                    torch.save(ckpt, wdir / f'epoch{epoch}.pt')
                del ckpt
            
            # Stop Single-GPU
            if RANK == -1 and stopper(epoch=epoch, fitness=fi):
                break
    
    # End training
    if RANK in (-1, 0):
        print(f'{epoch - start_epoch + 1} epochs complete
