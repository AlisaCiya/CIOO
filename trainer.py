"""
Unified Incremental Learning Trainer

将基础训练和增量训练合并为统一的类架构。
用法：
  基础训练: python train.py --nc_steps [15] --name n
  增量训练: python train.py --nc_steps [15,5] --name n
"""

import os
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import csv
import tqdm
import yaml
import torch
import warnings
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from torch.utils import data
from torch.nn.utils import clip_grad_norm_ as clip

from nets import nn
from utils import util


from utils.dataset import Dataset
from nets.getmodel import newmodel, oldmodel
from utils.util import PrototypeExtractor


class IncrementalTrainer:
    """统一的增量学习训练器"""
    
    def __init__(self, args):
        self.args = args
        self.is_base_step = len(args.nc_steps) == 1
        
        # 组件占位
        self.model: Optional[torch.nn.Module] = None
        self.model_old: Optional[torch.nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None
        self.scaler: Optional[torch.amp.GradScaler] = None
        self.ema: Optional[util.EMA] = None
        self.loader: Optional[data.DataLoader] = None
        self.criterion = None
        
        # 蒸馏相关（仅增量训练）
        self.kdloss = None
        self.proto_loss_fn = None  # 独立的 Prototype 蒸馏模块
        
        # Prototype 相关（缓存标志避免每步 getattr）
        self.proto_extractor = None
        self.use_proto_distill = getattr(args, 'proto_distill', False)

        
        # 训练状态
        self.best_map = 0.0
        self.best_mean_map = 0.0
        
    def setup(self):
        """初始化所有组件"""
        torch.backends.cudnn.benchmark = True
        util.init_seeds()
        
        self._setup_model()
        self._setup_teacher()
        self._setup_distributed()
        self._setup_dataloader()
        self._setup_optimizer()
        self._setup_losses()
        self._freeze_layers()
        self._setup_prototype()
        
        print(f"\n{'='*50}")
        print(f"训练模式: {'基础训练' if self.is_base_step else '增量训练'}")
        print(f"类别步骤: {self.args.nc_steps}")
        print(f"{'='*50}\n")
        
    def _setup_model(self):
        """初始化学生模型"""
        self.model = newmodel(self.args)
        
        eval_weights = getattr(self.args, 'eval_weights', None)
        if eval_weights:
            if not os.path.isfile(eval_weights):
                raise FileNotFoundError(f"评估权重不存在: {eval_weights}")
            self.model.load_weights(eval_weights, verbose=True)
        elif not self.is_base_step:
            # 增量训练：加载旧模型权重作为起点
            self.model.load_weights(self.args.oldmodel, verbose=True)
        elif getattr(self.args, 'weights', None) and os.path.isfile(self.args.weights):
            # 基础训练：加载预训练权重（使用统一的 load_weights 方法）
            self.model.load_weights(self.args.weights, verbose=True)
        
        self.model.cuda()
        util.freeze_layer(self.model)  # Freeze DFL Layer
        
    def _setup_teacher(self):
        """初始化教师模型（仅增量训练）"""
        if self.is_base_step:
            self.model_old = None
            return
        
        self.model_old = oldmodel(self.args)
        self.model_old.load_weights(self.args.oldmodel, verbose=True)
        self.model_old.cuda().eval()
        
        # 完全冻结教师模型
        for param in self.model_old.parameters():
            param.requires_grad = False
        print("✓ 教师模型已加载并冻结")
        
    def _setup_distributed(self):
        """设置训练组件"""
        self.scaler = torch.amp.GradScaler('cuda', enabled=True)
        self.ema = util.EMA(self.model)

    def setup_eval(self):
        """仅构建评估所需组件"""
        torch.backends.cudnn.benchmark = True
        util.init_seeds()
        self._setup_model()
        self.ema = None

    def evaluate(self, model_path: Optional[str] = None, save_csv: bool = True):
        """单模型评估入口"""
        if model_path:
            self.args.eval_weights = model_path

        self.setup_eval()
        m_pre, m_rec, map50, mean_ap, per_class = self.validate(return_details=True)
        overall = (m_pre, m_rec, map50, mean_ap)

        if save_csv:
            eval_path = Path(getattr(self.args, 'eval_weights', self.args.out))
            csv_path = eval_path.with_suffix('.eval.csv')
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            with open(csv_path, 'w', newline='', encoding='utf-8') as handle:
                writer = csv.DictWriter(handle, fieldnames=[
                    'class_id', 'name', 'instances', 'precision', 'recall', 'mAP50', 'mAP50-95'
                ])
                writer.writeheader()
                for row in per_class:
                    writer.writerow(row)
            print(f"评估结果已保存到: {csv_path}")

        return overall, per_class
        
    def _setup_dataloader(self):
        """初始化数据加载器"""
        dataset = Dataset(self.args, True, 
                         disjoint=getattr(self.args, 'disjoint', False))
        
        num_workers = min(16, max(1, os.cpu_count() // 2 - 1))
        
        self.loader = data.DataLoader(
            dataset,
            self.args.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=Dataset.collate_fn,
            prefetch_factor=4,
            persistent_workers=True
        )
        
    def _setup_optimizer(self):
        """初始化优化器和调度器"""
        accumulate = max(round(64 / self.args.batch_size), 1)
        
        if self.is_base_step:
            self.optimizer = util.optimizer_step0(self.args, self.model, self.args.decay)
        else:
            self.optimizer = util.smart_optimizer(self.args, self.model, self.args.decay)
        
        # 学习率调度器
        linear = lambda x: (max(1 - x / self.args.epochs, 0) * (1.0 - 0.01) + 0.01)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=linear)
        self._linear_fn = linear
        self._accumulate_base = accumulate
        
    def _setup_losses(self):
        """初始化损失函数"""
        self.criterion = util.DetectionLoss(self.model)
        
        if not self.is_base_step:
            # 蒸馏损失
            from utils.kdloss import IncrementalKDLossV2
            self.kdloss = IncrementalKDLossV2(
                temperature=4.0,
                box_weight=2.0,
                cls_weight=2.4,
                bg_weight=0.1,
                feat_weight=1.7,
                use_iou_weight=True,
                use_bg_distill=True,
                use_kl_cls=True
            )
            
            # Prototype 蒸馏（可选，独立模块）
            if getattr(self.args, 'proto_distill', False):
                from utils.kdloss import PrototypeDistillLoss
                
                # 自动推断 prototype 路径
                proto_path = getattr(self.args, 'proto_path', None)
                if proto_path is None:
                    proto_path = self.args.oldmodel.replace('.pt', '_proto.pt')
                
                if os.path.exists(proto_path):
                    self.proto_loss_fn = PrototypeDistillLoss(fg_thr=0.4)
                    self.proto_loss_fn.load_prototypes(proto_path)
                    print(f"✓ Prototype 蒸馏已启用，权重: {getattr(self.args, 'proto_weight', 1.0)}")
                else:
                    print(f"⚠️ Prototype 文件不存在: {proto_path}")
                    print(f"   请先使用 --prototype 提取旧类 prototype")
            
    
    def _freeze_layers(self):
        """冻结指定层"""
        if not self.is_base_step:
            # 增量训练：冻结旧分类头
            model = self.model.module if hasattr(self.model, 'module') else self.model
            if hasattr(model, 'detect') and hasattr(model.detect, 'cls_heads'):
                if len(model.detect.cls_heads) > 1:
                    for i, head in enumerate(model.detect.cls_heads[:-1]):
                        for param in head.parameters():
                            param.requires_grad = False
                        print(f"✓ 分类头 {i} 已冻结")
        
        # Backbone 始终可训练
        model = self.model.module if hasattr(self.model, 'module') else self.model
        for param in model.backbone.parameters():
            param.requires_grad = True
    
    def _setup_prototype(self):
        """初始化 Prototype 提取器（可选）"""
        if not getattr(self.args, 'prototype', False):
            return
        
        nc = sum(self.args.nc_steps)
        self.proto_extractor = PrototypeExtractor(self.model, nc=nc, device='cuda')
        print(f"✓ Prototype 提取器已初始化 (nc={nc})")
    
    def _get_class_names(self):
        """根据数据集返回类别名称"""
        data_dir = getattr(self.args, 'data_dir', 'VOC').upper()
        
        if 'VOC' in data_dir or 'PASCAL' in data_dir:
            return {
                0: 'aeroplane', 1: 'bicycle', 2: 'bird', 3: 'boat', 4: 'bottle',
                5: 'bus', 6: 'car', 7: 'cat', 8: 'chair', 9: 'cow',
                10: 'diningtable', 11: 'dog', 12: 'horse', 13: 'motorbike', 14: 'person',
                15: 'pottedplant', 16: 'sheep', 17: 'sofa', 18: 'train', 19: 'tvmonitor'
            }
        elif 'COCO' in data_dir:
            return {
                0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane',
                5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light',
                10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench',
                14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow',
                20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack',
                25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee',
                30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite',
                34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard',
                38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup',
                42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana',
                47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot',
                52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair',
                57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table',
                61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote',
                66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven',
                70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock',
                75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'
            }
        else:
            # 默认返回数字索引
            total_classes = sum(self.args.nc_steps)
            return {i: str(i) for i in range(total_classes)}
            
    def train(self):
        """主训练循环"""
        args = self.args
        num_batch = len(self.loader)
        warm_up = max(round(3 * num_batch), 100)
        opt_step = -1
        
        # 日志路径（与 args.out 同目录）
        out_dir = os.path.dirname(args.out)
        os.makedirs(out_dir, exist_ok=True)
        csv_path = args.out.replace('.pt', '.csv')
        
        with open(csv_path, 'a') as log:
            # 写入时间戳
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            log.write(f"\n# Training started at {timestamp}\n")
            log.write(f"# nc_steps={args.nc_steps}, epochs={args.epochs}, batch_size={args.batch_size}\n")
            
            logger = csv.DictWriter(log, fieldnames=[
                'epoch', 'box', 'cls', 'dfl', 'Recall', 'Precision', 'mAP@50', 'mAP'
            ])
            logger.writeheader()
            
            for epoch in range(args.epochs):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self.scheduler.step()
                
                if self.model_old:
                    self.model_old.eval()
                self.model.train()
                
                # 最后10个epoch关闭mosaic
                if args.epochs - epoch == 10:
                    self.loader.dataset.mosaic = False
                
                p_bar = enumerate(self.loader)
                print("\n" + "%11s" * 5 % ("Epoch", "GPU", "box", "cls", "dfl"))
                p_bar = tqdm.tqdm(enumerate(self.loader), total=num_batch)
                
                t_loss = None
                accumulate = self._accumulate_base
                
                for i, batch in p_bar:
                    glob_step = i + num_batch * epoch
                    
                    # Warmup
                    if glob_step <= warm_up:
                        xi = [0, warm_up]
                        accumulate = max(1, int(np.interp(glob_step, xi, 
                                        [1, 64 / args.batch_size]).round()))
                        for j, x in enumerate(self.optimizer.param_groups):
                            x["lr"] = np.interp(glob_step, xi, 
                                [0.0, x["initial_lr"] * self._linear_fn(epoch)])
                            if "momentum" in x:
                                x["momentum"] = np.interp(glob_step, xi, [0.8, 0.937])
                    
                    # 训练步骤
                    if self.is_base_step:
                        loss, loss_items = self._train_step_base(batch)
                    else:
                        loss, loss_items = self._train_step_incremental(batch)
                    
                    t_loss = (t_loss * i + loss_items) / (i + 1) if t_loss is not None else loss_items
                    
                    # 反向传播
                    self.scaler.scale(loss).backward()
                    
                    if glob_step - opt_step >= accumulate:
                        self.scaler.unscale_(self.optimizer)
                        clip(self.model.parameters(), max_norm=10.0)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.optimizer.zero_grad(set_to_none=True)
                        if self.ema:
                            self.ema.update(self.model)
                        opt_step = glob_step
                    
                    fmt = "%11s" * 2 + "%11.4g" * 3
                    memory = f'{torch.cuda.memory_reserved() / 1e9:.3g}G'
                    p_bar.set_description(fmt % (f"{epoch + 1}/{args.epochs}", memory, *t_loss))
                
                # 验证和保存
                m_pre, m_rec, map50, mean_map = self.validate()
                box, cls, dfl = map(float, t_loss)
                
                logger.writerow({
                    'epoch': str(epoch + 1).zfill(3),
                    'box': f'{box:.3f}',
                    'cls': f'{cls:.3f}',
                    'dfl': f'{dfl:.3f}',
                    'mAP': f'{mean_map:.3f}',
                    'mAP@50': f'{map50:.3f}',
                    'Recall': f'{m_rec:.3f}',
                    'Precision': f'{m_pre:.3f}'
                })
                log.flush()
                
                # 保存检查点
                self._save_checkpoint(epoch, map50, mean_map)
        
        # 训练结束后提取 Prototype
        self._extract_prototypes()
        
        print("Training complete.")
    
    def _train_step_base(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        """基础训练步骤（无蒸馏）"""
        images = batch["img"].cuda(non_blocking=True).float() / 255
        
        with torch.amp.autocast('cuda'):
            pred = self.model(images)
            loss, loss_items = self.criterion(pred, batch)
        
        return loss, loss_items
    
    def _train_step_incremental(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        """增量训练步骤（含蒸馏）"""
        images = batch["img"].cuda(non_blocking=True).float() / 255
        
        # 教师模型前向（在 autocast 外，使用 float32）
        with torch.no_grad():
            feature_teacher, pred_old = self.model_old.forward_teacher(images.float())
        
        # 学生模型前向
        with torch.amp.autocast('cuda'):
            # 如果启用 prototype 蒸馏，使用 forward_for_prototype
            if self.proto_loss_fn is not None:
                pred, feature_student, proto_feats = self.model.forward_for_prototype(images)
            else:
                feature_student, pred = self.model.forward_with_features(images)
                proto_feats = None
            
            # 检测损失
            loss, loss_items = self.criterion(pred, batch)
            
            # KD 蒸馏损失（不含 Prototype）
            kd_loss = self.kdloss(
                pred, pred_old, 
                old_nc_sum=sum(self.args.nc_steps[:-1]),
                student_neck_feats=feature_student,
                teacher_neck_feats=feature_teacher
            )
            
            loss = loss + kd_loss * 35
            
            # Prototype 蒸馏损失（独立计算）
            if self.proto_loss_fn is not None and proto_feats is not None:
                proto_loss = self.proto_loss_fn(proto_feats, pred_old)
                proto_weight = getattr(self.args, 'proto_weight', 1.0)
                loss = loss + proto_loss * proto_weight
        
        return loss, loss_items
    
    def validate(self, model=None, return_details: bool = False):
        """验证模型性能"""
        if model is None:
            model = self.ema.ema if self.ema else self.model
        
        model.eval()
        
        iou_v = torch.linspace(0.5, 0.95, 10)
        n_iou = iou_v.numel()
        metric = {"tp": [], "conf": [], "pred_cls": [], "target_cls": [], "target_img": []}
        
        # 验证时使用所有类别
        val_args = type(self.args)()
        for k, v in vars(self.args).items():
            setattr(val_args, k, v)
        val_args.allowed_classes = set(range(sum(self.args.nc_steps)))
        
        dataset = Dataset(val_args, False, 
                         disjoint=getattr(self.args, 'disjoint', False))
        
        num_workers = min(16, max(1, os.cpu_count() // 2 - 1))
        loader = data.DataLoader(
            dataset,
            batch_size=16,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=Dataset.collate_fn,
            prefetch_factor=4,
            persistent_workers=True
        )
        
        for batch in tqdm.tqdm(loader, desc=('%10s' * 5) % ('', 'precision', 'recall', 'mAP50', 'mAP')):
            image = batch["img"].cuda(non_blocking=True).float() / 255
            for k in ["idx", "cls", "box"]:
                batch[k] = batch[k].cuda(non_blocking=True)
            
            with torch.no_grad():
                outputs = util.non_max_suppression(model(image))
                metric = util.update_metrics(outputs, batch, n_iou, iou_v, metric)
        
        stats = {k: torch.cat(v, 0).cpu().numpy() for k, v in metric.items()}
        stats.pop("target_img", None)
        result = {}
        
        if len(stats) and stats["tp"].any():
            # 根据数据集选择类别名称
            names = self._get_class_names()
            result = util.compute_ap(
                tp=stats['tp'],
                conf=stats['conf'],
                pred=stats['pred_cls'],
                target=stats['target_cls'],
                plot=getattr(self.args, 'plot', False),
                save_dir='weights/',
                names=names,
                return_details=return_details
            )
            m_pre, m_rec, map50, mean_ap = (
                result['precision'], result['recall'], 
                result['mAP50'], result['mAP50-95']
            )
        else:
            m_pre, m_rec, map50, mean_ap = 0.0, 0.0, 0.0, 0.0
        
        print(('%10s' + '%10.3g' * 4) % ('', m_pre, m_rec, map50, mean_ap))
        model.float()
        model.train()
        
        if return_details:
            return m_pre, m_rec, map50, mean_ap, result.get('per_class', [])
        return m_pre, m_rec, map50, mean_ap
    
    def _save_checkpoint(self, epoch: int, map50: float, mean_map: float):
        """保存检查点"""
        ckpt = {
            'model': self.ema.ema.state_dict() if self.ema else self.model.state_dict(),
        }
        if map50 > self.best_map:
            self.best_map = map50
            self.best_mean_map = mean_map
            torch.save(ckpt, self.args.out)
            print(f"\n🏆 新的最佳模型! mAP50={map50:.4f}")

        
        del ckpt
    
    def _extract_prototypes(self):
        """训练结束后提取 Prototype 特征"""
        if self.proto_extractor is None:
            return
        
        print("\n" + "="*50)
        print("提取 Prototype 特征...")
        print("="*50)
        
        model = self.ema.ema if self.ema else self.model
        model.eval()
        
        # 遍历训练数据集提取特征
        for batch in tqdm.tqdm(self.loader, desc="提取 Prototype"):
            images = batch["img"].cuda(non_blocking=True).float() / 255
            self.proto_extractor.update(model, images, batch)
        
        # 打印统计
        self.proto_extractor.print_stats()
        
        # 保存
        proto_path = self.args.out.replace('.pt', '_proto.pt')
        self.proto_extractor.save(proto_path)
        
        model.train()
    
    def run(self):
        """运行完整训练流程"""
        self.setup()
        
        # 仅提取模式：跳过训练，直接提取 prototype
        if getattr(self.args, 'extract_only', False):
            if not getattr(self.args, 'prototype', False):
                print("⚠️ --extract-only 需要配合 --prototype 使用")
                return
            print("\n" + "="*50)
            print("独立提取模式：跳过训练，直接提取 Prototype")
            print("="*50)
            self._extract_prototypes()
            return
        
        self.train()
