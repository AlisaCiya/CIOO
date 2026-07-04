import os
import math
import time
import copy
import random
import warnings
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F


def init_seeds(seed=1789):  #1789
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False





def freeze_layer(model):
    layer_names = [".dfl"]
    freeze_layer_names = [f"model.{x}." for x in []] + layer_names
    for k, v in model.named_parameters():
        if any(x in k for x in freeze_layer_names):
            print(f"Freezing layer '{k}'")
            v.requires_grad = False






def wh2xy(x):
    assert x.shape[-1] == 4, f"expected 4 but input shape is {x.shape}"
    if isinstance(x, torch.Tensor):
        y = torch.empty_like(x, dtype=torch.float32)
    else:
        y = np.empty_like(x, dtype=np.float32)
    xy = x[..., :2]
    wh = x[..., 2:] / 2
    y[..., :2] = xy - wh
    y[..., 2:] = xy + wh
    return y


def make_anchors(feats, strides, offset=0.5):
    anchor_points, stride_tensor = [], []
    assert feats is not None
    dtype, device = feats[0].dtype, feats[0].device
    for i, stride in enumerate(strides):
        h, w = feats[i].shape[2:] if isinstance(feats, list) else (
            int(feats[i][0]), int(feats[i][1]))
        sx = torch.arange(end=w, device=device, dtype=dtype) + offset
        sy = torch.arange(end=h, device=device, dtype=dtype) + offset
        sy, sx = torch.meshgrid(sy, sx, indexing="ij")
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(
            torch.full((h * w, 1), stride, dtype=dtype, device=device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)


def smart_optimizer(args, model, decay=1e-5):
    """
    智能优化器 - 分块参数配置，新分类器单独处理
    
    参数分组策略：
    1. 标准分组: backbone、head 的 weights、bias、BN（排除新分类器）
    2. 旧分类器: 学习率 = lr
    3. 新分类器: 学习率 = lr * 10（增量学习加速）
    """
    g = [], [], []
    # 从 model 获取类别数（更可靠，支持增量学习）
    nc = model.detect.nc if hasattr(model.detect, 'nc') else 80
    #nc=5
    lr_fit = round(0.002 * 5 / (4 + nc), 6)
    name, lr, momentum = "AdamW", lr_fit, 0.9
    
    bn = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)

    # ========================================
    # 第1步：先收集新分类器和旧分类器参数的 ID（用于后续过滤）
    # ========================================
    new_clf_param_ids = set()
    old_clf_param_ids = set()

    if hasattr(model, 'get_new_classifier_params'):
        new_clf_param_ids = set(id(p) for p in model.get_new_classifier_params())

    if hasattr(model, 'get_old_classifier_params'):
        old_clf_param_ids = set(id(p) for p in model.get_old_classifier_params())

    # ========================================
    # 第2步：分组参数（排除新分类器和旧分类器）
    # ========================================
    for module_name, module in model.named_modules():
        for param_name, param in module.named_parameters(recurse=False):
            # ⚠️ 关键：过滤掉新分类器和旧分类器参数
            if id(param) in new_clf_param_ids or id(param) in old_clf_param_ids:
                continue
            
            if module_name:
                fullname = f"{module_name}.{param_name}"
            else:
                fullname = f"{param_name}"
            if "bias" in fullname:
                g[2].append(param)
            elif isinstance(module, bn):
                g[1].append(param)
            else:
                g[0].append(param)

    # ========================================
    # 第3步：创建优化器 - 标准参数组
    # ========================================
    optimizer = torch.optim.AdamW(g[2], lr=lr*0.0405, betas=(0.9, 0.999),  #  11 max  0.0405
                                  weight_decay=0.0)
    optimizer.add_param_group({"params": g[0], "weight_decay": decay})
    optimizer.add_param_group({"params": g[1], "weight_decay": 0.0})

    # 为所有参数组设置 initial_lr
    for group in optimizer.param_groups:
        group.setdefault('initial_lr', group['lr'])

    # ========================================
    # 第4步：添加旧分类器参数
    # ========================================
    if hasattr(model, 'get_old_classifier_params'):
        old_clf_params = list(model.get_old_classifier_params())
        if len(old_clf_params) > 0:
            optimizer.add_param_group({"params": old_clf_params, "lr": lr*0.001})
            # 设置 initial_lr
            #print(f"[优化器] 旧分类器参数已添加: {len(old_clf_params)} 个, lr={lr:.6f}")
    
    # # ========================================
    # # 第5步：添加新分类器参数（学习率提高 10 倍）✨
    # # ========================================
    if hasattr(model, 'get_new_classifier_params'):
        new_clf_params = list(model.get_new_classifier_params())
        if len(new_clf_params) > 0:
            new_lr = lr*11  # 提高 10 倍学习率
            optimizer.add_param_group({"params": new_clf_params, "lr": new_lr})
            print(f"[优化器] 新分类器参数已添加: {len(new_clf_params)} 个, lr={new_lr:.6f} ")

    return optimizer

def optimizer_step0(lr, model, decay=1e-5):

    g = [], [], []
    # 从 model 获取类别数（更可靠，支持增量学习）
    nc = model.detect.nc if hasattr(model.detect, 'nc') else 80

    lr_fit = round(0.002 * 5 / (4 + nc), 6)
    print(f"✓ 推荐学习率 lr={lr_fit}")
    name, lr, momentum = "AdamW", lr_fit, 0.9
    lr=0.0001
    bn = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)


    for module_name, module in model.named_modules():
        for param_name, param in module.named_parameters(recurse=False):

            if module_name:
                fullname = f"{module_name}.{param_name}"
            else:
                fullname = f"{param_name}"
            if "bias" in fullname:
                g[2].append(param)
            elif isinstance(module, bn):
                g[1].append(param)
            else:
                g[0].append(param)

    optimizer = torch.optim.AdamW(g[2], lr=lr, betas=(0.9, 0.999),  
                                  weight_decay=0.0)
    optimizer.add_param_group({"params": g[0], "weight_decay": decay})
    optimizer.add_param_group({"params": g[1], "weight_decay": 0.0})

    return optimizer


# ------------------------- MODEL EMA Start---------------------------------#



def copy_attr(a, b, include=(), exclude=()):
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith(
                "_") or k in exclude:
            continue
        else:
            setattr(a, k, v)


class EMA:
    def __init__(self, model, decay=0.9999, tau=2000, updates=0):
        self.ema = copy.deepcopy(model).eval()
        self.updates = updates
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.enabled = True

    def update(self, model):
        if self.enabled:
            self.updates += 1
            d = self.decay(self.updates)

            msd = model.state_dict()
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1 - d) * msd[k].detach()

    def update_attr(self, model, include=(),
                    exclude=("process_group", "reducer")):
        if self.enabled:
            copy_attr(self.ema, model, include, exclude)


# ------------------------- MODEL EMA End ---------------------------------#


# ----------------------- Detection Loss Start --------------
def bbox_iou(box1, box2, eps=1e-7):
    b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    x = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0)
    y = (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp_(0)
    inter = x * y
    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union

    cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)
    ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)
    c2 = cw.pow(2) + ch.pow(2) + eps
    a = (b2_x1 + b2_x2 - b1_x1 - b1_x2)
    b = (b2_y1 + b2_y2 - b1_y1 - b1_y2)
    rho2 = (a.pow(2) + b.pow(2)) / 4

    v = (4 / math.pi ** 2) * ((w2 / h2).atan() - (w1 / h1).atan()).pow(2)
    with torch.no_grad():
        alpha = v / (v - iou + (1 + eps))
    return iou - (rho2 / c2 + v * alpha)


class DFLoss(nn.Module):
    def __init__(self, reg_max=16):
        super().__init__()
        self.reg_max = reg_max

    def __call__(self, pred_dist, target):
        target = target.clamp_(0, self.reg_max - 1 - 0.01)
        tl, tr = target.long(), target.long() + 1
        wl, wr = tr - target, 1 - (tr - target)
        loss_l = F.cross_entropy(pred_dist, tl.view(-1), reduction="none")
        loss_r = F.cross_entropy(pred_dist, tr.view(-1), reduction="none")
        loss = (loss_l.view(tl.shape) * wl + loss_r.view(tl.shape) * wr)
        return loss.mean(-1, keepdim=True)


class BoxLoss(nn.Module):
    def __init__(self, reg_max=16):
        super().__init__()
        self.dfl_loss = DFLoss(reg_max)

    def forward(self, p_dist, p_box, anchors, gt_box, scores, scores_sum,
                mask):
        reg_max = self.dfl_loss.reg_max
        weight = scores.sum(-1)[mask].unsqueeze(-1)
        iou = bbox_iou(p_box[mask], gt_box[mask])
        loss_box = ((1.0 - iou) * weight).sum() / scores_sum

        a, b = gt_box.chunk(2, -1)
        distance = torch.cat((anchors - a, b - anchors), -1)
        target = distance.clamp_(0, (reg_max - 1) - 0.01)
        pred = p_dist[mask].view(-1, reg_max)
        loss_dfl = self.dfl_loss(pred, target[mask])
        loss_dfl = (loss_dfl * weight).sum() / scores_sum

        return loss_box, loss_dfl


class Assigner(nn.Module):
    def __init__(self, top_k=10, nc=80, alpha=0.5, beta=6.0, eps=1e-9):
        super().__init__()
        self.nc = nc
        self.eps = eps
        self.beta = beta
        self.top_k = top_k
        self.alpha = alpha

    @torch.no_grad()
    def forward(self, score, p_box, anchors, gt_labels, gt_box, mask , nc1=None):
        if nc1 is not None:
            self.nc = nc1
        bs = score.shape[0]
        na = p_box.shape[-2]
        n_max_boxes = gt_box.shape[1]

        if n_max_boxes == 0:
            # ⚠️ 返回 3 个值，形状匹配 [bs, na, nc]
            return (
                torch.zeros((bs, na, 4), dtype=p_box.dtype, device=p_box.device),
                torch.zeros((bs, na, self.nc), dtype=score.dtype, device=score.device),
                torch.zeros((bs, na), dtype=torch.bool, device=p_box.device))

        lt, rb = gt_box.view(-1, 1, 4).chunk(2, 2)
        box_delta = torch.cat((anchors[None] - lt, rb - anchors[None]), dim=2)
        mask_in_gts = box_delta.view(gt_box.shape[0], gt_box.shape[1],
                                     anchors.shape[0], -1)
        mask_in_gts = mask_in_gts.amin(3).gt_(1e-9)
        mask_gts = (mask_in_gts * mask).bool()
        overlaps = torch.zeros([bs, n_max_boxes, na], dtype=p_box.dtype,
                               device=p_box.device)
        bbox_scores = torch.zeros([bs, n_max_boxes, na], dtype=score.dtype,
                                  device=score.device)

        ind = torch.zeros([2, bs, n_max_boxes], dtype=torch.long)
        ind[0] = torch.arange(end=bs).view(-1, 1).expand(-1, n_max_boxes)
        ind[1] = gt_labels.squeeze(-1)
        bbox_scores[mask_gts] = score[ind[0], :, ind[1]][mask_gts]
        pd_boxes = p_box.unsqueeze(1).expand(-1, n_max_boxes, -1, -1)[mask_gts]
        gt_boxes = gt_box.unsqueeze(2).expand(-1, -1, na, -1)[mask_gts]
        overlaps[mask_gts] = bbox_iou(gt_boxes, pd_boxes).squeeze(-1).clamp_(0)

        metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)
        top_mask = mask.expand(-1, -1, self.top_k).bool()
        top_metrics, top_id = torch.topk(metric, self.top_k, dim=-1,
                                         largest=True)
        if top_mask is None:
            top_mask = (top_metrics.max(-1, keepdim=True)[
                            0] > self.eps).expand_as(top_id)
        top_id.masked_fill_(~top_mask, 0)

        count_tensor = torch.zeros(metric.shape, dtype=torch.int8,
                                   device=top_id.device)
        ones = torch.ones_like(top_id[:, :, :1], dtype=torch.int8,
                               device=top_id.device)
        for k in range(self.top_k):
            count_tensor.scatter_add_(-1, top_id[:, :, k: k + 1], ones)

        count_tensor.masked_fill_(count_tensor > 1, 0)
        mask_pos = count_tensor.to(metric.dtype) * mask_in_gts * mask
        fg_mask = mask_pos.sum(-2)
        if fg_mask.max() > 1:
            mask_multi_gts = (fg_mask.unsqueeze(1) > 1).expand(-1, n_max_boxes,
                                                               -1)

            max_over = torch.zeros(mask_pos.shape, dtype=mask_pos.dtype,
                                   device=mask_pos.device)
            max_over.scatter_(1, overlaps.argmax(1).unsqueeze(1), 1)

            mask_pos = torch.where(mask_multi_gts, max_over, mask_pos).float()
            fg_mask = mask_pos.sum(-2)
        gt_idx = mask_pos.argmax(-2)

        batch_ind = \
            torch.arange(end=bs, dtype=torch.int64, device=gt_labels.device)[
                ..., None]
        gt_idx = gt_idx + batch_ind * n_max_boxes
        target_labels = gt_labels.long().flatten()[gt_idx]

        target_bboxes = gt_box.view(-1, gt_box.shape[-1])[gt_idx]
        target_labels.clamp_(0)
        sc = (target_labels.shape[0], target_labels.shape[1], self.nc)
        target_scores = torch.zeros(sc, dtype=torch.int64,
                                    device=target_labels.device)
        target_scores.scatter_(2, target_labels.unsqueeze(-1), 1)
        scores_mask = fg_mask[:, :, None].repeat(1, 1, self.nc)
        target_scores = torch.where(scores_mask > 0, target_scores, 0)

        # Normalize
        metric *= mask_pos
        pos_metrics = metric.amax(dim=-1, keepdim=True)
        pos_overlaps = (overlaps * mask_pos).amax(dim=-1, keepdim=True)
        norm_metric = (metric * pos_overlaps / (pos_metrics + self.eps))
        target_scores = target_scores * (norm_metric.amax(-2).unsqueeze(-1))
        return target_bboxes, target_scores, fg_mask.bool()


class DetectionLoss:
    def __init__(self, model):
        """
        Args:
            model: 检测模型
            step: 当前所处的任务阶段 (0-indexed)
                 例如: step=0 表示 Step0，step=1 表示 Step1
                 如果为 None，则使用 model.detect.step
                 该参数控制 loss 计算中只使用该阶段的分类器输出
        """
        device = next(model.parameters()).device

        m = model.detect
        self.nc = m.nc
        
        # 获取类别信息和阶段数
        if hasattr(m, 'nc_steps'):
            self.nc_steps = m.nc_steps
            self.step =  m.step
        else:
            # 单步学习的兼容性处理
            self.nc_steps = [self.nc]
            self.step = 0
        
        # 计算当前分类器的标签偏移量
        # 对于 nc_steps=[15, 5]
        #   - Step 0: label_offset = sum([]) = 0 (第0个分类器处理类别 [0-14])
        #   - Step 1: label_offset = sum([15]) = 15 (第1个分类器处理类别 [15-19] 转换为 [0-4])
        self.label_offset = sum(self.nc_steps[:self.step])
        self.current_nc = self.nc_steps[self.step]
        
        # 打印配置信息
        print(f"[DetectionLoss] 分类器配置: nc_steps={self.nc_steps}, 当前step={self.step}")
        print(f"  增量范围: [{self.label_offset}:{self.label_offset + self.current_nc}] (第{self.step}个分类器的{self.current_nc}维)")
        print(f"  标签变换: 全局标签 [{self.label_offset}-{self.label_offset + self.current_nc - 1}] → 本地索引 [0-{self.current_nc - 1}] (减去偏移 {self.label_offset})")
        
        self.device = device
        self.stride = m.stride
        self.reg_max = m.reg_max
        self.no = 4 * self.reg_max + sum(self.nc_steps)

        self.assigner = Assigner(nc=self.current_nc)
        self.bbox_loss = BoxLoss(m.reg_max).cuda()
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)
      
    def preprocess(self, gt, bs, scale):
        nl, ne = gt.shape
        if nl == 0:
            out = torch.zeros(bs, 0, ne - 1, device=self.device)
        else:
            i = gt[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(bs, counts.max(), ne - 1, device=self.device)
            for j in range(bs):
                matches = i == j
                n = matches.sum()
                if n:
                    out[j, :n] = gt[matches, 1:]
            out[..., 1:5] = wh2xy(out[..., 1:5].mul_(scale))
        return out

    def bbox_decode(self, anchor, pred_dist):
        b, a, c = pred_dist.shape
        pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3)
        pred_dist = pred_dist.matmul(self.proj.type(pred_dist.dtype))
        lt, rb = pred_dist.chunk(2, -1)
        x1y1, x2y2 = anchor - lt, anchor + rb
        return torch.cat((x1y1, x2y2), -1)

 


    def __call__(self, pred, batch):
        """
        Args:
            pred: 模型输出
            batch: 批次数据
        
        Returns:
            loss: 总损失
            loss_items: 损失详情 (box, cls, dfl)
        """
        loss = torch.zeros(3, device=self.device)
        feats = pred[1] if isinstance(pred, tuple) else pred

        x = torch.cat([f.view(feats[0].shape[0], self.no, -1) for f in feats], 2)
        
        # ===== 【关键步骤】根据当前 step 提前筛选维度范围 =====
        # 目的：只保留当前 step 需要的维度，去掉其他 step 的分类器输出
        # 例如 nc_steps=[15, 5]，step=1 时：
        #   原始 x: [batch, 84, HW]  (64 box + 15 cls_step0 + 5 cls_step1)
        #   筛选后: [batch, 69, HW]  (64 box + 5 cls_step1)
        
        box_dims = 4 * self.reg_max  # 64
        cls_offset_start = box_dims + sum(self.nc_steps[:self.step])  # 当前 step 分类器开始位置
        cls_offset_end = cls_offset_start + self.nc_steps[self.step]  # 当前 step 分类器结束位置
        
        # print(f"[PRE-FILTERING] 在 Loss 函数开始处进行维度筛选")
        # print(f"  原始输出形状: {x.shape}")
        # print(f"  step={self.step}, nc_steps={self.nc_steps}")
        # print(f"  Box 范围: [0:{box_dims}]")
        # print(f"  第 {self.step} 个分类器范围: [{cls_offset_start}:{cls_offset_end}]")
        
        # 拼接 Box（全部保留）+ 当前 step 的分类器（只保留这个）
        x = torch.cat([
            x[:, :box_dims, :],                    # Box: [batch, 64, HW]
            x[:, cls_offset_start:cls_offset_end, :]  # Cls: [batch, current_nc, HW]
        ], dim=1)
        
        # print(f"  筛选后的输出形状: {x.shape}")
        # print(f"  新的维度结构: [0:{box_dims}]=Box, [{box_dims}:{box_dims + self.nc_steps[self.step]}]=Cls")
        # print(f"  ✓ 只保留了第 {self.step} 步的分类器，去掉了其他 step 的输出")
        
        # 现在 x 的结构是: [batch, box_dims + current_nc, HW]
        # 直接用标准的 YOLO loss 格式处理
        
        # 分割: box (4*reg_max) + cls (current_nc)
        pred_distri, pred_scores_all = x.split((self.reg_max * 4, self.nc_steps[self.step]), 1)

        pred_scores_all = pred_scores_all.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        
        # print(f"  [After Permute]")
        # print(f"    pred_distri 形状: {pred_distri.shape}  (Box, 64维)")
        # print(f"    pred_scores_all 形状: {pred_scores_all.shape}  (仅第 {self.step} 步的分类器, {self.nc_steps[self.step]}维)")

        dtype, bs = pred_scores_all.dtype, pred_scores_all.shape[0]
        img_size = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype)
  
        img_size = img_size * self.stride[0]
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        idx, cls, box = batch["idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["box"]
        
        # 【DEBUG】查看批次中的实际标签

        
        # ===== 【关键】在这里立即重映射标签 =====
        # 目的：确保从此之后，所有流程中的标签都是本地索引 [0, current_nc)
        step_global_start = sum(self.nc_steps[:self.step])
        cls_remapped = cls - self.label_offset


        targets = torch.cat((idx, cls_remapped, box), 1).to(self.device)

        
        targets = self.preprocess(targets, bs, img_size[[1, 0, 1, 0]])


        gt_labels, gt_bboxes = targets.split((1, 4), 2)


        
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)


        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)
        
        # ===== 标签已在源头重映射，现在直接使用本地标签 =====
        # gt_labels 已经是本地索引 [0, current_nc)
        # 对应 pred_scores_all 的维度 [0, current_nc)
        
        gt_labels_local = gt_labels
        

        
        # ===== 为当前 step 的分类分配目标 =====
        # gt_labels_local 已经是本地标签 [0, current_nc)
        # pred_scores_all 也已经只包含当前 step 的分类器输出 [0, current_nc)
        # 它们的维度完全匹配
        

        target_bboxes, target_scores, fg_mask = self.assigner(
            pred_scores_all.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor, gt_labels_local, gt_bboxes, mask_gt)

        # ===== 分类损失: 使用选定的分类器输出 =====
        # pred_scores_all 现在就是当前 step 的分类器输出
        _loss = self.bce(pred_scores_all, target_scores.to(dtype))
        loss[1] = _loss.sum() / max(target_scores.sum(), 1)
        
        # 使用当前 step 的目标分数进行边界框损失
        target_scores_for_bbox = target_scores

        # ===== 边界框损失 =====
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(pred_distri,
                                              pred_bboxes,
                                              anchor_points,
                                              target_bboxes,
                                              target_scores_for_bbox,
                                              max(target_scores_for_bbox.sum(), 1),
                                              fg_mask)

        loss[0] *= 7.5
        loss[1] *= 0.5  #0.5
        loss[2] *= 1.5
    

        return loss.sum() * bs, loss.detach()


# ----------------------- Detection Loss End --------------

# ----------------------- Compute AP Start -----------------
def non_max_suppression(pred, conf_th=0.001, iou_th=0.7):
    import torchvision
    max_det = 300
    max_wh = 7680
    max_nms = 30000

    pred = pred[0] if isinstance(pred, (list, tuple)) else pred

    bs = pred.shape[0]  # batch size
    nc = pred.shape[1] - 4  # number of classes
    xc = pred[:, 4:(4 + nc)].amax(1) > conf_th

    start_time = time.time()
    time_limit = 10.0 + 0.05 * bs

    pred = pred.transpose(-1, -2)
    pred[..., :4] = wh2xy(pred[..., :4])

    output = [torch.zeros((0, 6), device=pred.device)] * bs
    for xi, x in enumerate(pred):
        x = x[xc[xi]]
        if not x.shape[0]:
            continue

        box, cls = x.split((4, nc), 1)

        if nc > 1:
            i, j = torch.where(cls > conf_th)
            x = torch.cat((box[i], x[i, 4 + j, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = cls.max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_th]

        n = x.shape[0]  # number of boxes
        if not n:
            continue
        if n > max_nms:
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]

        c = x[:, 5:6] * max_wh  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]

        idx = torchvision.ops.nms(boxes, scores, iou_th)  # NMS
        idx = idx[:max_det]

        output[xi] = x[idx]
        if (time.time() - start_time) > time_limit:
            print(f"WARNING ⚠️ NMS time limit {time_limit:.3f}s exceeded")
            break

    return output


def smooth(y, f=0.05):
    nf = round(
        len(y) * f * 2) // 2 + 1
    p = np.ones(nf // 2)
    yp = np.concatenate((p * y[0], y, p * y[-1]), 0)
    return np.convolve(yp, np.ones(nf) / nf, mode="valid")


def compute_ap(tp, conf, pred, target, plot=False, on_plot=None,
               save_dir=Path(), names={}, return_details=False):
    save_dir = Path(save_dir)
    pr_val = []
    i = np.argsort(-conf)
    tp, conf, pred = tp[i], conf[i], pred[i]
    x = np.linspace(0, 1, 1000)
    unique_cls, nt = np.unique(target, return_counts=True)
    name_lookup = {int(k): v for k, v in names.items()}
    unique_class_ids = [int(c) for c in unique_cls.tolist()]

    p = np.zeros((unique_cls.shape[0], 1000))
    r = np.zeros((unique_cls.shape[0], 1000))
    ap = np.zeros((unique_cls.shape[0], tp.shape[1]))

    for ci, c in enumerate(unique_cls):
        i = pred == c
        nl, no = nt[ci], i.sum()
        if no == 0 or nl == 0: continue

        # Recall
        tpc = tp[i].cumsum(0)
        fpc = (1 - tp[i]).cumsum(0)

        recall = tpc / (nt[ci] + 1e-16)
        r[ci] = np.interp(-x, -conf[i], recall[:, 0], left=0)

        # Precision
        precision = tpc / (tpc + fpc)
        p[ci] = np.interp(-x, -conf[i], precision[:, 0], left=1)

        # AP from recall-precision curve
        for j in range(tp.shape[1]):
            mrec = np.concatenate(([0.0], recall[:, j], [1.0]))
            mp_re = np.concatenate(([1.0], precision[:, j], [0.0]))
            mp_re = np.flip(np.maximum.accumulate(np.flip(mp_re)))
            px = np.linspace(start=0, stop=1, num=101)
            ap[ci, j] = np.trapz(np.interp(px, mrec, mp_re), px)

            if j == 0:
                pr_val.append(np.interp(x, mrec, mp_re))

    pr_val = np.array(pr_val)

    f1 = 2 * p * r / (p + r + 1e-16)
    plot_names = {
        index: name_lookup.get(int(class_id), str(int(class_id)))
        for index, class_id in enumerate(unique_class_ids)
    }
    if plot:
        plot_pr_curve(x, pr_val, ap, save_dir / f"PR_curve.png", plot_names,
                      plot=on_plot)
        plot_mc_curve(x, f1, save_dir / f"F1_curve.png", plot_names, y="F1",
                      plot=on_plot)
        plot_mc_curve(x, p, save_dir / f"P_curve.png", plot_names, y="Precision",
                      plot=on_plot)
        plot_mc_curve(x, r, save_dir / f"R_curve.png", plot_names, y="Recall",
                      plot=on_plot)

    i = smooth(f1.mean(0), 0.1).argmax()
    p, r, f1 = p[:, i], r[:, i], f1[:, i]

    mean_ap, map50 = ap.mean(), ap[:, 0].mean()
    m_pre, m_rec = p.mean(), r.mean()  # precision, recall

    class_to_index = {int(class_id): index for index, class_id in enumerate(unique_class_ids)}
    all_class_ids = sorted(set(name_lookup.keys()) | set(unique_class_ids)) if name_lookup else unique_class_ids
    per_class = []
    for class_id in all_class_ids:
        if class_id in class_to_index:
            index = class_to_index[class_id]
            class_instances = int(nt[index])
            class_precision = float(p[index])
            class_recall = float(r[index])
            class_map50 = float(ap[index, 0])
            class_mean_ap = float(ap[index].mean())
        else:
            class_instances = 0
            class_precision = 0.0
            class_recall = 0.0
            class_map50 = 0.0
            class_mean_ap = 0.0

        per_class.append({
            "class_id": int(class_id),
            "name": name_lookup.get(int(class_id), str(int(class_id))),
            "instances": class_instances,
            "precision": class_precision,
            "recall": class_recall,
            "mAP50": class_map50,
            "mAP50-95": class_mean_ap,
        })

    # 先输出总体指标，再输出各类别详情
    print(f"\n{'ClassID':<8} {'Class':<20} {'Instances':<12} {'Precision':<12} {'Recall':<12} {'mAP50':<12} {'mAP50-95':<12}")
    print(f"{'all':<8} {'all':<20} {nt.sum():<12} {m_pre:<12.3f} {m_rec:<12.3f} {map50:<12.3f} {mean_ap:<12.3f}")
    for row in per_class:
        print(f"{row['class_id']:<8} {row['name']:<20} {row['instances']:<12} {row['precision']:<12.3f} {row['recall']:<12.3f} {row['mAP50']:<12.3f} {row['mAP50-95']:<12.3f}")

    # tp = (r * nt).round()  # true positive
    # fp = (tp / (p + 1e-16) - tp).round() # false positive
    # weight = [0.0, 0.0, 0.1, 0.9]
    # fitness = (np.array([m_pre, m_rec, map50, mean_ap]) * weight).sum()

    result = {"precision": m_pre, "recall": m_rec, "mAP50": map50,
              "mAP50-95": mean_ap}
    if return_details:
        result["per_class"] = per_class
    return result


# ----------------------- Compute AP End ---------------------

# ----------------------------- Metrics & Plotting Start -----
def update_metrics(preds, batch, niou, iou_v, stats):
    for i, pred in enumerate(preds):
        stat = dict(conf=torch.zeros(0).cuda(), pred_cls=torch.zeros(0).cuda(),
                    tp=torch.zeros(len(pred), niou, dtype=torch.bool).cuda())

        idx = batch["idx"] == i
        box = batch["box"][idx]
        cls = batch["cls"][idx]
        cls = cls.squeeze(-1)

        if len(cls):
            img_shape = batch["img"].shape[2:]
            tensor = torch.tensor(img_shape).cuda()[[1, 0, 1, 0]]
            box = wh2xy(box) * tensor
            scale_boxes(box, batch["shape"][i], batch["pad"][i])

        stat["target_cls"] = cls
        stat["target_img"] = cls.unique()
        if len(pred) == 0:
            if len(cls):
                for k in stats.keys():
                    stats[k].append(stat[k])
            continue

        output = pred.clone()
        scale_boxes(output[:, :4], batch["shape"][i], batch["pad"][i])

        stat["conf"] = output[:, 4]
        stat["pred_cls"] = output[:, 5]

        # Evaluate
        if len(cls):
            iou = box_iou(box, output[:, :4])
            stat["tp"] = match_predictions(iou_v, output, cls, iou)

        for k in stats.keys():
            stats[k].append(stat[k])
    return stats


def box_iou(box1, box2, eps=1e-7):
    (a1, a2) = box1.float().unsqueeze(1).chunk(2, 2)
    (b1, b2) = box2.float().unsqueeze(0).chunk(2, 2)
    inter = (torch.min(a2, b2) - torch.max(a1, b1))
    inter = inter.clamp_(0).prod(2)
    union = (a2 - a1).prod(2) + (b2 - b1).prod(2) - inter
    return inter / (union + eps)


def scale_boxes(boxes, shape, r_pad):
    gain, pad = r_pad[0][0], r_pad[1]
    boxes[..., :4] -= torch.tensor([pad[0], pad[1], pad[0], pad[1]]).cuda()
    boxes[..., :4] /= gain

    boxes[..., [0, 2]] = boxes[..., [0, 2]].clamp(0, shape[1])
    boxes[..., [1, 3]] = boxes[..., [1, 3]].clamp(0, shape[0])
    return boxes


def plot_pr_curve(px, py, ap, save_dir, names, plot):
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    py = np.stack(py, axis=1)

    if 0 < len(names) < 21:
        [ax.plot(px, y, linewidth=1, label=f"{names[i]} {ap[i, 0]:.3f}") for
         i, y in enumerate(py.T)]
    else:
        ax.plot(px, py, linewidth=1, color="grey")

    ax.plot(px, py.mean(1), linewidth=3, color="blue",
            label=f"all classes {ap[:, 0].mean():.3f} mAP@0.5")
    ax.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1),
           title="Precision-Recall Curve")
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")

    fig.savefig(save_dir, dpi=250)
    plt.close(fig)
    if plot:
        plot(save_dir)


def plot_mc_curve(px, py, save_dir, names, y, plot):
    names = names or {}
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)

    if 0 < len(names) < 21:
        [ax.plot(px, y, linewidth=1, label=f"{names[i]}") for i, y in
         enumerate(py)]
    else:
        ax.plot(px, py.T, linewidth=1, color="grey")

    y_smooth = smooth(py.mean(0), 0.05)
    max_val, max_id = y_smooth.max(), px[y_smooth.argmax()]

    ax.plot(px, y_smooth, linewidth=3, color="blue",
            label=f"all classes {max_val:.2f} at {max_id:.3f}")
    ax.set(xlabel="Confidence", ylabel=y, xlim=(0, 1), ylim=(0, 1),
           title=f"{y}-Confidence Curve")

    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    fig.savefig(save_dir, dpi=250)
    plt.close(fig)
    if plot:
        plot(save_dir)


def match_predictions(iou_v, pred, cls, iou):
    correct = np.zeros((len(pred[:, 5]), len(iou_v)), dtype=bool)
    # Detach before converting to numpy to avoid RuntimeError
    
    iou = (iou * (cls[:, None] == pred[:, 5])).cpu().numpy()

    for i, th in enumerate(iou_v.cpu().tolist()):
        match = np.array(np.nonzero(iou >= th)).T
        if match.size > 0:
            match = match[iou[match[:, 0], match[:, 1]].argsort()[::-1]]
            match = match[np.unique(match[:, 1], return_index=True)[1]]
            match = match[np.unique(match[:, 0], return_index=True)[1]]
            correct[match[:, 1].astype(int), i] = True
    return torch.tensor(correct, dtype=torch.bool, device=pred[:, 5].device)


# ----------------------- Prototype Extractor Start -----------------

class PrototypeExtractor:
    """
    Prototype 特征提取器
    
    用于增量学习中计算各类别的 prototype 特征均值。
    复用 Assigner 进行 GT-anchor 匹配，从分类前特征中按类别聚合。
    
    用法:
        extractor = PrototypeExtractor(model, nc=20)
        
        # 训练过程中累积
        for images, batch in dataloader:
            extractor.update(model, images, batch)
        
        # 获取 prototype
        prototypes = extractor.get_prototypes()  # {class_id: [C,] tensor}
    """
    
    def __init__(self, model, nc, device='cuda'):
        """
        Args:
            model: YOLO 模型
            nc: 总类别数
            device: 计算设备
        """
        self.nc = nc
        self.device = device
        self.stride = model.detect.stride
        self.reg_max = model.detect.reg_max
        
        # 获取特征通道数 (从 shared_blocks 推断)
        # shared_blocks 输出通道 = cls_channels
        sample_block = model.detect.shared_blocks[0][1][-1]  # 最后一个 Conv
        self.feat_dim = sample_block.conv.out_channels
        
        # 累积器: 每个类别的特征总和与计数
        self.feat_sum = torch.zeros(nc, self.feat_dim, device=device)
        self.feat_count = torch.zeros(nc, device=device)
        
        # Assigner 用于 GT-anchor 匹配
        self.assigner = Assigner(nc=nc)
        
        # 用于 bbox 解码的投影向量
        self.proj = torch.arange(self.reg_max, dtype=torch.float, device=device)
    
    def reset(self):
        """重置累积器"""
        self.feat_sum.zero_()
        self.feat_count.zero_()
    
    def _bbox_decode(self, anchor, pred_dist):
        """解码边界框"""
        b, a, c = pred_dist.shape
        pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3)
        pred_dist = pred_dist.matmul(self.proj.type(pred_dist.dtype))
        lt, rb = pred_dist.chunk(2, -1)
        x1y1, x2y2 = anchor - lt, anchor + rb
        return torch.cat((x1y1, x2y2), -1)
    
    @torch.no_grad()
    def update(self, model, images, batch):
        """
        处理一个 batch，累积 prototype 特征
        
        Args:
            model: YOLO 模型 (需要有 forward_for_prototype 方法)
            images: 输入图像 [B, 3, H, W]
            batch: 标注数据 dict, 包含 'idx', 'cls', 'box'
        """
        # 1. 前向传播，获取预测和分类前特征
        pred, neck_feats, proto_feats = model.forward_for_prototype(images)
        
        # pred: 多尺度预测列表 [p3, p4, p5]
        # proto_feats: 分类前特征列表 [feat_p3, feat_p4, feat_p5]
        
        bs = images.shape[0]
        device = images.device
        
        # 2. 拼接多尺度预测
        no = model.detect.no  # 输出通道数
        nc_steps = model.detect.nc_steps
        x = torch.cat([f.view(bs, no, -1) for f in pred], 2)
        
        # 分离 box 和 cls
        box_dims = 4 * self.reg_max
        pred_distri = x[:, :box_dims, :].permute(0, 2, 1).contiguous()
        pred_scores = x[:, box_dims:, :].permute(0, 2, 1).contiguous()
        
        # 3. 生成 anchors
        anchor_points, stride_tensor = make_anchors(pred, self.stride, 0.5)
        
        # 4. 解码边界框
        pred_bboxes = self._bbox_decode(anchor_points, pred_distri)
        
        # 5. 预处理 GT
        idx = batch["idx"].view(-1, 1)
        cls = batch["cls"].view(-1, 1)
        box = batch["box"]
        
        img_size = torch.tensor(pred[0].shape[2:], device=device, dtype=pred_scores.dtype)
        img_size = img_size * self.stride[0]
        
        targets = torch.cat((idx, cls, box), 1).to(device)
        targets = self._preprocess(targets, bs, img_size[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        
        # 6. 使用 Assigner 分配 GT 到 anchor
        target_bboxes, target_scores, fg_mask = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels, gt_bboxes, mask_gt, nc1=self.nc
        )
        
        # 7. 从 proto_feats 中提取前景 anchor 对应的特征
        # 计算每个尺度的 anchor 数量
        n_anchors_per_scale = [f.shape[2] * f.shape[3] for f in proto_feats]
        
        # 拼接所有尺度的特征到 [B, C, N_total]
        feats_flat = []
        for feat in proto_feats:
            b, c, h, w = feat.shape
            feats_flat.append(feat.view(b, c, -1))
        feats_concat = torch.cat(feats_flat, dim=2)  # [B, C, N_anchors]
        
        # 8. 按类别聚合特征（向量化实现，使用 scatter_add）
        for b_idx in range(bs):
            fg_indices = fg_mask[b_idx].nonzero(as_tuple=True)[0]
            if len(fg_indices) == 0:
                continue
            
            # 获取前景 anchor 的特征 [N_fg, C]
            fg_feats = feats_concat[b_idx, :, fg_indices].T  # [N_fg, C]
            
            # 获取前景 anchor 的类别 (从 target_scores 中取 argmax)
            fg_scores = target_scores[b_idx, fg_indices, :]  # [N_fg, nc]
            fg_classes = fg_scores.argmax(dim=1)  # [N_fg]
            
            # 使用 scatter_add 向量化累积（替代 for 循环）
            # 扩展 fg_classes 到 [N_fg, C] 用于 scatter
            cls_idx = fg_classes.unsqueeze(1).expand(-1, fg_feats.shape[1])  # [N_fg, C]
            
            # 累积特征和
            self.feat_sum.scatter_add_(0, cls_idx, fg_feats)
            
            # 累积计数（使用 bincount）
            counts = torch.bincount(fg_classes, minlength=self.nc).float()
            self.feat_count += counts
    
    def _preprocess(self, gt, bs, scale):
        """预处理 GT 数据 (复用 DetectionLoss.preprocess 逻辑)"""
        nl, ne = gt.shape
        if nl == 0:
            out = torch.zeros(bs, 0, ne - 1, device=self.device)
        else:
            i = gt[:, 0]
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(bs, counts.max(), ne - 1, device=self.device)
            for j in range(bs):
                matches = i == j
                n = matches.sum()
                if n:
                    out[j, :n] = gt[matches, 1:]
            out[..., 1:5] = wh2xy(out[..., 1:5].mul_(scale))
        return out
    
    def get_prototypes(self):
        """
        获取各类别的 prototype 特征均值
        
        Returns:
            dict: {class_id: prototype_tensor [C,]}
                  只返回有样本的类别
        """
        prototypes = {}
        for cls_id in range(self.nc):
            if self.feat_count[cls_id] > 0:
                prototypes[cls_id] = self.feat_sum[cls_id] / self.feat_count[cls_id]
        return prototypes
    
    def get_prototype_matrix(self):
        """
        获取 prototype 矩阵
        
        Returns:
            prototypes: [nc, C] tensor, 无样本的类别为零向量
            counts: [nc,] tensor, 每个类别的样本数
        """
        prototypes = torch.zeros_like(self.feat_sum)
        valid_mask = self.feat_count > 0
        prototypes[valid_mask] = self.feat_sum[valid_mask] / self.feat_count[valid_mask].unsqueeze(1)
        return prototypes, self.feat_count.clone()
    
    def save(self, path):
        """保存 prototype 到文件"""
        torch.save({
            'feat_sum': self.feat_sum,
            'feat_count': self.feat_count,
            'nc': self.nc,
            'feat_dim': self.feat_dim
        }, path)
        print(f"✓ Prototype 已保存: {path}")
    
    def load(self, path):
        """从文件加载 prototype"""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.feat_sum = ckpt['feat_sum'].to(self.device)
        self.feat_count = ckpt['feat_count'].to(self.device)
        print(f"✓ Prototype 已加载: {path}")
    
    def print_stats(self):
        """打印统计信息"""
        print(f"\n[Prototype 统计]")
        print(f"  总类别数: {self.nc}")
        print(f"  特征维度: {self.feat_dim}")
        valid_classes = (self.feat_count > 0).sum().item()
        print(f"  有效类别: {valid_classes}/{self.nc}")
        print(f"  总样本数: {self.feat_count.sum().item():.0f}")
        
        if valid_classes > 0:
            print(f"\n  各类别样本数:")
            for cls_id in range(self.nc):
                if self.feat_count[cls_id] > 0:
                    print(f"    类别 {cls_id}: {self.feat_count[cls_id].item():.0f}")


# ----------------------- Prototype Extractor End -----------------


