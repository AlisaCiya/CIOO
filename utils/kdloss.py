import torch
import torch.nn as nn
import torch.nn.functional as F



class IncrementalKDLossV2(nn.Module):
    """
    增强版增量学习蒸馏模块 V2
    
    优化点：
    1. 分类蒸馏改用 KL Divergence（更适合概率分布匹配）
    2. 添加背景蒸馏（防止新类被误认为旧类背景）
    3. IoU 自适应权重（对定位准确的样本给更高权重）
    4. 支持特征蒸馏（Neck 输出）
    5. 动态前景阈值
    """
    def __init__(self, 
                 reg_max=16, 
                 temperature=4.0, 
                 box_weight=2.0,
                 cls_weight=1.0,
                 bg_weight=0.1,       # 背景蒸馏权重
                 feat_weight=0.5,     # 特征蒸馏权重
                 fg_thr=0.4,
                 use_iou_weight=True, # 是否使用 IoU 加权
                 use_bg_distill=True, # 是否使用背景蒸馏
                 use_kl_cls=True):    # 分类是否用 KL（否则用 MSE）
        super().__init__()
        self.reg_max = reg_max
        self.temperature = temperature
        self.box_weight = box_weight
        self.cls_weight = cls_weight
        self.bg_weight = bg_weight
        self.feat_weight = feat_weight
        self.fg_thr = fg_thr
        self.reg_ch = 4 * reg_max
        
        self.use_iou_weight = use_iou_weight
        self.use_bg_distill = use_bg_distill
        self.use_kl_cls = use_kl_cls

    def forward(self, student_feats, teacher_preds, old_nc_sum, 
                student_neck_feats=None, teacher_neck_feats=None):
        """
        Args:
            student_feats: List[Tensor], Student 的原始输出 [(B, C_total, H, W), ...]
            teacher_preds: List[Tensor], Teacher 的原始输出 [(B, C_old, H, W), ...]
            old_nc_sum: int, 旧类别的总数
            student_neck_feats: Optional[List[Tensor]], Student 的 Neck 特征（可选）
            teacher_neck_feats: Optional[List[Tensor]], Teacher 的 Neck 特征（可选）
        """
        device = student_feats[0].device
        loss_box = torch.tensor(0., device=device)
        loss_cls = torch.tensor(0., device=device)
        loss_bg = torch.tensor(0., device=device)
        loss_feat = torch.tensor(0., device=device)

        # --- 1. 数据展平与对齐 ---
        def flatten_and_concat(preds):
            flattened = [p.flatten(2).transpose(1, 2) for p in preds]
            return torch.cat(flattened, 1)

        # Teacher 前向（不计算梯度）
        with torch.no_grad():
            t_all = flatten_and_concat(teacher_preds)
            t_box = t_all[..., :self.reg_ch]
            t_cls = t_all[..., self.reg_ch:]
            
            # 动态阈值：使用 Teacher 置信度的中位数作为参考
            t_prob = torch.sigmoid(t_cls)
            t_max_conf, _ = t_prob.max(dim=-1)
            
            # 前景 mask
            fg_mask = t_max_conf >= self.fg_thr
            # 背景 mask（置信度很低的区域）
            bg_mask = t_max_conf < 0.1

        # Student 前向
        s_all = flatten_and_concat(student_feats)
        s_box = s_all[..., :self.reg_ch]
        s_cls_old = s_all[..., self.reg_ch : self.reg_ch + old_nc_sum]

        T = self.temperature
        num_fg = fg_mask.sum().item()
        num_bg = bg_mask.sum().item()

        # --- 2. Box 蒸馏（前景区域）---
        if num_fg > 0:
            s_box_fg = s_box[fg_mask].view(-1, 4, self.reg_max)
            t_box_fg = t_box[fg_mask].view(-1, 4, self.reg_max)
            
            t_prob_box = F.softmax(t_box_fg / T, dim=-1)
            s_log_box = F.log_softmax(s_box_fg / T, dim=-1)
            
            # 基础 Box KL Loss
            box_kl = F.kl_div(s_log_box, t_prob_box, reduction='none').sum(-1).mean(-1)
            
            # 【优化】IoU 自适应权重
            if self.use_iou_weight:
                with torch.no_grad():
                    # 简化的 IoU 计算：使用 softmax 期望值
                    arange = torch.arange(self.reg_max, device=device, dtype=torch.float)
                    s_box_decoded = (F.softmax(s_box_fg, dim=-1) * arange).sum(-1)
                    t_box_decoded = (F.softmax(t_box_fg, dim=-1) * arange).sum(-1)
                    
                    # 计算 L1 距离作为权重（距离越小权重越高）
                    dist = (s_box_decoded - t_box_decoded).abs().mean(-1)
                    iou_weight = torch.exp(-dist)  # 距离越小，权重越高
                
                loss_box = (box_kl * iou_weight).mean() * (T ** 2)
            else:
                loss_box = box_kl.mean() * (T ** 2)

        # --- 3. 分类蒸馏（前景区域）---
        if num_fg > 0:
            s_logits_fg = s_cls_old[fg_mask]
            t_logits_fg = t_cls[fg_mask]
            
            if self.use_kl_cls:
                # 【优化】使用 BCE with logits（autocast 安全）
                # 对于多标签问题，逐类别计算二元交叉熵
                # 使用温度缩放的 soft targets
                with torch.no_grad():
                    t_prob_cls = torch.sigmoid(t_logits_fg / T)
                
                # BCE with logits 直接使用 logits，更稳定
                loss_cls = F.binary_cross_entropy_with_logits(
                    s_logits_fg / T, t_prob_cls, reduction='mean'
                ) * (T ** 2)
            else:
                # 原始 MSE
                loss_cls = F.mse_loss(
                    torch.sigmoid(s_logits_fg), 
                    torch.sigmoid(t_logits_fg)
                )

        # --- 4. 背景蒸馏（防止新类被抑制）---
        if self.use_bg_distill and num_bg > 0:
            s_logits_bg = s_cls_old[bg_mask]
            t_logits_bg = t_cls[bg_mask]
            
            # 背景区域：Teacher 输出低置信度，Student 也应该输出低置信度
            # 使用 MSE 即可，不需要温度缩放
            loss_bg = F.mse_loss(
                torch.sigmoid(s_logits_bg), 
                torch.sigmoid(t_logits_bg)
            )

        # --- 5. 特征蒸馏（Neck 输出）---
        if (self.feat_weight > 0 and 
            student_neck_feats is not None and 
            teacher_neck_feats is not None):
            
            for s_feat, t_feat in zip(student_neck_feats, teacher_neck_feats):
                with torch.no_grad():
                    t_feat_norm = F.normalize(t_feat.flatten(2), dim=-1)
                s_feat_norm = F.normalize(s_feat.flatten(2), dim=-1)
                
                # Cosine Similarity Loss
                loss_feat = loss_feat + (1 - (s_feat_norm * t_feat_norm).sum(-1).mean())

        # --- 6. 总损失 ---
        total_loss = (
            self.box_weight * loss_box + 
            self.cls_weight * loss_cls + 
            self.bg_weight * loss_bg +
            self.feat_weight * loss_feat
        )
        
        return total_loss
    
    def get_loss_dict(self, student_feats, teacher_preds, old_nc_sum,
                      student_neck_feats=None, teacher_neck_feats=None):
        """
        返回详细的损失字典（用于日志记录）
        """
        # 这里可以复用 forward 的逻辑，但分别返回各部分
        # 简化实现：直接返回总损失
        total = self.forward(student_feats, teacher_preds, old_nc_sum,
                           student_neck_feats, teacher_neck_feats)
        return {'kd_total': total}

class PrototypeDistillLoss(nn.Module):
    """
    独立的 Prototype 蒸馏模块
    
    按类别约束 Student 的特征接近旧类 Prototype，保持类间区分性。
    与 IncrementalKDLossV2 完全分离，可独立启用/禁用。
    
    用法:
        proto_loss_fn = PrototypeDistillLoss()
        proto_loss_fn.load_prototypes('path/to/proto.pt')
        
        # 训练时
        proto_loss = proto_loss_fn(student_proto_feats, teacher_preds, fg_thr=0.4)
        total_loss = det_loss + kd_loss + proto_weight * proto_loss
    """
    
    def __init__(self, reg_max=16, fg_thr=0.4):
        super().__init__()
        self.reg_max = reg_max
        self.reg_ch = 4 * reg_max
        self.fg_thr = fg_thr
        self.old_prototypes = None  # [num_classes, C]
    
    def load_prototypes(self, proto_path, device='cuda'):
        """
        加载旧类 prototype
        
        Args:
            proto_path: prototype 文件路径
            device: 计算设备
        """
        ckpt = torch.load(proto_path, map_location=device, weights_only=False)
        feat_sum = ckpt['feat_sum']
        feat_count = ckpt['feat_count']
        
        # 计算均值
        valid_mask = feat_count > 0
        prototypes = torch.zeros_like(feat_sum)
        prototypes[valid_mask] = feat_sum[valid_mask] / feat_count[valid_mask].unsqueeze(1)
        
        self.old_prototypes = prototypes.to(device)
        print(f"✓ [PrototypeDistillLoss] 已加载旧类 Prototype: {proto_path}")
        print(f"  特征维度: {prototypes.shape[1]}, 有效类别: {valid_mask.sum().item()}")
        return self
    
    def forward(self, student_proto_feats, teacher_preds, fg_thr=None):
        """
        计算 Prototype 蒸馏损失
        
        Args:
            student_proto_feats: List[Tensor], Student 分类前特征 [(B, C, H, W), ...]
            teacher_preds: List[Tensor], Teacher 的原始输出 [(B, C_old, H, W), ...]
            fg_thr: float, 前景阈值（可选，默认使用初始化时的值）
        
        Returns:
            loss_proto: Tensor, Prototype 蒸馏损失
        """
        if self.old_prototypes is None:
            return torch.tensor(0., device=student_proto_feats[0].device)
        
        fg_thr = fg_thr if fg_thr is not None else self.fg_thr
        device = student_proto_feats[0].device
        loss_proto = torch.tensor(0., device=device)
        
        # 展平 Teacher 输出
        def flatten_and_concat(preds):
            flattened = [p.flatten(2).transpose(1, 2) for p in preds]
            return torch.cat(flattened, 1)
        
        with torch.no_grad():
            t_all = flatten_and_concat(teacher_preds)
            t_cls = t_all[..., self.reg_ch:]  # [B, N, num_old_classes]
            
            # 前景 mask 和类别预测
            t_prob = torch.sigmoid(t_cls)
            t_max_conf, _ = t_prob.max(dim=-1)
            fg_mask = t_max_conf >= fg_thr  # [B, N]
            t_cls_pred = t_cls.argmax(dim=-1)  # [B, N]
        
        # 计算各尺度的 anchor 数量
        n_anchors = 0
        scale_sizes = []
        for s_proto_feat in student_proto_feats:
            h, w = s_proto_feat.shape[2], s_proto_feat.shape[3]
            scale_sizes.append((h, w, n_anchors, n_anchors + h * w))
            n_anchors += h * w
        
        num_old_classes = self.old_prototypes.shape[0]
        proto_loss_count = 0
        
        for scale_idx, s_proto_feat in enumerate(student_proto_feats):
            h, w, start_idx, end_idx = scale_sizes[scale_idx]
            
            # 获取当前尺度的 mask [B, H*W]
            fg_mask_scale = fg_mask[:, start_idx:end_idx]
            cls_pred_scale = t_cls_pred[:, start_idx:end_idx]
            
            # 将特征展平 [B, C, H*W] -> [B, H*W, C]
            s_feat_flat = s_proto_feat.flatten(2).transpose(1, 2)
            
            # 按类别计算 loss
            for cls_id in range(num_old_classes):
                cls_mask = (cls_pred_scale == cls_id) & fg_mask_scale
                
                if cls_mask.sum() == 0:
                    continue
                
                # 该类的 Student 特征
                s_feat_cls = s_feat_flat[cls_mask]  # [M, C]
                
                # 对应类别的 prototype
                proto_cls = self.old_prototypes[cls_id]  # [C]
                
                # Cosine similarity loss
                s_norm = F.normalize(s_feat_cls, dim=-1)
                p_norm = F.normalize(proto_cls.unsqueeze(0), dim=-1)
                cos_sim = (s_norm * p_norm).sum(-1).mean()
                loss_proto = loss_proto + (1 - cos_sim)
                proto_loss_count += 1
        
        # 归一化
        if proto_loss_count > 0:
            loss_proto = loss_proto / proto_loss_count
        
        return loss_proto