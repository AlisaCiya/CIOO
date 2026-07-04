import math
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import util


def fuse_conv(conv, norm):
    fused_conv = torch.nn.Conv2d(conv.in_channels,
                                 conv.out_channels,
                                 kernel_size=conv.kernel_size,
                                 stride=conv.stride,
                                 padding=conv.padding,
                                 groups=conv.groups,
                                 bias=True).requires_grad_(False).to(
        conv.weight.device)

    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_norm = torch.diag(
        norm.weight.div(torch.sqrt(norm.eps + norm.running_var)))
    fused_conv.weight.copy_(
        torch.mm(w_norm, w_conv).view(fused_conv.weight.size()))

    b_conv = torch.zeros(conv.weight.size(0),
                         device=conv.weight.device) if conv.bias is None else conv.bias
    b_norm = norm.bias - norm.weight.mul(norm.running_mean).div(
        torch.sqrt(norm.running_var + norm.eps))
    fused_conv.bias.copy_(
        torch.mm(w_norm, b_conv.reshape(-1, 1)).reshape(-1) + b_norm)

    return fused_conv


class Concat(nn.Module):
    """Concatenate a list of tensors along dimension."""

    def __init__(self, dimension=1):
        """Concatenates a list of tensors along a specified dimension."""
        super().__init__()
        self.d = dimension

    def forward(self, x):
        """Forward pass for the YOLOv8 mask Proto module."""
        return torch.cat(x, self.d)


class Conv(nn.Module):
    def __init__(self, inp, oup, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(inp, oup, k, s, self._pad(k, p), d, g, False)
        self.norm = nn.BatchNorm2d(oup)
        self.act = nn.SiLU(inplace=True) if act is True else nn.Identity()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))

    @staticmethod
    def _pad(k, p=None):
        if p is None:
            p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
        return p


class Residual(nn.Module):
    def __init__(self, inp, g=1, k=(3, 3), e=0.5):
        super().__init__()
        self.conv1 = Conv(inp, int(inp * e), k[0], 1)
        self.conv2 = Conv(int(inp * e), inp, k[1], 1, g=g)

    def forward(self, x):
        return x + self.conv2(self.conv1(x))


class CSPBlock(torch.nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = Conv(in_ch, out_ch // 2)
        self.conv2 = Conv(in_ch, out_ch // 2)
        self.conv3 = Conv(2 * (out_ch // 2), out_ch)
        self.res_m = torch.nn.Sequential(Residual(out_ch // 2, e=1.0),
                                         Residual(out_ch // 2, e=1.0))

    def forward(self, x):
        y = self.res_m(self.conv1(x))
        return self.conv3(torch.cat((y, self.conv2(x)), dim=1))


class CSP(torch.nn.Module):
    def __init__(self, in_ch, out_ch, n, csp, r=2):
        super().__init__()
        self.conv1 = Conv(in_ch, 2 * (out_ch // r))
        self.conv2 = Conv((2 + n) * (out_ch // r), out_ch)

        if not csp:
            self.res_m = torch.nn.ModuleList(
                Residual(out_ch // r) for _ in range(n))
        else:
            self.res_m = torch.nn.ModuleList(
                CSPBlock(out_ch // r, out_ch // r) for _ in range(n))

    def forward(self, x):
        y = list(self.conv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.res_m)
        return self.conv2(torch.cat(y, dim=1))


class SPP(nn.Module):
    def __init__(self, inp, k=5):
        super().__init__()
        self.conv1 = Conv(inp, inp // 2, 1, 1)
        self.conv2 = Conv(inp // 2 * 4, inp, 1, 1)
        self.m = nn.MaxPool2d(k, stride=1, padding=k // 2)

    def forward(self, x):
        y = [self.conv1(x)]
        y.extend(self.m(y[-1]) for _ in range(3))
        return self.conv2(torch.cat(y, 1))


class Attention(nn.Module):
    """Multi-head attention with FlashAttention optimization (PyTorch 2.0+)"""
    
    def __init__(self, dim, num_head=8):
        super().__init__()
        self.num_head = num_head
        self.head_dim = dim // num_head
        self.key_dim = self.head_dim // 2
        self.scale = self.key_dim ** -0.5
        h = dim + self.key_dim * num_head * 2

        # Convolution for query, key, and value
        self.qkv_conv = Conv(dim, h, 1, act=False)

        # Projection and Positional encoding convolution
        self.proj_conv = Conv(dim, dim, 1, act=False)
        self.pe_conv = Conv(dim, dim, 3, g=dim, act=False)

    def forward(self, x):
        b, ch, h, w = x.shape

        qkv = self.qkv_conv(x)
        qkv = qkv.view(b, self.num_head, self.key_dim * 2 + self.head_dim, h * w)
        q, k, v = qkv.split([self.key_dim, self.key_dim, self.head_dim], dim=2)
        
        # 使用 scaled_dot_product_attention 获得 FlashAttention 优化
        # q, k, v: (B, num_head, dim, seq_len) -> 需要转置为 (B, num_head, seq_len, dim)
        q_t = q.transpose(-2, -1)  # (B, num_head, seq_len, key_dim)
        k_t = k.transpose(-2, -1)  # (B, num_head, seq_len, key_dim)
        v_t = v.transpose(-2, -1)  # (B, num_head, seq_len, head_dim)
        
        # F.scaled_dot_product_attention 自动应用 scale 和 softmax
        attn_out = F.scaled_dot_product_attention(q_t, k_t, v_t, scale=self.scale)
        # attn_out: (B, num_head, seq_len, head_dim)
        
        out = attn_out.transpose(-2, -1).reshape(b, ch, h, w)
        
        return self.proj_conv(out + self.pe_conv(v.reshape(b, ch, h, w)))


class PSABlock(nn.Module):
    def __init__(self, inp, num_head=4):
        super().__init__()
        self.att = Attention(inp, num_head)
        self.ffn = nn.Sequential(Conv(inp, inp * 2, 1),
                                 Conv(inp * 2, inp, 1, act=False))

    def forward(self, x):
        x = x + self.att(x)
        return x + self.ffn(x)


class PSA(nn.Module):
    def __init__(self, inp, oup, n=1):
        super().__init__()
        assert inp == oup
        self.conv1 = Conv(inp, 2 * (inp // 2))
        self.conv2 = Conv(2 * (inp // 2), inp)

        self.m = nn.Sequential(
            *(PSABlock(inp // 2, inp // 128) for _ in range(n)))

    def forward(self, x):
        a, b = self.conv1(x).chunk(2, 1)
        return self.conv2(torch.cat((a, self.m(b)), 1))


class DWConv(Conv):
    def __init__(self, inp, oup, k=1, s=1, d=1, act=True):
        super().__init__(inp, oup, k, s, g=math.gcd(inp, oup), d=d, act=act)


class DFL(nn.Module):
    """Distribution Focal Loss layer with optimized buffer management."""
    
    def __init__(self, inp=16):
        super().__init__()
        self.inp = inp
        self.conv = nn.Conv2d(inp, 1, 1, bias=False).requires_grad_(False)
        # 使用 register_buffer 确保权重跟随设备移动
        weight = torch.arange(inp, dtype=torch.float).view(1, inp, 1, 1)
        self.conv.weight.data.copy_(weight)

    def forward(self, x):
        b, _, a = x.shape
        out = x.view(b, 4, self.inp, a).transpose(2, 1)
        return self.conv(out.softmax(1)).view(b, 4, a)


class Detect(nn.Module):
    anchors = torch.empty(0)
    strides = torch.empty(0)

    def __init__(self, filters=(), classes=None):
        """
        Args:
            filters: 各尺度的输入通道数
            classes: 增量学习的类别数列表，形式为 [nc_step0, nc_step1, ...]
                    例如: [15] 表示 Step 0 输出15个类别
                         [15, 5] 表示 Step 0 输出15个，Step 1 输出5个新类别
                    如果为 None, 则默认使用单步学习 [80] (COCO)
        """
        super().__init__()
        self.reg_max = 16
        self.nl = len(filters)  # 多尺度数量
        

            # 增量学习模式: classes = [15, 5] 表示各阶段输出的类别数
        self.nc_steps = classes
        self.all_nc = sum(classes)  # 全局类别空间是所有类别总和
        
        current_nc = self.nc_steps[-1]  # 最后一个阶段的类别数
        self.nc = current_nc  # 当前阶段的类别数
        self.step = len(self.nc_steps) - 1  # 当前处于哪个阶段（0-indexed）
        
        # 输出维度: 4*reg_max (边界框) + 各步类别数之和
        self.no = 4 * self.reg_max + sum(self.nc_steps)
        self.stride = torch.zeros(self.nl)

        box = max((filters[0] // 4, 64))
        cls = max(filters[0], min(self.all_nc, 100))

        # Box 头 (共享, 所有类别共用)
        self.box = nn.ModuleList(
            nn.Sequential(Conv(x, box, 3), Conv(box, box, 3),
                          nn.Conv2d(box, 4 * self.reg_max, 1)) for x in
            filters)

        # Cls 头 (多个, 每个对应一个 step)
        # self.cls_heads: List[ModuleList] 
        #   - 外层 List: num_steps 个
        #   - 内层 ModuleList: nl 个尺度
        # self.cls_heads = nn.ModuleList()
        # for step_idx, nc_step in enumerate(self.nc_steps):
        #     cls_head_scales = nn.ModuleList(
        #         nn.Sequential(
        #             nn.Sequential(DWConv(x, x, 3), Conv(x, cls, 1)),
        #             nn.Sequential(DWConv(cls, cls, 3), Conv(cls, cls, 1)),
        #             nn.Conv2d(cls, nc_step, 1)
        #         )
        #         for x in filters
        #     )
        #     self.cls_heads.append(cls_head_scales)

            #          初始化时
        self.shared_blocks = nn.ModuleList(
            nn.Sequential(
                nn.Sequential(DWConv(x, x, 3), Conv(x, cls, 1)),   # block1
                nn.Sequential(DWConv(cls, cls, 3), Conv(cls, cls, 1))  # block2
            )
            for x in filters
        )

        # 每个 step 只存 classifier
        self.cls_heads = nn.ModuleList()
        for nc_step in self.nc_steps:
            cls_head_scales = nn.ModuleList(
                nn.Conv2d(cls, nc_step, 1)   # 只扩展 classifier
                for _ in filters
            )
            self.cls_heads.append(cls_head_scales)

        
        self.dfl = DFL(self.reg_max)

    def forward(self, x, logitsintrain=False):
        """
        前向传播
        
        Args:
            x: 各尺度的特征图列表 [feat_p3, feat_p4, feat_p5]
            logitsintrain: 是否在训练模式下返回 logits

        Returns (训练模式):
            x: 各尺度的预测 [pred_p3, pred_p4, pred_p5]
        
        Returns (推理模式):
            output: (B, N_anchors, 4 + sum(nc_steps))
            x: 原始多尺度预测
        """
        num_steps = self.step + 1
        nl = self.nl
        
        # 预分配列表
        box_preds = [None] * nl
        cls_preds = [[None] * nl for _ in range(num_steps)]
        
        # 单次遍历计算所有输出
        for i in range(nl):
            # Box 预测
            box_preds[i] = self.box[i](x[i])
            
            # Shared blocks 特征
            feat = self.shared_blocks[i][0](x[i])
            feat = self.shared_blocks[i][1](feat)
            
            # 各 step 的分类器输出
            for step_idx in range(num_steps):
                cls_preds[step_idx][i] = self.cls_heads[step_idx][i](feat)
        
        # 合并预测
        for i in range(nl):
            all_cls = [cls_preds[s][i] for s in range(num_steps)]
            x[i] = torch.cat([box_preds[i]] + all_cls, dim=1)
        
        # 计算推理输出
        if logitsintrain or not self.training:
            bs = x[0].shape[0]
            x_cat = torch.cat([xi.view(bs, self.no, -1) for xi in x], dim=2)
            self.anchors, self.strides = (
                j.transpose(0, 1) for j in util.make_anchors(x, self.stride)
            )
            box, cls = x_cat.split((self.reg_max * 4, sum(self.nc_steps)), dim=1)
            lt, rb = self.dfl(box).chunk(2, dim=1)
            x1y1 = self.anchors.unsqueeze(0) - lt
            x2y2 = self.anchors.unsqueeze(0) + rb
            c_xy = (x1y1 + x2y2) * 0.5
            wh = x2y2 - x1y1
            d_box = torch.cat((c_xy, wh), dim=1)
            output = torch.cat((d_box * self.strides, cls.sigmoid()), dim=1)
            
            if logitsintrain:
                return x  # logitsintrain=True: 返回原始多尺度预测列表
            if self.training:
                return output  # training=True: 返回处理后的 output
            return output, x  # 推理模式
        
        # 训练模式
        return x

    def forward_with_proto_feats(self, x):
        """
        前向传播，同时返回分类前的特征（用于 prototype 计算）
        
        Args:
            x: 各尺度的特征图列表 [feat_p3, feat_p4, feat_p5]
        
        Returns:
            x: 各尺度的预测（与 forward 一致）
            proto_feats: 分类前的特征列表 [feat_p3, feat_p4, feat_p5]
                        每个 shape: [B, cls_channels, H, W]
        """
        num_steps = self.step + 1
        nl = self.nl
        
        # 预分配列表
        box_preds = [None] * nl
        cls_preds = [[None] * nl for _ in range(num_steps)]
        proto_feats = [None] * nl  # 存储分类前的特征
        
        # 单次遍历计算所有输出
        for i in range(nl):
            # Box 预测
            box_preds[i] = self.box[i](x[i])
            
            # Shared blocks 特征（分类前）
            feat = self.shared_blocks[i][0](x[i])
            feat = self.shared_blocks[i][1](feat)
            proto_feats[i] = feat  # 保存分类前的特征
            
            # 各 step 的分类器输出
            for step_idx in range(num_steps):
                cls_preds[step_idx][i] = self.cls_heads[step_idx][i](feat)
        
        # 合并预测
        for i in range(nl):
            all_cls = [cls_preds[s][i] for s in range(num_steps)]
            x[i] = torch.cat([box_preds[i]] + all_cls, dim=1)
        
        # 训练模式返回原始多尺度预测 + proto 特征
        return x, proto_feats

    def bias_init(self):
        """初始化偏置"""
        m = self
        num_steps = self.step + 1
        for i, s in enumerate(m.stride):
            # Box 头偏置
            m.box[i][-1].bias.data[:] = 1.0
            
            # 各 Cls 头偏置
            for step_idx in range(num_steps):
                nc_step = self.nc_steps[step_idx]
                m.cls_heads[step_idx][i].bias.data[: nc_step] = \
                    math.log(5 / nc_step / (640 / s) ** 2)


class Backbone(nn.Module):
    def __init__(self, width, depth, csp):
        super().__init__()

        self.p1 = []
        self.p2 = []
        self.p3 = []
        self.p4 = []
        self.p5 = []

        self.p1.append(Conv(width[0], width[1], 3, 2))

        self.p2.append(Conv(width[1], width[2], 3, 2))
        self.p2.append(CSP(width[2], width[3], depth[0], csp[0], 4))

        self.p3.append(Conv(width[3], width[3], 3, 2))
        self.p3.append(CSP(width[3], width[4], depth[1], csp[0], 4))

        self.p4.append(Conv(width[4], width[4], 3, 2))
        self.p4.append(CSP(width[4], width[4], depth[2], csp[1]))

        self.p5.append(Conv(width[4], width[5], 3, 2))
        
        self.p5.append(CSP(width[5], width[5], depth[3], csp[1]))
        
        self.p5.append(SPP(width[5], 5))
        self.p5.append(PSA(width[5], width[5], depth[4]))

        self.p1 = nn.Sequential(*self.p1)
        self.p2 = nn.Sequential(*self.p2)
        self.p3 = nn.Sequential(*self.p3)
        self.p4 = nn.Sequential(*self.p4)
        self.p5 = nn.Sequential(*self.p5)

    def forward(self, x):
        p1 = self.p1(x)
        p2 = self.p2(p1)
        p3 = self.p3(p2)
        p4 = self.p4(p3)
        p5 = self.p5(p4)
        return p3, p4, p5


class Head(nn.Module):
    def __init__(self, width, depth, csp):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.concat = Concat()

        self.h1 = CSP(width[4] + width[5], width[4], depth[0], csp[0])

        self.h2 = CSP(width[4] + width[4], width[3], depth[0], csp[0])

        self.h3 = Conv(width[3], width[3], 3, 2, 1)
        self.h4 = CSP(width[3] + width[4], width[4], depth[0], csp[0])

        self.h5 = Conv(width[4], width[4], 3, 2, 1)
        self.h6 = CSP(width[4] + width[5], width[5], depth[0], csp[1])

       

    def forward(self, x):
        p3, p4, p5 = x
        h1 = self.h1(self.concat([self.up(p5), p4]))
        h2 = self.h2(self.concat([self.up(h1), p3]))
        h4 = self.h4(self.concat([self.h3(h2), h1]))
        h6 = self.h6(self.concat([self.h5(h4), p5]))
        return h2, h4, h6


def initialize_weights(model):
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            pass
        elif t is nn.BatchNorm2d:
            m.eps = 1e-3
            m.momentum = 0.03
        elif t in {nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU}:
            m.inplace = True


class YOLO(torch.nn.Module):
    def __init__(self, width, depth, csp, classes=None):
        super().__init__()
        self.backbone = Backbone(width, depth, csp)
        self.head = Head(width, depth, csp)

        img_dummy = torch.zeros(1, width[0], 256, 256)
        self.detect = Detect(
            filters=(width[3], width[4], width[5]),
            classes=classes
        )
        self.detect.stride = torch.tensor(
            [256 / x.shape[-2] for x in self.forward(img_dummy)])
        self.stride = self.detect.stride
        self.detect.bias_init()
        initialize_weights(self)

    def forward(self, x):
        """标准前向传播（训练/推理）"""
        neck_feats = self._extract_features(x)
        return self.detect(neck_feats)
    
    def _extract_features(self, x):
        """提取 backbone + head 特征（内部复用）"""
        x = self.backbone(x)
        x = self.head(x)
        return list(x)
    
    def fuse(self):
        for m in self.modules():
            if type(m) is Conv and hasattr(m, 'norm'):
                m.conv = fuse_conv(m.conv, m.norm)
                m.forward = m.forward_fuse
                delattr(m, 'norm')
        return self
    
    def forward_with_features(self, x):
        """
        前向传播，同时返回 neck 特征（用于知识蒸馏）
        
        Returns:
            neck_feats: List[Tensor] - neck 输出特征 [p3, p4, p5]
            pred: Tensor - detect 输出
        """
        neck_feats = self._extract_features(x)
        # 复制列表，防止 detect 原地修改
        pred = self.detect([f for f in neck_feats])
        return neck_feats, pred
    
    def forward_teacher(self, x):
        """
        教师模型前向（用于知识蒸馏，返回 logits）
        
        Returns:
            neck_feats: List[Tensor] - neck 输出特征
            logits: Tensor - detect logits 输出
        """
        neck_feats = self._extract_features(x)
        # 复制列表，防止 detect 原地修改
        logits = self.detect([f for f in neck_feats], logitsintrain=True)
        return neck_feats, logits
    
    def forward_for_prototype(self, x):
        """
        前向传播，返回用于 prototype 计算的特征
        
        用于增量学习中计算各类别的 prototype 特征均值。
        
        Args:
            x: 输入图像 [B, 3, H, W]
        
        Returns:
            pred: 检测输出（与 forward 一致）
            neck_feats: Neck 输出特征列表（用于特征蒸馏）
            proto_feats: 分类前的多尺度特征列表（用于 prototype）
                        [feat_p3, feat_p4, feat_p5]
                        每个 shape: [B, cls_channels, H_i, W_i]
        
        示例:
            pred, neck_feats, proto_feats = model.forward_for_prototype(images)
            # neck_feats: 用于特征蒸馏
            # proto_feats: 用于 prototype 蒸馏
        """
        neck_feats = self._extract_features(x)
        pred, proto_feats = self.detect.forward_with_proto_feats([f for f in neck_feats])
        return pred, neck_feats, proto_feats
    

    def load_weights(self, weights_path, verbose=True, shape_match=True):
        """
        加载预训练权重（支持名称匹配和形状匹配）
        
        Args:
            weights_path (str): 权重文件路径
            verbose (bool): 是否打印加载信息
            shape_match (bool): 是否启用形状匹配（用于增量学习跨阶段加载）
        
        Returns:
            int: 成功加载的参数数量
        
        支持格式:
            - {'model': state_dict}
            - {'state_dict': state_dict}
            - 直接的 state_dict
        """
        if not isinstance(weights_path, str) or not os.path.isfile(weights_path):
            if verbose:
                print(f"✗ 权重文件不存在: {weights_path}")
            return 0
        
        try:
            if verbose:
                print(f"\n📥 加载权重: {weights_path}")
            
            # 加载检查点
            ckpt = torch.load(weights_path, map_location='cpu', weights_only=False)
            
            # 提取 state_dict
            if isinstance(ckpt, dict):
                if 'model' in ckpt:
                    src = ckpt['model']
                    if hasattr(src, 'state_dict'):
                        src = src.state_dict()
                elif 'state_dict' in ckpt:
                    src = ckpt['state_dict']
                else:
                    src = ckpt
            else:
                src = ckpt
            if hasattr(src, 'state_dict'):
                src = src.state_dict()
            
            dst = self.state_dict()
            new_sd = dst.copy()
            loaded = 0
            
            # 1) 按名称和形状匹配
            used_src = set()
            for k, v in dst.items():
                if k in src and src[k].shape == v.shape:
                    new_sd[k] = src[k].to(v.device)
                    used_src.add(k)
                    loaded += 1
            
            # 2) 形状匹配（用于增量学习跨阶段加载）
            if shape_match:
                src_items = list(src.items())
                for dst_k, dst_v in dst.items():
                    if dst_k in used_src:
                        continue
                    if dst_k in src and src[dst_k].shape == dst_v.shape:
                        continue
                    for src_k, src_v in src_items:
                        if src_k in used_src:
                            continue
                        if src_v.shape == dst_v.shape:
                            new_sd[dst_k] = src_v.to(dst_v.device)
                            used_src.add(src_k)
                            loaded += 1
                            break
            
            # 加载权重
            self.load_state_dict(new_sd, strict=False)
            
            if verbose:
                missing = len(dst) - loaded
                print(f"   ✓ 加载参数: {loaded}/{len(dst)}")
                if missing > 0:
                    print(f"   ⚠️ 未匹配: {missing} (将使用随机初始化)")
                print()
            
            return loaded
        
        except Exception as e:
            if verbose:
                print(f"   ✗ 加载失败: {e}\n")
            return 0

    def _iter_module_params(self, module, exclude_patterns=None):
            """
            通用参数迭代器，减少重复代码
            
            Args:
                module: 要遍历的模块
                exclude_patterns: 排除的模式列表
            """
            exclude_patterns = exclude_patterns or []
            for name, m in module.named_modules():
                # 检查是否需要排除
                if any(pattern in name for pattern in exclude_patterns):
                    continue
                if isinstance(m, (nn.Conv2d, nn.BatchNorm2d, nn.SyncBatchNorm)):
                    for p in m.parameters():
                        if p.requires_grad:
                            yield p

    def get_old_classifier_params(self):
        """获取旧分类器参数（除了最后一个 step）"""
        if not hasattr(self.detect, 'cls_heads'):
            return
        num_old_steps = len(self.detect.cls_heads) - 1
        if num_old_steps <= 0:
            return
        for step_idx in range(num_old_steps):
            yield from self._iter_module_params(self.detect.cls_heads[step_idx])

    def get_new_classifier_params(self):
        """获取新分类器参数（最后一个 step）"""
        if not hasattr(self.detect, 'cls_heads') or len(self.detect.cls_heads) == 0:
            return iter([])
        return self._iter_module_params(self.detect.cls_heads[-1])



def yolo_v11_n(classes=None):
    """
    初始化 YOLOv11 Nano 模型
    
    Args:
        classes: 各阶段的类别数列表，如 [20] 或 [15, 5]
                如果为 None，默认 [80] (COCO)
    """
    if classes is None:
        classes = [20]
    
    csp = [False, True]
    depth = [1, 1, 1, 1, 1]
    width = [3, 16, 32, 64, 128, 256]
    
    return YOLO(width, depth, csp, classes)


def yolo_v11_s(classes=None):
    """初始化 YOLOv11 Small 模型，参数同 yolo_v11_n()"""
    if classes is None:
        classes = [80]
    
    csp = [False, True]
    depth = [1, 1, 1, 1, 1]
    width = [3, 32, 64, 128, 256, 512]
    
    return YOLO(width, depth, csp, classes)


def yolo_v11_m(classes=None):
    """初始化 YOLOv11 Medium 模型，参数同 yolo_v11_n()"""
    if classes is None:
        classes = [80]
    
    csp = [True, True]
    depth = [1, 1, 1, 1, 1]
    width = [3, 64, 128, 256, 512, 512]
    
    return YOLO(width, depth, csp, classes)


def yolo_v11_l(classes=None):
    """初始化 YOLOv11 Large 模型，参数同 yolo_v11_n()"""
    if classes is None:
        classes = [80]
    
    csp = [True, True]
    depth = [2, 2, 2, 2, 2]
    width = [3, 64, 128, 256, 512, 512]
    
    return YOLO(width, depth, csp, classes)


def yolo_v11_x(classes=None):
    """初始化 YOLOv11 Extra Large 模型，参数同 yolo_v11_n()"""
    if classes is None:
        classes = [80]
    
    csp = [True, True]
    depth = [2, 2, 2, 2, 2]
    width = [3, 96, 192, 384, 768, 768]
    
    return YOLO(width, depth, csp, classes)


# ============================================================================
# 工具函数
# ============================================================================


def get_model_size(model, print_info=True):
    """
    计算模型大小 (参数数量和内存占用)
    
    Args:
        model: YOLO 模型实例
        print_info: 是否打印详细信息
    
    Returns:
        dict: 包含:
            - 'total_params': 总参数数
            - 'trainable_params': 可训练参数数
            - 'model_size_mb': 模型大小 (MB)
    
    示例:
        size_info = get_model_size(model)
        print(f"模型大小: {size_info['model_size_mb']:.1f} MB")
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # 计算内存占用 (以 float32 计算)
    model_size_mb = (total_params * 4) / (1024 * 1024)
    
    if print_info:
        print(f"\n[模型大小统计]")
        print(f"  总参数数: {total_params:,}")
        print(f"  可训练参数: {trainable_params:,}")
        print(f"  不可训练参数: {total_params - trainable_params:,}")
        print(f"  模型大小: {model_size_mb:.1f} MB")
        print()
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'model_size_mb': model_size_mb,
    }


