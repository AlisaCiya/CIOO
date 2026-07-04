"""
统一增量学习训练入口

用法：
  基础训练: python train.py --nc_steps [15] --name n --epochs 40
  增量训练: python train.py --nc_steps [15,5] --name n --epochs 30
"""

import os
import ast
import argparse
from trainer import IncrementalTrainer


def _make_step_tag(nc_steps: list) -> str:
    """
    生成步骤标签
    例如: [10, 5] -> '0-9-14', [10] -> '0-9'
    """
    if not nc_steps:
        return "0"
    total = 0
    ends = []
    for n in nc_steps:
        total += int(n)
        ends.append(total - 1)
    parts = ["0"] + [str(e) for e in ends]
    return "-".join(parts)


def parse_args():
    parser = argparse.ArgumentParser(description='Unified Incremental Learning Trainer')
    
    # 模型配置
    parser.add_argument('--name', default='n', type=str, 
                       help='模型规模 (n/s/m)')
    parser.add_argument('--nc_steps', type=ast.literal_eval, default=[15,5],
                       help='类别步骤，[15] 基础训练，[15,5] 增量训练')
    parser.add_argument('--weights', type=str, default='./weights/0-14n.pt',
                       help='预训练权重路径（仅基础训练）')
    parser.add_argument('--oldmodel', type=str, default=None,
                       help='旧模型路径（增量训练自动推断）')
    
    # 训练配置
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=16)
    # 数据配置
    parser.add_argument('--inp-size', type=int, default=640)
    parser.add_argument('--data-dir', type=str, default='VOC')
    parser.add_argument('--disjoint', type=lambda x: x.lower() == 'true', default=True,
                       help='启用 disjoint 模式 (true/false)')
    
    # 数据增强参数（原 args.yaml）
    parser.add_argument('--buffer', type=bool, default=True, help='大数据集使用缓存')
    parser.add_argument('--mosaic', type=float, default=1.0)
    parser.add_argument('--mixup', type=float, default=0.0)
    parser.add_argument('--scale', type=float, default=0.5)
    parser.add_argument('--translate', type=float, default=0.1)
    parser.add_argument('--degree', type=float, default=0.0)
    parser.add_argument('--shear', type=float, default=0.0)
    parser.add_argument('--psp', type=float, default=0.0)
    parser.add_argument('--hsv-h', type=float, default=0.015)
    parser.add_argument('--hsv-s', type=float, default=0.7)
    parser.add_argument('--hsv-v', type=float, default=0.4)
    parser.add_argument('--flip-ud', type=float, default=0.0)
    parser.add_argument('--flip-lr', type=float, default=0.5)
    parser.add_argument('--bgr', type=float, default=0.0)
    
    # 损失权重
    parser.add_argument('--box', type=float, default=7.5)
    parser.add_argument('--cls', type=float, default=0.5)
    parser.add_argument('--dfl', type=float, default=1.5)
    parser.add_argument('--decay', type=float, default=0.0005)
    
    # 其他
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--prototype', action='store_true',
                       help='启用 prototype 特征提取（用于增量学习）')
    parser.add_argument('--extract-only', action='store_true',
                       help='仅提取 prototype（跳过训练，需配合 --weights 指定已训练模型）')
    parser.add_argument('--proto-distill', action='store_true',
                       help='启用 prototype 蒸馏（增量训练时约束旧类特征）')
    parser.add_argument('--proto-weight', type=float, default=2,  #! 744.7
                       help='prototype 蒸馏权重（默认 1.0）')
    parser.add_argument('--proto-path', type=str, default=None,
                       help='旧类 prototype 文件路径（默认自动推断）')
    
    args = parser.parse_args()
    
    # 自动推断路径
    is_base_step = len(args.nc_steps) == 1
    args.path = 'weights'
    if is_base_step:
        pass
    else:
        if not args.oldmodel:
            args.oldmodel = f"./weights/{_make_step_tag(args.nc_steps[:-1])}{args.name}.pt"
    
    args.out = f"weights/{_make_step_tag(args.nc_steps)}{args.name}.pt"
    os.makedirs(args.path, exist_ok=True)
    
    # 设置 allowed_classes
    if args.nc_steps:
        total = sum(args.nc_steps)
        last_step = args.nc_steps[-1]
        start = total - last_step
        end = total - 1
        
        print(f"✓ 增量学习类别范围: {start}-{end} (共 {last_step} 类)")
        args.allowed_classes = set(range(start, end + 1))
        print(f"✓ 允许的类别索引: {list(args.allowed_classes)}")
    
    return args


def main():
    args = parse_args()
    
    # 运行训练
    trainer = IncrementalTrainer(args)
    trainer.run()


if __name__ == "__main__":
    main()
