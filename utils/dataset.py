import os
import cv2
import math
import torch
import hashlib
import numpy as np
from PIL import Image
from pathlib import Path
from utils import augment
from torch.utils import data
from concurrent.futures import ThreadPoolExecutor

from ultralytics.utils.instance import Instances

img_ext = {"bmp", "jpeg", "jpg", "png", "tif", "tiff"}

# 【优化】预分配的空 segments 数组（复用避免重复创建）
_EMPTY_SEGMENTS = np.zeros((0, 1000, 2), dtype=np.float32)


class Dataset(data.Dataset):
    """
    优化后的数据集类，支持：
    - 标签缓存 (.cache 文件)
    - 可选的图片 RAM 缓存
    - 延迟图片验证
    - Disjoint 模式
    """
    
    # 类级别的缓存版本号，修改数据处理逻辑时需要更新
    CACHE_VERSION = "v2.0"
    
    def __init__(self, args, augments=True, disjoint=False, cache_images=False):
        """
        初始化数据集
        
        Args:
            args: 配置参数（包含 yaml 数据增强参数）
            augments: 是否使用数据增强
            disjoint: 是否使用 disjoint 模式（只保留当前阶段类别，过滤空标签图片）
            cache_images: 是否将图片缓存到 RAM（加速训练但占用内存）
        """
        super(Dataset, self).__init__()
        self.args = args
        self.augment = augments
        self.mosaic = augments
        self.disjoint = disjoint
        self.cache_images = cache_images
        
        # 【优化】预计算 allowed_classes 为 frozenset（更快的查找）
        allowed = getattr(args, "allowed_classes", None)
        self._allowed_set = frozenset(allowed) if allowed else None
        self._allowed_array = np.array(list(allowed), dtype=np.float32) if allowed else None

        file = 'train.txt' if self.augment else 'val.txt'
        
        filenames = f'{args.data_dir}/{file}'
        all_images = self.load_image(filenames)
        cache_result = self.load_labels(args, all_images, disjoint=disjoint, 
                                         allowed_set=self._allowed_set,
                                         allowed_array=self._allowed_array)
        
        self.images = cache_result["images"]
        self.labels = cache_result["labels"]

        self.num_img = len(self.labels)
        self.indices = np.arange(self.num_img)
        self.transforms = self.build_transforms()
        
        # 【优化】图片 RAM 缓存
        self._image_cache = {}
        if self.cache_images:
            self._cache_images_to_ram()

        if not self.augment:
            bi = np.floor(self.indices / self.args.batch_size)
            bi = bi.astype(int)
            nb = bi[-1] + 1
            shape = np.array([x.pop("shape") for x in self.labels])
            ratio = shape[:, 0] / shape[:, 1]
            self.images = [self.images[i] for i in ratio.argsort()]
            self.labels = [self.labels[i] for i in ratio.argsort()]

            shapes = [[1, 1]] * nb
            for i in range(nb):
                ari = ratio[ratio.argsort()][bi == i]
                mini, maxi = ari.min(), ari.max()
                if maxi < 1:
                    shapes[i] = [maxi, 1]
                elif mini > 1:
                    shapes[i] = [1, 1 / mini]

            self.batch_shapes = np.ceil(
                np.array(shapes) * self.args.inp_size / 32 + 0.5)
            self.batch_shapes = self.batch_shapes.astype(int) * 32
            self.batch = bi

    def _cache_images_to_ram(self):
        """【优化】将所有图片预加载到 RAM"""
        import psutil
        
        # 检查可用内存
        mem = psutil.virtual_memory()
        available_gb = mem.available / (1024 ** 3)
        
        # 估算需要的内存（粗略估计每张图片 1MB）
        estimated_gb = len(self.images) * 1 / 1024
        
        if estimated_gb > available_gb * 0.8:
            print(f"[Dataset] WARNING: Not enough RAM for image caching "
                  f"(need ~{estimated_gb:.1f}GB, available {available_gb:.1f}GB)")
            self.cache_images = False
            return
        
        print(f"[Dataset] Caching {len(self.images)} images to RAM...")
        
        def load_image(idx):
            """加载单张图片"""
            try:
                img = cv2.imread(self.images[idx])
                return idx, img
            except Exception:
                return idx, None
        
        # 使用多线程并行加载
        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(load_image, range(len(self.images))))
        
        for idx, img in results:
            if img is not None:
                self._image_cache[idx] = img
        
        cached = len(self._image_cache)
        print(f"[Dataset] Cached {cached}/{len(self.images)} images to RAM")

    def __getitem__(self, index):
        tr_dataset = self.transforms(self.get_image_and_label(index))
        return tr_dataset

    def __len__(self):
        return len(self.labels)

    @staticmethod
    def load_image(path):
        samples = []
        for p in [path]:
            p = Path(p)
            with open(p) as f:
                samples += [x.replace("./", str(p.parent) + os.sep)
                            if x.startswith("./") else x for x in
                            f.read().strip().splitlines()]

        return sorted(x.replace("/", os.sep) for x in samples if
                      x.split(".")[-1].lower() in img_ext)

    def read_image(self, index):
        """【优化】优先从缓存读取图片"""
        # 尝试从缓存读取
        if index in self._image_cache:
            image = self._image_cache[index]
        else:
            image = cv2.imread(self.images[index])
        
        if image is None:
            raise ValueError(f"Failed to load image: {self.images[index]}")
        
        h0, w0 = image.shape[:2]
        r = self.args.inp_size / max(h0, w0)
        if r != 1:
            w, h = (min(math.ceil(w0 * r), self.args.inp_size),
                    min(math.ceil(h0 * r), self.args.inp_size))
            image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
        return image, (h0, w0), image.shape[:2]

    @staticmethod
    def _get_cache_path(args, label_dir, disjoint, allowed_set):
        """【优化】生成唯一的缓存文件路径"""
        # 基于配置生成唯一的缓存标识
        cache_key_parts = [Dataset.CACHE_VERSION]
        
        if allowed_set:
            # 使用类别的哈希值
            cls_str = ",".join(map(str, sorted(allowed_set)))
            cls_hash = hashlib.md5(cls_str.encode()).hexdigest()[:8]
            cache_key_parts.append(f"cls_{cls_hash}")
        
        if disjoint:
            cache_key_parts.append("disjoint")
        
        remap = getattr(args, "class_remap", None)
        if remap:
            remap_str = str(sorted(remap.items()))
            remap_hash = hashlib.md5(remap_str.encode()).hexdigest()[:8]
            cache_key_parts.append(f"remap_{remap_hash}")
        
        cache_suffix = "_".join(cache_key_parts) + ".cache"
        return label_dir.with_suffix("." + cache_suffix)

    @staticmethod
    def load_labels(args, images, disjoint=False, allowed_set=None, allowed_array=None):
        """
        加载并处理标签（带缓存支持）
        
        Args:
            args: 配置参数
            images: 图片路径列表
            disjoint: 是否使用 disjoint 模式
            allowed_set: 预计算的 allowed_classes frozenset
            allowed_array: 预计算的 allowed_classes numpy array
        
        Returns:
            dict: {"images": [...], "labels": [...]}
        """
        if not images:
            return {"labels": [], "images": []}
        
        a = f"{os.sep}images{os.sep}"
        b = f"{os.sep}labels{os.sep}"
        label_paths = [b.join(x.rsplit(a, 1)).rsplit(".", 1)[0] + ".txt" for x in images]
        label_dir = Path(label_paths[0]).parent
        
        # 【优化】尝试加载缓存
        cache_path = Dataset._get_cache_path(args, label_dir, disjoint, allowed_set)
        
        if cache_path.exists():
            try:
                cached = torch.load(cache_path, weights_only=False)
                # 验证缓存有效性
                if (cached.get("version") == Dataset.CACHE_VERSION and
                    cached.get("num_images") == len(images)):
                    print(f"[Dataset] Loaded cache from {cache_path}")
                    return {"images": cached["images"], "labels": cached["labels"]}
            except Exception as e:
                print(f"[Dataset] Cache load failed: {e}, rebuilding...")
        
        # 构建新缓存
        cache = {"labels": [], "images": []}
        remap = getattr(args, "class_remap", None)
        
        # 统计信息
        total_images = len(images)
        valid_images = 0
        skipped_empty = 0
        skipped_error = 0

        for img, label in zip(images, label_paths):
            try:
                # 【优化】延迟验证 - 只读取图片尺寸，不完整验证
                try:
                    with Image.open(img) as pil_img:
                        shape = pil_img.size
                        shape = (shape[1], shape[0])  # (H, W)
                except Exception as e:
                    raise ValueError(f"Cannot open image: {e}")
                
                if shape[0] <= 9 or shape[1] <= 9:
                    raise ValueError(f"image size {shape} <10 pixels")

                if os.path.isfile(label):
                    with open(label) as f:
                        lines = f.read().strip().splitlines()
                        lb = [x.split() for x in lines if len(x)]
                        lb = np.array(lb, dtype=np.float32) if lb else np.zeros((0, 5), dtype=np.float32)
                    nl = len(lb)
                    
                    if nl:
                        assert lb.min() >= 0, f"negative label values {lb[lb < 0]}"
                        assert lb.shape[1] == 5, f"labels require 5 columns, {lb.shape[1]} columns detected"
                        assert lb[:, 1:].max() <= 1, f"non-normalized or out of bounds coordinates"
                        
                        # 【优化】使用预计算的 allowed_array 进行过滤
                        if allowed_array is not None and len(allowed_array) > 0:
                            m = np.isin(lb[:, 0], allowed_array)
                            lb = lb[m]
                            nl = len(lb)
                            
                            # 类别重映射
                            if nl and remap is not None:
                                cls_new = np.array([remap.get(int(c), -1) for c in lb[:, 0]], dtype=np.float32)
                                keep = cls_new >= 0
                                lb = lb[keep]
                                nl = len(lb)
                                if nl:
                                    lb[:, 0] = cls_new[keep]
                        
                        # Disjoint 模式：过滤空标签图片
                        if disjoint and nl == 0:
                            skipped_empty += 1
                            continue
                        
                        if nl:
                            # 验证类别范围
                            expected_nc = len(args.names) if hasattr(args, 'names') and args.names else (
                                sum(args.nc_steps) if hasattr(args, 'nc_steps') and args.nc_steps else 80
                            )
                            assert lb[:, 0].max() < expected_nc, (
                                f"Label class {int(lb[:, 0].max())} exceeds {expected_nc}")
                            
                            # 去重
                            _, i = np.unique(lb, axis=0, return_index=True)
                            if len(i) < nl:
                                lb = lb[i]
                    else:
                        lb = np.zeros((0, 5), dtype=np.float32)
                        if disjoint:
                            skipped_empty += 1
                            continue
                else:
                    lb = np.zeros((0, 5), dtype=np.float32)
                    if disjoint:
                        skipped_empty += 1
                        continue
                
                lb = lb[:, :5]
                
                cache["images"].append(img)
                cache["labels"].append({
                    'image': img,
                    "shape": shape,
                    'cls': lb[:, 0:1],
                    'box': lb[:, 1:],
                    "norm": True,
                    "format": "xywh"
                })
                valid_images += 1
                
            except Exception as e:
                skipped_error += 1
                print(f"Skipping {img}: {e}")
        
        # 保存缓存
        try:
            cache_data = {
                "version": Dataset.CACHE_VERSION,
                "num_images": len(images),
                "images": cache["images"],
                "labels": cache["labels"]
            }
            torch.save(cache_data, cache_path)
            print(f"[Dataset] Saved cache to {cache_path}")
        except Exception as e:
            print(f"[Dataset] Failed to save cache: {e}")
        
        # 输出统计信息
        mode_str = "Disjoint" if disjoint else "Standard"
        print(f"[Dataset] {mode_str} mode: {valid_images}/{total_images} images loaded")
        if disjoint and skipped_empty > 0:
            print(f"[Dataset] Skipped {skipped_empty} images with empty labels")
        if skipped_error > 0:
            print(f"[Dataset] Skipped {skipped_error} images due to errors")
        
        return cache

    def get_image_and_label(self, index):
        """【优化】使用浅拷贝代替深拷贝"""
        # 浅拷贝 dict，只需要复制字典本身，不需要复制内部的 numpy 数组
        label = dict(self.labels[index])
        
        # 单独复制可变的 numpy 数组（cls 和 box）
        label['cls'] = label['cls'].copy()
        label['box'] = label['box'].copy()
        
        label["img"], label["shape"], label["new_shape"] = self.read_image(index)
        label["pad"] = (label["new_shape"][0] / label["shape"][0],
                        label["new_shape"][1] / label["shape"][1])

        if not self.augment:
            label["rect_shape"] = self.batch_shapes[self.batch[index]]

        bboxes = label.pop("box")
        bbox_format = label.pop("format")
        normalized = label.pop("norm")
        
        # 【优化】复用预分配的空 segments 数组
        label["instances"] = Instances(bboxes, _EMPTY_SEGMENTS, bbox_format=bbox_format,
                                       normalized=normalized)

        return label

    def build_transforms(self):
        if self.mosaic:
            transforms = augment.transforms(self, self.args)
        else:
            transforms = augment.Compose([augment.LetterBox(new_shape=(
                self.args.inp_size, self.args.inp_size), scaleup=False)])

        transforms.append(augment.Format())
        return transforms

    @staticmethod
    def collate_fn(batch):
        """【优化】使用更高效的批处理合并"""
        new_batch = {}
        keys = batch[0].keys()
        
        for k in keys:
            values = [b[k] for b in batch]
            
            if k == "img":
                new_batch[k] = torch.stack(values, 0)
            elif k in {"cls", "box"}:
                new_batch[k] = torch.cat(values, 0)
            elif k == "idx":
                # 预先计算偏移量
                idx_list = [v + i for i, v in enumerate(values)]
                new_batch[k] = torch.cat(idx_list, 0)
            else:
                new_batch[k] = values
        
        return new_batch

    