# Class-Incremental Object Detection via Prototype Inversion and Distribution Re-balancing
CIOD resourse
## Dataset Format

All datasets in this project should be converted to the **YOLO format** before training.

For datasets such as **VOC** and **COCO**, please convert the original annotations to YOLO-format labels first.

The expected directory structure is:

```text
<dataset_name>/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

For example:

```text
VOC/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

```text
COCO/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```
