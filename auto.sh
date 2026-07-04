# python train.py --nc_steps [5,5] --name n --epochs 30
# sleep 10
# python train.py --nc_steps [5,5,5] --name n --epochs 30
# sleep 10
# python train.py --nc_steps [5,5,5,5] --name n --epochs 30
# sleep 10

# python train.py --nc_steps [40] --name s --epochs 30 --batch-size 16 --weights './weights/yolo11s.pt' --data-dir 'COCO'   
# sleep 10
# python train.py --nc_steps [40,40] --name s --epochs 30 --batch-size 16 --data-dir 'COCO'   
# sleep 10

python eval.py --weights weights/0-39m.pt --nc_steps [40] --name m --data-dir COCO
sleep 5
python eval.py --weights weights/0-39-79m.pt --nc_steps [40,40] --name m --data-dir COCO
sleep 5
python eval.py --weights weights/0-39-59m.pt --nc_steps [40,20] --name m --data-dir COCO
sleep 5
python eval.py --weights weights/0-39-59-79m.pt --nc_steps [40,20,20] --name m --data-dir COCO
sleep 5
python eval.py --weights weights/0-39-49m.pt --nc_steps [40,10] --name m --data-dir COCO
sleep 5
python eval.py --weights weights/0-39-49-59m.pt --nc_steps [40,10,10] --name m --data-dir COCO
sleep 5
python eval.py --weights weights/0-39-49-59-69m.pt --nc_steps [40,10,10,10] --name m --data-dir COCO
sleep 5
python eval.py --weights weights/0-39-49-59-69-79m.pt --nc_steps [40,10,10,10,10] --name m --data-dir COCO
sleep 5
python eval.py --weights weights/0-69m.pt --nc_steps [70] --name m --data-dir COCO
sleep 5
python eval.py --weights weights/0-69-79m.pt --nc_steps [70,10] --name m --data-dir COCO
sleep 5