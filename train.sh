python train.py --data data_bdd100k.yaml --weights '' --cfg yolov5s.yaml --epochs 50 --img 640 --rect --save-period 2
#python train.py --data data_bdd100k.yaml --weights yolov5s.pt --epochs 3 --img 640 --rect
python train.py  --data data_bdd100k.yaml --weights '' --cfg yolov5s.yaml --epochs 1  --rect
