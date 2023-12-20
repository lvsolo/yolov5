# -*- coding: utf-8 -*-
import re 
import os 
import json 
from tqdm import tqdm
from pprint import pprint 
def search_file(data_dir, pattern=r'\.jpg$'):
    root_dir = os.path.abspath(data_dir)
    for root, dirs, files in os.walk(root_dir):
        for f in files:
            if re.search(pattern, f, re.I):
                abs_path = os.path.join(root, f)
                #print('new file %s' % absfn)
                yield abs_path
 
class Bdd2yolov5:
    def __init__(self):
        self.bdd100k_width = 1280
        self.bdd100k_height = 720
        self.select_categories = ["person", "rider", "car", "bus", "truck", "bike",
               "motor", "traffic light", "traffic sign","train"]
        self.cat2id = {name:ind for ind, name in enumerate(self.select_categories)}
        print("self.cat2id:", self.cat2id)

 
    @property
    def all_categorys(self):
        return ["person", "rider", "car", "bus", "truck", "bike", 
               "motor", "traffic light", "traffic sign","train"]
 
    def _filter_by_attr(self, attr=None):
        if attr is None:
            return False 
        #过滤掉晚上的图片
        if attr['timeofday'] == 'night':
            return True 
        return False 
 
    def _filter_by_box(self, w, h):
        #size ratio 
        #过滤到过于小的小目标
        threshold = 0.001
        if float(w*h)/(self.bdd100k_width*self.bdd100k_height) < threshold:
            return True 
        return False 
 
    def bdd2yolov5(self, path):
        if 'train' in path:
            prefix = '../images/100k/train/'
            dst_dir = './images/train/'
            dst_label_dir = 'labels/train/'
        elif 'val' in path:
            prefix = '../images/100k/val/'
            dst_dir = './images/val/'
            dst_label_dir = 'labels/val/'
        else:
            assert 0
        with open(path) as fp:
            j = json.load(fp)
            pprint(j[0])
            dw = 1.0 / self.bdd100k_width
            dh = 1.0 / self.bdd100k_height
            for fr in tqdm(j):#["frames"]):
                if self._filter_by_attr(fr['attributes']):
                    continue
                lines = ""
                for obj in fr["labels"]:
                    if obj["category"] in self.select_categories:
                        idx = self.cat2id[obj["category"]]
                        cx = (obj["box2d"]["x1"] + obj["box2d"]["x2"]) / 2.0
                        cy = (obj["box2d"]["y1"] + obj["box2d"]["y2"]) / 2.0
                        w  = obj["box2d"]["x2"] - obj["box2d"]["x1"]
                        h  = obj["box2d"]["y2"] - obj["box2d"]["y1"]
                        if w<=0 or h<=0:
                            continue
                        if self._filter_by_box(w,h):
                            continue
                        #根据图片尺寸进行归一化
                        cx,cy,w,h = cx*dw,cy*dh,w*dw,h*dh
                        line = f"{idx} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n"
                        lines += line
                #转换后的以*.txt结尾的标注文件我就直接和*.json放一具目录了
                #yolov5中用到的时候稍微挪一下就行了
                yolo_txt = dst_label_dir + fr['name'].replace(".jpg",".txt")
                with open(yolo_txt, 'w') as fp2:
                     fp2.writelines(lines)
                     fp2.close()
                #print("%s has been dealt!" % path)
#                print(fr['name'])
                os.system('cp ' + prefix + fr['name'] + ' ' + dst_dir + fr['name'])
 
 
if __name__ == "__main__":
    bdd_label_dir = "../labels/"
    cvt=Bdd2yolov5()
    for path in search_file(bdd_label_dir, r"\.json$"):
        cvt.bdd2yolov5(path)
