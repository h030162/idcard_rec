
import os
import sys
import cv2
import numpy as np
import math
import time
import numpy as np
import onnxruntime as ort
import argparse
import glob
from predict_rec import TextRecognizer, init_args, parse_args
from ultralytics import YOLO
labelnames = ['name','sex', 'nation', 'year','month', 'day', 'id', 'address0', 'address1', 'address2']
if __name__ == '__main__': 
    args = parse_args()
    recognizer = TextRecognizer(args)
    det = YOLO("models/det/det2.pt")
    imglist = glob.glob(f"{args.image_dir}/*.jpg")
    print(imglist)
    for path in imglist:
        frame = cv2.imread(path)
        results = det(frame, verbose=False)
        imgs = {}
        addressimgs = [None,None,None,None]
        expandpix_x = 5
        expandpix_y = 5
        #找到身份证号的框,并将身份证号框向外扩展几个像素
        for pts, cls_id in zip(results[0].obb.xyxy, results[0].obb.cls):
            pts = pts.cpu().numpy()
            cls_id = cls_id.cpu().numpy()
            cls_id = int(cls_id)
            if cls_id == labelnames.index("id"):
                id_w = int(pts[2]-pts[0])
                id_h = int(pts[3]-pts[1])
                expandpix_x = id_w/(18*2)
                #expandpix_y = id_h/8
                break
        for pts, cls_id in zip(results[0].obb.xyxy, results[0].obb.cls):
            pts = pts.cpu().numpy()
            cls_id = cls_id.cpu().numpy()
            cls_id = int(cls_id)
            pt_top_left = (int(pts[0] - expandpix_x), int(pts[1] - expandpix_y))
            pt_bottom_right = (int(pts[2] + expandpix_x), int(pts[3] + expandpix_y))
            cv2.rectangle(frame, pt_top_left, pt_bottom_right, (0, 255, 0), 1)
            # 裁剪图片
            img_crop = frame[pt_top_left[1]:pt_bottom_right[1], pt_top_left[0]:pt_bottom_right[0]]
            if cls_id < 7:
                imgs[labelnames[cls_id]] = img_crop
            else:
                addressimgs[cls_id-6] = img_crop
        addressimg = None
        for img in addressimgs:
            if img is not None:
                if addressimg is None:
                    addressimg = img
                else:
                    dst_h = addressimg.shape[0]
                    dst_w = img.shape[1]*dst_h/img.shape[0]
                    img = cv2.resize(img, (int(dst_w), int(dst_h)))                
                    addressimg = np.concatenate((addressimg, img), axis=1)
        imglist = []
        keys = []
        for key in imgs:
            if imgs[key] is not None:
                imglist.append(imgs[key])
                keys.append(key)
        keys.append("address")
        imglist.append(addressimg)
        res,_ = recognizer(imglist)    
        outputs = {}
        for key,v in zip(keys, res):
            outputs[key] = v[0]    
        mapoutput = {}
        if "name" in outputs:
            mapoutput["姓名"] = outputs["name"]
        if "sex" in outputs:
            mapoutput["性别"] = outputs["sex"]
        if "nation" in outputs:
            mapoutput["民族"] = outputs["nation"]
        if "year" in outputs and "month" in outputs and "day" in outputs:
            mapoutput["出生"] = f'{outputs["year"]}-{outputs["month"]}-{outputs["day"]}'
        if "id" in outputs:
            mapoutput["身份证号"] = outputs["id"]
        if "address" in outputs:
            mapoutput["住址"] = outputs["address"]
        print(mapoutput)
        #cv2.imshow("img", frame)
        #cv2.waitKey(0)
        cv2.imwrite(f"result/{os.path.basename(path)}", frame)




