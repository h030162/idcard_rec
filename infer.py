from ultralytics import YOLO
#from PIL import Image
import cv2
import time
import glob
import os,sys
model = YOLO('models/det/cardpose.pt')
imgfoldpath = "imgs"
alljpgs = glob.glob(f"{imgfoldpath}/det/*")
print(len(alljpgs))
labelnames=["card"]
for imgid, jpgname in enumerate(alljpgs):    
    basename = os.path.basename(jpgname)
    print(imgid, len(alljpgs), basename)
    results = model(source=jpgname, verbose=False)  # save predictions as labels    
    res_plotted = results[0].plot()
    #cv2.imshow("result", res_plotted)
    #cv2.waitKey()
    cv2.imwrite(f"result/kp.jpg", res_plotted)
    

