import glob
import cv2
import numpy as np
cv_img = []
d=1
#Reading images from BSD68 dataset as greyscale(color has no information in this dataset) and storing them to create true train set with clean and noisy images

for img in glob.glob("train/*.png"):
    filename = "blur/img%d.png"%d
    im= cv2.imread(img,0)
    bl_im=cv2.blur(im,(3,3))#Average blurring
    cv2.imwrite(filename,bl_im)
    d=d+1
