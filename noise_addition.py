import glob
import cv2
import numpy as np
cv_img = []
d=0
#Reading images from BSD68 dataset as greyscale(color has no information in this dataset) and storing them to create true train set with clean and noisy images

for img in glob.glob("train/*.png"):
    filename = "traint/images/img%d.png"%d
    n= cv2.imread(img,0)
    row,col=n.shape
    mean=0
    sigma=15#as per paper
    noise=np.random.normal(mean,sigma,(row,col))
    noisy_image=n+noise
    noisy_image=noisy_image.astype(np.uint8)
    cv2.imwrite(filename,n)
    filename2 = "traint/noisy/img%d.png"%d
    cv2.imwrite(filename2,noisy_image)
    filename3 = "traint/noise/img%d.png"%d
    cv2.imwrite(filename3,noise.astype(np.uint8))
    d=d+1
    cv_img.append(n)
d=0    
#noisy test set creation
for img in glob.glob("Set68/*.png"):
    
    n= cv2.imread(img,0)
    row,col=n.shape
    sigma=np.random.uniform(0,50)
    noise=np.random.normal(0,sigma,(row,col))
    cv2.imwrite("test/img%d.png"%d,n+noise)
    d=d+1