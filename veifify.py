import torch
model=torch.load('best2.pkl').cuda()
import cv2
import numpy as np
import torchvision.datasets as dset
import torchvision.transforms as transforms
from skimage.measure import compare_ssim as ssim
train_image_folder = dset.ImageFolder(root='Set12',transform=transforms.ToTensor())
train_test_image_folder=dset.ImageFolder(root='Set12', transform=transforms.Compose([transforms.Resize((180,180))]))

from torch.utils.data import DataLoader
from torch.autograd import Variable
loader_test = DataLoader(dataset=train_test_image_folder,batch_size=12, shuffle=False)
for a,b in train_test_image_folder:
    x=np.asarray(a).astype(float)#There is some proble in PIL to tensor conversion
    x=x/255
    x=x[:,:,:1].reshape(1,1,180,180)#Greyscale conversion and need approriate dim of (x,1,180,180)
    blur=cv2.blur(x[0,:,:,:],(5,5))
    blur=blur.reshape(1,1,180,180)
    images=Variable(torch.tensor(blur)[:,:1,:,:].cuda())
    target=Variable(torch.tensor(x)[:,:1,:,:].cuda())
    images = images.type(torch.cuda.FloatTensor)
    target=target.type(torch.cuda.FloatTensor)
    output=model(images)
    x=output.cpu().data.numpy()
    y=target.cpu().data.numpy()
    print(ssim(x[0,0,:,:],y[0,0,:,:]))
