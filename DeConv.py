import torch.nn as nn
import torch
#torch.cuda.empty_cache()
import numpy as np
import cv2
#Model is nothing but modified version of standard VGG netowrk

class DeConv(nn.Module):
    def __init__(self):
        super(DeConv, self).__init__()
        
        # Convolution 1
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(1,49), stride=1, padding=(0,24))#Win-k+2P+1=Wout
        #self.drop=nn.Dropout2d(0.4)
        self.T = nn.Tanh()
        
        
        # Convolution 2
        self.cnn2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(49,1), stride=1, padding=(24,0))
        self.T2 = nn.Tanh()
        #self.drop2=nn.Dropout2d(0.4)
        
        # Linear O/p
        self.cnn3=nn.Conv2d(in_channels=32,out_channels=1,kernel_size=5,padding=2)
    
    def forward(self, x):
        # Convolution 1
        
        
        out = self.cnn1(x)
        
        out = self.T(out)
        
        # Convolution 2 
        out = self.cnn2(out)
        
        out = self.T2(out)
        out = self.cnn3(out)
        
        return out
class ODeConv(nn.Module):
    def __init__(self):
        super(ODeConv, self).__init__()
        
        # Convolution 1
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=(1,121), stride=1, padding=(0,60))#Win-k+2P+1=Wout
        #self.drop=nn.Dropout2d(0.4)
        
        
        # Convolution 2
        self.cnn2 = nn.Conv2d(in_channels=20, out_channels=20, kernel_size=(121,1), stride=1, padding=(60,0))
        #self.drop2=nn.Dropout2d(0.4)
        
        
        
        self.cnn3=nn.Conv2d(in_channels=20,out_channels=128,kernel_size=17,padding=8)
        
        self.cnn4=nn.Conv2d(in_channels=128,out_channels=128,kernel_size=1,padding=0)
        self.cnn5=nn.Conv2d(in_channels=128,out_channels=1,kernel_size=9,padding=4)
        
    def forward(self, x):
        # Convolution 1
        
        
        out = self.cnn1(x)
        
       
        # Convolution 2 
        out = self.cnn2(out)
        
      
        out = self.cnn3(out)
        out = self.cnn4(out)
        out = self.cnn5(out)
        return out
def psnr(i1, i2):
    mse = torch.mean( (i1 - i2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 1
    return 20 * torch.log10(PIXEL_MAX / torch.sqrt(mse))

import torchvision.datasets as dset
import torchvision.transforms as transforms
train_image_folder = dset.ImageFolder(root='train',transform=transforms.ToTensor())
from torch.utils.data import DataLoader
from torch.autograd import Variable
model=DeConv().cuda()
criterion = nn.MSELoss()
from torch import optim
import matplotlib.pyplot as plt
optimizer = optim.Adam(model.parameters(), lr=0.001)#0.001
#Clean images will be corrupted and model will learn the distribution of noise given the corrupted images
loader_train = DataLoader(dataset=train_image_folder,batch_size=20, shuffle=True)
epochs=20#1 imgae is for 5 epochs
iter=0
best_loss=np.inf
l=[]

for epoch in range(epochs):
    for i, (im1,l1) in enumerate(loader_train):
        x=im1.data.numpy()
        blur=[]
        for k in range(im1.size()[0]):
            blur.append(cv2.blur(x[k,:,:,:],(5,5)))
        blur=np.array(blur)
        images=Variable(torch.tensor(blur)[:,:1,:,:].cuda())
        target = Variable(im1[:,:1,:,:].cuda())
        print(iter)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        if iter % 10 == 0:
            print('Iteration: {}. Loss: {}. PSNR{}'.format(iter, loss.data[0],psnr(outputs,target)))
            l.append(loss.data[0])
            plt.plot(l)
            plt.ylabel('loss')
            plt.xlabel('iterations')
        if(loss.data[0]<best_loss):
                best_loss=loss.data[0]
                print('saving model')
                torch.save(model, 'best4.pkl')
        iter=iter+1