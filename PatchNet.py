import imp
from sqlalchemy import outparam
import torch.nn as nn
from torch import device, cuda
from torchvision.models.resnet import resnet18

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork,self).__init__()
        self.model = resnet18(pretrained=True,progress=True)
        
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features,9)
    def forward_once(self,x):
        output = self.model(x)
        return output
    def forward(self,input1,input2):
        output1 = self.model(input1)
        output2 = self.model(input2)
        return output1, output2



net = SiameseNetwork()
first_parameter = next(net.parameters())
input_shape = first_parameter.size()
print(net)
from PIL import Image
x1 = Image.open('D:\\university\GP\\PatchNet\\train\\1\\frame1_47.jpg')
x2 = Image.open('D:\\university\GP\\PatchNet\\train\\1\\frame1_15.jpg')
import torchvision.transforms as transforms
transform = transforms.ToTensor()
x1 = transform(x1).unsqueeze(0)
x2 = transform(x2).unsqueeze(0)

y1,y2 = net.forward(x1,x2)

print(y1.dist(y2))
