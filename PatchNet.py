import torch
import torch.nn as nn
import torch.nn.functional as F
from lossFunction import AsymAdditiveMarginSoftmax
from torchvision.models import resnet18

from torchvision.models.resnet import resnet18

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.encoder = resnet18(pretrained=True)
        self.encoder.requires_grad = False
        self.encoder.fc = nn.Sequential()
        self.ams = AsymAdditiveMarginSoftmax(512,9)
    def forward_once(self,input,labels):
        f = self.encoder(input)
        out, loss = self.ams(f,labels)
        return out, f, loss
    
    def forward(self, input1, input2,labels):
        out1, f1, loss1 = self.forward_once(input1,labels)
        out2, f2, loss2 = self.forward_once(input2,labels)
        f1_norm = F.normalize(f1,p=2,dim=1)
        f2_norm = F.normalize(f2,p=2,dim=1)
        loss =  loss1 + loss2 + torch.mean(torch.norm(f1_norm-f2_norm))
        return out1, out2, loss
        
class PatchNet(nn.Module):
    def __init__(self):
        super(PatchNet,self).__init__()
        self.encoder = resnet18()
        self.encoder.fc = nn.Sequential()
        self.fc = nn.Linear(512,9, bias=False)
    def forward(self,x):
        x = self.encoder(x)
        self.fc.weight.data = F.normalize(self.fc.weight.data, p=2, dim=1) 

        x = F.normalize(x, p=2, dim=1)

        wf = self.fc(x)
        x = F.softmax(30 * wf)
        return x