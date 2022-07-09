import torch
import torch.nn as nn
import torch.nn.functional as F


live_lbls = [0,1,6]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class AsymAdditiveMarginSoftmax(nn.Module):
    def __init__(self, in_features, out_features, s=30, ml=0.4,ms=0.1):
        super(AsymAdditiveMarginSoftmax, self).__init__()
        self.s = s
        self.ml = ml
        self.ms = ms
        self.in_features = in_features
        self.out_features = out_features
        self.fc = nn.Linear(in_features, out_features, bias=False)
    def forward(self, x, labels):
        assert len(x) == len(labels)
        assert torch.min(labels) >= 0
        assert torch.max(labels) < self.out_features
        m = torch.tensor([[self.ml if lbl in live_lbls else self.ms for lbl in labels]]).to(device)
        self.fc.weight.data = F.normalize(self.fc.weight.data, p=2, dim=1)
        x = F.normalize(x, p=2, dim=1)
        wf = self.fc(x)
        out = F.softmax(self.s*wf)
        numerator = self.s * (torch.diagonal(wf.transpose(0, 1)[labels]) - m)
        excl = torch.cat([torch.cat((wf[i, :y], wf[i, y+1:])).unsqueeze(0) for i, y in enumerate(labels)], dim=0)
        denominator = torch.exp(numerator) + torch.sum(torch.exp(self.s * excl), dim=1)
        L = numerator - torch.log(denominator)
        return out, -torch.mean(L)
        
