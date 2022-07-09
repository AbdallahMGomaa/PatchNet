from PIL import Image
from torch.utils.data import Dataset


class SiameseNetworkDataset(Dataset):
    def __init__(self,imageFolderDataset,transform):
        self.imageFolderDataset = imageFolderDataset    
        self.transform = transform
    def __getitem__(self,index):   
        data = self.imageFolderDataset.imgs[index]
        img = Image.open(data[0])
        img0 = self.transform[0](img)
        img1 = self.transform[1](img)
        return img0, img1, data[1]
    def __len__(self):
        return len(self.imageFolderDataset.imgs)
class TestDataset(Dataset):
    def __init__(self,imageFolderDataset,transform):
        self.imageFolderDataset = imageFolderDataset    
        self.transform = transform
    def __getitem__(self,index):   
        data = self.imageFolderDataset.imgs[index]
        img = Image.open(data[0])
        img = self.transform(img)
        return img, data[1]
    def __len__(self):
        return len(self.imageFolderDataset.imgs)