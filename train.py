import time
import copy
from torchvision import datasets, transforms, utils
import os
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from dataset import SiameseNetworkDataset, TestDataset
import numpy as np
from PatchNet import SiameseNetwork, PatchNet
import torch.optim as optim
from torch.optim import lr_scheduler


data_transforms = {
    'train': [
        transforms.Compose([
          transforms.RandomRotation(180),
          transforms.RandomHorizontalFlip(p=0.5),
          transforms.RandomCrop(100),
          transforms.ToTensor(),
        ]),
        transforms.Compose([
          transforms.RandomRotation(180),
          transforms.RandomHorizontalFlip(p=0.5),
          transforms.RandomCrop(100),
          transforms.ToTensor(),
        ]),
    ],
    'test': transforms.Compose([
          transforms.RandomRotation(180),
          transforms.RandomHorizontalFlip(p=0.5),
          transforms.RandomCrop(100),
          transforms.ToTensor(),
        ]),
}
data_dir = 'D:\\university\\GP\\PatchNet'

image_datasets = {
    'train': datasets.ImageFolder(os.path.join(data_dir, 'train')),
     'test': datasets.ImageFolder(os.path.join(data_dir, 'test'))
}
siameseDataset = SiameseNetworkDataset(image_datasets['train'],data_transforms['train'])
testDataset = TestDataset(image_datasets['test'], data_transforms['test'])
dataloaders = {
    'train': torch.utils.data.DataLoader(siameseDataset,shuffle=True,num_workers=2),
    'test': torch.utils.data.DataLoader(testDataset,shuffle=True,num_workers=2)
}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train','test']}
class_names = image_datasets['train'].classes

def imshow(img, text=None):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic',fontweight='bold',
            bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()    

def show_plot(iteration,loss):
    plt.plot(iteration,loss)
    plt.show()




device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

loss_history_train = []
loss_history_test = []
acc_history_train = []
acc_history_test = []

def train_model(model, optimizer, scheduler, num_epochs):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0


    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 40)
        for phase in ['train' , 'test']:
            if phase == 'train':
                model.train()
            running_loss = 0.0
            running_corrects = 0
            for dataloader in dataloaders[phase]:
                optimizer.zero_grad()
                if phase == 'train':
                  inputs1, inputs2, labels = dataloader
                  inputs1 = inputs1.to(device)
                  inputs2 = inputs2.to(device)
                  labels = labels.to(device)
                  out1, out2, loss = model(inputs1, inputs2, labels)
                  loss.backward()
                  optimizer.step()
                  _, preds = torch.max(out1,1)
                  running_corrects += torch.sum(preds == labels.data)
                  running_loss += loss.item() * inputs1.size(0)
                else:
                  inputs, labels = dataloader
                  inputs = inputs.to(device)
                  labels = labels.to(device)
                  with torch.no_grad():
                    out, f, loss = net.forward_once(inputs, labels)
                  _, preds = torch.max(out,1)
                  running_corrects += torch.sum(preds == labels.data)
                  running_loss += loss.item() * inputs.size(0)
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]
            if phase == 'train':
              scheduler.step()
              loss_history_train.append(epoch_loss)
              acc_history_train.append(epoch_acc.cpu().detach().numpy())
            else:
              loss_history_test.append(epoch_loss)
              acc_history_test.append(epoch_acc.cpu().detach().numpy())
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')
    model.load_state_dict(best_model_wts)
    return model



if __name__ == '__main__':
    vis_dataloader = torch.utils.data.DataLoader(siameseDataset,
                        shuffle=True,
                        num_workers=2,
                        batch_size=8)
    example_batch = next(iter(vis_dataloader))
    concatenated = torch.cat((example_batch[0], example_batch[1]),0)
    plt.figure()
    imshow(utils.make_grid(concatenated))
    print(example_batch[2].numpy().reshape(-1))



    vis_dataloader = torch.utils.data.DataLoader(testDataset,
                            shuffle=True,
                            num_workers=2,
                            batch_size=8)
    example_batch = next(iter(vis_dataloader))
    concatenated = example_batch[0]
    plt.figure()
    imshow(utils.make_grid(concatenated))
    print(example_batch[1].numpy().reshape(-1))
    net = SiameseNetwork().to(device)
    optimizer = optim.SGD(net.parameters(), lr = 0.002 )
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=90, gamma=0.5)
    net = train_model(net,optimizer,exp_lr_scheduler,200)
    plt.figure()
    plt.plot(loss_history_train,c='r')
    plt.plot(loss_history_test,c='b')
    plt.show()
    plt.figure()
    plt.plot(acc_history_train,c='r')
    plt.plot(acc_history_test,c='b')
    plt.show()
    model = PatchNet()
    model.encoder = net.encoder
    model.fc = net.ams.fc
    torch.save(model,'patchnet.pt')
    running_corrects = 0
    for i, (img, label) in enumerate(dataloaders['test']):

        # Send the images and labels to CUDA
        img, label = img.cuda(), label.cuda()

        # Zero the gradients
        with torch.no_grad():
            # Pass in the two images into the network and obtain two outputs
            output = model(img, label)
            _, preds = torch.max(output, 1)
            running_corrects += torch.sum(preds == label.data)
    acc = running_corrects/dataset_sizes['test']
    print(f'Acc: {acc:.4f}')