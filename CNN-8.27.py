import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.nn import Module
import torch.nn.functional as F
#准备数据
batch_size = 4
train_data = datasets.MNIST(
    root = 'data',
    train = True,
    download = True,
    transform=ToTensor()
)
train_datalader = torch.utils.data.DataLoader(
    train_data,shuffle = True,batch_size = batch_size
)
test_data = datasets.MNIST(
    root = 'data',
    train = False,
    download = True,
    transform = ToTensor()
)
test_dataloader = torch.utils.data.DataLoader(
    test_data,shuffle = False,batch_size= batch_size
)

#创建相关函数

class CNN(Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1,20,kernel_size=3)
        self.conv2 = torch.nn.Conv2d(20,40,kernel_size=3)
        self.conv3 = torch.nn.Conv2d(40,80,kernel_size =3)
        self.pooling = torch.nn.MaxPool2d(2)
        self.fc1 = torch.nn.Linear(80,40)
        self.fc2 = torch.nn.Linear(40,20)
        self.fc3 = torch.nn.Linear(20,10)
    def forward(self,x):
        batch_size = x.size(0)
        x = self.pooling(F.relu(self.conv1(x)))
        x = self.pooling(F.relu(self.conv2(x)))
        x = self.pooling(F.relu(self.conv3(x)))
        x = x.view(batch_size,-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = CNN()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(),lr = 0.005)
#train
for epoch in range(1):
    for x,y in train_datalader:
        y_ = model(x)
        loss = criterion(y_,y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        print(loss.item())








