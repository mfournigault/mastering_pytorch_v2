import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.cn1 = nn.Conv2d(1, 16, 3, 1)
        self.cn2 = nn.Conv2d(16, 32, 3, 1)
        self.dp1 = nn.Dropout2d(0.10)
        self.dp2 = nn.Dropout2d(0.25)
        self.fc1 = nn.Linear(4608, 64) # 4608 is 12x12x32
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.cn1(x)
        x = F.relu(x)
        x = self.cn2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dp1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dp2(x)
        x = self.fc2(x)
        op = F.log_softmax(x, dim=1)
        return op
    
def train(model, device, train_dataloader, optimizer, epoch):
    model.train()
    for b_i, (X, y) in enumerate(train_dataloader): 
        X, y = X.to(device), y.to(device) # send data to device
        optimizer.zero_grad() # zero the gradients
        pred_prob = model(X) # forward pass
        loss = F.nll_loss(pred_prob, y) # negative log-likelihood loss
        loss.backward() # backward pass
        optimizer.step() # update the weights
        if b_i % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss:{:.6f}'.format(
                epoch, b_i * len(X), len(train_dataloader.dataset),
                100. * b_i / len(train_dataloader), loss.item())) # print training logs every 10 batches
def test(model, device, test_dataloader):
    model.eval()
    test_loss = 0
    success = 0
    with torch.no_grad(): # we don't need gradients for the testing phase
        for X, y in test_dataloader:
            X, y = X.to(device), y.to(device) # send data to device
            pred_prob = model(X) # forward pass
            test_loss += F.nll_loss(pred_prob, y, reduction='sum').item() # sum up batch loss
            pred = pred_prob.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            success += pred.eq(y.view_as(pred)).sum().item() # compare predictions to true label
    test_loss /= len(test_dataloader.dataset)
    print('\nTest dataset: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, success, len(test_dataloader.dataset),
        100. * success / len(test_dataloader.dataset)))
    
def main():
    # The mean and standard deviation values are calculated
    # as the mean of all pixel values of all images in the
    # training dataset
    train_dataloader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
            transform=transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.1302,),(0.3069,))])
            ), # train_X.mean()/256. and train_X.std()/256.
        batch_size=32, shuffle=True)
    
    test_dataloader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False,
            transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1302,),(0.3069,))])
            ),
        batch_size=500, shuffle=False)
    
    torch.manual_seed(0)
    device = torch.device("cpu")
    model = ConvNet()
    optimizer = optim.Adadelta(model.parameters(), lr=0.5)

    for epoch in range(1, 3):
        train(model, device, train_dataloader, optimizer, epoch)
        test(model, device, test_dataloader)
    
    test_samples = enumerate(test_dataloader)
    b_i, (sample_data, sample_targets) = next(test_samples)
    plt.imshow(sample_data[0][0], cmap='gray', interpolation='none')
    print(f"Model prediction is : {model(sample_data).data.max(1)[1][0]}")
    print(f"Ground truth is : {sample_targets[0]}")

if __name__ == '__main__':
    main()