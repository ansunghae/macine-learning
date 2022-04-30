from cgi import test
from pickletools import optimize
from turtle import forward
from h11 import Data
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn as nn
from torchvision.transforms import ToTensor
import random

train_dataset = datasets.MNIST(root='data', train=True, download=True, transform=ToTensor())
test_dataset = datasets.MNIST(root='data', train=False, download=True, transform=ToTensor())

batch_size = 100

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device : {device}')

input_size = 28*28
output_size = 10

class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        out = self.flatten(x)
        out = self.linear(out)
        return out

model = NeuralNet().to(device)

learning_rate = 0.001
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)


def train(train_dataloader, model, loss_fn, optimizer):
    for batch, (X,y) in enumerate(train_dataloader):
        X,y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch%100==0:
            loss = loss.item()
            print(f"loss : {loss}, batch : {batch}")

def test(test_dataloader, model, loss_fn):
    size = len(test_dataloader)
    num_batches = len(test_dataloader)
    test_loss, correct = 0, 0


    with torch.no_grad():
        for X, y in test_dataloader:
            X, y = X.to(device), y.to(device)

            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1)==y).type(torch.float).sum().item()

    test_loss/=num_batches
    correct/=size
    print(f"Test Error :\n Accuracy : {(correct)}%, Avg loss : {test_loss:>8f}\n")

epochs = 10
for t in range(epochs):
    print(f"Epoch : {t+1}\n----------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done")


real_test_dataset = datasets.MNIST(root='data', train=False, download=True)
rand = random.randint(0, 9999)
X, y = test_dataset[rand][0], test_dataset[rand][1]
A = real_test_dataset[rand][0]

with torch.no_grad():
    pred = model(X)
    predicate, actual = pred[0].argmax(0), y
    print(f"predicate : {predicate}, actual : {actual}")
    plt.imshow(A, cmap='gray')
    plt.show()