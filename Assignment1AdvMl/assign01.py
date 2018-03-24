# Notice that in this assignment you are not allowed to use torch optimizer and nn module.
import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
from math import log10

# Hyper Parameters

input_size = 784
hidden_size = 128
num_epochs = 10
batch_size = 100
learning_rate = 0.001


def relu(x):
    x[x < 0] = 0
    return x

def reluGrad(x):
    x[x < 0] = 0
    x[x >= 0] = 1
    return x


# Train the Model

# MNIST Dataset
train_dataset = dsets.MNIST(root='./data',
                            train=True,
                            transform=transforms.ToTensor(),
                            download=True)

test_dataset = dsets.MNIST(root='./data',
                           train=False,
                           transform=transforms.ToTensor())

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

# initialize your parameters
W1 = torch.randn(input_size, hidden_size).type(torch.FloatTensor)
W2 = torch.randn(hidden_size, input_size).type(torch.FloatTensor)

for epoch in range(num_epochs):
    for i, (images, _,) in enumerate(train_loader):
        # Grad für jedes img berechnen. Dann aufsumieren udn avg.

        # Convert torch tensor to Variable
        images = images.view(-1, 28 * 28)
        targets = images.clone()
        # forward
        xw1 = torch.mm(targets,W1)
        model = torch.mm(relu(xw1), W2)

        # loss calculation
        loss = (model - targets).pow(2).sum()
        # gradient calculation and update parameters
        # grad w1 = x^t * 2(x^-x) * w2^t (*) non'(x*w1)
        gradW1 = torch.mm(torch.t(images), torch.mul(torch.mm((2 * (model - targets)), torch.t(W2)), reluGrad(xw1)))

        # grad w2 = non(x*w1)^t * 2(x^-x)
        gradW2 = torch.mm(torch.t(xw1),(2 *(images - model)))

        #Update weights.
        W1 = W1 - learning_rate * gradW1
        W2 = W2 - learning_rate * gradW2

        # check your loss
        if (i + 1) % 1 == 0:
            print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
                  % (epoch + 1, num_epochs, i + 1, len(train_dataset) // batch_size, loss))

# Test the Model
avg_psnr = 0
for (images, epoch,) in test_loader:
    images = images.view(-1, 28 * 28)
    targets = images.clone()
    # get your predictions
    predictions = torch.mm(relu(torch.mm(images,W1)),W2)
    # calculate PSNR
    mse = torch.mean((predictions - targets).pow(2))
    psnr = 10 * log10(1 / mse)
    avg_psnr += psnr
print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(test_loader)))