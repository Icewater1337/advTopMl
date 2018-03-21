# Notice that in this assignment you are not allowed to use torch optimizer and nn module.
import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
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


def reulGrad(x):
    x[x < 0] = 0
    x[x >= 0] = 1
    return x


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
W1 = torch.Tensor(input_size, hidden_size)
W2 = torch.Tensor(hidden_size, input_size)

# Train the Model
for epoch in range(num_epochs):
    for i, (images, _,) in enumerate(train_loader):
        # Grad für jedes img berechnen. Dann aufsumieren udn avg.

        # Convert torch tensor to Variable
        images = images.view(-1, 28 * 28)
        targets = images.clone()
        # forward
        x1 = relu(torch.mm(images, W1))
        x2 = torch.mm(x1, W2)

        # loss calculation
        loss = torch.dot((x2 - images).view(-1), (x2 - images).view(-1))
        # gradient calculation and update parameters
        gradW1 = torch.dot((2 * (x2 - images)).view(-1), images.view(-1)) * torch.mm(x1, W2)
        gradW2 = torch.dot((2 * (x2 - images)).view(-1), x1)

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
    # predictions =
    # calculate PSNR
    # mse = torch.mean((predictions - targets).pow(2))
    # psnr = 10 * log10(1 / mse)
# avg_psnr += psnr
print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(test_loader)))
