# Notice that in this assignment you are not allowed to use torch optimizer and nn module.
import torch
import sys
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
from math import log10
import matplotlib.pyplot as plt
import numpy as np
# Hyper Parameters

input_size = 784
num_epochs = 10
batch_size = 100
learning_rate = 0.001#float(sys.argv[1])
hidden_size = 128#int(sys.argv[2])
#psnrs = []
#sumWeights = []

#torch.manual_seed(2)

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


def relu(x):
    x[x < 0] = 0
    return x


def reluGrad(x):
    x[x <= 0] = 0
    x[x > 0] = 1
    return x


# Train the Model
#hiddenLayerSizes= [32,64,128,256,512,1028,4112]
#for hidden_size in hiddenLayerSizes:


# initialize your parameters
W1 = np.sqrt(2. / input_size) * torch.randn(input_size, hidden_size)
W2 = np.sqrt(2. / hidden_size) * torch.randn(hidden_size, input_size)

for epoch in range(num_epochs):
    for i, (images, _,) in enumerate(train_loader):
        # Grad für jedes img berechnen. Dann aufsumieren udn avg.

        # Convert torch tensor to Variable
        images = images.view(-1, 28 * 28)
        targets = images.clone()
        # forward
        xw1 = torch.mm(targets, W1)
        model = torch.mm(relu(xw1), W2)

        # loss calculation
        loss = (model - targets).pow(2).sum()
        # gradient calculation and update parameters
        # grad w1 = x^t * 2(x^-x) * w2^t (*) non'(x*w1)
        gradW1 = torch.mm(torch.t(targets), torch.mul(torch.mm((2 * (model - targets)), torch.t(W2)), reluGrad(xw1)))

        # grad w2 = non(x*w1)^t * 2(x^-x)
        gradW2 = torch.mm(torch.t(xw1), (2 * (model - targets)))

        # Update weights.
        W1 = W1 - learning_rate * gradW1
        W2 = W2 - learning_rate * gradW2

        # check your loss
 #       if (i + 1) % 1 == 0:
  #          print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
   #              % (epoch + 1, num_epochs, i + 1, len(train_dataset) // batch_size, loss))

# Test the Model
avg_psnr = 0
for (images, epoch,) in test_loader:
    images = images.view(-1, 28 * 28)
    targets = images.clone()
    # get your predictions
    predictions = torch.mm(relu(torch.mm(images, W1)), W2)
    # calculate PSNR
    mse = torch.mean((predictions - targets).pow(2))
    psnr = 10 * log10(1 / mse)
    avg_psnr += psnr
#print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(test_loader)))
#print("Mse: " + str(mse))
#print("Learning rate: " + str(learning_rate))
#print("#HiddenLayers: " + str(hidden_size))

'''
#Code for plot:
# I had the impression that running everything with a loop changes something.
# I had no idea why so I ran all the different layer sizes and learning ratios individually.
# And wrote down the results.
hiddenLayerSizes = [32,64,128,256,512,1028,4112]
lr00 = [9.4779,9.4779,9.4779,9.4779,9.4779,9.4779,9.4779]
lr00000 = [9.4771, 11.5889, 11.9680, 12.3683, 13.3507, 14.4391, 16.00267]
lr000000 = [9.4738, 9.4694, 9.5915, 10.472, 10.8574, 10.8295, 11.7909]
plt.plot(hiddenLayerSizes, lr00)
plt.plot(hiddenLayerSizes, lr00000)
plt.plot(hiddenLayerSizes, lr000000)
plt.xlabel('number of hidden layer neurons')
plt.ylabel('psnr')
plt.title('psnr in relation to different hidden layer sizes')
plt.legend(['learning rate = 0.001', 'learning rate = 0.000001', 'learning rate = 0.0000001'], loc='upper left')
plt.savefig("plot.png")
plt.show()
'''
