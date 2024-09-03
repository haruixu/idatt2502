import torch
from torch import nn
import torchvision
import matplotlib.pyplot as plt

mnist_train = torchvision.datasets.MNIST('./data', train=True, download=True)
# Reshape input to 60_000x784 from 60_000x28x28
x_train = mnist_train.data.reshape(-1, 784).float()
y_train = torch.zeros((mnist_train.targets.shape[0], 10))  # 60_000x10 matrix
y_train[torch.arange(mnist_train.targets.shape[0]),
        mnist_train.targets] = 1  # Populate output - classifies the correct number

mnist_test = torchvision.datasets.MNIST('./data', train=False, download=True)
x_test = mnist_test.data.reshape(-1, 784).float()  # Reshape input
y_test = torch.zeros((mnist_test.targets.shape[0], 10))  # Create output tensor
y_test[torch.arange(mnist_test.targets.shape[0]),
       mnist_test.targets] = 1  # Populate output


class MnstRegressionModel(nn.Module):
    def __init__(self):

        super(MnstRegressionModel, self).__init__()
        # Model variables
        self.W = torch.zeros((784, 10), requires_grad=True)
        self.b = torch.zeros((1, 10), requires_grad=True)

    def logits(self, x):
        return x @ self.W + self.b

    def f(self, x):
        return nn.functional.softmax(self.logits(x))

    def loss(self, x, y):
        return nn.functional.cross_entropy(self.logits(x), y.argmax(1))

    def accuracy(self, x, y):
        return torch.mean(torch.eq(self.f(x).argmax(1),
                                   y.argmax(1)).float())


model = MnstRegressionModel()

# Optimize: adjust W and b to minimize loss using stochastic gradient descent
optimizer = torch.optim.SGD([model.W, model.b], lr=0.0001)
for epoch in range(1000):
    if model.accuracy(x_test, y_test) > 0.91:
        # printing with item removes the tensor wrapper
        print("finished with accuracy %f" % model.accuracy(x_test, y_test))
        break

    if epoch % 10 == 0:
        print("epoch: %d, loss: %f, accuracy: %f" % (epoch, model.loss(x_train, y_train).detach().item(),
              model.accuracy(x_test, y_test).item()))
    model.loss(x_train, y_train).backward()  # Compute loss gradients
    optimizer.step()  # Perform optimization by adjusting W and b,
    optimizer.zero_grad()  # Clear gradients for next step

f, ax = plt.subplots(1, 10)
# NOTE: Remember that W is a 784x10 matrix, each column representing pixels for an entire 28x28 image, which was warped to a 784x1 matrix
for i in range(10):
    image = model.W[:, i].detach().reshape(28, 28)
    plt.imsave(("w_%d.png" % i), image)
    ax[0, i] = plt.imshow(image)
