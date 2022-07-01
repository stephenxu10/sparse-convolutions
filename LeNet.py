"""
Train a LeNet Model (two convolutional layers and three fully-connected layers) on Fashion-MNIST to obtain
the best accuracy we can.
"""
import numpy as np
import torch
from torch import nn
import datetime
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms
from convolutions import LinearCNNLayer
from SparseNet import SparseLeNet
from GoldStandard import GoldNet

device = "cuda" if torch.cuda.is_available() else "cpu"


def _dict2lists(data):
    """
    Convert a dictionary into a list of keys and values, sorted by
    key.

    Arguments:
    data -- dictionary

    Returns:
    A tuple of two lists: the first is the keys, the second is the values
    """
    xvals = list(data.keys())
    xvals.sort()
    yvals = []
    for x in xvals:
        yvals.append(data[x])
    return xvals, yvals


def _plot_dict_line(d, label=None):
    """
    Plot data in the dictionary d on the current plot as a line.

    Arguments:
    d     -- dictionary
    label -- optional legend label

    Returns:
    None
    """
    xvals, yvals = _dict2lists(d)
    if label:
        plt.plot(xvals, yvals, label=label)
    else:
        plt.plot(xvals, yvals)


def plot_lines(data, title, xlabel, ylabel, labels=None, filename=None):
    """
    Plot a line graph with the provided data.

    Arguments:
    data     -- a list of dictionaries, each of which will be plotted
                as a line with the keys on the x-axis and the values on
                the y-axis.
    title    -- title label for the plot
    xlabel   -- x-axis label for the plot
    ylabel   -- y-axis label for the plot
    labels   -- optional list of strings that will be used for a legend
                this list must correspond to the data list
    filename -- optional name of file to which plot will be
                saved (in png format)

    Returns:
    None
    """
    # Check that the data is a list
    if not isinstance(data, list):
        msg = "data must be a list, not {0}".format(type(data).__name__)
        raise TypeError(msg)

    # Create a new figure
    fig = plt.figure()

    # Plot the data
    if labels:
        mylabels = labels[:]
        for _ in range(len(data)-len(labels)):
            mylabels.append("")
        for d, l in zip(data, mylabels):
            _plot_dict_line(d, l)
        # Add legend
        plt.legend(loc='best')
        gca = plt.gca()
        legend = gca.get_legend()
        plt.setp(legend.get_texts(), fontsize='medium')
    else:
        for d in data:
            _plot_dict_line(d)

    # Set the lower y limit to 0 or the lowest number in the values
    mins = [min(l.values()) for l in data]
    ymin = min(mins)
    plt.ylim(ymin=ymin)

    plt.xticks(np.arange(1, epochs, 1))

    # Label the plot
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # Draw grid lines
    plt.grid(True)

    # Show the plot
    fig.show()

    # Save to file
    if filename:
        plt.savefig(filename)


training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=False,
    transform=transforms.ToTensor()
)

testing_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=False,
    transform=transforms.ToTensor()
)

# Define some hyper-parameters
batch_size = 64
epochs = 20

train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(testing_data, batch_size=batch_size)


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = LinearCNNLayer(28, 5, 1, 32, "normal", False)
        self.act1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = LinearCNNLayer(12, 5, 32, 16, "normal", False)
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(16*4*4, 64)
        self.act3 = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.act4 = nn.ReLU()
        self.fc3 = nn.Linear(32, 10)

    def forward(self, x):
        out = self.pool1(self.act1(self.conv1(x)))
        out = self.pool2(self.act2(self.conv2(out)))
        out = out.view(-1, 16*4*4)
        out = self.act3(self.fc1(out))
        out = self.act4(self.fc2(out))
        out = self.fc3(out)
        return out


lin_cnn = LeNet().to(device)
sparse_cnn = SparseLeNet().to(device)
golden = GoldNet().to(device)

loss_fn = nn.CrossEntropyLoss()


def train_loop(dataloader, network, loss_function, optim):
    size = len(dataloader.dataset)

    for batch, (img_batch, result_batch) in enumerate(dataloader):
        img_batch = img_batch.to(device)
        result_batch = result_batch.to(device)

        pred = network(img_batch)
        loss = loss_function(pred, result_batch)

        optim.zero_grad()
        loss.backward()
        optim.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(pred)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, network, loss_function):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for img_batch, result_batch in dataloader:
            img_batch = img_batch.to(device)
            result_batch = result_batch.to(device)

            pred = network(img_batch)
            test_loss += loss_function(pred, result_batch).item()

            correct += (pred.argmax(1) == result_batch).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(datetime.datetime.now())
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return 100 * correct


def create_dict(model):
    output = {}

    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")

        if t <= 25:
            optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
        elif t <= 40:
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=0.005)

        train_loop(train_dataloader, model, loss_fn, optimizer)
        output[t+1] = test_loop(test_dataloader, model, loss_fn)

    return output


# normal_data = [create_dict(golden), create_dict(lin_cnn), create_dict(sparse_cnn)]
# print(normal_data)

"""
normal_data = [{1: 73.14, 2: 80.47, 3: 82.48, 4: 83.32, 5: 85.14, 6: 85.9, 7: 86.27, 8: 86.89, 9: 87.65, 
10: 88.04, 11: 88.2, 12: 88.56, 13: 88.82, 14: 88.79, 15: 89.14, 16: 89.01, 17: 89.27, 18: 89.26, 19: 89.5, 20: 89.27}, 
{1: 76.23, 2: 82.23, 3: 84.41, 4: 85.86, 5: 86.87, 6: 87.36, 7: 88.14, 8: 88.03, 9: 88.37, 
10: 88.59, 11: 88.97, 12: 89.08, 13: 88.88, 14: 88.63, 15: 88.38, 16: 88.03, 17: 88.49, 18: 88.68, 19: 88.36, 20: 88.91}, 
{1: 69.21, 2: 77.8, 3: 80.79, 4: 82.8, 5: 82.95, 6: 84.32, 7: 84.77, 8: 86.13, 9: 86.43,
 10: 86.92, 11: 87.4, 12: 87.58, 13: 88.51, 14: 88.49, 15: 88.33, 16: 88.9, 17: 89.06, 18: 89.08, 19: 89.21, 20: 89.26}]
"""

# plot_lines(normal_data, "Epoch vs Accuracy Progression in 3 LeNet Models", "Epoch", "Accuracy", ["Standard", "LinearCNNLayer",
# "SparseConv2D"],
# "normal_plots")

"""
shuffle_data = [create_dict(golden), create_dict(lin_cnn), create_dict(sparse_cnn)]
plot_lines(shuffle_data, "Epoch vs Accuracy in 3 LeNet Models with Shuffle", "Epoch", "Accuracy", ["Standard", "LinearCNNLayer",
 "SparseConv2D"], "shuffle_plots")


shuffle_data = 
[{1: 75.63, 2: 81.40, 3: 84.40, 4: 85.64, 5: 86.2, 6: 87.15, 7: 87.68, 8: 88.44, 9: 88.28, 
10: 89.03, 11: 89.26, 12: 88.95, 13: 89.37, 14: 89.51, 15: 88.95, 16: 89.60, 17: 89.60, 18: 89.46, 19: 89.88, 20: 89.81}, 
{1: 68.8, 2: 81.92, 3: 83.51, 4: 84.33, 5: 85.27, 6: 85.0, 7: 86.21, 8: 86.02, 9: 86.24, 
10: 85.91, 11: 86.03, 12: 85.85, 13: 86.2, 14: 85.96, 15: 86.29, 16: 86.46, 17: 85.87, 18: 86.11, 19: 85.62, 20: 86.03}, 
{1: 71.41, 2: 80.92, 3: 82.97, 4: 84.46, 5: 86.35, 6: 86.83, 7: 87.46, 8: 87.63, 9: 88.13, 
10: 87.67, 11: 87.89, 12: 88.44, 13: 88.55, 14: 88.09, 15: 88.37, 16: 88.43, 17: 88.47, 18: 87.27, 19: 88.5, 20: 87.96}]

print(shuffle_data)
"""
"""
scatter_data = [create_dict(golden), create_dict(lin_cnn), create_dict(sparse_cnn)]
plot_lines(scatter_data, "Epoch vs Accuracy in 3 LeNet Models with Scatter", "Epoch", "Accuracy",
           ["Standard", "LinearCNNLayer", "SparseConv2D"], "scatter_data")

print(scatter_data)



scatter_data = [{1: 77.0, 2: 80.36, 3: 84.2, 4: 85.65, 5: 86.85, 6: 87.26, 7: 87.86, 8: 88.1, 9: 87.99, 
10: 87.94, 11: 88.05, 12: 88.36, 13: 88.51, 14: 88.61, 15: 89.06, 16: 89.33, 17: 89.5, 18: 89.38, 19: 89.61, 20: 89.27}, 
{1: 71.56, 2: 80.16, 3: 82.82, 4: 83.82, 5: 84.93, 6: 85.1, 7: 85.12, 8: 85.29, 9: 85.49, 
10: 85.9, 11: 85.71, 12: 85.85, 13: 85.78, 14: 85.17, 15: 86.3, 16: 86.28, 17: 85.61, 18: 85.94, 19: 85.36, 20: 86.11}, 
{1: 69.75, 2: 81.49, 3: 84.08, 4: 85.04, 5: 85.47, 6: 85.05, 7: 84.94, 8: 85.01, 9: 85.05, 
10: 84.69, 11: 85.40, 12: 85.76, 13: 85.08, 14: 85.40, 15: 85.68, 16: 85.5, 17: 85.74, 18: 84.91, 19: 85.52, 20: 85.19}]

"""