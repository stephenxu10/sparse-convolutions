import torch
import numpy as np
import torch.nn as nn

input_s = 3
kernel_s = 2
in_channel = 1
out_channel = 1
batches = 1
connect = "normal"
biased = True
output_s = input_s - kernel_s + 1

device = "cuda" if torch.cuda.is_available() else "cpu"


class LinearCNNLayer(nn.Module):
    def __init__(self, size_in, size_kern, in_channels, out_channels, connect_type, bias):
        super().__init__()
        self.size_in, self.size_kern = size_in, size_kern
        self.in_chan, self.out_chan = in_channels, out_channels
        self.connect_type = connect_type

        self.size_out = self.size_in - self.size_kern + 1

        self.pos_in, self.param_idx, self.sparse_size = self.generate_sparse_weights()

        _conv = nn.Conv2d(self.in_chan, self.out_chan, size_kern)
        self.weight = nn.Parameter(
            _conv.weight.data.reshape(-1), requires_grad=True,
        )

        if bias:
            self.bias = nn.Parameter(torch.zeros(self.out_chan), requires_grad=True)
        else:
            self.bias = None

    def randomize_weights(self, idxs):
        if self.connect_type == "shuffle":
            indexes = []
            nnz = self.in_chan * (self.size_kern * self.size_out) ** 2
            channel = np.array(idxs)[0: nnz, 0].reshape(-1, self.size_out ** 2)

            for k in range(self.out_chan):
                x = np.stack([np.random.permutation(channel[:, i]) for i in range(channel.shape[1])], axis=1)
                for j in range(x.shape[1]):
                    col = x[:, j]

                    for c in col:
                        indexes.append([c, k * self.size_out ** 2 + j])
            return indexes

        elif self.connect_type == "scatter":
            indexes = []
            nnz = self.in_chan * (self.size_kern * self.size_out) ** 2
            channel = np.array(idxs)[0: nnz, 0].reshape(-1, self.size_out ** 2)

            for k in range(self.out_chan):
                x = np.stack([np.random.permutation(channel[i]) for i in range(channel.shape[0])], axis=0)

                for j in range(x.shape[1]):
                    col = x[:, j]
                    for c in col:
                        indexes.append([c, k * self.size_out ** 2 + j])

            return indexes

        else:
            return idxs

    def generate_sparse_weights(self):
        """
        given an input size and a convolution kernel size, the function
        returns a weight tensor that will be used for a linear layer.

        Assumes:
            - The input tensor and kernel have the dimensions of a square.
            - padding = 0
            - stride = 1
        """
        sparse_size = torch.Size((self.in_chan * self.size_in ** 2, self.out_chan * self.size_out ** 2))

        indices = []
        values = []

        if self.size_kern > self.size_in:
            print("The kernel is too large for the input tensor!")
            return None

        else:
            weight_col = 0
            for y in range(self.out_chan):
                for i in range(self.size_out):
                    for j in range(self.size_out):
                        curr_row = i
                        curr_col = j

                        for k in range(self.in_chan):
                            for k1 in range(self.size_kern):
                                for k2 in range(self.size_kern):
                                    indices.append([curr_row * self.size_in + curr_col + k * self.size_in ** 2 + k2,
                                                    weight_col])
                                    values.append(k1 * self.size_kern + k2
                                                  + (k + self.in_chan * y) * self.size_kern ** 2)

                                curr_col = j
                                curr_row += 1

                            curr_row = i
                            curr_col = j

                        weight_col += 1

        indices = torch.Tensor(self.randomize_weights(indices)).to(device)
        values = torch.Tensor(values).type(torch.LongTensor).to(device)

        return indices.t(), values, sparse_size

    def forward(self, x):
        sparse_weights = torch.sparse_coo_tensor(self.pos_in, self.weight[self.param_idx], self.sparse_size).t()

        x = x.view(-1, self.in_chan * self.size_in ** 2)
        output = torch.sparse.mm(sparse_weights, x.t()).t()
        output = output.reshape(-1, self.out_chan, self.size_out, self.size_out)

        if self.bias is not None:
            output = output + self.bias[:, None, None]

        return output

    def test(self, x):
        """
        Given an input tensor,  this function evaluates the linear CNN model with the PyTorch 2D Convolution.
        Prints relevant results and outputs if the output and weight gradients match.

        x - an input tensor of size [batches, in_ch, side, side]
        """

        print(f"The input tensor is: \n{x}\n")

        print("====================== SPARSE LAYER RESULTS ========================")
        sparse_model = LinearCNNLayer(self.size_in, self.size_kern, self.in_chan, self.out_chan, self.connect_type,
                                      biased)
        print(f"Model weights: \n {sparse_model.weight} \n")
        print(f"Biases: \n {sparse_model.bias}\n")

        print("-----------------------------------------------")
        prediction = sparse_model(x)
        print(f"Prediction: \n {prediction} \n")
        prediction.sum().backward()

        print("------------------------------------------------")

        print(f"Weight gradients: \n {sparse_model.weight.grad} \n")

        if sparse_model.bias is not None:
            print(f"Bias gradients: \n {sparse_model.bias.grad} \n")

        print("====================== 2D CONVOLUTION RESULTS ========================")
        conv = nn.Conv2d(self.in_chan, self.out_chan, self.size_kern, bias=biased)
        print(conv.weight)

        conv.weight = nn.Parameter(sparse_model.weight.reshape(self.out_chan, self.in_chan, self.size_kern, self.size_kern))
        conv.bias = sparse_model.bias
        cnn_pred = conv(x)
        print(f"Prediction: \n {cnn_pred} \n")
        cnn_pred.sum().backward()

        g_c = conv.weight.grad.reshape(-1)
        print("--------------------------------------------------------------------")
        print(f"Weight gradients: \n {g_c} \n")

        if sparse_model.bias is not None:
            print(f"Bias gradients: \n {conv.bias.grad} \n")

        print(torch.allclose(prediction, cnn_pred))
        print(torch.allclose(g_c, sparse_model.weight.grad))
        print(torch.allclose(conv.bias.grad, sparse_model.bias.grad))


# linear_CNN = LinearCNNLayer(input_s, kernel_s, in_channel, out_channel, connect, biased)
# tester = torch.randn(batches, in_channel, input_s, input_s)
# linear_CNN.test(tester)
