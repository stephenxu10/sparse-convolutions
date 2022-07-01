"""
author: Zhe Li
email: zhe.li@bcm.edu
"""

import torch
import torch.nn as nn

from utils import sparse_support

Tensor = torch.Tensor
device = "cuda" if torch.cuda.is_available() else "cpu"

class SparseConv2d(nn.Module):

    def __init__(self,
        H_in: int, W_in: int,
        in_channels: int, out_channels: int,
        kernel_size: int, stride: int = 1, padding: int = 0,
        padding_mode: str = 'zeros', connect_type: str = 'normal',
        in_consistent: bool = True, out_consistent: bool = True,
        bias: bool = False,
    ):
        r"""
        Args
        ----
        H_in, W_in:
            The height and width of inputs.
        in_channels, out_channels:
            The number of input and output channels.
        kernel_size, stride, padding:
            The kernel size, stride and padding of a baseline convolution.
        padding_mode:
            The padding mode, can be 'zeros' or 'circular'.
        connect_type:
            The sparse connection type, can be 'normal', 'shuffle' or 'scatter'.
            See `utils.random_pos_channel` for more information.
        in_consistent, out_consistent:
            Whether the spatial pattern is consistent across input or output
            channels.
        bias:
            Whether to use bias.

        """
        super(SparseConv2d, self).__init__()
        self.H_in, self.W_in = H_in, W_in
        self.in_channels, self.out_channels = in_channels, out_channels
        self.kernel_size, self.stride, self.padding = kernel_size, stride, padding
        self.padding_mode = padding_mode
        self.connect_type = connect_type
        self.in_consistent, self.out_consistent = in_consistent, out_consistent

        _conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.weight = nn.Parameter(
            _conv.weight.data.reshape(-1), requires_grad=True,
        )
        if bias:
            self.bias = nn.Parameter(
                torch.zeros(out_channels), requires_grad=True,
            )
        else:
            self.bias = None

        self.H_out, self.W_out, w_idxs, self.w_size = sparse_support(
            H_in, W_in, in_channels, out_channels, kernel_size, stride, padding,
            padding_mode, connect_type, in_consistent, out_consistent,
        )
        self.pos_idxs = nn.Parameter(
            torch.LongTensor(w_idxs[:2]), requires_grad=False
        )
        self.param_idxs = nn.Parameter(
            torch.LongTensor(w_idxs[2]), requires_grad=False,
        )

    def extra_repr(self):
        return '\n'.join([
            '{}, {}, kernel_size={}, stride={}, padding={},'.format(
                self.in_channels, self.out_channels,
                self.kernel_size, self.stride, self.padding,
            ),
            '{}, H_in={}, W_in={}, H_out={}, W_out={}'.format(
                self.connect_type, self.H_in, self.W_in, self.H_out, self.W_out,
            ),
        ])

    def forward(self, inputs: Tensor) -> Tensor:
        r"""Forward pass.

        Args
        ----
        inputs: (N, C, H, W)
            Batch inputs to the layer.

        Returns
        -------
        outputs: (N, C', H, W)
            Batch outputs of the layer.

        """
        _, C, H, W = inputs.shape
        assert C==self.in_channels and H==self.H_in and W==self.W_in, "Invalid input shape"
        s_weight = torch.sparse.FloatTensor(
            self.pos_idxs, self.weight[self.param_idxs], torch.Size(self.w_size),
        )
        inputs = inputs.view(-1, self.in_channels*self.H_in*self.W_in).to(device)
        outputs = torch.sparse.mm(s_weight, inputs.t()).t().to(device)
        outputs = outputs.view(-1, self.out_channels, self.H_out, self.W_out)
        if self.bias is not None:
            outputs = outputs+self.bias[:, None, None]
        return outputs


