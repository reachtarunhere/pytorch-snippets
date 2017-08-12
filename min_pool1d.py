import torch
import torch.nn as nn


class MinPool1d(nn.Module):
    """ Simple Hack for 1D min pooling. Input size = (N, C, L_in)
        Output size = (N, C, L_out) where N = Batch Size, C = No. Channels
        L_in = size of 1D channel, L_out = output size after pooling.

        This implementation does not support custom strides, padding or dialation
        Input shape compatibilty by kernel_size needs to be ensured"""
    
    def __init__(self, kernel_size=3):
        super(MinPool1d, self).__init__()
        self.kernel_size = kernel_size
    
    def forward(self, l):
        N, C, L = [l.size(i) for i in range(3)]
        l = l.view(N, C, int(L/self.kernel_size), self.kernel_size)
        return l.min(dim=3)[0].view(N,C, -1)
