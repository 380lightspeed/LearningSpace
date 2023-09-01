import torch
from torch.autograd import Variable
import torch.nn as nn

class CustomLinearLayer(nn.Module):
    def __init__(self, input_dims, output_dims):
        super().__init__()
        self.weights = Variable(torch.randn(input_dims, output_dims), requires_grad=True)
        self.bias = Variable(torch.randn(1,output_dims), requires_grad=True)

    def forward(self, x):
        y = torch.matmul(x, self.weights)
        return y + self.bias