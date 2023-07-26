import torch
import torch.nn as nn

class CustomRNNLayer(nn.module):

    def __init__(self, input_dims, output_dims):
        self.input_dims = input_dims
        self.output_dims = output_dims
        super(CustomRNNLayer,self).__init__()

    def forward(self):
        pass
