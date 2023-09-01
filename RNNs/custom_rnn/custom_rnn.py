import torch
import torch.nn as nn
from torch.autograd import Variable

class CustomRNNLayer(nn.Module):

    def __init__(self, input_dims, output_dims):
        # self.x_dims = input_dims
        # self.y_dims = output_dims
        self.whh = Variable(torch.randn(input_dims,output_dims), requires_grad=True)
        self.wxh = Variable(torch.randn(input_dims,output_dims), requires_grad=True)
        self.why = Variable(torch.randn(1, output_dims), requires_grad=True)

        super(CustomRNNLayer,self).__init__()

    def forward(self,xs):
        print(xs.shape)
        hidden_state = torch.zeros(1,xs.shape[1])
        hidden_states = []
        for x in xs:
            hidden_state = torch.tanh(torch.matmul(x[0].unsqueeze(0), self.wxh) + torch.matmul(hidden_state, self.whh))
            hidden_states.append(hidden_state)
        return hidden_states
