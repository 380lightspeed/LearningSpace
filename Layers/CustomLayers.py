import torch
from torch.autograd import Variable
import torch.nn as nn

class CustomLinearLayer(nn.Module):
    def __init__(self, 
                 input_dims, 
                 output_dims, 
                 bias=True):
        '''
        input_dims -> Integer. Input Dimensions of the input vector (N, input_dims)
        output_dims -> Integer. Output dimensions of the output vector (N, output_dims)
        bias -> Bool. If True, add bias to the linear layer
        '''
        super().__init__()
        self.weights = Variable(torch.randn(input_dims, output_dims), requires_grad=True)
        self.bias = bias
        if self.bias:
            self.b = Variable(torch.randn(1,output_dims), requires_grad=True)

    def forward(self, x):
        '''
        x -> Input vector (N, input_dims)
        returns -> Output vector (N, output_dims)
        '''
        
        y = torch.matmul(x, self.weights) 
        if self.bias:
            y = y + self.b
        return y
    
class CustomRNNLayer(nn.Module):
    def __init__(self, 
                 input_dims, 
                 hidden_state_dims,
                 return_sequences=False,
                 output_dims=1,
                 hidden_state_activation=nn.Tanh(),
                 output_seq_activation=None):
        '''
        input_dims -> Integer. Input Dimensions of the input vector (N, input_dims).
        hidden_state_dims -> Integer. Hidden State Dimensions (N, hidden_state_dims).
        output_dims -> Integer. Output dimensions of the output vector (N, output_dims).
        return_sequences -> Bool. If True return output sequence for each timestep.
        activation
        '''
        super().__init__()
        self.input_dims = input_dims
        self.hidden_state_dims = hidden_state_dims
        self.return_sequences = return_sequences
        
        self.linear_x = CustomLinearLayer(input_dims, hidden_state_dims)
        self.linear_h = CustomLinearLayer(hidden_state_dims, hidden_state_dims)
        
        self.hidden_state_activation = hidden_state_activation
        self.output_seq_activation = output_seq_activation
        if self.return_sequences:
            self.linear_y = CustomLinearLayer(hidden_state_dims, output_dims)
        

    def forward(self, seq_x):
        '''
        seq_x -> Input vectors. expected shape ==> (N, seq_length, input_dims)
        returns
        hidden_states -> hidden_state dimensions ==> (N,seq_length,hidden_state_dims)
        
        output_sequence -> if return_sequence is set as True ==> (N,seq_length,output_dims)
        '''
        batch_len = seq_x.shape[0]
        seq_len = seq_x.shape[1]
        
        h_init = torch.randn(batch_len,self.hidden_state_dims)
        hidden_states = []
        ht_1 = h_init
        for t, i in enumerate(range(seq_len)):
            xt = seq_x[:, t, :]
            ht = self.hidden_state_activation(self.linear_x(xt) + self.linear_h(ht_1))
            hidden_states.append(ht)
            ht_1 = ht
        
        hidden_states = torch.stack(hidden_states, dim=1)
        print(hidden_states.shape)
        if self.return_sequences:
            y = self.linear_y(hidden_states)
            if self.output_seq_activation is not None:
                y = self.output_seq_activation(y)
            return hidden_states, y
        return hidden_states            