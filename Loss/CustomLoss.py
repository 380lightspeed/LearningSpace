import logging
import torch
import torch.nn as nn


class MAELoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(y_ground, y_pred):

        try:
            assert y_ground.shape == y_pred.shape
        except Exception as e:
            logging.warning(f"SHAPE MISMATCH - y_ground.shape={y_ground.shape}, probabilities.shape={y_pred.shape}. Broadcasting could lead to errors.")
        e = y_ground - y_pred
        ae = torch.abs(e)
        mae = ae.mean()
        return mae

class MSELoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(y_ground, y_pred):

        try:
            assert y_ground.shape == y_pred.shape
        except Exception as e:
            logging.warning(f"SHAPE MISMATCH - y_ground.shape={y_ground.shape}, probabilities.shape={y_pred.shape}. Broadcasting could lead to errors.")
        e = y_ground - y_pred
        se = torch.pow(e, 2)
        mse = se.mean()
        return mse

class LogLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(y_ground:torch.tensor, probabilities:torch.tensor):

        '''
        y_ground: Ground Truth Labels. Binary Tensor of shape (N). 
                  These can have values 0 and 1.

        probabilities: Predicted Probabilities of label being 1.

        returns: logarithmic loss 
        '''

        try:
            assert y_ground.shape == probabilities.shape
        except Exception as e:
            logging.warning(f"SHAPE MISMATCH - y_ground.shape={y_ground.shape}, probabilities.shape={probabilities.shape}. Broadcasting could lead to errors.")
        loss = -(y_ground * torch.log(probabilities) + (1-y_ground) * torch.log(1-probabilities)).mean()
        return loss