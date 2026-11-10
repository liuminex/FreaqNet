import torch
import torch.nn as nn

class MAPELoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, y_pred, y_true):
        # eps prevents division by zero if an actual value is 0
        absolute_percentage_errors = torch.abs((y_true - y_pred) / (torch.abs(y_true) + self.eps))
        return torch.mean(absolute_percentage_errors)


class SMAPELoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, y_pred, y_true):
        denominator = (torch.abs(y_true) + torch.abs(y_pred)) / 2.0
        # eps prevents 0 / 0 if both target and prediction are exactly 0
        absolute_percentage_errors = torch.abs(y_pred - y_true) / (denominator + self.eps)
        return torch.mean(absolute_percentage_errors)


def get_criterion(CONFIG):
    criterion_name = CONFIG['criterion']
    device = CONFIG['device']

    if criterion_name == "mse":
        # Mean Squared Error - sensitive to outliers
        return nn.MSELoss().to(device)
    
    elif criterion_name == "mae" or criterion_name == "l1":
        # Mean Absolute Error - highly robust to outliers
        return nn.L1Loss().to(device)
    
    elif criterion_name == "smoothL1":
        # L1-like when error is large, L2-like when error is close to 0
        return nn.SmoothL1Loss(beta=0.5).to(device)
        
    elif criterion_name == "huber":
        # Similar to smoothL1 but parameterizes delta directly
        return nn.HuberLoss(delta=1.0).to(device)
        
    elif criterion_name == "mape":
        # Mean Absolute Percentage Error
        return MAPELoss().to(device)
        
    elif criterion_name == "smape":
        # Symmetric Mean Absolute Percentage Error
        return SMAPELoss().to(device)
    
    else:
        raise ValueError(f"Unsupported criterion name: {criterion_name}")
