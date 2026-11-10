import os
import torch


def get_optimizer(CONFIG, model):

    optimizer_name = CONFIG['optimizer'].lower()  # .lower() prevents case-sensitive bugs

    if optimizer_name == "adam":
        # Standard Adaptive Moment Estimation
        return torch.optim.Adam(
            model.parameters(), 
            lr=CONFIG['init_lr'],
            weight_decay=CONFIG.get('weight_decay', 0.0)
        )
    
    elif optimizer_name == "adamw":
        # Adam with decoupled weight decay - fixes regularization issues in standard Adam
        # Highly recommended for Transformers and modern CNNs
        return torch.optim.AdamW(
            model.parameters(), 
            lr=CONFIG['init_lr'],
            weight_decay=CONFIG.get('weight_decay', 1e-2)
        )
        
    elif optimizer_name == "sgd":
        # Stochastic Gradient Descent with momentum
        return torch.optim.SGD(
            model.parameters(), 
            lr=CONFIG['init_lr'], 
            momentum=CONFIG.get('momentum', 0.9),
            weight_decay=CONFIG.get('weight_decay', 0.0)
        )
        
    elif optimizer_name == "rmsprop":
        # Root Mean Squared Propagation - great for RNNs and reinforcement learning
        return torch.optim.RMSprop(
            model.parameters(), 
            lr=CONFIG['init_lr'],
            alpha=0.99,
            momentum=CONFIG.get('momentum', 0.0)
        )
        
    elif optimizer_name == "adagrad":
        # Adaptive Gradient Algorithm - scales learning rate based on feature frequency
        # Excellent for sparse data (NLP, embedding layers, recommender systems)
        return torch.optim.Adagrad(
            model.parameters(), 
            lr=CONFIG['init_lr'],
            weight_decay=CONFIG.get('weight_decay', 0.0)
        )
        
    elif optimizer_name == "adadelta":
        # An extension of Adagrad that adapts learning rates based on a moving window
        return torch.optim.Adadelta(
            model.parameters(), 
            lr=CONFIG.get('init_lr', 1.0) # Adadelta often uses a default LR of 1.0
        )
        
    elif optimizer_name == "radam":
        # Rectified Adam - dynamically turns on/off warm-up, making training highly stable
        return torch.optim.RAdam(
            model.parameters(), 
            lr=CONFIG['init_lr'],
            weight_decay=CONFIG.get('weight_decay', 0.0)
        )
    
    else:
        raise ValueError(f"Unsupported optimizer name: {optimizer_name}")