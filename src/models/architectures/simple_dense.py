import torch
import torch.nn as nn


class SimpleDense(nn.Module):

    def __init__(self, input_shape, output_shape, units=500, n_layers=3):
        super().__init__()

        # in_features = number of input features (dimensionality of input data) (= window size)
        in_features = input_shape if isinstance(input_shape, int) else int(input_shape)

        layers = []

        """
        nn.Linear(in_features, out_features)
            creates a fully connected layer that performs the affine transformation: y = xW^T + b
            cleates:
                - Weight matrix of shape (out_features, in_features)
                - Bias vector of shape (out_features,)
        """
        layers.append(nn.Linear(in_features, units))

        for _ in range(n_layers - 1):
            layers.append(nn.ReLU()) # Nonlinear activation: max(0, x), introduces nonlinearity
            layers.append(nn.Linear(units, units)) # another fully connected layer mapping units -> units

        self.hidden = nn.ModuleList(layers) # register the layers
        
        self.out = nn.Linear(units, output_shape) # out layer (units -> output dimension)

        self.relu = nn.ReLU() # create a reusable ReLU instance (used during forward pass after Linear layers)

        for layer in self.hidden:
            if isinstance(layer, nn.Linear):

                # xavier sets the weights from a uniform distribution scaled to preserve signal variance across layers
                nn.init.xavier_uniform_(layer.weight)

                # initializes bias vector to zeros
                nn.init.zeros_(layer.bias)

        # same initialization for output layer
        nn.init.xavier_uniform_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x):
        for layer in self.hidden:
            if isinstance(layer, nn.Linear):

                # if the layer is lenear, apply linear transformation followed by ReLU activation
                x = self.relu(layer(x))
            else:
                
                # if the layer is already an activation (ReLU), apply it directly
                x = layer(x)

        x = self.out(x)
        return x


def model_simple_dense(ws, fh, hidden_units_per_layer, n_layers):

    model = SimpleDense(
        input_shape=ws,
        output_shape=fh,
        units=hidden_units_per_layer,
        n_layers=n_layers
    )

    return {
        'model': model,
        'id': f"simple_dense_{n_layers}l_{hidden_units_per_layer}u",
        'window_size': ws,
        'forecasting_horizon': fh
    }



