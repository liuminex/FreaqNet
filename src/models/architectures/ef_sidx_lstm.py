import torch
import torch.nn as nn

N_FREQS = 7          # keep in sync with extract_features.N_FREQS (or import it)
FREQ_COL = 0
PERIOD_COL = 1       # unused by the model (determined by the frequency ID)
SEASONAL_START = 2
N_SEASONAL = 12      # keep equal to extract_features.SEASONAL_VECTOR_LEN
N_EXOGS = 2 + N_SEASONAL   # [freq_id, period, sidx_1..sidx_12] = 14


class SIdxLSTM(nn.Module):
    def __init__(self, input_shape, output_shape, hidden_dim=128, layers_per_block=2,
                 n_blocks=3, embed_dim=2, n_seasonal=N_SEASONAL, n_freqs=N_FREQS):
        """
        input_shape: ws (window size)
        output_shape: fh (forecasting horizon)
        """
        super().__init__()
        self.n_blocks = n_blocks
        self.layers_per_block = layers_per_block
        self.n_seasonal = n_seasonal

        # 1. Frequency embedding (one entry per frequency ID)
        self.exog_embed = nn.Embedding(num_embeddings=n_freqs, embedding_dim=embed_dim)

        # 2. Input projection per step (1 value + embed_dim + n_seasonal)
        total_step_features = 1 + embed_dim + n_seasonal
        self.input_proj = nn.Linear(total_step_features, hidden_dim)

        # 3. Residual LSTM blocks
        self.blocks = nn.ModuleList()
        for _ in range(n_blocks):
            block_layers = nn.ModuleList()
            for _ in range(layers_per_block):
                block_layers.append(nn.LSTM(hidden_dim, hidden_dim, batch_first=True))
            self.blocks.append(block_layers)

        # 4. Output layer
        self.out = nn.Linear(hidden_dim, output_shape)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)

        for block in self.blocks:
            for layer in block:
                if isinstance(layer, nn.LSTM):
                    for name, param in layer.named_parameters():
                        if 'weight_ih' in name:
                            nn.init.xavier_uniform_(param.data)
                        elif 'weight_hh' in name:
                            nn.init.orthogonal_(param.data)
                        elif 'bias' in name:
                            nn.init.zeros_(param.data)
                            hidden_dim = param.size(0) // 4
                            param.data[hidden_dim:2 * hidden_dim] = 1.0  # forget gate bias

        nn.init.xavier_uniform_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x, exogs):
        """
        x:     [batch, ws]
        exogs: [batch, 14] = [freq_id, period, sidx_1, ..., sidx_12]
        """
        batch_size, seq_len = x.shape

        # --- 1. Frequency embedding ---
        freq_ids = exogs[:, FREQ_COL].round().long()                           # [B]
        embedded_freq = self.exog_embed(freq_ids)                              # [B, embed_dim]

        # --- 2. Seasonal features ---
        seasonal_features = exogs[:, SEASONAL_START:SEASONAL_START + self.n_seasonal]  # [B, 12]

        # --- 3. Main input as an explicit sequence ---
        x_seq = x.unsqueeze(-1)                                                # [B, ws, 1]

        # --- 4. Broadcast static exogenous features across window steps ---
        static_exogs = torch.cat([embedded_freq, seasonal_features], dim=1)    # [B, embed_dim + 12]
        static_seq = static_exogs.unsqueeze(1).expand(-1, seq_len, -1)         # [B, ws, embed_dim + 12]

        # --- 5. Combine on the feature dimension ---
        x_combined = torch.cat([x_seq, static_seq], dim=2)                     # [B, ws, 1 + embed_dim + 12]

        # --- 6. Step projection ---
        h = self.input_proj(x_combined)                                        # [B, ws, hidden_dim]

        # --- 7. Residual recurrent blocks ---
        for block in self.blocks:
            residual = h
            for lstm in block:
                h, _ = lstm(h)
            h = h + residual

        # --- 8. Linear forecast horizon map ---
        return self.out(h[:, -1, :])                                           # [B, fh]


# Factory function
def model_sidx_lstm(ws, fh, hidden_units=128, n_layers=2, n_blocks=3, embed_dim=2):
    model = SIdxLSTM(
        input_shape=ws,
        output_shape=fh,
        hidden_dim=hidden_units,
        layers_per_block=n_layers,
        n_blocks=n_blocks,
        embed_dim=embed_dim,
        n_seasonal=N_SEASONAL,
        n_freqs=N_FREQS,
    )

    return {
        'model': model,
        'id': f"sidx_lstm_{n_blocks}b_{n_layers}l_{hidden_units}u",
        'n_exogs': N_EXOGS,          # 14 (was 42)
        'window_size': ws,
        'forecasting_horizon': fh,
    }