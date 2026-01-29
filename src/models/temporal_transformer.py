import torch
import torch.nn as nn

class TemporalTransformerEncoder(nn.Module):
    def __init__(self, n_features: int, d_model:int = 128, nhead: int=8, num_layers: int=4, dim_feedforward: int=256, 
                 dropout: float=0.1, out_horizons: int = 24, max_len: int = 512):
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model
        self.out_horizons = out_horizons

        # Input projection: model dimension
        self.in_proj = nn.Linear(n_features, d_model)

        # Learnable positional embeddings
        self.pos_emb = nn.Embedding(max_len, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='gelu',
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, out_horizons)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input : (B, T, F) where B=batch size, T=sequence length, F=number of features
        Output: (B, out_horizons)
        """
        B, T, F = x.shape
        if T > self.pos_emb.num_embeddings:
            raise ValueError(f"Input sequence length {T} exceeds maximum length {self.pos_emb.num_embeddings}")
        
        h = self.in_proj(x)  # (B, T, d_model)
        pos_idx = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        h = h + self.pos_emb(pos_idx)  # Add positional embeddings

        h =self.encoder(h)  # (B, T, d_model)

        #Pool : Using Last token because that is most relevant for forecasting next steps
        last = h[:, -1, :]  # (B, d_model)
        last = self.norm(last)

        yhat = self.head(last)  # (B, out_horizons)
        return yhat