import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5147):
        super(PositionalEncoding, self).__init__()
        self.max_len = max_len
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(0)
        if seq_len > self.max_len:
            # Reinitialize positional encoding if the sequence length exceeds max_len
            self.max_len = seq_len
            pe = torch.zeros(self.max_len, x.size(2))
            position = torch.arange(0, self.max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, x.size(2), 2).float() * (-math.log(10000.0) / x.size(2)))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0).transpose(0, 1)
            self.register_buffer('pe', pe)

        x = x + self.pe[:x.size(0), :]
        return x

class AudioDeepfakeTransformer(nn.Module):
    def __init__(self, input_dim=48, n_heads=4, n_layers=4, hidden_dim=256, num_classes=1):
        super(AudioDeepfakeTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim)
        encoder_layers = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=n_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=n_layers)
        self.fc_out = nn.Linear(hidden_dim, num_classes)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.num_classes = num_classes
        
    def forward(self, x):
        # x shape: (batch_size, n, 48)
        x = self.embedding(x)  # Transform input dimension to hidden_dim
        x = self.pos_encoder(x.permute(1, 0, 2))  # Apply positional encoding (n, batch_size, hidden_dim)
        x = self.transformer_encoder(x)  # Apply transformer encoder
        x = x.permute(1, 2, 0)  # (batch_size, hidden_dim, n)
        x = self.pool(x).squeeze(-1)  # Pooling to get (batch_size, hidden_dim)
        output = self.fc_out(x)  # Final classification layer
        output = torch.sigmoid(output)
        # Ensure output has the shape [batch_size, 1] for binary cross-entropy
        return output
