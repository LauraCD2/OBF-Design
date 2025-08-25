import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 33):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)  # even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # odd indices
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        pe = self.pe[:x.size(0)] # [seq_len, embedding_dim]
        pe = pe.unsqueeze(1)     # [seq_len, 1, embedding_dim]
        x = x + pe
        return x

class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=33):
        super().__init__()
        self.pe = nn.Embedding(max_len, d_model)
        
    def forward(self, x):
        position = torch.arange(x.size(0), device=x.device).unsqueeze(1)
        return x + self.pe(position)
    

class TSTransformerEncoderClassiregressor(nn.Module):
    def __init__(self, feat_dim, input_dim, d_model, n_heads, num_layers, dim_feedforward, num_classes,
                 dropout=0.1, pos_encoding='fixed', activation='gelu', norm='BatchNorm', freeze=False, patch_size=4):
        super(TSTransformerEncoderClassiregressor, self).__init__()


        self.fist_batchnorm = nn.BatchNorm1d(input_dim, momentum=0.9)

        self.d_model = d_model
        self.n_heads = n_heads

        self.patch_size = patch_size
        # Calculate padding and max sequence length after patching
        if input_dim % self.patch_size == 0:
            self.padding = 0
            self.max_len = input_dim // self.patch_size
        else:
            self.padding = self.patch_size - (input_dim % self.patch_size)
            self.max_len = (input_dim + self.padding) // self.patch_size
            
        self.feat_dim = feat_dim * self.patch_size
        
        # Projection layer
        self.project_inp = nn.Linear(self.feat_dim, d_model)
        
        # Positional encoding
        self.pos_enc = LearnedPositionalEncoding(d_model, self.max_len)
        
        # LayerNorm after projection
        self.norm_after_project = nn.LayerNorm(d_model)
        
        # Transformer encoder
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward, dropout, activation),
            num_layers)
        
        # Dropout after positional encoding
        self.dropout_after_pos_enc = nn.Dropout(dropout)
        
        # Dropout before output layer
        self.dropout1 = nn.Dropout(dropout)
        
        # Normalization before output
        if norm == 'BatchNorm':
            self.norm_before_output = nn.BatchNorm1d(d_model * self.max_len, momentum=0.9)
        elif norm == 'LayerNorm':
            self.norm_before_output = nn.LayerNorm(d_model * self.max_len)
        else:
            raise ValueError("Unsupported norm type. Choose 'BatchNorm' or 'LayerNorm'.")

        # Output layer
        self.output_layer = nn.Linear(d_model * self.max_len, num_classes)
    
    def token_embedding(self, X):
        # Padding and patching logic
        pad_start = self.padding // 2
        pad_end = self.padding - pad_start
        X = torch.nn.functional.pad(X, (0, 0, pad_start, pad_end))
        X = X.unfold(1, self.patch_size, self.patch_size)
        X = X.flatten(-2)
        return X

    def forward(self, X):

        X = self.fist_batchnorm(X)

        X = X[..., None]  # Add channel dimension
        X = self.token_embedding(X)
        
        # Prepare input for transformer
        inp = X.permute(1, 0, 2)  # [seq_len, batch_size, features]
        inp = self.project_inp(inp) * math.sqrt(self.d_model)
        inp = self.norm_after_project(inp)
        inp = self.pos_enc(inp)  # Add positional encoding
        inp = self.dropout_after_pos_enc(inp)
        
        # Transformer processing
        output = self.transformer_encoder(inp)
        output = output.permute(1, 0, 2)
        output = self.dropout1(output)
        
        # Prepare output
        output = output.reshape(output.shape[0], -1)
        output = self.norm_before_output(output)
        output = self.output_layer(output)
        return output