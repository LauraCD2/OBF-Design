import torch
import torch.nn as nn
import torch.nn.functional as F

class NHITSBlock(nn.Module):
    def __init__(self, input_dim, seq_len, mlp_units=512, num_layers=3, dropout=0.1):
        super().__init__()
        layers = []
        in_dim = input_dim * seq_len  # Flatten temporal dimension
        
        for _ in range(num_layers-1):
            layers.extend([
                nn.Linear(in_dim, mlp_units),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.LayerNorm(mlp_units)
            ])
            in_dim = mlp_units
            
        layers.append(nn.Linear(mlp_units, 1))
        self.mlp = nn.Sequential(*layers)
        self.pool = nn.AvgPool1d(kernel_size=2, stride=2)  # Multi-scale processing

    def forward(self, x):
        batch_size = x.size(0)
        x_pooled = self.pool(x.permute(0, 2, 1)).permute(0, 2, 1)  # [batch, seq_len//2, features]
        x_flat = x_pooled.flatten(1)  # [batch, (seq_len//2)*features]
        return self.mlp(x_flat)

class NHITS(nn.Module):
    def __init__(self, input_dim, num_classes, feature_dim, num_blocks=3, mlp_units=256):
        seq_len = input_dim
        super().__init__()
        self.blocks = nn.ModuleList([
            NHITSBlock(feature_dim, seq_len//(2**i), mlp_units) 
            for i in range(num_blocks)
        ])
        
        # Final projection to [0,1]
        self.output_scale = nn.Sequential(
            nn.Linear(num_blocks, 32),
            nn.GELU(),
            nn.Linear(32, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x shape: [batch, seq_len, features]'
        x = x[..., None]
        block_outputs = []
        for i, block in enumerate(self.blocks):
            # Create input for this block
            if i > 0:
                x = F.avg_pool1d(x.transpose(1,2), kernel_size=2).transpose(1,2)
            block_out = block(x)
            block_outputs.append(block_out)
        
        # Combine block outputs
        combined = torch.cat(block_outputs, dim=-1)  # [batch, num_blocks]
        return self.output_scale(combined)