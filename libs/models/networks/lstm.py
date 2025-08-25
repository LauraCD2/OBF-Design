
import torch
import torch.nn as nn

class Lstm(nn.Module):
    def __init__(self, input_dim, num_classes, n_layers, dropout_rate, hidden_size=64):
        super(Lstm, self).__init__()
        
        self.lstm = nn.LSTM(input_size=input_dim,
                            hidden_size=hidden_size,
                            num_layers=n_layers,
                            batch_first=True,
                            dropout=dropout_rate if n_layers > 1 else 0,
                            bidirectional=False)
        self.batchnorm = nn.BatchNorm1d(hidden_size)  
        self.fc = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        
        x = x.unsqueeze(1)  
        x, (hn, cn) = self.lstm(x)
        x = x[:, -1, :]  
        x = self.batchnorm(x) 
        x = self.dropout(x)
        x = self.fc(x)
        return x


        
    
    

