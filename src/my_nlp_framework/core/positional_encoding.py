import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, embed_size, dropout=0.1, max_length=5000):
        """
        Initializes the PositionalEncoding module.

        Args:
            embed_size (int): The size of the embedding dimension.
            dropout (float): The dropout rate.
            max_length (int): The maximum length of the input sequences.
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create a positional encoding matrix
        pe = torch.zeros(max_length, embed_size)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_size, 2).float() * (-math.log(10000.0) / embed_size))
        
        # Apply sine to even indices in pe
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cosine to odd indices in pe
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        # register_buffer makes it part of the model's state_dict, but not a parameter to be trained
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Adds positional encoding to the input tensor.

        Args:
            x (torch.Tensor): The input tensor of shape (seq_len, batch_size, embed_size).

        Returns:
            torch.Tensor: The output tensor with positional encoding added.
        """
        # x is expected to be of shape (seq_len, batch_size, embed_size)
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)