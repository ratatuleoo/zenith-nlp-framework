import torch
import torch.nn as nn

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k

    def forward(self, query, key, value, mask=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention = torch.nn.functional.softmax(scores, dim=-1)
        output = torch.matmul(attention, value)
        return output, attention
    
    

