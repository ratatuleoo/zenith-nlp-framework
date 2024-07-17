# src/my_nlp_framework/models/transformer_model.py
import torch
import torch.nn as nn
from my_nlp_framework.core.attention import ScaledDotProductAttention
from my_nlp_framework.core.feed_forward import FeedForwardNetwork

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, ff_hidden_dim, dropout):
        super(TransformerBlock, self).__init__()
        self.attention = ScaledDotProductAttention(embed_size)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = FeedForwardNetwork(embed_size, ff_hidden_dim, embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention, _ = self.attention(query, key, value, mask)
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out

class Transformer(nn.Module):
    def __init__(self, embed_size, num_layers, heads, vocab_size, ff_hidden_dim, dropout, max_length):
        super(Transformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)
        self.layers = nn.ModuleList(
            [TransformerBlock(embed_size, heads, ff_hidden_dim, dropout) for _ in range(num_layers)]
        )
        self.fc_out = nn.Linear(embed_size, vocab_size)

    def forward(self, x, mask):
        seq_length = x.shape[1]
        positions = torch.arange(0, seq_length).expand(x.shape[0], seq_length).to(x.device)
        out = self.dropout(self.embedding(x) + self.position_embedding(positions))

        for layer in self.layers:
            out = layer(out, out, out, mask)

        return self.fc_out(out)
