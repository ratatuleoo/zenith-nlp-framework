import torch
import pytest
from my_nlp_framework.models.transformer_model import MultiHeadAttention

def test_multi_head_attention_forward_pass():
    """
    Tests that the MultiHeadAttention layer can perform a forward pass without errors.
    """
    embed_size = 256
    heads = 8
    batch_size = 4
    seq_length = 10

    # Create an instance of the attention layer
    attention_layer = MultiHeadAttention(embed_size, heads)

    # Create dummy input tensors
    query = torch.rand(batch_size, seq_length, embed_size)
    key = torch.rand(batch_size, seq_length, embed_size)
    value = torch.rand(batch_size, seq_length, embed_size)
    mask = None  # No mask for this simple test

    # Perform a forward pass
    try:
        output = attention_layer(value, key, query, mask)
        
        # Check that the output shape is correct
        assert output.shape == (batch_size, seq_length, embed_size)

    except Exception as e:
        pytest.fail(f"MultiHeadAttention forward pass failed with an exception: {e}")
