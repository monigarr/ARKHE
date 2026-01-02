"""
Module: collatz_transformer
Package: math_research.ml.models

Description:
    Transformer model for predicting Collatz sequence long steps.
    Based on research from "Transformers know more than they can tell:
    Learning the Collatz sequence" paper.

Author: MoniGarr
Author Email: monigarr@MoniGarr.com
Author Website: MoniGarr.com

Author Research Interests:
    - AI/ML Research and Development
    - Extended Reality (XR) Applications
    - 3D Graphics and Visualization
    - Robotics and Autonomous Systems
    - Computer Vision
    - Navigation Systems
    - Natural Language Processing (NLP)
    - Low Resource Languages (spoken in English communities)

Usage:
    >>> import torch
    >>> from math_research.ml.models import CollatzTransformer
    >>> 
    >>> model = CollatzTransformer(
    ...     vocab_size=24,
    ...     d_model=512,
    ...     nhead=8,
    ...     num_layers=6
    ... )
    >>> 
    >>> src = torch.randint(0, 24, (32, 20))  # batch_size=32, seq_len=20
    >>> output = model(src)

Dependencies:
    - torch>=2.0.0
    - math_research.ml.models.base_model

Version: 0.1.0
Last Modified: 2025-01-09
License: MIT

Notes:
    This model is designed to predict the "long Collatz step" - mapping
    an odd integer to its long Collatz successor. The model uses encoder-decoder
    architecture with positional encoding.
"""

import torch
import torch.nn as nn
import math
from typing import Optional
from math_research.ml.models.base_model import BaseSequenceModel


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer models.
    
    Adds positional information to input embeddings using sinusoidal encoding.
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        """
        Initialize positional encoding.
        
        Args:
            d_model: Model dimension
            max_len: Maximum sequence length
            dropout: Dropout rate
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            
        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class CollatzTransformer(BaseSequenceModel):
    """
    Transformer model for Collatz sequence long step prediction.
    
    Uses encoder-decoder architecture to predict the long Collatz successor
    of an input odd integer represented as a sequence of digits in a given base.
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_seq_length: int = 128,
        activation: str = "relu",
    ):
        """
        Initialize Collatz transformer model.
        
        Args:
            vocab_size: Vocabulary size (equal to base for digit encoding)
            d_model: Model dimension
            nhead: Number of attention heads
            num_layers: Number of transformer encoder layers
            dim_feedforward: Feedforward network dimension
            dropout: Dropout rate
            max_seq_length: Maximum sequence length
            activation: Activation function ('relu' or 'gelu')
        """
        super().__init__(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_seq_length=max_seq_length,
        )
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_length, dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )
        
        # Output projection
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        # Initialize parameters
        self._init_parameters()
    
    def _init_parameters(self) -> None:
        """Initialize model parameters."""
        nn.init.normal_(self.embedding.weight, mean=0, std=0.02)
        nn.init.xavier_uniform_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)
    
    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ):
        """
        Forward pass through the model.
        
        Args:
            src: Input sequence [batch_size, seq_len]
            src_mask: Attention mask [seq_len, seq_len]
            src_key_padding_mask: Key padding mask [batch_size, seq_len]
            return_attention: If True, returns dict with 'logits' and 'attentions'.
            
        Returns:
            Output logits [batch_size, seq_len, vocab_size] or dict with 'logits' and 'attentions' if return_attention=True.
        """
        # Embedding
        x = self.embedding(src) * math.sqrt(self.d_model)  # [batch_size, seq_len, d_model]
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoder with optional attention extraction
        if return_attention:
            attentions = []
            for layer in self.transformer_encoder.layers:
                attn_output, attn_weights = layer.self_attn(
                    x, x, x,
                    key_padding_mask=src_key_padding_mask,
                    attn_mask=src_mask,
                    need_weights=True,
                    average_attn_weights=False,
                )
                attentions.append(attn_weights.detach())
                x = x + layer.dropout1(attn_output)
                x = layer.norm1(x)
                x2 = layer.linear2(layer.dropout2(layer.activation(layer.linear1(x))))
                x = x + layer.dropout2(x2)
                x = layer.norm2(x)
        else:
            x = self.transformer_encoder(x, mask=src_mask, src_key_padding_mask=src_key_padding_mask)

        # Output projection
        output = self.output_projection(x)

        if return_attention:
            return {"logits": output, "attentions": attentions}
        return output
    
    def generate(
        self,
        src: torch.Tensor,
        max_length: Optional[int] = None,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Generate output sequence from input.
        
        Args:
            src: Input sequence [batch_size, seq_len]
            max_length: Maximum generation length (default: src length)
            temperature: Sampling temperature
            
        Returns:
            Generated sequence [batch_size, max_length]
        """
        if max_length is None:
            max_length = src.size(1)
        
        self.eval()
        with torch.no_grad():
            output = self.forward(src)
            # Take logits and sample/greedy decode
            probs = torch.softmax(output / temperature, dim=-1)
            generated = torch.argmax(probs, dim=-1)  # Greedy decoding
        
        return generated

