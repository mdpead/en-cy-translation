from torch import nn
import torch
import math


class Linear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        x = torch.matmul(x, self.weight.t()) + self.bias
        return x


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.linear1 = Linear(d_model, d_ff)
        self.linear2 = Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True, unbiased=False)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_h = d_model // num_heads
        self.W_q = Linear(d_model, d_model)
        self.W_k = Linear(d_model, d_model)
        self.W_v = Linear(d_model, d_model)
        self.W_o = Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, attn_mask):
        # Create Q,K,V tensors
        Q = self.W_q(q)
        K = self.W_k(k)
        V = self.W_v(v)

        # Split into multi heads (batch_no, num_heads, seq_no, d_h)
        Q_mh = Q.reshape(Q.shape[0], Q.shape[1], self.num_heads, self.d_h).permute(0, 2, 1, 3)
        K_mh = K.reshape(K.shape[0], K.shape[1], self.num_heads, self.d_h).permute(0, 2, 1, 3)
        V_mh = V.reshape(V.shape[0], V.shape[1], self.num_heads, self.d_h).permute(0, 2, 1, 3)

        # Calculate attention matrices
        attn_raw = torch.matmul(Q_mh, K_mh.transpose(-1, -2)) / math.sqrt(self.d_h)
        attn_mask_mh = attn_mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        fill_value = torch.finfo(attn_raw.dtype).min  # works for fp16/bf16/fp32
        attn_masked = attn_raw.masked_fill(~attn_mask_mh, fill_value)
        attn = torch.softmax(attn_masked, -1)
        attn = self.dropout(attn)

        # Apply to value vectors and recombine
        A_mh = torch.matmul(attn, V_mh)
        A = A_mh.permute(0, 2, 1, 3).reshape(A_mh.shape[0], A_mh.shape[2], -1)

        # Project out
        O = self.W_o(A)

        return O


class Embedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.E = nn.Parameter(torch.randn(vocab_size, d_model) * 0.02)
        self.d_model = d_model

    def forward(self, x):
        x = self.E[x] * math.sqrt(self.d_model)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_length):
        super().__init__()
        self.register_buffer("PE", self.create_pe(d_model, max_length))

    def create_pe(self, d_model, max_length):
        PE = torch.zeros((max_length, d_model))
        positions = torch.arange(0, max_length)
        dims = torch.arange(0, d_model // 2)

        evens = torch.sin(positions.unsqueeze(-1) / (10000 ** (2 * dims / d_model)).unsqueeze(0))
        odds = torch.cos(positions.unsqueeze(-1) / (10000 ** (2 * dims / d_model)).unsqueeze(0))

        PE[:, 0::2] = evens
        PE[:, 1::2] = odds

        return PE

    def forward(self, x):
        x = x + self.PE[: x.shape[1], :].unsqueeze(0)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feedforward = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.layer_norms = nn.ModuleList([LayerNorm(d_model) for _ in range(0, 2)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, attn_mask):
        x = src
        attn_out = self.attention(x, x, x, attn_mask)
        x = self.layer_norms[0](x + self.dropout(attn_out))
        ff_out = self.feedforward(x)
        x = self.layer_norms[1](x + self.dropout(ff_out))
        return x


class Encoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_enc_layers, dropout):
        super().__init__()
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_enc_layers)]
        )

    def make_attn_mask(self, q_padding_mask, k_padding_mask):
        padding_mask = q_padding_mask.unsqueeze(-1) & k_padding_mask.unsqueeze(-2)
        return padding_mask

    def forward(self, src, src_padding_mask):
        attn_mask = self.make_attn_mask(src_padding_mask, src_padding_mask)
        for layer in self.layers:
            src = layer(src, attn_mask)
        return src


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feedforward = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.layer_norms = nn.ModuleList([LayerNorm(d_model) for _ in range(0, 3)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, tgt, self_attn_mask, cross_attn_mask):
        x = tgt
        self_attn_out = self.self_attention(tgt, tgt, tgt, self_attn_mask)
        x = self.layer_norms[0](x + self.dropout(self_attn_out))
        cross_attn_out = self.cross_attention(x, src, src, cross_attn_mask)
        x = self.layer_norms[1](x + self.dropout(cross_attn_out))
        ff_out = self.feedforward(x)
        x = self.layer_norms[2](x + self.dropout(ff_out))
        return x


class Decoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_dec_layers, dropout):
        super().__init__()
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_dec_layers)]
        )

    def make_self_attn_mask(self, q_padding_mask, k_padding_mask):
        padding_mask = q_padding_mask.unsqueeze(-1) & k_padding_mask.unsqueeze(-2)
        causal_mask = torch.tril(
            torch.ones(
                q_padding_mask.shape[1], k_padding_mask.shape[1], device=q_padding_mask.device
            ).bool()
        ).unsqueeze(0)
        self_attn_mask = causal_mask & padding_mask
        return self_attn_mask

    def make_cross_attn_mask(self, q_padding_mask, k_padding_mask):
        cross_attn_mask = q_padding_mask.unsqueeze(-1) & k_padding_mask.unsqueeze(-2)
        return cross_attn_mask

    def forward(self, src, tgt, src_padding_mask, tgt_padding_mask):
        # Make attention masks
        self_attn_mask = self.make_self_attn_mask(tgt_padding_mask, tgt_padding_mask)
        cross_attn_mask = self.make_cross_attn_mask(tgt_padding_mask, src_padding_mask)

        for layer in self.layers:
            tgt = layer(src, tgt, self_attn_mask, cross_attn_mask)
        return tgt


class Output(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.linear = Linear(d_model, vocab_size)

    def forward(self, tgt_dec):
        x = self.linear(tgt_dec)
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        d_model,
        num_heads,
        d_ff,
        num_enc_layers,
        num_dec_layers,
        vocab_size,
        max_length,
        dropout,
    ):
        super().__init__()
        self.src_embedding = Embedding(vocab_size, d_model)
        self.tgt_embedding = Embedding(vocab_size, d_model)
        self.positional_encoder = PositionalEncoding(d_model, max_length)
        self.encoder = Encoder(d_model, num_heads, d_ff, num_enc_layers, dropout)
        self.decoder = Decoder(d_model, num_heads, d_ff, num_dec_layers, dropout)
        self.output = Output(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def encode(self, src, src_padding_mask):
        src_enc = self.encoder(src, src_padding_mask)
        return src_enc

    def decode(self, src_enc, tgt, src_padding_mask, tgt_padding_mask):
        tgt_dec = self.decoder(src_enc, tgt, src_padding_mask, tgt_padding_mask)
        return tgt_dec

    def forward(self, src, tgt, src_padding_mask, tgt_padding_mask):
        src_emb = self.dropout(self.positional_encoder((self.src_embedding(src))))
        src_enc = self.encode(src_emb, src_padding_mask)
        tgt_emb = self.dropout(self.positional_encoder(self.tgt_embedding(tgt)))
        tgt_dec = self.decode(src_enc, tgt_emb, src_padding_mask, tgt_padding_mask)
        out = self.output(tgt_dec)
        return out


def init_weights(m):
    if isinstance(m, (nn.Linear, Linear)):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.Embedding, Embedding)):
        nn.init.normal_(m.E, mean=0.0, std=0.02)


def build_transformer(model_config):
    transformer = Transformer(
        model_config["d_model"],
        model_config["num_heads"],
        model_config["d_ff"],
        model_config["num_enc_layers"],
        model_config["num_dec_layers"],
        model_config["vocab_size"],
        model_config["max_length"],
        model_config["dropout"],
    )
    transformer = transformer.apply(init_weights)
    return transformer
