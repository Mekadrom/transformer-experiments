use anyhow::Result;
use tch::{nn, nn::ModuleT, nn::OptimizerConfig, Device, Tensor};

mod rotary_embedding_torch;

#[derive(Debug)]
struct EncoderLayer {
    self_attn: nn::MultiheadAttention,
    expand: nn::Linear,
    compress: nn::Linear,
    attn_norm: nn::LayerNorm,
    fcn_norm: nn::LayerNorm,
}

#[derive(Debug)]
struct DecoderLayer {
    self_attn: nn::MultiheadAttention,
    cross_attn: nn::MultiheadAttention,
    expand: nn::Linear,
    compress: nn::Linear,
    self_attn_norm: nn::LayerNorm,
    cross_attn_norm: nn::LayerNorm,
    fcn_norm: nn::LayerNorm,
}

#[derive(Debug)]
struct Encoder {
    embedding: nn::Embedding,
    layers: Vec<EncoderLayer>,
    layer_norm: nn::LayerNorm,
}

#[derive(Debug)]
struct Decoder {
    embedding: nn::Embedding,
    layers: Vec<DecoderLayer>,
    layer_norm: nn::LayerNorm,
    generator: nn::Linear,
}

#[derive(Debug)]
struct Transformer {
    conv1: nn::Conv2D,
    conv2: nn::Conv2D,
    fc1: nn::Linear,
    fc2: nn::Linear,
}

struct MultiHeadAttention {
    self_attn: bool,
    in_decoder: bool,
    cast_queries: nn::Linear,
    cast_keys: nn::Linear,
    cast_values: nn::Linear,
    positional_encoding: RotaryEmbedding,
}