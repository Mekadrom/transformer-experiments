from .multiconv_attn import MultiConvAttention
from .multihead_attn import MultiHeadAttention
from .positional_encoding import PositionalEncoding
from .positionwise_fcn import PositionWiseFCNetwork
from .sum import Sum
from .tuple_identity import TupleIdentity

import admin_torch
import math
import torch
import torch.nn as nn

class EncoderLayer(nn.Module):
    def __init__(self, args, positional_encoding):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(args, args.mha_d_output, positional_encoding, True, False)
        if args.mca_d_output > 0:
            self.mca = MultiConvAttention(args, False)
        else:
            self.mca = None

        if args.use_admin:
            self.attn_residual = admin_torch.as_module(args.n_layers)
        else:
            self.attn_residual = Sum()

        self.fcn = PositionWiseFCNetwork(args, in_decoder=False)

    def forward(self, encoder_sequences, encoder_sequence_lengths):
        # Store input for adding later
        input_to_add = encoder_sequences.clone()

        multihead_attn, _ = self.mha(encoder_sequences, encoder_sequences, encoder_sequences, encoder_sequence_lengths)
        if self.mca is not None:
            conv_attn = self.mca(encoder_sequences, encoder_sequences)
            encoder_sequences = torch.cat([multihead_attn, conv_attn], dim=-1) # (N, pad_length, d_model)
        else:
            encoder_sequences = multihead_attn

        encoder_sequences = self.attn_residual(input_to_add, encoder_sequences)

        return self.fcn(sequences=encoder_sequences)

class Encoder(nn.Module):
    def __init__(self, args, vocab_size, positional_encoding):
        super(Encoder, self).__init__()

        self.args = args

        # disable gradients for buffer/sinusoidal positional encoding if gradients are not configured to be enabled
        if type(positional_encoding) == torch.Tensor:
            positional_encoding.requires_grad = args.learnable_positional_encoding
            positional_encoding = positional_encoding.to(args.device)

        self.embedding = nn.Embedding(vocab_size, args.d_model)
        self.apply_dropout = nn.Dropout(args.dropout)
        self.layer_norm = nn.LayerNorm(args.d_model)
        self.encoder_layers = self.make_encoder_layers(positional_encoding, args.n_encoder_layers, args.encoder_param_sharing_type, args.m_encoder_independent_layers)

        if type(positional_encoding) == torch.Tensor and len(positional_encoding.shape) == 3:
            self.positional_encoding_apply = PositionalEncoding(positional_encoding)
        else:
            self.positional_encoding_apply = TupleIdentity()

    def make_encoder_layers(self, positional_encoding, n_layers, param_sharing_type, m_independent_layers):
        def new_encoder_layer():
            return EncoderLayer(self.args, positional_encoding)

        layers = []
        for i in range(n_layers):
            if i == 0:
                layers.append(new_encoder_layer())
            elif param_sharing_type == 'sequence':
                if (i - 1) % math.floor(n_layers / m_independent_layers) == 0:
                    layers.append(new_encoder_layer())
                else:
                    layers.append(layers[i - 1])
            elif param_sharing_type == 'cycle':
                if i <= m_independent_layers:
                    layers.append(new_encoder_layer())
                else:
                    res_idx = ((i - 1) % m_independent_layers) + 1
                    layers.append(layers[res_idx])
            elif param_sharing_type == 'cycle-rev':
                if i <= m_independent_layers:
                    layers.append(new_encoder_layer())
                elif i <= m_independent_layers * (math.ceil(n_layers / m_independent_layers) - 1):
                    res_idx = ((i - 1) % m_independent_layers) + 1
                    layers.append(layers[res_idx])
                else:
                    res_idx = m_independent_layers - ((i - 1) % m_independent_layers)
                    layers.append(layers[res_idx])
            elif param_sharing_type == 'ffn-cycle-rev':
                if i <= m_independent_layers:
                    layers.append(new_encoder_layer())
                elif i <= m_independent_layers * (math.ceil(n_layers / m_independent_layers) - 1):
                    res_idx = ((i - 1) % m_independent_layers) + 1
                    new_layer = new_encoder_layer()
                    new_layer.fcn = layers[res_idx].fcn
                    layers.append(new_layer)
                else:
                    res_idx = m_independent_layers - ((i - 1) % m_independent_layers)
                    new_layer = new_encoder_layer()
                    new_layer.fcn = layers[res_idx].fcn
                    layers.append(new_layer)
            elif param_sharing_type == 'heads-cycle-rev':
                if i <= m_independent_layers:
                    layers.append(new_encoder_layer())
                elif i <= m_independent_layers * (math.ceil(n_layers / m_independent_layers) - 1):
                    res_idx = ((i - 1) % m_independent_layers) + 1
                    new_layer = new_encoder_layer()
                    new_layer.mha = layers[res_idx].mha
                    layers.append(new_layer)
                else:
                    res_idx = m_independent_layers - ((i - 1) % m_independent_layers)
                    new_layer = new_encoder_layer()
                    new_layer.mha = layers[res_idx].mha
                    layers.append(new_layer)
            elif param_sharing_type == 'all':
                layers.append(layers[0])
            else:
                layers.append(new_encoder_layer())
        return nn.ModuleList(layers)

    def perform_embedding_transformation(self, encoder_sequences):
        d_model = self.embedding.weight.size(1)
        return self.embedding(encoder_sequences) * math.sqrt(d_model) # (N, pad_length, d_model)

    def apply_positional_embedding(self, encoder_sequences):
        # 1D buffer/sinusoidal encoding is applied here. 2D buffer/sinusoidal encoding and rotary encoding are applied in the MultiHeadAttention layer(s)
        return self.positional_encoding_apply(encoder_sequences)
    
    def apply_encoder_layer(self, encoder_sequences, encoder_sequence_lengths, encoder_layer):
        return encoder_layer(encoder_sequences, encoder_sequence_lengths)

    def forward(self, encoder_sequences, encoder_sequence_lengths):
        encoder_sequences = self.perform_embedding_transformation(encoder_sequences) # (N, pad_length, d_model)
        encoder_sequences = self.apply_positional_embedding(encoder_sequences) # (N, pad_length, d_model)
        encoder_sequences = self.apply_dropout(encoder_sequences) # (N, pad_length, d_model)

        for encoder_layer in self.encoder_layers:
            encoder_sequences = self.apply_encoder_layer(encoder_sequences, encoder_sequence_lengths, encoder_layer)

        # post-LN
        encoder_sequences = self.layer_norm(encoder_sequences) # (N, pad_length, d_model)

        return encoder_sequences

class DecoderLayer(nn.Module):
    def __init__(self, args, positional_encoding):
        super(DecoderLayer, self).__init__()

        self.args = args

        self.self_mha = MultiHeadAttention(args, args.mha_d_output, positional_encoding, True, True)
        if args.mca_d_output > 0:
            self.self_mca = MultiConvAttention(args, True)
        else:
            self.self_mca = None

        self.cross_mha = MultiHeadAttention(args, args.d_model, positional_encoding, False, True)

        self.attn_residual = Sum()

        self.fcn = PositionWiseFCNetwork(args, in_decoder=True)

    def forward(self, decoder_sequences, decoder_sequence_lengths, encoder_sequences, encoder_sequence_lengths):
        mha_self_attn, _ = self.self_mha(decoder_sequences, decoder_sequences, decoder_sequences, decoder_sequence_lengths) # (N, pad_length, d_model), trash attention_weights
        if self.self_mca is not None:
            conv_self_att = self.cross_mca(decoder_sequences, decoder_sequences) # (N, pad_length, d_model)
        
            decoder_sequences = torch.cat([mha_self_attn, conv_self_att], dim=-1) # (N, pad_length, d_model)
        else:
            decoder_sequences = mha_self_attn

        decoder_sequences, _ = self.cross_mha(decoder_sequences, encoder_sequences, encoder_sequences, encoder_sequence_lengths) # (N, pad_length, d_model)
        decoder_sequences = self.attn_residual(decoder_sequences, decoder_sequences) # (N, pad_length, d_model)

        return self.fcn(sequences=decoder_sequences) # (N, pad_length, d_model)

class Decoder(nn.Module):
    def __init__(self, args, vocab_size, positional_encoding):
        super(Decoder, self).__init__()

        self.args = args

        # disable gradients for buffer/sinusoidal positional encoding if gradients are not configured to be enabled
        if type(positional_encoding) == torch.Tensor:
            positional_encoding.requires_grad = args.learnable_positional_encoding
            positional_encoding = positional_encoding.to(args.device)

        self.embedding = nn.Embedding(vocab_size, args.d_model)
        self.apply_dropout = nn.Dropout(args.dropout)
        self.layer_norm = nn.LayerNorm(args.d_model)
        self.decoder_layers = self.make_decoder_layers(positional_encoding, args.n_decoder_layers, args.decoder_param_sharing_type, args.m_decoder_independent_layers)
        self.classifier = nn.Linear(args.d_model, vocab_size)

        if type(positional_encoding) == torch.Tensor and len(positional_encoding.shape) == 3:
            self.positional_encoding_apply = PositionalEncoding(positional_encoding)
        else:
            self.positional_encoding_apply = TupleIdentity()

    def make_decoder_layers(self, positional_encoding, n_layers, param_sharing_type, m_independent_layers):
        def new_decoder_layer():
            return DecoderLayer(self.args, positional_encoding)
        
        layers = []
        for i in range(n_layers):
            if i == 0:
                layers.append(new_decoder_layer())
            elif param_sharing_type == 'sequence':
                if (i - 1) % math.floor(n_layers / m_independent_layers) == 0:
                    layers.append(new_decoder_layer())
                else:
                    layers.append(layers[i - 1])
            elif param_sharing_type == 'cycle':
                if i <= m_independent_layers:
                    layers.append(new_decoder_layer())
                else:
                    res_idx = ((i - 1) % m_independent_layers) + 1
                    layers.append(layers[res_idx])
            elif param_sharing_type == 'cycle-rev':
                if i <= m_independent_layers:
                    layers.append(new_decoder_layer())
                elif i <= m_independent_layers * (math.ceil(n_layers / m_independent_layers) - 1):
                    res_idx = ((i - 1) % m_independent_layers) + 1
                    layers.append(layers[res_idx])
                else:
                    res_idx = m_independent_layers - ((i - 1) % m_independent_layers)
                    layers.append(layers[res_idx])
            elif param_sharing_type == 'ffn-cycle-rev':
                if i <= m_independent_layers:
                    layers.append(new_decoder_layer())
                elif i <= m_independent_layers * (math.ceil(n_layers / m_independent_layers) - 1):
                    res_idx = ((i - 1) % m_independent_layers) + 1
                    new_layer = new_decoder_layer()
                    new_layer.fcn = layers[res_idx].fcn
                    layers.append(new_layer)
                else:
                    res_idx = m_independent_layers - ((i - 1) % m_independent_layers)
                    new_layer = new_decoder_layer()
                    new_layer.fcn = layers[res_idx].fcn
                    layers.append(new_layer)
            elif param_sharing_type == 'heads-cycle-rev':
                if i <= m_independent_layers:
                    layers.append(new_decoder_layer())
                elif i <= m_independent_layers * (math.ceil(n_layers / m_independent_layers) - 1):
                    res_idx = ((i - 1) % m_independent_layers) + 1
                    new_layer = new_decoder_layer()
                    new_layer.self_mha = layers[res_idx].self_mha
                    new_layer.cross_mha = layers[res_idx].cross_mha
                    new_layer.cross_mca = layers[res_idx].cross_mca
                    layers.append(new_layer)
                else:
                    res_idx = m_independent_layers - ((i - 1) % m_independent_layers)
                    new_layer = new_decoder_layer()
                    new_layer.self_mha = layers[res_idx].self_mha
                    new_layer.cross_mha = layers[res_idx].cross_mha
                    new_layer.cross_mca = layers[res_idx].cross_mca
                    layers.append(new_layer)
            elif param_sharing_type == 'all':
                layers.append(layers[0])
            else:
                layers.append(new_decoder_layer())
        return nn.ModuleList(layers)

    def apply_embedding_transformation(self, decoder_sequences):
        d_model = self.embedding.weight.size(1)
        return self.embedding(decoder_sequences) * math.sqrt(d_model) # (N, pad_length, d_model)
    
    def apply_positional_embedding(self, decoder_sequences):
        # 1D buffer/sinusoidal encoding is applied here. 2D buffer/sinusoidal encoding and rotary encoding are applied in the MultiHeadAttention layer(s)
        return self.positional_encoding_apply(decoder_sequences)
    
    def apply_decoder_layer(self, decoder_sequences, decoder_sequence_lengths, encoder_sequences, encoder_sequence_lengths, decoder_layer):
        return decoder_layer(decoder_sequences, decoder_sequence_lengths, encoder_sequences, encoder_sequence_lengths)

    def forward(self, decoder_sequences, decoder_sequence_lengths, encoder_sequences, encoder_sequence_lengths):
        decoder_sequences = self.apply_embedding_transformation(decoder_sequences) # (N, pad_length, d_model)
        decoder_sequences = self.apply_positional_embedding(decoder_sequences) # (N, pad_length, d_model)
        decoder_sequences = self.apply_dropout(decoder_sequences)

        for decoder_layer in self.decoder_layers:
            decoder_sequences = self.apply_decoder_layer(decoder_sequences, decoder_sequence_lengths, encoder_sequences, encoder_sequence_lengths, decoder_layer)

        decoder_sequences = self.layer_norm(decoder_sequences)  # (N, pad_length, d_model)
        decoder_sequences = self.classifier(decoder_sequences)  # (N, pad_length, vocab_size)

        return decoder_sequences

class Transformer(nn.Module):
    def __init__(self, args, src_vocab_size, tgt_vocab_size, positional_encoding):
        super(Transformer, self).__init__()

        self.args = args

        self.encoder = Encoder(args, src_vocab_size, positional_encoding)
        self.decoder = Decoder(args, tgt_vocab_size, positional_encoding)

    def init_weights(self, use_shared_qkv=False, tie_embeddings=True):
        # Glorot uniform initialization with a gain of self.init_weights_gain
        for p in self.parameters():
            # Glorot initialization needs at least two dimensions on the tensor
            if p.dim() > 1:
                if self.args.init_weights_from in ['glorot_uniform', 'xavier_uniform']:
                    nn.init.xavier_uniform_(p, gain=self.args.init_weights_gain)
                elif self.args.init_weights_from in ['glorot_normal', 'xavier_normal']:
                    nn.init.xavier_normal_(p, gain=self.args.init_weights_gain)
                elif self.args.init_weights_from == 'kaiming_uniform':
                    nn.init.kaiming_uniform_(p)
                elif self.args.init_weights_from == 'kaiming_normal':
                    nn.init.kaiming_normal_(p)
                elif self.args.init_weights_from == 'orthogonal':
                    nn.init.orthogonal_(p)
                else:
                    raise Exception(f"Unknown weight initialization method: {self.args.init_weights_from}")

        # Share weights between the embedding layers and the logit layer
        nn.init.normal_(self.encoder.embedding.weight, mean=0., std=math.pow(self.args.d_model, -0.5))
        self.decoder.embedding.weight = self.encoder.embedding.weight

        if tie_embeddings:
            self.decoder.classifier.weight = self.decoder.embedding.weight

        if use_shared_qkv:
            # makes every transformer layer use the same qkv "database" (generally use this with much larger query and value embedding sizes)
            encoder_query_cast_weights = self.encoder.encoder_layers[0].mha.cast_queries.weight
            encoder_query_cast_biases = self.encoder.encoder_layers[0].mha.cast_queries.bias
            encoder_key_cast_weights = self.encoder.encoder_layers[0].mha.cast_keys.weight
            encoder_key_cast_biases = self.encoder.encoder_layers[0].mha.cast_keys.bias
            encoder_value_cast_weights = self.encoder.encoder_layers[0].mha.cast_values.weight
            encoder_value_cast_biases = self.encoder.encoder_layers[0].mha.cast_values.bias
            encoder_output_cast_weights = self.encoder.encoder_layers[0].mha.cast_output.weight
            encoder_output_cast_biases = self.encoder.encoder_layers[0].mha.cast_output.bias

            for encoder_layer in self.encoder.encoder_layers:
                encoder_layer.mha.cast_queries.weight = encoder_query_cast_weights
                encoder_layer.mha.cast_queries.bias = encoder_query_cast_biases
                encoder_layer.mha.cast_keys.weight = encoder_key_cast_weights
                encoder_layer.mha.cast_keys.bias = encoder_key_cast_biases
                encoder_layer.mha.cast_values.weight = encoder_value_cast_weights
                encoder_layer.mha.cast_values.bias = encoder_value_cast_biases
                encoder_layer.mha.cast_output.weight = encoder_output_cast_weights
                encoder_layer.mha.cast_output.bias = encoder_output_cast_biases

            decoder_query_cast_weights = self.decoder.decoder_layers[0].self_mha.cast_queries.weight
            decoder_query_cast_biases = self.decoder.decoder_layers[0].self_mha.cast_queries.bias
            decoder_key_cast_weights = self.decoder.decoder_layers[0].self_mha.cast_keys.weight
            decoder_key_cast_biases = self.decoder.decoder_layers[0].self_mha.cast_keys.bias
            decoder_value_cast_weights = self.decoder.decoder_layers[0].self_mha.cast_values.weight
            decoder_value_cast_biases = self.decoder.decoder_layers[0].self_mha.cast_values.bias
            decoder_output_cast_weights = self.decoder.decoder_layers[0].self_mha.cast_output.weight
            decoder_output_cast_biases = self.decoder.decoder_layers[0].self_mha.cast_output.bias

            for decoder_layer in self.decoder.decoder_layers:
                decoder_layer.self_mha.cast_queries.weight = decoder_query_cast_weights
                decoder_layer.self_mha.cast_queries.bias = decoder_query_cast_biases
                decoder_layer.self_mha.cast_keys.weight = decoder_key_cast_weights
                decoder_layer.self_mha.cast_keys.bias = decoder_key_cast_biases
                decoder_layer.self_mha.cast_values.weight = decoder_value_cast_weights
                decoder_layer.self_mha.cast_values.bias = decoder_value_cast_biases
                decoder_layer.self_mha.cast_output.weight = decoder_output_cast_weights
                decoder_layer.self_mha.cast_output.bias = decoder_output_cast_biases

            decoder_query_cast_weights = self.decoder.decoder_layers[0].cross_mha.cast_queries.weight
            decoder_query_cast_biases = self.decoder.decoder_layers[0].cross_mha.cast_queries.bias
            decoder_key_cast_weights = self.decoder.decoder_layers[0].cross_mha.cast_keys.weight
            decoder_key_cast_biases = self.decoder.decoder_layers[0].cross_mha.cast_keys.bias
            decoder_value_cast_weights = self.decoder.decoder_layers[0].cross_mha.cast_values.weight
            decoder_value_cast_biases = self.decoder.decoder_layers[0].cross_mha.cast_values.bias
            decoder_output_cast_weights = self.decoder.decoder_layers[0].cross_mha.cast_output.weight
            decoder_output_cast_biases = self.decoder.decoder_layers[0].cross_mha.cast_output.bias

            for decoder_layer in self.decoder.decoder_layers:
                decoder_layer.cross_mha.cast_queries.weight = decoder_query_cast_weights
                decoder_layer.cross_mha.cast_queries.bias = decoder_query_cast_biases
                decoder_layer.cross_mha.cast_keys.weight = decoder_key_cast_weights
                decoder_layer.cross_mha.cast_keys.bias = decoder_key_cast_biases
                decoder_layer.cross_mha.cast_values.weight = decoder_value_cast_weights
                decoder_layer.cross_mha.cast_values.bias = decoder_value_cast_biases
                decoder_layer.cross_mha.cast_output.weight = decoder_output_cast_weights
                decoder_layer.cross_mha.cast_output.bias = decoder_output_cast_biases

        print("Model initialized.")

    def forward(self, encoder_sequences, decoder_sequences, encoder_sequence_lengths, decoder_sequence_lengths):
        encoder_sequences = self.encoder(encoder_sequences, encoder_sequence_lengths) # (N, encoder_sequence_pad_length, d_model)
        decoder_sequences = self.decoder(decoder_sequences, decoder_sequence_lengths, encoder_sequences, encoder_sequence_lengths) # (N, decoder_sequence_pad_length, vocab_size)
        return decoder_sequences
