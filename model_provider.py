from megatransformer import megatransformer, rmsnorm, config
from torch import nn

class CausalTransformerModelProvider:
    def provide_transformer(self, args, vocab_size, tie_embeddings) -> megatransformer.MegaTransformer:
        norm = rmsnorm.RMSNorm if args.norm_type == 'rms' else nn.LayerNorm

        attn_config = config.AttentionConfig(
            n_heads=args.n_heads,
            n_gqa_groups=args.n_gqa_groups,
            d_queries=args.d_queries,
            d_values=args.d_values,
            q_bias=args.q_bias,
            k_bias=args.k_bias,
            v_bias=args.v_bias,
            heads_activation_function=args.heads_activation,
            use_infinite_attention=args.use_infinite_attention,
            infinite_attention_n_segments=args.infinite_attention_n_segments,
            infinite_attention_update=args.infinite_attention_update,
            use_grok_scaled_attn=args.use_grok_scaled_attn,
        )

        ffn_config = config.FFNConfig(
            ffn_type=args.fcn_type,
            d_inner=args.d_inner,
            moe_replace=args.moe_replace,
            moe_top_k=args.moe_top_k,
            millions_moe_n_heads=args.millions_moe_n_heads,
            millions_moe_d_keys=args.millions_moe_d_keys,
            millions_moe_input_dropout=args.millions_moe_input_dropout,
            millions_moe_query_dropout=args.millions_moe_query_dropout,
            millions_moe_value_dropout=args.millions_moe_value_dropout,
            activation_function=args.activation_function,
            ffn_bias=args.fcn_bias,
        )

        decoder_config = config.EncoderDecoderConfig(
            device=args.decoder_device,
            vocab_size=vocab_size,
            n_layers=args.n_decoder_layers,
            self_attn_config=attn_config,
            embedding_compression_dim=args.embedding_compression_dim,
            per_lang_embedding_layers=args.per_lang_embedding_layers,
            embedding_activation=args.embedding_activation,
            param_sharing_type=args.decoder_param_sharing_type,
            m_independent_layers=args.m_decoder_independent_layers,
        )

        model_config = config.TransformerConfig(
            decoder_config=decoder_config,
            tokenizer=args.tokenizer_run_name,
            ffn_config=ffn_config,
            maxlen=args.maxlen,
            d_model=args.d_model,
            dropout=args.dropout,
            use_admin=args.use_admin,
            positional_encoding_type=args.positional_encoding_type,
            positional_encoding_dim=args.positional_encoding_dim,
            learnable_positional_encoding=args.learnable_positional_encoding,
            tie_embeddings=tie_embeddings,
            padding_value=args.padding_value,
            norm_eps=args.norm_eps,
            norm=norm,
            init_weights_from=args.init_weights_from,
            init_weights_gain=args.init_weights_gain,
        )

        model = megatransformer.Decoder(model_config)

        model = model.to(args.decoder_device)
        
        return model

class TranslationTransformerModelProvider:
    def provide_transformer(self, args, src_vocab_size, tgt_vocab_size, tie_embeddings) -> megatransformer.MegaTransformer:
        norm = rmsnorm.RMSNorm if args.norm_type == 'rms' else nn.LayerNorm

        attn_config = config.AttentionConfig(
            n_heads=args.n_heads,
            n_gqa_groups=args.n_gqa_groups,
            d_queries=args.d_queries,
            d_values=args.d_values,
            q_bias=args.q_bias,
            k_bias=args.k_bias,
            v_bias=args.v_bias,
            heads_activation_function=args.heads_activation,
            use_infinite_attention=args.use_infinite_attention,
            infinite_attention_n_segments=args.infinite_attention_n_segments,
            infinite_attention_update=args.infinite_attention_update,
            use_grok_scaled_attn=args.use_grok_scaled_attn,
        )

        ffn_config = config.FFNConfig(
            ffn_type=args.fcn_type,
            d_inner=args.d_inner,
            moe_replace=args.moe_replace,
            moe_top_k=args.moe_top_k,
            millions_moe_n_heads=args.millions_moe_n_heads,
            millions_moe_d_keys=args.millions_moe_d_keys,
            millions_moe_input_dropout=args.millions_moe_input_dropout,
            millions_moe_query_dropout=args.millions_moe_query_dropout,
            millions_moe_value_dropout=args.millions_moe_value_dropout,
            activation_function=args.activation_function,
            ffn_bias=args.fcn_bias,
        )

        encoder_config = config.EncoderDecoderConfig(
            device=args.encoder_device,
            vocab_size=src_vocab_size,
            n_layers=args.n_encoder_layers,
            self_attn_config=attn_config,
            embedding_compression_dim=args.embedding_compression_dim,
            per_lang_embedding_layers=args.per_lang_embedding_layers,
            embedding_activation=args.embedding_activation,
            param_sharing_type=args.encoder_param_sharing_type,
            m_independent_layers=args.m_encoder_independent_layers,
        )

        decoder_config = config.EncoderDecoderConfig(
            device=args.decoder_device,
            vocab_size=tgt_vocab_size,
            n_layers=args.n_decoder_layers,
            self_attn_config=attn_config,
            cross_attn_config=attn_config,
            embedding_compression_dim=args.embedding_compression_dim,
            per_lang_embedding_layers=args.per_lang_embedding_layers,
            embedding_activation=args.embedding_activation,
            param_sharing_type=args.decoder_param_sharing_type,
            m_independent_layers=args.m_decoder_independent_layers,
        )

        model_config = config.TransformerConfig(
            encoder_config=encoder_config,
            decoder_config=decoder_config,
            tokenizer=args.tokenizer_run_name,
            ffn_config=ffn_config,
            maxlen=args.maxlen,
            d_model=args.d_model,
            dropout=args.dropout,
            use_admin=args.use_admin,
            positional_encoding_type=args.positional_encoding_type,
            positional_encoding_dim=args.positional_encoding_dim,
            learnable_positional_encoding=args.learnable_positional_encoding,
            tie_embeddings=tie_embeddings,
            padding_value=args.padding_value,
            norm_eps=args.norm_eps,
            norm=norm,
            init_weights_from=args.init_weights_from,
            init_weights_gain=args.init_weights_gain,
        )

        model = megatransformer.MegaTransformer(model_config)

        model.decoder = model.decoder.to(args.decoder_device)
        model.encoder = model.encoder.to(args.encoder_device)
        
        return model
