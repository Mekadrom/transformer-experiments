from megatransformer import megatransformer, rmsnorm, config, transformer_utils
from torch import nn

import os
import yaml

class CausalTransformerModelProvider:
    def provide_transformer(self, args, run_dir, tokenizer, tie_embeddings) -> megatransformer.MegaTransformer:
        norm = rmsnorm.RMSNorm if args.norm_type == 'rms' else nn.LayerNorm

        tokenizer.pad_token = tokenizer.eos_token

        print(f"pad_token_id: {args.ignore_token_id}")

        attn_config = config.AttentionConfig(
            n_heads=args.n_heads,
            n_gqa_groups=args.n_gqa_groups,
            d_queries=args.d_queries,
            d_values=args.d_values,
            q_bias=args.q_bias,
            k_bias=args.k_bias,
            v_bias=args.v_bias,
            heads_activation_function=args.heads_activation,
            attn_impl=args.attn_impl,
            infinite_attention_n_segments=args.infinite_attention_n_segments,
            infinite_attention_update=args.infinite_attention_update,
            use_grok_scaled_attn=args.use_grok_scaled_attn
        )

        ffn_config = config.FFNConfig(
            ffn_type=args.ffn_type,
            d_inner=args.d_inner,
            moe_replace=args.moe_replace,
            moe_top_k=args.moe_top_k,
            millions_moe_n_heads=args.millions_moe_n_heads,
            millions_moe_d_keys=args.millions_moe_d_keys,
            millions_moe_dropout=args.millions_moe_dropout,
            activation_function=args.activation_function,
            ffn_bias=args.ffn_bias,
        )
        
        decoder_config = config.EncoderDecoderConfig(
            device=args.decoder_device,
            vocab_size=tokenizer.vocab_size,
            n_layers=args.n_decoder_layers,
            self_attn_config=attn_config,
            embedding_compression_dim=args.embedding_compression_dim,
            per_lang_embedding_layers=args.per_lang_embedding_layers,
            embedding_activation=args.embedding_activation,
            param_sharing_type=args.decoder_param_sharing_type,
            m_independent_layers=args.m_decoder_independent_layers,
            embed_scale=args.embedding_scale,
            pre_self_attn_norm=bool(args.pre_self_attn_norm),
            post_self_attn_norm=bool(args.post_self_attn_norm),
            pre_ffn_norm=bool(args.pre_ffn_norm),
            post_ffn_norm=bool(args.post_ffn_norm),
            moe_diversity_loss_coefficient=args.moe_diversity_loss_coefficient,
        )

        model_config = config.TransformerConfig(
            ignore_token_id=args.ignore_token_id,
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
            label_smoothing=args.label_smoothing,
            norm_eps=args.norm_eps,
            norm=norm,
            init_weights_from=args.init_weights_from,
            init_weights_gain=args.init_weights_gain,
        )

        new_config_path = os.path.join(run_dir, 'config.yaml')
        with open(new_config_path, 'w') as f:
            yaml.dump(model_config, f)

        if bool(args.use_huginn):
            decoder_config.n_huginn_prelude_layers = args.n_huginn_prelude_layers,
            decoder_config.n_huginn_thinking_layers = args.n_huginn_thinking_layers,
            decoder_config.n_huginn_coda_layers = args.n_huginn_coda_layers,
            decoder_config.mean_huginn_thinking_steps = args.mean_huginn_thinking_steps,
            decoder_config.huginn_thought_initialization_method = args.huginn_thought_initialization_method,
            decoder_config.huginn_adapter_method = args.huginn_adapter_method,
            decoder_config.huginn_exit_criteria = args.huginn_exit_criteria,
            decoder_config.huginn_exit_criteria_threshold = args.huginn_exit_criteria_threshold,
            model = megatransformer.HuginnDecoder(args.decoder_device, model_config)
        else:
            model = megatransformer.Decoder(args.decoder_device, model_config)

        model = model.to(args.decoder_device)

        transformer_utils.init_weights(
            model,
            d_model=args.d_model,
            d_head=args.d_model // args.n_heads,
            d_ff=args.d_inner,
            n_layers=args.n_decoder_layers,
            init_weights_from=args.init_weights_from,
            init_weights_gain=args.init_weights_gain,
            tie_embeddings=tie_embeddings,
            scale_residual=args.init_weights_scale_residual,
        )

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
            attn_impl=args.attn_impl,
            infinite_attention_n_segments=args.infinite_attention_n_segments,
            infinite_attention_update=args.infinite_attention_update,
            use_grok_scaled_attn=args.use_grok_scaled_attn,
        )

        ffn_config = config.FFNConfig(
            ffn_type=args.ffn_type,
            d_inner=args.d_inner,
            moe_replace=args.moe_replace,
            moe_top_k=args.moe_top_k,
            millions_moe_n_heads=args.millions_moe_n_heads,
            millions_moe_d_keys=args.millions_moe_d_keys,
            millions_moe_dropout=args.millions_moe_dropout,
            activation_function=args.activation_function,
            ffn_bias=args.ffn_bias,
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
            moe_diversity_loss_coefficient=args.moe_diversity_loss_coefficient,
        )

        model_config = config.TransformerConfig(
            ignore_token_id=args.ignore_token_id,
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
            label_smoothing=args.label_smoothing,
            norm_eps=args.norm_eps,
            norm=norm,
            init_weights_from=args.init_weights_from,
            init_weights_gain=args.init_weights_gain,
        )

        model = megatransformer.MegaTransformer(model_config)

        model.decoder = model.decoder.to(args.decoder_device)
        model.encoder = model.encoder.to(args.encoder_device)

        transformer_utils.init_weights(model, args.d_model, args.init_weights_from, args.init_weights_gain, tie_embeddings)
        
        return model
