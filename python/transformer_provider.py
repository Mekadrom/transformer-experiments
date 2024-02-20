from modules.transformer import Transformer

class TransformerModelProvider:
    def __init__(self):
        super().__init__()

    def provide(self, args, src_vocab_size, tgt_vocab_size, positional_encoding, admin_profiling_batch, tie_embeddings):
        model = Transformer(
            args=args,
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
            d_model=args.d_model,
            n_heads=args.n_heads,
            d_queries=args.d_queries,
            d_values=args.d_values,
            qkv_config=args.qkv_config,
            d_inner=args.d_inner,
            n_encoder_layers=args.n_encoder_layers,
            n_decoder_layers=args.n_decoder_layers,
            dropout=args.dropout,
            encoder_param_sharing_type=args.encoder_param_sharing_type,
            decoder_param_sharing_type=args.decoder_param_sharing_type,
            m_encoder_independent_layers=args.m_encoder_independent_layers,
            m_decoder_independent_layers=args.m_decoder_independent_layers,
            positional_encoding_dim=args.positional_encoding_dim,
            positional_encoding=positional_encoding,
            activation_function=args.activation_function,
            init_weights_from=args.init_weights_from,
            init_weights_gain=args.init_weights_gain,
            use_admin=args.use_admin,
            admin_profiling_batch=admin_profiling_batch,
            device=args.device,
            learnable_positional_encoding=args.learnable_positional_encoding,
            tie_embeddings=tie_embeddings,
        )

        model = model.to(args.device)
        model.init_weights(admin_profiling_batch=admin_profiling_batch, tie_embeddings=tie_embeddings)

        return model