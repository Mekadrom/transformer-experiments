from modules.multicast_attn import MultiCastAttention
from utils import *

import matplotlib.pyplot as plt
import os
import seaborn as sns
import torch

def gen_figure_for_page(layer_num, encoder_attention_weights, decoder_self_attn_weights, decoder_cross_attn_weights, attendee_tokens, attending_tokens):
    fig, axes = plt.subplots(nrows=max(len(encoder_attention_weights), len(weights)), ncols=3, figsize=(10, 20))

    def heatmap(data, attendee_tokens, attending_tokens, ax):
        return sns.heatmap(data, square=True, annot=True, annot_kws={"fontsize":6}, fmt=".4f", xticklabels=attendee_tokens, yticklabels=attending_tokens, ax=ax)

    for i, weights in enumerate(encoder_attention_weights):
        s = heatmap(weights, attendee_tokens, attending_tokens, axes[i, 0])
        s.set(xlabel="Input Sequence", ylabel="Output Sequence")

    axes[0, 0].set_title(f"Encoder Layer {layer_num} Self Attention")

    for i, weights in enumerate(decoder_self_attn_weights):
        s = heatmap(weights, attendee_tokens, attending_tokens, axes[i, 1])
        s.set(xlabel="Input Sequence", ylabel="Output Sequence")

    axes[0, 1].set_title(f"Decoder Layer {layer_num} Self Attention")

    for i, weights in enumerate(decoder_cross_attn_weights):
        s = heatmap(weights, attendee_tokens, attending_tokens, axes[i, 2])
        s.set(xlabel="Input Sequence", ylabel="Output Sequence")

    axes[0, 2].set_title(f"Decoder Layer {layer_num} Cross Attention")

    return fig, axes

def visualize_attention_weights(args, model, src_bpe_model, tgt_bpe_model, src, tgt):
    input_sequence = torch.LongTensor(src_bpe_model.encode(src, eos=False)).unsqueeze(0) # (1, input_sequence_length)
    input_tokens = [src_bpe_model.decode([id.item()])[0] for id in input_sequence.squeeze(0)]
    input_sequence_length = torch.LongTensor([input_sequence.size(1)]).unsqueeze(0) # (1)
    target_sequence = torch.LongTensor(tgt_bpe_model.encode(tgt, eos=True)).unsqueeze(0) # (1, target_sequence_length)
    target_tokens = [tgt_bpe_model.decode([id.item()])[0] for id in target_sequence.squeeze(0)]
    target_sequence_length = torch.LongTensor([target_sequence.size(1)]).unsqueeze(0) # (1)

    input_sequence = model.encoder.perform_embedding_transformation(input_sequence) # (N, pad_length, d_model)
    input_sequence = model.encoder.apply_positional_embedding(input_sequence) # (N, pad_length, d_model)
    # input_sequence = model.apply_dropout(input_sequence) # (N, pad_length, d_model) # don't apply dropout for visualization

    encoder_layer_weights = []
    decoder_layer_self_attn_weights = []
    decoder_layer_cross_attn_weights = []

    for e, encoder_layer in enumerate(model.encoder.encoder_layers):
        input_sequence, attention_weights = encoder_layer[0](query_sequences=input_sequence, key_sequences=input_sequence, value_sequences=input_sequence, key_value_sequence_lengths=input_sequence_length)

        attention_weights = attention_weights.cpu().detach()
        attention_weights = attention_weights.contiguous().view(1, args.n_heads, attention_weights.size(1), attention_weights.size(2))

        encoder_attn_weights = []

        # shape of attention_weights will be (1, n_heads, input_sequence_length, input_sequence_length) for self attention (like in encoder layers and beginning of each decoder layer)
        for i in range(attention_weights.size(1)):
            encoder_attn_weights.append(attention_weights[:, i, :, :].squeeze(0).cpu().detach().numpy())
        input_sequence = encoder_layer[1](sequences=input_sequence) # (N, pad_length, d_model)

        encoder_layer_weights.append(encoder_attn_weights)

    input_sequence = model.encoder.layer_norm(input_sequence)

    target_sequence = model.decoder.apply_embedding_transformation(target_sequence) # (N, pad_length, d_model)
    target_sequence = model.decoder.apply_positional_embedding(target_sequence) # (N, pad_length, d_model)
    # target_sequence = model.apply_dropout(target_sequence) # (N, pad_length, d_model) # don't apply dropout for visualization

    for d, decoder_layer in enumerate(model.decoder.decoder_layers):
        target_sequence, attention_weights = decoder_layer[0](query_sequences=target_sequence, key_sequences=target_sequence, value_sequences=target_sequence, key_value_sequence_lengths=target_sequence_length) # (N, pad_length, d_model)
        
        attention_weights = attention_weights.cpu().detach()
        attention_weights = attention_weights.contiguous().view(1, args.n_heads, attention_weights.size(1), attention_weights.size(2))

        decoder_self_attn_weights = []

        # shape of attention_weights will be (1, n_heads, target_sequence_length, target_sequence_length) for self attention (like in encoder layers and beginning of each decoder layer)
        for i in range(attention_weights.size(1)):
            decoder_self_attn_weights.append(attention_weights[:, i, :, :].squeeze(0).cpu().detach().numpy())
        target_sequence, attention_weights = decoder_layer[1](query_sequences=target_sequence, key_sequences=input_sequence, value_sequences=input_sequence, key_value_sequence_lengths=input_sequence_length) # (N, pad_length, d_model)

        attention_weights = attention_weights.cpu().detach()
        attention_weights = attention_weights.contiguous().view(1, args.n_heads, attention_weights.size(1), attention_weights.size(2))

        decoder_cross_attn_weights = []

        # shape of attention_weights will be (1, n_heads, target_sequence_length, input_sequence_length) for encoder-decoder attention
        for i in range(attention_weights.size(1)):
            decoder_cross_attn_weights.append(attention_weights[:, i, :, :].squeeze(0).cpu().detach().numpy())
        target_sequence = decoder_layer[2](sequences=target_sequence) # (N, pad_length, d_model)

        decoder_layer_self_attn_weights.append(decoder_layer_self_attn_weights)
        decoder_layer_cross_attn_weights.append(decoder_layer_cross_attn_weights)

    figures = []

    for i in range(len(encoder_layer_weights)):
        fig, axes = gen_figure_for_page(i, encoder_layer_weights[i], decoder_layer_self_attn_weights[i], decoder_layer_cross_attn_weights[i], input_tokens, target_tokens)
        figures.append(fig)

    return figures

if __name__ == "__main__":
    args, unk = get_args()

    args.device = 'cpu'

    run_dir = os.path.join('runs', args.run_name)

    if not os.path.exists(run_dir):
        os.makedirs(run_dir)

    bpe_run_dir = os.path.join('runs', args.tokenizer_run_name)

    src_bpe_model, tgt_bpe_model = load_tokenizers(bpe_run_dir)

    state_dict = torch.load(os.path.join(run_dir, 'averaged_transformer_checkpoint.pth.tar'), map_location=torch.device('cpu'))

    model = state_dict['model']

    
    if 'positional_encoding' in state_dict:
        positional_encoding = state_dict['positional_encoding']
    else:
        positional_encoding = get_positional_encoding(args)

    model.eval()

    model.args = args
    model.positional_encoding = positional_encoding
    model.encoder.args = args
    model.encoder.positional_encoding = positional_encoding
    model.decoder.args = args
    model.decoder.positional_encoding = positional_encoding

    print(model)

    for encoder_layer in model.encoder.encoder_layers:
        encoder_layer[0].args = args
        encoder_layer[0].positional_encoding = positional_encoding
        if type(encoder_layer[0]) == MultiCastAttention:
            for self_attn_layer in encoder_layer[0].layers:
                self_attn_layer.args = args
                self_attn_layer.positional_encoding = positional_encoding
        encoder_layer[1].args = args

    for decoder_layer in model.decoder.decoder_layers:
        decoder_layer[0].in_decoder = True
        decoder_layer[0].args = args
        decoder_layer[0].positional_encoding = positional_encoding
        if type(decoder_layer[0]) == MultiCastAttention:
            for self_attn_layer in decoder_layer[0].layers:
                self_attn_layer.in_decoder = True
                self_attn_layer.args = args
                self_attn_layer.positional_encoding = positional_encoding
        decoder_layer[1].in_decoder = True
        decoder_layer[1].args = args
        decoder_layer[1].positional_encoding = positional_encoding
        if type(decoder_layer[1]) == MultiCastAttention:
            for cross_attn_layer in decoder_layer[1].layers:
                cross_attn_layer.in_decoder = True
                cross_attn_layer.args = args
                cross_attn_layer.positional_encoding = positional_encoding
        decoder_layer[2].args = args

    while True:
        src = input("Enter source sentence: ")
        tgt = input("Enter target sentence: ")

        figures = visualize_attention_weights(args, model, src_bpe_model, tgt_bpe_model, src, tgt)

        def navigate_figures(figures, start_index=0):
            current_index = start_index

            def show_figure(index):
                plt.close(figures[current_index])
                figures[index].show()

            def on_next(event):
                nonlocal current_index
                if current_index + 1 < len(figures):
                    current_index += 1
                    show_figure(current_index)

            def on_prev(event):
                nonlocal current_index
                if current_index > 0:
                    current_index -= 1
                    show_figure(current_index)

            def on_close(event):
                plt.close(figures[current_index])

            # Create navigation buttons
            next_button = plt.axes([0.85, 0.05, 0.1, 0.075])
            prev_button = plt.axes([0.7, 0.05, 0.1, 0.075])
            close_button = plt.axes([0.55, 0.05, 0.1, 0.075])
            bnext = plt.Button(next_button, 'Next')
            bnext.on_clicked(on_next)
            bprev = plt.Button(prev_button, 'Previous')
            bprev.on_clicked(on_prev)
            bclose = plt.Button(close_button, 'Close')
            bclose.on_clicked(on_close)

            # Show the first figure
            figures[start_index].show()

        # Call the function with the figures list
        navigate_figures(figures)