from modules.multicast_attn import MultiCastAttention
from plotly.subplots import make_subplots
from utils import *

import argparse
import numpy as np
import os
import plotly.graph_objects as go
import torch

torch.set_printoptions(threshold=50000, linewidth=80)

def heatmap(data, x_labels=None, y_labels=None):
    trace = go.Heatmap(
        z=data,
        x=[(" " * i) + x_label for i, x_label in enumerate(x_labels)],
        y=[(" " * i) + y_label for i, y_label in enumerate(y_labels)],
        colorscale='Viridis',
        hovertemplate='x: %{x}<br>y: %{y}<br>z: %{z:.64e}<extra></extra>'
    )
    return trace

def create_figure(encoder_layer_weights, decoder_layer_self_attn_weights, decoder_layer_cross_attn_weights, input_sequence, output_sequence):
    num_layers = len(encoder_layer_weights)
    num_heads = max(*[len(encoder_layer_weight) for encoder_layer_weight in encoder_layer_weights] + [len(decoder_layer_self_attn_weight) for decoder_layer_self_attn_weight in decoder_layer_self_attn_weights] + [len(decoder_layer_cross_attn_weight) for decoder_layer_cross_attn_weight in decoder_layer_cross_attn_weights])
    fig = make_subplots(rows=num_heads, cols=3, subplot_titles=("Encoder Self-Attention", "Decoder Self-Attention", "Decoder Cross-Attention"), vertical_spacing=0.002, horizontal_spacing=0.00001)

    for layer_num in range(num_layers):
        encoder_self_attn_weights = encoder_layer_weights[layer_num]
        decoder_self_attn_weights = decoder_layer_self_attn_weights[layer_num]
        decoder_cross_attn_weights = decoder_layer_cross_attn_weights[layer_num]

        assert len(encoder_self_attn_weights) == len(decoder_self_attn_weights) == len(decoder_cross_attn_weights), "Number of heads for each layer must be the same"

        for i in range(len(encoder_self_attn_weights)):
            fig.add_trace(
                heatmap(encoder_self_attn_weights[i], input_sequence, input_sequence),
                row=i + 1, col=1
            )

        for i in range(len(decoder_self_attn_weights)):
            fig.add_trace(
                heatmap(decoder_self_attn_weights[i], output_sequence, output_sequence),
                row=i + 1, col=2
            )

        for i in range(len(decoder_cross_attn_weights)):
            fig.add_trace(
                heatmap(decoder_cross_attn_weights[i], input_sequence, output_sequence),
                row=i + 1, col=3
            )

    fig.update_layout(
        updatemenus=[
            dict(
                type="dropdown",
                buttons=[
                    dict(label=f"Layer {i+1}",
                         method="update",
                         args=[{"visible": [(trace_num // (num_heads * 3) == i) for trace_num in range(num_layers * num_heads * 3)]}])
                    for i in range(num_layers)
                ],
                x=0.5,
                xanchor="left",
                y=1.03,
                yanchor="bottom"
            ),
        ],
        height = 450 * num_heads,
        plot_bgcolor='rgba(0,0,0,0)',
    )

    # make all heatmaps use square aspect ratio
    for i in fig['layout']['annotations']:
        i['font'] = dict(size=20)
        # move titles further up
        i['y'] = 1.1
    for i in range(num_heads * 3):
        if i % (num_heads * 3) in [0, 1, 2]:
            fig['layout'][f"xaxis{i+1}"]['showticklabels'] = True
        else:
            fig['layout'][f"xaxis{i+1}"]['showticklabels'] = False

        fig['layout'][f"yaxis{i+1}"]['scaleanchor'] = f'x{i+1}'
        fig['layout'][f"yaxis{i+1}"]['scaleratio'] = 1
        fig['layout'][f"xaxis{i+1}"]['scaleanchor'] = f'y{i+1}'
        fig['layout'][f"xaxis{i+1}"]['scaleratio'] = 1
        fig['layout'][f"yaxis{i+1}"]['showgrid'] = False
        fig['layout'][f"xaxis{i+1}"]['showgrid'] = False
        fig['layout'][f"yaxis{i+1}"]['showline'] = False
        fig['layout'][f"xaxis{i+1}"]['showline'] = False
        fig['layout'][f"yaxis{i+1}"]['zeroline'] = False
        fig['layout'][f"xaxis{i+1}"]['zeroline'] = False
        # move x axis to top
        fig['layout'][f"xaxis{i+1}"]['side'] = 'top'
        # remove background
        fig['layout'][f"yaxis{i+1}"]['gridcolor'] = 'rgba(0,0,0,0)'
        fig['layout'][f"xaxis{i+1}"]['gridcolor'] = 'rgba(0,0,0,0)'
        # reverse yaxis
        fig['layout'][f"yaxis{i+1}"]['autorange'] = 'reversed'
        # set font size to 6
        fig['layout'][f"yaxis{i+1}"]['tickfont'] = dict(size=8)
        fig['layout'][f"xaxis{i+1}"]['tickfont'] = dict(size=8)

    fig.update_traces(visible=False)

    # Show the first layer's traces by default
    for i in range(num_heads * 3):
        fig.data[i].visible = True

    return fig

def extract_attention_weights(model, src_bpe_model, tgt_bpe_model, src, tgt):
    src_sequence = torch.LongTensor(src_bpe_model.encode(src, eos=False)).unsqueeze(0) # (1, input_sequence_length)
    src_tokens = [src_bpe_model.decode([id.item()])[0] for id in src_sequence.squeeze(0)]
    src_sequence_length = torch.LongTensor([src_sequence.size(1)]).unsqueeze(0) # (1)
    tgt_sequence = torch.LongTensor(tgt_bpe_model.encode(tgt, eos=True)).unsqueeze(0) # (1, target_sequence_length)
    tgt_tokens = [tgt_bpe_model.decode([id.item()])[0] for id in tgt_sequence.squeeze(0)]
    tgt_sequence_length = torch.LongTensor([tgt_sequence.size(1)]).unsqueeze(0) # (1)

    src_sequence = model.encoder.perform_embedding_transformation(src_sequence) # (N, pad_length, d_model)
    src_sequence = model.encoder.apply_positional_embedding(src_sequence) # (N, pad_length, d_model)

    encoder_layer_weights = []
    decoder_layer_self_attn_weights = []
    decoder_layer_cross_attn_weights = []

    for e, encoder_layer in enumerate(model.encoder.encoder_layers):
        src_sequence, attention_weights = encoder_layer[0](query_sequences=src_sequence, key_sequences=src_sequence, value_sequences=src_sequence, key_value_sequence_lengths=src_sequence_length)
        src_sequence = encoder_layer[1](sequences=src_sequence) # (N, pad_length, d_model)

        encoder_layer_weights.append(attention_weights.cpu().detach().numpy())

    src_sequence = model.encoder.layer_norm(src_sequence)
    tgt_sequence = model.decoder.apply_embedding_transformation(tgt_sequence) # (N, pad_length, d_model)
    tgt_sequence = model.decoder.apply_positional_embedding(tgt_sequence) # (N, pad_length, d_model)

    for d, decoder_layer in enumerate(model.decoder.decoder_layers):
        tgt_sequence, self_attn_weights = decoder_layer[0](query_sequences=tgt_sequence, key_sequences=tgt_sequence, value_sequences=tgt_sequence, key_value_sequence_lengths=tgt_sequence_length) # (N, pad_length, d_model)
        tgt_sequence, cross_attn_weights = decoder_layer[1](query_sequences=tgt_sequence, key_sequences=src_sequence, value_sequences=src_sequence, key_value_sequence_lengths=src_sequence_length) # (N, pad_length, d_model)
        tgt_sequence = decoder_layer[2](sequences=tgt_sequence) # (N, pad_length, d_model)

        decoder_layer_self_attn_weights.append(self_attn_weights.cpu().contiguous().detach().numpy())
        decoder_layer_cross_attn_weights.append(cross_attn_weights.cpu().detach().numpy())

    return src_tokens, tgt_tokens, encoder_layer_weights, decoder_layer_self_attn_weights, decoder_layer_cross_attn_weights

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description='Visualize attention weights')

    argparser.add_argument('--run_name', type=str, required=True)
    argparser.add_argument('--model_name', type=str, default="averaged_transformer_checkpoint.pth.tar")
    argparser.add_argument('--tokenizer_run_name', type=str, required=True)

    args, unk = argparser.parse_known_args()

    run_dir = os.path.join('runs', args.run_name)

    if not os.path.exists(run_dir):
        os.makedirs(run_dir)

    bpe_run_dir = os.path.join('runs', args.tokenizer_run_name)

    src_bpe_model, tgt_bpe_model = load_tokenizers(bpe_run_dir)

    state_dict = torch.load(os.path.join(run_dir, args.model_name), map_location=torch.device('cpu'))

    model = state_dict['model']
    model = model.to('cpu')

    model.eval()

    print(model)

    while True:
        src = input("Enter source sentence: ")
        print(f"predicted: {beam_search_translate(src, model, src_bpe_model, tgt_bpe_model, device='cpu')}")
        tgt = input("Enter target sentence: ")

        src_tokens, tgt_tokens, encoder_layer_weights, decoder_layer_self_attn_weights, decoder_layer_cross_attn_weights = extract_attention_weights(model, src_bpe_model, tgt_bpe_model, src, tgt)

        fig = create_figure(encoder_layer_weights, decoder_layer_self_attn_weights, decoder_layer_cross_attn_weights, src_tokens, tgt_tokens)

        fig.show()
