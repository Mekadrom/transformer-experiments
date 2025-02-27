from megatransformer import config

import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, config: config.TransformerConfig):
        super(MLP, self).__init__()
        assert config.d_model % config.decoder_config.self_attn_config.n_heads == 0

        self.c_fc = nn.Linear(config.d_model, config.ffn_config.d_inner)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(config.ffn_config.d_inner, config.d_model)

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))

class CausalSelfAttention(nn.Module):
    def __init__(self, config: config.TransformerConfig):
        super(CausalSelfAttention, self).__init__()

        self.c_attn = nn.Linear(config.d_model, 3 * config.d_model)
        self.c_proj = nn.Linear(config.d_model, config.d_model)

        self.n_heads = config.decoder_config.self_attn_config.n_heads
        self.d_model = config.d_model

        self.register_buffer("bias", torch.tril(torch.ones(config.maxlen, config.maxlen)).view(1, 1, config.maxlen, config.maxlen))

    def forward(self, x):
        B, T, C = x.size()

        qkv = self.c_attn(x)
        q, k, v = qkv.split(C, dim=2)
        q = q.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        k = k.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        v = v.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) / (k.size(-1) ** 0.5)
        attn = attn.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        y = attn @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        return self.c_proj(y)

class Block(nn.Module):
    def __init__(self, config: config.TransformerConfig):
        super(Block, self).__init__()

        self.ln_1 = config.norm(config.d_model, config.norm_eps)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = config.norm(config.d_model, config.norm_eps)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    def __init__(self, config: config.TransformerConfig):
        super(GPT, self).__init__()

        self.maxlen = config.maxlen

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.decoder_config.vocab_size, config.d_model),
            wpe = nn.Embedding(config.maxlen, config.d_model),
            h = nn.ModuleList([Block(config) for _ in range(config.decoder_config.n_layers)]),
            ln_f = config.norm(config.d_model, config.norm_eps)
        ))
        self.lm_head = nn.Linear(config.d_model, config.decoder_config.vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, idx):
        B, T = idx.size()
        assert T <= self.maxlen, f"Input length {T} exceeds maximum length {self.maxlen}"

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(idx)
        x = tok_emb + pos_emb

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)
        return self.lm_head(x)

    @classmethod
    def from_pretrained(cls, model_type):
        assert model_type in {'gpt2'}
        from transformers import GPT2LMHeadModel
        print(f"Loading {model_type} from HuggingFace Transformers")

        config_args = config.TransformerConfig(
            decoder_config=config.EncoderDecoderConfig(
                device='cuda',
                n_layers=12,
                vocab_size=50257,
                self_attn_config=config.AttentionConfig(
                    n_heads=12,
                )
            ),
            ffn_config=config.FFNConfig(
                d_inner=3072,
            ),
            maxlen=1024,
            d_model=768,
        )

        model = GPT(config_args)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]

        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape, f"Shape mismatch: {sd_hf[k].shape} != {sd[k].shape} for {k}"
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        model.load_state_dict(sd)
        return model

def evaluate(tokenizer, model, n_sequences=5, max_length=50):
    start = "Hello, I'm a language model,"
    tokens = tokenizer.encode(start, return_tensors='pt').to('cuda')
    tokens = tokens.repeat(n_sequences, 1)

    while tokens.size(1) < max_length:
        with torch.no_grad():
            logits = model(tokens)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)

            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)

            next_tokens = torch.multinomial(topk_probs, 1)

            gathered = torch.gather(topk_indices, 1, next_tokens)

            tokens = torch.cat((tokens, gathered), dim=1)

    for i in range(n_sequences):
        print(tokenizer.decode(tokens[i].tolist(), skip_special_tokens=True))

if __name__ == '__main__':
    from datasets import load_dataset
    from transformers import AutoTokenizer, pipeline
    from torch.utils.data import DataLoader
    from torch.utils.tensorboard import SummaryWriter

    import os

    class CustomDataLoader(DataLoader):
        def __init__(self, tokenizer, dataset_name, dataset_config_name, split, batch_size, sequence_length):
            self.tokenizer = tokenizer
            self.dataset = load_dataset(dataset_name, dataset_config_name, split=split)
            self.batch_size = batch_size
            self.sequence_length = sequence_length

        def __iter__(self):
            for i in range(0, len(self.dataset), self.batch_size):
                batch = self.dataset[i:i+self.batch_size]
                examples = [example for example in batch['text'] if example.strip() != '']
                batch = self.tokenizer(examples, padding='max_length', max_length=self.sequence_length+1, return_tensors='pt', truncation=True)
                batch['labels'] = batch['input_ids'].clone()
                batch['input_ids'] = batch['input_ids'][:, :-1]
                batch['labels'] = batch['labels'][:, 1:]
                yield batch

    tokenizer = AutoTokenizer.from_pretrained('gpt2')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # model = GPT.from_pretrained('gpt2')
    model = GPT(config.TransformerConfig(
        decoder_config=config.EncoderDecoderConfig(
            device=device,
            n_layers=12,
            vocab_size=50257,
            self_attn_config=config.AttentionConfig(
                n_heads=12,
            )
        ),
        ffn_config=config.FFNConfig(
            d_inner=3072,
        ),
        maxlen=1024,
        d_model=768,
    ))

    model.lm_head.weight = model.transformer.wte.weight
    model.train()
    model = model.to(device)

    writer = SummaryWriter(os.path.join('runs', 'gpt2', 'wikitext-103-raw-v1'))

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataloader = CustomDataLoader(tokenizer, 'wikitext', 'wikitext-103-raw-v1', 'train', batch_size=8, sequence_length=1024)
    val_dataloader = CustomDataLoader(tokenizer, 'wikitext', 'wikitext-103-raw-v1', 'validation', batch_size=8, sequence_length=1024)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01, betas=(0.9, 0.95))

    n_steps = 100000

    step = 0
    while step < n_steps:
        for batch in train_dataloader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()

            logits = model(input_ids)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))

            loss.backward()

            loss_item = loss.item()
            ppl = torch.exp(loss).item()

            writer.add_scalar('train/loss', loss_item, step)
            writer.add_scalar('train/ppl', ppl, step)

            optimizer.step()

            step += 1

            if step >= n_steps:
                break

            if step % 10 == 0:
                print(f"Step {step}, Loss: {loss_item}")
                model.eval()
                evaluate(tokenizer, model, n_sequences=5, max_length=50)
                model.train()

    print("Done!")
