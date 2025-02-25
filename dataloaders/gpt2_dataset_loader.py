import torch
import numpy as np
from datasets import load_dataset
from torch.utils.data import IterableDataset, DataLoader
from transformers import PreTrainedTokenizer
import os

class GPT2TextDataset(IterableDataset):
    """
    Dataset for GPT2-style training - concatenates documents with EOS tokens
    and creates chunks of the specified sequence length.
    """
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        dataset_name: str,
        dataset_config_name: str,
        split: str,
        sequence_length: int,
    ):
        self.tokenizer = tokenizer
        self.dataset_name = dataset_name
        self.dataset_config_name = dataset_config_name
        self.split = split
        self.sequence_length = sequence_length
        self.dataset = load_dataset(dataset_name, dataset_config_name, split=split, streaming=True)
    
    def __iter__(self):
        buffer = []
        buffer_size = 0
        
        for example in self.dataset:
            text = example["text"]
            if not text or len(text.strip()) == 0:
                continue
            
            # Tokenize text and add EOS token
            tokens = self.tokenizer.encode(text)
            tokens.append(self.tokenizer.eos_token_id)
            
            # Add to buffer
            buffer.extend(tokens)
            buffer_size += len(tokens)
            
            # Once we have enough tokens, yield chunks
            while buffer_size >= self.sequence_length + 1:  # +1 for labels shift
                chunk = buffer[:self.sequence_length + 1]
                buffer = buffer[self.sequence_length:]
                buffer_size -= self.sequence_length
                
                input_ids = torch.tensor(chunk[:-1])
                labels = torch.tensor(chunk[1:])
                
                yield {
                    "input_ids": input_ids,
                    "labels": labels,
                    "attention_mask": torch.ones_like(input_ids)
                }

def collate_batch_for_gpt2(batch):
    """
    Collate function specifically designed for GPT-2 style training.
    Since all sequences should be the same length, we just stack them.
    """
    input_ids = torch.stack([item["input_ids"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask
    }

def create_gpt2_dataloaders(
    tokenizer: PreTrainedTokenizer,
    dataset_name: str,
    dataset_config_name: str,
    batch_size: int,
    sequence_length: int,
):
    """
    Creates dataloaders in the style of GPT-2 training.
    Returns train, validation, and test dataloaders.
    """
    dataloaders = {}
    
    for split in ["train", "validation", "test"]:
        try:
            dataset = GPT2TextDataset(
                tokenizer=tokenizer,
                dataset_name=dataset_name,
                dataset_config_name=dataset_config_name,
                split=split,
                sequence_length=sequence_length,
            )
            
            dataloaders[split] = DataLoader(
                dataset,
                batch_size=batch_size,
                collate_fn=collate_batch_for_gpt2,
                num_workers=min(4, os.cpu_count() or 1),
                pin_memory=True,
                persistent_workers=True,
                prefetch_factor=2,
            )
        except ValueError:
            # Split doesn't exist
            pass
    
    return (
        dataloaders.get("train"),
        dataloaders.get("validation"),
        dataloaders.get("test")
    )

# Example usage
if __name__ == "__main__":
    from transformers import AutoTokenizer
    from torch.utils.tensorboard import SummaryWriter
    from torch.amp import autocast
    from tqdm import tqdm

    import argparse
    import utils

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--run_name', type=str, default="test")
    argparser.add_argument('--steps', type=int, default=100000)
    argparser.add_argument('--warmup_steps', type=int, default=1000)
    argparser.add_argument('--batch_size', type=int, default=8)
    argparser.add_argument('--batch_accum_factor', type=int, default=4)
    argparser.add_argument('--sequence_length', type=int, default=1024)
    argparser.add_argument('--tokenizer', type=str, default="gpt2")
    argparser.add_argument('--dataset_name', type=str, default="wikitext")
    argparser.add_argument('--dataset_config_name', type=str, default="wikitext-103-raw-v1")
    argparser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    argparser.add_argument('--print_freq', type=int, default=4000)

    argparser.add_argument('--base_lr', type=float, default=1.5e-5)
    argparser.add_argument('--weight_decay', type=float, default=0.01)
    argparser.add_argument('--beta1', type=float, default=0.9)
    argparser.add_argument('--beta2', type=float, default=0.999)
    argparser.add_argument('--epsilon', type=float, default=1e-8)
    argparser.add_argument('--clip_grad_norm', type=float, default=1.0)

    argparser.add_argument('--d_model', type=int, default=512)
    argparser.add_argument('--n_heads', type=int, default=8)
    argparser.add_argument('--n_layers', type=int, default=8)
    argparser.add_argument('--d_ff_factor', type=int, default=4)
    argparser.add_argument('--dropout', type=float, default=0.1)

    args = argparser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    
    train_loader, val_loader, test_loader = create_gpt2_dataloaders(
        tokenizer=tokenizer,
        dataset_name="wikitext",
        dataset_config_name="wikitext-103-raw-v1",
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
    )

    class GPT2(torch.nn.Module):
        def __init__(self, device, tokenizer: PreTrainedTokenizer, d_model, n_heads, n_layers, d_ff_factor, dropout, activation_fn, sequence_length):
            super(GPT2, self).__init__()
            self.tokenizer = tokenizer

            self.embedding = torch.nn.Embedding(tokenizer.vocab_size, d_model)
            self.pos_embedding = utils.get_buffered_positional_encoding(args, d_model, 'cpu', sequence_length)
            self.pos_embedding = self.pos_embedding.to(device)
            self.pos_embedding.requires_grad = False

            self.dropout = torch.nn.Dropout(dropout)

            self.transformer = torch.nn.TransformerEncoder(
                torch.nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=n_heads,
                    dim_feedforward=d_model*d_ff_factor,
                    activation=activation_fn,
                    batch_first=True
                ),
                num_layers=n_layers
            )

            self.lm_head = torch.nn.Linear(d_model, tokenizer.vocab_size)

            self.register_buffer('causal_mask', torch.nn.Transformer.generate_square_subsequent_mask(sequence_length, device='cuda', dtype=torch.bool))
            self.causal_mask.requires_grad = False

        def forward(self, src, src_key_padding_mask):
            x = self.embedding(src)
            x = x + self.pos_embedding[:, :x.size(1), :]
            causal_slice = self.causal_mask[:src.size(1), :src.size(1)].detach()
            x = self.transformer(x, mask=causal_slice, src_key_padding_mask=src_key_padding_mask, is_causal=True)
            x = self.lm_head(x)
            return x
        
        def generate(self, start_tokens, max_length):
            # greedy decoding
            with torch.no_grad():
                for _ in range(max_length-start_tokens.size(1)):
                    logits = self(start_tokens, None)
                    next_token = torch.argmax(logits[:, -1], dim=-1, keepdim=True)
                    start_tokens = torch.cat([start_tokens, next_token], dim=-1)

                    if next_token == self.tokenizer.eos_token_id:
                        break
            return start_tokens
        
    model = GPT2(args.device, tokenizer, args.d_model, args.n_heads, args.n_layers, args.d_ff_factor, args.dropout, torch.nn.functional.gelu, args.sequence_length).to(args.device).to(dtype=torch.bfloat16)

    # initialize weights
    model.embedding.weight.data.normal_(mean=0.0, std=0.02)
    for p in model.transformer.parameters():
        if p.dim() > 1:
            torch.nn.init.xavier_uniform_(p)
    model.lm_head.weight = model.embedding.weight

    model = torch.compile(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.base_lr,
        weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2),
        eps=args.epsilon
    )

    lr_scheduler = utils.create_warmup_cosine_scheduler(optimizer, args.warmup_steps, args.steps, min_lr=0.)

    run_dir = os.path.join('runs', 'test', args.run_name)
    os.makedirs(run_dir, exist_ok=True)

    summary_writer = SummaryWriter(run_dir)

    steps = 0
    with autocast('cuda'):
        while steps < args.steps:
            for i, batch in enumerate(tqdm(train_loader)):
                input_ids = batch["input_ids"].to(args.device)
                labels = batch["labels"].to(args.device)
                attention_mask = batch["attention_mask"].to(args.device).bool()

                outputs = model(input_ids, attention_mask)

                loss = torch.nn.functional.cross_entropy(outputs.view(-1, tokenizer.vocab_size), labels.view(-1), reduction='mean')

                loss.backward()

                loss = loss.item()
                ppl = np.exp(loss)

                summary_writer.add_scalar("train/loss", loss, steps)
                summary_writer.add_scalar("train/ppl", ppl, steps)

                if i % args.batch_accum_factor == 0:
                    summary_writer.add_scalar("train/grad", torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm), steps)

                    optimizer.step()
                    optimizer.zero_grad()

                    lr_scheduler.step()
                    summary_writer.add_scalar("train/lr", optimizer.param_groups[0]['lr'], steps)

                    steps += 1
                    if steps >= args.steps:
                        break

                if i % args.print_freq == 0:
                    print(f"Step: {steps}, Last Loss: {loss}, PPL: {ppl}")
                    example_input = "The Higgs boson, sometimes called the Higgs particle, is"
                    input_tokens = tokenizer.encode(example_input)
                    gen_output = model.generate(
                        torch.tensor(input_tokens).unsqueeze(0).to(args.device),
                        max_length=args.sequence_length
                    )
                    gen_text = tokenizer.decode(gen_output[0].tolist())
                    print(f"Generated: {gen_text}")

            # validation
            model.eval()
            with torch.no_grad():
                avg_loss = 0.0
                avg_ppl = 0.0
                n_batches = 0
                for batch in tqdm(val_loader):
                    input_ids = batch["input_ids"].to(args.device)
                    labels = batch["labels"].to(args.device)
                    attention_mask = batch["attention_mask"].to(args.device).bool()

                    outputs = model(input_ids, attention_mask)

                    loss = torch.nn.functional.cross_entropy(outputs.view(-1, tokenizer.vocab_size), labels.view(-1))

                    avg_loss += loss.item()
                    avg_ppl += np.exp(loss.item())
                    n_batches += 1

                avg_loss /= n_batches
                avg_ppl /= n_batches

                summary_writer.add_scalar("val/loss", avg_loss, steps)
                summary_writer.add_scalar("val/ppl", avg_ppl, steps)

            torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict()}, os.path.join(run_dir, f"model_{steps}.pt"))
            model.train()
