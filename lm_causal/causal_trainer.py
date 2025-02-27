from datasets import load_dataset, Dataset
from megatransformer import megatransformer, grokfast, transformer_utils, visualization_helper
from prettytable import PrettyTable
from torch.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
from trainer import base_trainer
from transformers import AutoTokenizer
from utils import multigpu_training_wrappers, avg_meter, utils, model_provider

import os
import random
import re
import time
import torch

class CausalTrainer(base_trainer.BaseTrainer):
    def __init__(self, args):
        self.grads = None

        super(CausalTrainer, self).__init__(args)

    def load_tokenizers(self, identifier):
        tokenizer = AutoTokenizer.from_pretrained(identifier[5:])
        return tokenizer, None

    def load_model_and_optimizer(self, run_dir, checkpoint_model_name='transformer_checkpoint.pth.tar'):
        print('Initializing model...')

        tie_embeddings = True
        if hasattr(self.args, 'tie_embeddings'):
            tie_embeddings = bool(self.args.tie_embeddings)

        if os.path.exists(os.path.join(run_dir, checkpoint_model_name)):
            checkpoint = torch.load(os.path.join(run_dir, checkpoint_model_name))
            if hasattr(self.args, 'start_step') and self.args.start_step == 0:
                self.args.start_step = checkpoint['step'] + 1
                print('\nLoaded checkpoint from step %d.\n' % self.args.start_step)

            model = model_provider.CausalTransformerModelProvider().provide_transformer(self.args, run_dir, self.src_tokenizer, tie_embeddings=tie_embeddings)

            model.load_state_dict(checkpoint['model'].state_dict())

            if 'optimizer' in checkpoint:
                optimizer = checkpoint['optimizer']
            else:
                optimizer = None
        else:
            print("Starting from scratch...")
            model = model_provider.CausalTransformerModelProvider().provide_transformer(self.args, run_dir, self.src_tokenizer, tie_embeddings=tie_embeddings)

            optimizer = torch.optim.AdamW(
                params=model.parameters(),
                lr=self.args.lr,
                betas=[self.args.beta1, self.args.beta2],
                weight_decay=float(self.args.weight_decay),
                eps=self.args.epsilon
            )

        if hasattr(self.args, 'multidevice') and bool(self.args.multidevice):
            self.model = multigpu_training_wrappers.MultiGPUCausalWrapper(
                model=model,
                optimizer=optimizer,
                gpu_ids=list(range(torch.cuda.device_count())),
                sync_steps=self.args.multidevice_sync_steps
            )
        
        return model, optimizer

    def load_data(self):
        if hasattr(self.args, 'batch_size'):
            self.batch_size = self.args.batch_size
            self.batch_accumulation_steps = self.args.batch_accumulation_steps
        else:
            tokens_in_batch = self.args.tokens_in_batch
            self.batch_size = tokens_in_batch // self.args.maxlen

        def make_dataloader(tokenizer, dataset_name, dataset_config_name, split, batch_size, maxlen, batch_tokenization_padding):
            class CausalLMDataset(Dataset):
                def __init__(self, tokenizer, dataset_name, dataset_config_name, split, sequence_length):
                    self.tokenizer = tokenizer
                    self.sequence_length = sequence_length
                    
                    # Load and preprocess dataset
                    dataset = load_dataset(dataset_name, dataset_config_name, split=split)
                    
                    # Tokenize entire dataset at once with map function
                    self.tokenized_dataset = dataset.map(
                        lambda examples: self.tokenize_function(examples),
                        batched=True,
                        remove_columns=['text']
                    )
                
                def tokenize_function(self, examples):
                    # Tokenize with appropriate padding and truncation
                    return self.tokenizer(
                        examples['text'], 
                        padding=batch_tokenization_padding,
                        truncation=True,
                        max_length=self.sequence_length + 1,
                        return_tensors='pt'
                    )
                
                def __len__(self):
                    return len(self.tokenized_dataset)
                
                def __getitem__(self, idx):
                    item = self.tokenized_dataset[idx]
                    input_ids = item['input_ids'][:-1]
                    labels = item['input_ids'][1:]
                    attention_mask = item['attention_mask'][:-1]
                    
                    # Mask padding tokens in labels
                    labels[labels == self.tokenizer.pad_token_id] = -100
                    
                    return {
                        'input_ids': input_ids,
                        'attention_mask': attention_mask,
                        'labels': labels
                    }
                
            dataset = CausalLMDataset(tokenizer, dataset_name, dataset_config_name, split, maxlen)
            return DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=2
            )

        train_dataloader = make_dataloader(self.src_tokenizer, self.args.dataset_name, self.args.dataset_config_name, 'train', self.batch_size, self.args.maxlen, self.args.batch_tokenization_padding)
        val_dataloader = make_dataloader(self.src_tokenizer, self.args.dataset_name, self.args.dataset_config_name, 'validation', self.batch_size, self.args.maxlen, self.args.batch_tokenization_padding)
        test_dataloader = make_dataloader(self.src_tokenizer, self.args.dataset_name, self.args.dataset_config_name, 'test', self.batch_size, self.args.maxlen, self.args.batch_tokenization_padding)

        print(len(train_dataloader.dataset), len(val_dataloader.dataset), len(test_dataloader.dataset))

        return train_dataloader, val_dataloader, test_dataloader

    def train(self, model_name_prefix=''):
        if self.args.start_step == 0 and not bool(self.args.debug):
            print("Visualizing attention weights before training...")
            # get attention weight visualization before any updates are made to the model
            with torch.no_grad():
                self.model.eval()
                self.viz_model(0, self.model, seq=random.choice(self.args.print_example_tgts))

        super().train()

    def forward_pass(self, model, input_ids: torch.Tensor, labels: torch.Tensor, attention_mask: torch.Tensor=None):
        with autocast('cuda' if 'cuda' in self.args.decoder_device else 'cpu'):
            outputs = model(input_ids=input_ids, labels=labels, decoder_attention_mask=attention_mask, return_dict=True)

        logits = outputs.logits

        self.summary_writer.add_scalar('train/logits_mean', logits.mean().item(), self.step)
        self.summary_writer.add_scalar('train/logits_std', logits.std().item(), self.step)
        self.summary_writer.add_scalar('train/logits_max', logits.max().item(), self.step)
        self.summary_writer.add_scalar('train/logits_min', logits.min().item(), self.step)
        self.summary_writer.add_scalar('train/logits_norm', logits.norm().item(), self.step)

        loss = outputs.loss
        causal_loss = outputs.prediction_loss
        moe_loss = outputs.moe_loss
        gating_variances = outputs.decoder_gating_variances

        loss.backward()

        return loss, causal_loss, moe_loss, gating_variances
    
    def training(self, model: megatransformer.MegaTransformer):
        # training mode enables dropout
        model.train()

        data_time = avg_meter.AverageMeter()
        step_time = avg_meter.AverageMeter()
        losses = avg_meter.AverageMeter()
        causal_losses = avg_meter.AverageMeter()
        moe_losses = avg_meter.AverageMeter()
        avg_gating_variances = avg_meter.AverageMeter()
        ppls = avg_meter.AverageMeter()

        start_data_time = time.time()
        start_step_time = time.time()

        for i, batch in enumerate(tqdm(self.train_loader, desc='Training')):
            input_ids = batch['input_ids']
            labels = batch['labels']
            # attention_mask = batch['attention_mask']

            input_ids = input_ids.to(self.args.decoder_device)
            labels = labels.to(self.args.decoder_device)
            # attention_mask = attention_mask.to(self.args.decoder_device)

            # input_ids = self.target_sequence_transform(labels, input_ids)
            # labels = self.target_sequence_transform(input_ids, labels)
            
            if self.args.debug:
                first_example_input_seq = input_ids[0].tolist()#self.src_tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=False)
                # labels_to_print = labels[0].clone().detach()
                # labels_to_print[labels_to_print == -100] = self.src_tokenizer.pad_token_id
                first_example_label_seq = labels[0].tolist()#self.src_tokenizer.decode(labels_to_print.tolist(), skip_special_tokens=False)
                # first_example_mask = attention_mask[0].tolist()

                print(f"############# Input #############: {first_example_input_seq}")
                print(f"############# Label #############: {first_example_label_seq}")
                # print(f"############# Mask  ##############: {first_example_mask}")

            data_time.update(time.time() - start_data_time)

            loss, causal_loss, moe_loss, gating_variances = self.forward_pass(model, input_ids, labels)#, attention_mask)
            ppl = torch.exp(loss).item()
            loss = loss.item()
            causal_loss = causal_loss.item()
            losses.update(loss)
            causal_losses.update(causal_loss)
            moe_losses.update(moe_loss.item())
            avg_gating_variances.update(gating_variances.mean().item())
            ppls.update(ppl)

            # Update model (i.e. perform a training step) only after gradients are accumulated from batches_per_step batches
            if (i + 1) % self.batches_per_step == 0:
                if self.args.clip_grad_norm is not None and self.args.clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.clip_grad_norm)
                
                if self.args.use_grokfast == 'ema':
                    self.grads = grokfast.gradfilter_ema(utils.sanitize_model(model), grads=self.grads, alpha=self.args.grokfast_alpha, lamb=self.args.grokfast_lambda)
                elif self.args.use_grokfast == 'ma':
                    self.grads = grokfast.gradfilter_ma(utils.sanitize_model(model), grads=self.grads, window_size=self.args.grokfast_window_size, lamb=self.args.grokfast_lambda)

                self.optimizer.step()

                grad_norms = utils.get_grad_norms(model)
                avg_grad_norm = sum(grad_norms) / len(grad_norms)
                self.summary_writer.add_scalar('train/avg_grad_norm', avg_grad_norm, self.step)

                self.optimizer.zero_grad()

                self.step += 1

                if self.lr_scheduler is None:
                    utils.change_lr(self.optimizer, new_lr=utils.get_lr(self.step, self.args.d_model, self.warmup_steps))
                else:
                    self.lr_scheduler.step()

                lr = self.lr_scheduler.get_last_lr()[0]
                self.summary_writer.add_scalar('train/lr', lr, self.step)

                step_time.update(time.time() - start_step_time)

                if self.step % self.print_frequency == 0:
                    print('Step {step}/{n_steps}-----Data Time {data_time.val:.3f} ({data_time.avg:.3f})-----Step Time {step_time.val:.3f} ({step_time.avg:.3f})-----'
                          'Loss {losses.val:.4f} ({losses.avg:.4f})-----Early Stopping Counter: {early_stop_counter}/{early_stop_patience}'.format(step=self.step, n_steps=self.n_steps, step_time=step_time, data_time=data_time, losses=losses, early_stop_counter=self.early_stopping.counter if self.early_stopping is not None else 0, early_stop_patience=self.early_stopping.patience if self.early_stopping is not None else 0))
                    for i, (src, tgt) in enumerate(zip(self.args.print_examples, self.args.print_example_tgts)):
                        predictions = self.evaluate(src, tgt)
                        for j, prediction in enumerate(predictions):
                            self.summary_writer.add_text(f"train/prediction_{i}_{j}", prediction, self.step)
                            self.summary_writer.add_text(f"train/expected_{i}_{j}", tgt, self.step)

                self.summary_writer.add_scalar('train/perplexity', ppl, self.step)
                self.summary_writer.add_scalar('train/loss', loss, self.step)
                self.summary_writer.add_scalar('train/causal_loss', causal_loss, self.step)
                if moe_loss > 0:
                    self.summary_writer.add_scalar('MoE Loss', moe_losses.avg, self.step)
                    self.summary_writer.add_scalar('Decoder Gating Variance', avg_gating_variances.avg, self.step)

                start_step_time = time.time()

                # early stopping requires the ability to average the last few checkpoints so just save all of them
                if (self.step > 0.95 * self.n_steps or bool(self.args.early_stop)) and self.step % 2000 == 0:
                    utils.save_checkpoint(self.step, self.model, self.optimizer, prefix=f"{self.run_dir}/step{str(self.step)}_")
            start_data_time = time.time()
    
    def validation(self, model):
        model.eval()

        with torch.no_grad():
            self.viz_model(self.step, model, random.choice(self.args.print_example_tgts))
            if self.val_loader is not None:
                losses = avg_meter.AverageMeter()
                ppls = avg_meter.AverageMeter()

                for batch in tqdm(self.val_loader):
                    input_ids = batch['input_ids']
                    labels = batch['labels']
                    attention_mask = batch['attention_mask']
                    
                    input_ids = input_ids.to(self.args.decoder_device)
                    labels = labels.to(self.args.decoder_device)
                    attention_mask = attention_mask.to(self.args.decoder_device)
                    
                    outputs = model(input_ids=input_ids, labels=labels, decoder_attention_mask=attention_mask, return_dict=True)

                    loss = outputs.loss

                    losses.update(loss.item())
                    ppls.update(torch.exp(loss).item())

                self.summary_writer.add_scalar('val/perplexity', ppls.avg, self.step)
                self.summary_writer.add_scalar('val/avg_loss', losses.avg, self.step)
                print("\nValidation loss: %.3f\n\n" % losses.avg)

                return losses.avg, ppls.avg
            return 0, 0

    def evaluate(self, src, tgt):
        predictions = []
        for _ in range(2):
            seq = self.src_tokenizer.encode(src, return_tensors='pt').to(self.args.decoder_device)
            result = self.model.generate(
                seq,
                max_length=self.args.maxlen,
                do_sample=True,
                top_p=0.95,
                temperature=0.9,
                eos_token_id=self.src_tokenizer.eos_token_id
            ).squeeze(0).tolist()

            text = self.src_tokenizer.decode(result, skip_special_tokens=True)
            text = re.sub(r'\s([,.!?;:])', r'\1', text)
            predictions.append(text)

        self.model.train()
        debug_validate_table = PrettyTable(["Rank", "Prediction", "Expected"])
        for i, prediction in enumerate(predictions):
            debug_validate_table.add_row([i + 1, prediction, tgt])

        console_size = os.get_terminal_size()
        debug_validate_table.max_width = (console_size.columns // 3) - 15
        debug_validate_table.min_width = (console_size.columns // 3) - 15

        print(debug_validate_table)

        return predictions

    def viz_model(self, step, model, seq, **kwargs):
        print("Visualizing model...")
        all_tokens = [''] * self.src_tokenizer.vocab_size
        for token, id in self.src_tokenizer.get_vocab().items():
            all_tokens[id] = token

        model.eval()
        with torch.no_grad():
            with autocast('cuda' if 'cuda' in self.args.decoder_device else 'cpu'):
                transformer_utils.record_model_param_stats(self.args, self.summary_writer, model, step, embedding_tokens=all_tokens)

                input_ids = self.src_tokenizer.encode(
                    seq,
                    max_length=self.args.maxlen,
                    truncation=True,
                    return_tensors='pt'
                ).to(self.args.decoder_device)

                attention_mask = torch.zeros_like(input_ids)

                tokens = [self.src_tokenizer.decode([token]) for token in input_ids.squeeze(0).tolist()]

                seq_len = input_ids.size(1)
                
                input_ids = input_ids.to(self.args.decoder_device)
                model.embed_tokens = model.embed_tokens.to(self.args.decoder_device)
                model.lm_head = model.lm_head.to(self.args.decoder_device)

                seq = visualization_helper.viz_decoder(self.args.decoder_device, model, input_ids, seq_len, attention_mask, input_ids, seq_len, None, tokens, tokens, self.summary_writer.add_image, step, annot=False)
