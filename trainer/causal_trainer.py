from datasets import load_dataset, Value
from functools import partial
from model_provider import CausalTransformerModelProvider
from megatransformer import megatransformer
from megatransformer import grokfast
from prettytable import PrettyTable
from torch.utils.data import DataLoader
from tqdm import tqdm
from trainer import base_trainer
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from multigpu_training_wrappers import MultiGPUCausalWrapper

import avg_meter
import os
import time
import torch
import utils

class CausalTrainer(base_trainer.BaseTrainer):
    def __init__(self, args):
        super(CausalTrainer, self).__init__(args)

        self.grads = None

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
            if hasattr(self.args, 'start_epoch') and self.args.start_epoch == 0:
                self.args.start_epoch = checkpoint['epoch'] + 1
                print('\nLoaded checkpoint from epoch %d.\n' % self.args.start_epoch)

            model = CausalTransformerModelProvider().provide_transformer(self.args, self.src_tokenizer, tie_embeddings=tie_embeddings)

            model.load_state_dict(checkpoint['model'].state_dict())

            if 'optimizer' in checkpoint:
                optimizer = checkpoint['optimizer']
            else:
                optimizer = None
        else:
            print("Starting from scratch...")
            model = CausalTransformerModelProvider().provide_transformer(self.args, self.src_tokenizer, tie_embeddings=tie_embeddings)

            weight_decay = self.args.weight_decay if hasattr(self.args, 'weight_decay') else 0.0
            optimizer = torch.optim.AdamW(params=[p for p in model.parameters() if p.requires_grad], lr=self.args.lr, betas=[self.args.beta1, self.args.beta2], weight_decay=weight_decay, eps=self.args.epsilon)

        if hasattr(self.args, 'multidevice') and bool(self.args.multidevice):
            self.model = MultiGPUCausalWrapper(
                model=model,
                optimizer=optimizer,
                gpu_ids=list(range(torch.cuda.device_count())),
                sync_steps=self.args.multidevice_sync_steps
            )
        
        return model, optimizer

    def load_data(self):
        collator = DataCollatorForLanguageModeling(tokenizer=self.src_tokenizer, mlm=False, mlm_probability=0.0)
        total_dataset = load_dataset("tiny_shakespeare", split="train", trust_remote_code=True)
        total_dataset = total_dataset.map(lambda x: Value('strings').unicode_split(x['text'], 'UTF-8'))
        if len(total_dataset) > 1:
            train_dataset, test_dataset = total_dataset.train_test_split(test_size=0.05)
            val_dataset, test_dataset = test_dataset.train_test_split(test_size=0.5)
            train_dataloader = DataLoader(train_dataset, collate_fn=collator, batch_size=self.args.batch_size, shuffle=True)
            val_dataloader = DataLoader(val_dataset, collate_fn=collator, batch_size=self.args.batch_size, shuffle=False)
            test_dataloader = DataLoader(test_dataset, collate_fn=collator, batch_size=self.args.batch_size, shuffle=False)
            return train_dataloader, val_dataloader, test_dataloader
        else:
            if hasattr(self.args, 'batch_size'):
                batch_size = self.args.batch_size
            else:
                tokens_in_batch = self.args.tokens_in_batch
                batch_size = tokens_in_batch // self.args.maxlen
            train_dataloader = DataLoader(total_dataset, collate_fn=collator, batch_size=batch_size, shuffle=True)
            return train_dataloader, None, None
    
    def train(self, model_name_prefix=''):
        if self.args.start_epoch == 0:
            print("Visualizing attention weights before training...")
            # get attention weight visualization before any updates are made to the model
            with torch.no_grad():
                self.model.eval()
                self.viz_model(0, self.model, seq="Anyone who retains the ability to recognise beauty will never become old.")

        super().train()

    def forward_pass(self, model, input_ids: torch.Tensor, labels: torch.Tensor, key_padding_mask: torch.Tensor):
        outputs = model(target_ids=input_ids, labels=labels, decoder_attention_mask=key_padding_mask, return_dict=True)

        total_loss = outputs.total_loss
        causal_loss = outputs.main_loss
        moe_loss = outputs.moe_loss

        total_loss.backward()

        return total_loss, causal_loss, moe_loss
    
    def train_epoch(self, model: megatransformer.MegaTransformer, epoch):
        # training mode enables dropout
        model.train()

        data_time = avg_meter.AverageMeter()
        step_time = avg_meter.AverageMeter()
        total_losses = avg_meter.AverageMeter()
        causal_losses = avg_meter.AverageMeter()
        moe_losses = avg_meter.AverageMeter()
        avg_gating_variances = avg_meter.AverageMeter()

        start_data_time = time.time()
        start_step_time = time.time()

        for i, (input_ids, labels) in enumerate(self.train_loader):
            input_ids = self.target_sequence_transform(labels, input_ids)
            labels = self.target_sequence_transform(input_ids, labels)

            key_padding_mask = input_ids == 0

            data_time.update(time.time() - start_data_time)

            total_loss, causal_loss, moe_loss, gating_variances = self.forward_pass(model, epoch, input_ids, labels, key_padding_mask)
            total_losses.update(total_loss.item())
            causal_losses.update(causal_loss.item())
            moe_losses.update(moe_loss.item())
            avg_gating_variances.update(gating_variances.mean().item())

            # Update model (i.e. perform a training step) only after gradients are accumulated from batches_per_step batches
            if (i + 1) % self.batches_per_step == 0:
                if self.args.clip_grad_norm is not None and self.args.clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.clip_grad_norm)
                
                if self.args.use_grokfast == 'ema':
                    self.grads = grokfast.gradfilter_ema(utils.sanitize_model(model), grads=self.grads, alpha=self.args.grokfast_alpha, lamb=self.args.grokfast_lambda)
                elif self.args.use_grokfast == 'ma':
                    self.grads = grokfast.gradfilter_ma(utils.sanitize_model(model), grads=self.grads, window_size=self.args.grokfast_window_size, lamb=self.args.grokfast_lambda)

                self.optimizer.step()
                self.optimizer.zero_grad()

                self.steps += 1

                utils.change_lr(self.optimizer, new_lr=utils.get_lr(self.steps, self.args.d_model, self.warmup_steps))

                step_time.update(time.time() - start_step_time)

                if self.steps % self.print_frequency == 0:
                    print('Epoch {0}/{1}-----Batch {2}/{3}-----Step {4}/{5}-----Data Time {data_time.val:.3f} ({data_time.avg:.3f})-----Step Time {step_time.val:.3f} ({step_time.avg:.3f})-----'
                          'Loss {total_losses.val:.4f} ({total_losses.avg:.4f})-----Early Stopping Counter: {early_stop_counter}/{early_stop_patience}'.format(epoch + 1, self.epochs, i + 1,  self.train_loader.n_batches * len(self.train_loader.src_file_paths), self.steps, self.n_steps, step_time=step_time, data_time=data_time, total_losses=total_losses, early_stop_counter=self.early_stopping.counter if self.early_stopping is not None else 0, early_stop_patience=self.early_stopping.patience if self.early_stopping is not None else 0))
                    self.evaluate(seq='Anyone who retains the ability to ', tgt='Anyone who retains the ability to recognise beauty will never become old.')

                self.summary_writer.add_scalar('train/causal_loss', causal_losses.avg, self.steps)
                self.summary_writer.add_scalar('train/avg_loss', total_losses.avg, self.steps)
                if moe_loss > 0:
                    self.summary_writer.add_scalar('MoE Loss', moe_losses.avg, self.steps)
                    self.summary_writer.add_scalar('Decoder Gating Variance', avg_gating_variances.avg, self.steps)

                start_step_time = time.time()

                # epoch is 0-indexed
                # early stopping requires the ability to average the last few checkpoints so just save all of them
                if (epoch in [self.epochs - 1, self.epochs - 2] or bool(self.args.early_stop)) and self.steps % 1500 == 0:
                    utils.save_checkpoint(epoch, self.model, self.optimizer, prefix=f"{self.run_dir}/step{str(self.steps)}_")
            start_data_time = time.time()
    
    def validate_epoch(self, model):
        model.eval()

        with torch.no_grad():
            losses = avg_meter.AverageMeter()
            for input_ids, labels in tqdm(self.val_loader, total=self.val_loader.n_batches):
                input_ids = input_ids.to(self.decoder_device)
                targets = targets.to(self.decoder_device)

                key_padding_mask = (input_ids == 0).to(self.decoder_device)

                outputs = model(input_ids=input_ids, labels=labels, decoder_attention_mask=key_padding_mask, return_dict=True)

                total_loss = outputs.total_loss

                losses.update(total_loss.item())

            self.summary_writer.add_scalar('val/avg_loss', losses.avg, self.steps)
            print("\nValidation loss: %.3f\n\n" % losses.avg)

            self.viz_model(self.steps, model, "Anyone who retains the ability to recognise beauty will never become old.", "Wer die Fähigkeit behält, Schönheit zu erkennen, wird niemals alt.", src_lang_code="en", tgt_lang_code="de")

            return losses.avg

    def evaluate(self, seq, example_expected):
        predictions = utils.greedy_complete(self.args, seq, self.model, self.src_tokenizer, 5)

        debug_validate_table = PrettyTable(["Rank", "Prediction", "Expected"])
        for i, prediction in enumerate(predictions):
            debug_validate_table.add_row([i + 1, prediction, example_expected])

        console_size = os.get_terminal_size()
        debug_validate_table.max_width = (console_size.columns // 3) - 15
        debug_validate_table.min_width = (console_size.columns // 3) - 15

        print(debug_validate_table)

    def viz_model(self, step, model, seq, **kwargs):
        print("Visualizing model...")
        if self.args.use_infinite_attention:
            return # todo: temporary; would like to visualize memory block attention weights in the future
        
        model.eval()
        with torch.no_grad():
            tokenized = self.src_tokenizer(
                seq,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"  # return PyTorch tensors
            )

            input_ids = tokenized.input_ids
            attention_mask = tokenized.attention_mask

            tokens = self.src_tokenizer.convert_ids_to_tokens(input_ids[0])
            input_ids = input_ids.to(self.decoder_device)

            seq_len = input_ids.size(1)
            
            input_ids = input_ids.to(self.args.decoder_device)
            attention_mask = attention_mask.to(self.args.decoder_device)
            model.embed_tokens = model.embed_tokens.to(self.args.decoder_device)
            model.lm_head = model.lm_head.to(self.args.decoder_device)

            seq = self.viz_decoder(model, input_ids, seq_len, attention_mask, input_ids, seq_len, attention_mask, tokens, tokens, self.summary_writer, step)