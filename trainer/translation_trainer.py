from functools import partial
from model_provider import TranslationTransformerModelProvider
from megatransformer import megatransformer, grokfast, transformer_utils, visualization_helper
from multigpu_training_wrappers import MultiGPUTranslationWrapper
from prettytable import PrettyTable
from torch.amp import autocast
from tqdm import tqdm
from trainer import base_trainer

import avg_meter
import os
import time
import torch
import utils

class TranslationTrainer(base_trainer.BaseTrainer):
    def __init__(self, args):
        super(TranslationTrainer, self).__init__(args)

        self.grads = None

    def load_tokenizers(self, identifier):
        return utils.load_yttm_tokenizers(run_dir=identifier)

    def load_model_and_optimizer(self, run_dir, checkpoint_model_name='transformer_checkpoint.pth.tar'):
        print('Initializing model...')

        tie_embeddings = self.src_tokenizer == self.tgt_tokenizer
        if hasattr(self.args, 'tie_embeddings'):
            tie_embeddings = bool(self.args.tie_embeddings)

        if os.path.exists(os.path.join(run_dir, checkpoint_model_name)):
            checkpoint = torch.load(os.path.join(run_dir, checkpoint_model_name))
            if hasattr(self.args, 'start_step') and self.args.start_step == 0:
                self.args.start_step = checkpoint['step'] + 1
                print('\nLoaded checkpoint from step %d.\n' % self.args.start_step)

            model = TranslationTransformerModelProvider().provide_transformer(self.args, utils.vocab_size(self.args, self.src_tokenizer), utils.vocab_size(self.args, self.tgt_tokenizer), tie_embeddings=tie_embeddings)

            model.load_state_dict(checkpoint['model'].state_dict())

            if 'optimizer' in checkpoint:
                optimizer = checkpoint['optimizer']
            else:
                optimizer = None
        else:
            print("Starting from scratch...")
            model = TranslationTransformerModelProvider().provide_transformer(self.args, utils.vocab_size(self.args, self.src_tokenizer), utils.vocab_size(self.args, self.tgt_tokenizer), tie_embeddings=tie_embeddings)

            optimizer = torch.optim.Adam(params=[p for p in model.parameters() if p.requires_grad], lr=self.args.lr, betas=[self.args.beta1, self.args.beta2], eps=self.args.epsilon)

        if hasattr(self.args, 'multidevice') and bool(self.args.multidevice):
            self.model = MultiGPUTranslationWrapper(
                model=model,
                optimizer=optimizer,
                gpu_ids=list(range(torch.cuda.device_count())),
                sync_steps=self.args.multidevice_sync_steps
            )
        
        return model, optimizer

    def load_data(self):
        return utils.load_translation_data(self.args, int(self.args.tokens_in_batch), self.tokenizer_run_dir, self.src_tokenizer, self.tgt_tokenizer, pad_to_length=self.args.maxlen if self.args.attn_impl == 'infini' else None)
    
    def train(self, model_name_prefix=''):
        if self.args.start_step == 0:
            print("Visualizing attention weights before training...")
            # get attention weight visualization before any updates are made to the model
            with torch.no_grad():
                self.model.eval()
                self.viz_model(0, self.model, "Anyone who retains the ability to recognise beauty will never become old.", "Wer die Fähigkeit behält, Schönheit zu erkennen, wird niemals alt.", src_lang_code="en", tgt_lang_code="de")

        start = time.time()
        super().train()
        time_taken = time.time() - start

        print(f"Training complete. Scoring with sacrebleu...")
        return utils.sacrebleu_evaluate(self.args, self.run_dir, self.src_tokenizer, self.tgt_tokenizer, self.model, sacrebleu_in_python=True, test_loader=self.test_loader).score, time_taken, utils.count_parameters(self.model)

    def forward_pass(self, model,
                     src_seqs: torch.Tensor,
                     tgt_seqs: torch.Tensor,
                     src_key_padding_mask: torch.Tensor,
                     tgt_key_padding_mask: torch.Tensor):
        outputs = model(input_ids=src_seqs, labels=tgt_seqs, decoder_attention_mask=tgt_key_padding_mask, encoder_attention_mask=src_key_padding_mask, return_dict=True)

        loss = outputs.loss
        translation_loss = outputs.prediction_loss
        moe_loss = outputs.moe_loss
        encoder_gating_variances = outputs.encoder_gating_variances
        decoder_gating_variances = outputs.decoder_gating_variances

        loss.backward()
        
        return loss, translation_loss, moe_loss, encoder_gating_variances, decoder_gating_variances
    
    def training(self, model: megatransformer.MegaTransformer):
        # training mode enables dropout
        model.train()

        data_time = avg_meter.AverageMeter()
        step_time = avg_meter.AverageMeter()
        losses = avg_meter.AverageMeter()
        translation_losses = avg_meter.AverageMeter()
        moe_losses = avg_meter.AverageMeter()
        encoder_moe_gating_variance_losses = avg_meter.AverageMeter()
        decoder_moe_gating_variance_losses = avg_meter.AverageMeter()

        start_data_time = time.time()
        start_step_time = time.time()

        for i, (input_ids, labels) in enumerate(self.train_loader):
            input_ids = self.target_sequence_transform(labels, input_ids)
            labels = self.target_sequence_transform(input_ids, labels)

            src_key_padding_mask = input_ids == self.src_tokenizer.pad_token_id
            tgt_key_padding_mask = (labels == self.tgt_tokenizer.pad_token_id).to(self.args.decoder_device)

            data_time.update(time.time() - start_data_time)

            loss, forward_translation_loss, moe_loss, encoder_gating_variances, decoder_gating_variances = self.forward_pass(model, input_ids, labels, src_key_padding_mask, tgt_key_padding_mask)
            translation_losses.update(forward_translation_loss.item())
            losses.update(loss.item())
            moe_losses.update(moe_loss.item(), 1)
            encoder_moe_gating_variance_losses.update(encoder_gating_variances.item(), 1)
            decoder_moe_gating_variance_losses.update(decoder_gating_variances.item(), 1)

            if bool(self.args.multilang):
                loss, backward_translation_loss, moe_loss, encoder_gating_variances, decoder_gating_variances = self.forward_pass(model, labels, input_ids, tgt_key_padding_mask, src_key_padding_mask)
                translation_losses.update(backward_translation_loss.item())
                losses.update(loss.item())
                moe_losses.update(moe_loss.item(), 1)
                encoder_moe_gating_variance_losses.update(encoder_gating_variances.item(), 1)
                decoder_moe_gating_variance_losses.update(decoder_gating_variances.item(), 1)

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

                self.step += 1

                utils.change_lr(self.optimizer, new_lr=utils.get_lr(self.step, self.args.d_model, self.warmup_steps))

                step_time.update(time.time() - start_step_time)

                if self.step % self.print_frequency == 0:
                    print('Step {step}/{n_steps}-----Data Time {data_time.val:.3f} ({data_time.avg:.3f})-----Step Time {step_time.val:.3f} ({step_time.avg:.3f})-----'
                          'Loss {losses.val:.4f} ({losses.avg:.4f})-----Early Stopping Counter: {early_stop_counter}/{early_stop_patience}'.format(self.step, self.n_steps, step_time=step_time, data_time=data_time, losses=losses, early_stop_counter=self.early_stopping.counter if self.early_stopping is not None else 0, early_stop_patience=self.early_stopping.patience if self.early_stopping is not None else 0))
                    if self.args.multilang:
                        self.evaluate(src='Anyone who retains the ability to recognise beauty will never become old.', tgt='Wer die Fähigkeit behält, Schönheit zu erkennen, wird niemals alt.', src_lang_code='en', tgt_lang_code='de')
                        self.evaluate(src='Wer die Fähigkeit behält, Schönheit zu erkennen, wird niemals alt.', tgt='Anyone who retains the ability to recognise beauty will never become old.', src_lang_code='de', tgt_lang_code='en')
                        self.evaluate(src='Wer die Fähigkeit behält, Schönheit zu erkennen, wird niemals alt.', tgt='Ceux qui conservent la capacité de reconnaître la beauté ne vieillissent jamais.', src_lang_code='de', tgt_lang_code='fr')
                        self.evaluate(src='Anyone who retains the ability to recognise beauty will never become old.', tgt='Tie, kas saglabā spēju atpazīt skaistumu, nekad nenoveco.', src_lang_code='en', tgt_lang_code='lv')
                        self.evaluate(src='Ceux qui conservent la capacité de reconnaître la beauté ne vieillissent jamais.', tgt='Tie, kas saglabā spēju atpazīt skaistumu, nekad nenoveco.', src_lang_code='fr', tgt_lang_code='lv')
                    else:
                        self.evaluate(src='Anyone who retains the ability to recognise beauty will never become old.', tgt='Wer die Fähigkeit behält, Schönheit zu erkennen, wird niemals alt.')

                self.summary_writer.add_scalar('train/translation_loss', translation_losses.avg, self.step)
                self.summary_writer.add_scalar('train/avg_loss', loss.avg, self.step)
                if moe_loss > 0:
                    self.summary_writer.add_scalar('Encoder MoE Gating Variances', encoder_moe_gating_variance_losses.avg, self.step)
                    self.summary_writer.add_scalar('Decoder MoE Gating Variances', decoder_moe_gating_variance_losses.avg, self.step)

                start_step_time = time.time()

                # early stopping requires the ability to average the last few checkpoints so just save all of them
                if (self.step > 0.95 * self.n_steps or bool(self.args.early_stop)) and self.step % 2000 == 0:
                    utils.save_checkpoint(self.step, self.model, self.optimizer, prefix=f"{self.run_dir}/step{str(self.step)}_")
            start_data_time = time.time()
    
    def validation(self, model):
        model.eval()

        with torch.no_grad():
            losses = avg_meter.AverageMeter()
            for (input_ids, labels) in tqdm(self.val_loader, total=self.val_loader.n_batches):
                input_ids = input_ids.to(self.args.encoder_device)
                labels = labels.to(self.args.encoder_device)

                src_key_padding_mask = input_ids == self.src_tokenizer.pad_token_id
                tgt_key_padding_mask = (labels == self.tgt_tokenizer.pad_token_id).to(self.args.decoder_device)

                outputs = model(input_ids=input_ids, labels=labels, attention_mask=src_key_padding_mask, decoder_attention_mask=tgt_key_padding_mask, return_dict=True)

                loss = outputs.loss

                losses.update(loss.item())

            self.summary_writer.add_scalar('val/avg_loss', losses.avg, self.step)
            print("\nValidation loss: %.3f\n\n" % losses.avg)

            self.viz_model(self.step, model, "Anyone who retains the ability to recognise beauty will never become old.", "Wer die Fähigkeit behält, Schönheit zu erkennen, wird niemals alt.", src_lang_code="en", tgt_lang_code="de")

            return losses.avg

    def evaluate(self, src, tgt, src_lang_code=None, tgt_lang_code=None):
        if src_lang_code is None or tgt_lang_code is None:
            best, _ = utils.beam_search_translate(self.args, src, self.tgt_tokenizer.subword_to_id('<BOS>'), self.model, self.src_tokenizer, self.tgt_tokenizer, beam_size=4, length_norm_coefficient=0.6)
        else:
            src = f"{src_lang_code}__{src}"
            tgt = f"{tgt_lang_code}__{tgt}"
            best, _ = utils.beam_search_translate(self.args, src, utils.lang_code_to_id(tgt_lang_code), self.model, self.src_tokenizer, self.tgt_tokenizer, beam_size=4, length_norm_coefficient=0.6)

        debug_validate_table = PrettyTable(["Test Source", "Test Prediction", "Test Target"])
        debug_validate_table.add_row([src, best, tgt])

        console_size = os.get_terminal_size()
        debug_validate_table.max_width = (console_size.columns // 3) - 15
        debug_validate_table.min_width = (console_size.columns // 3) - 15

        print(debug_validate_table)

    def viz_model(self, step, model, src, **kwargs):
        tgt = kwargs.get('tgt', None)
        src_lang_code = kwargs.get('src_lang_code', None)
        tgt_lang_code = kwargs.get('tgt_lang_code', None)
        
        if src_lang_code is not None and tgt_lang_code is not None and self.args.multilang:
            src = f"{src_lang_code}__{src}"
            tgt = f"{tgt_lang_code}__{tgt}"

        src_encode = partial(utils.encode, bool(self.args.multilang), self.src_tokenizer)
        src_decode = partial(utils.decode, bool(self.args.multilang), self.src_tokenizer)

        tgt_encode = partial(utils.encode, bool(self.args.multilang), self.tgt_tokenizer)
        tgt_decode = partial(utils.decode, bool(self.args.multilang), self.tgt_tokenizer)

        model.eval()
        with torch.no_grad():
            with autocast('cuda' if 'cuda' in self.args.decoder_device else 'cpu'):
                transformer_utils.record_model_param_stats(self.args, self.summary_writer, model, step)

                src = torch.LongTensor(src_encode(sentences=src, eos=bool(self.args.multilang)))
                src_tokens = src_decode(ids=src)[-1][0]
                src = src.to(self.args.encoder_device)

                tgt = torch.LongTensor(tgt_encode(sentences=tgt, bos=True, eos=True))
                tgt_tokens = tgt_decode(ids=tgt)[-1][0]
                tgt = tgt.to(self.args.decoder_device)

                visualization_helper.viz_model(
                    self.args.encoder_device,
                    self.args.decoder_device,
                    model,
                    self.summary_writer.add_image,
                    step,
                    self.args.maxlen,
                    src, src_tokens,
                    tgt, tgt_tokens, 
                    src_pad_token_id=self.src_tokenizer.pad_token_id, tgt_pad_token_id=self.tgt_tokenizer.pad_token_id
                )
