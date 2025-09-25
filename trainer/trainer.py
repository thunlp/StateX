from pathlib import Path
from arguments import Args
from typing import Optional
import contextlib
import time, random

from torch import nn, Tensor
from torch.utils.data import DataLoader
from accelerate import Accelerator, DistributedType
import torch
from transformers.modeling_outputs import CausalLMOutputWithPast

from itertools import islice

class LMTrainer:
    def __init__(
        self,
        args: Args,
        output_dir: Path,
        accelerator: Accelerator,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
        train_loader: DataLoader,
        val_loader: None | DataLoader = None,
        include_num_input_tokens_seen: bool = True,
        state_passing: bool = False,
    ):
        self.args = args
        self.output_dir = output_dir
        self.accelerator = accelerator
        self.raw_model = model
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.lr_scheduler = lr_scheduler
        self.val_loader = val_loader

        self.include_num_input_tokens_seen = include_num_input_tokens_seen
        if self.include_num_input_tokens_seen:
            self.main_input_name = 'input_ids'
            self.n_input_tokens_seen = 0

        if getattr(self.raw_model, "estimate_mfu", None) is not None:
            self.do_estimate_mfu = True
        else:
            self.do_estimate_mfu = False
            
        if state_passing is not None and state_passing is True:
            self.accelerator.print("State passing is enabled.")

            self.state_passing = True
            random.seed(42 + self.accelerator.process_index)
            self.history_states = {}
        else:
            self.state_passing = False

    @torch.no_grad()
    def get_validation_loss(
        self,
        accelerator: Accelerator,
        model: nn.Module,
        data_loader: DataLoader,
        n_batches: int,
        device: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Perform evaluation on `n_batches` batches from a dataloader.
        """
        model.eval()
        losses = torch.zeros(n_batches)
        accelerator.print("==== Evaluation ====")
        i = 0
        for i, batch in enumerate(data_loader):
            if i == n_batches:
                break
            loss, logits = model(**batch)
            losses[i] = loss.item()
        accelerator.print("==== Evaluation Done ====")
        model.train()
        loss = losses.mean()
        return loss

    def estimate_mfu(
        self,
        iter_time: float,
    ) -> float:
        """
        Estimate the model efficiency in terms of FLOPS Utilized.
        """
        self.running_mfu = -1.0
        mfu = self.raw_model.estimate_mfu(
            self.args.grad_accum_steps
            * self.args.batch_size,  # The number of sequences this process has seen.
            iter_time,
            train_len=self.args.max_len,
        )
        if self.running_mfu == -1.0:
            return mfu
        else:
            return 0.9 * self.running_mfu + 0.1 * mfu

    def handle_resumption(self, data_iterator):
        '''
        Resume from a training checkpoint.
        '''
        # NOTE: I think this is not working correctly.
        if self.args.resume_path is not None and self.args.resume_step is not None:
            assert self.args.pretrained_path is None, (
                "You specified resume_model_path, but that will be be overridden by resume_path."
            )
            self.accelerator.print(f"Resuming from {self.args.resume_path} at step {self.args.resume_step}")
            self.accelerator.load_state(self.args.resume_path)
            # Skip the data loader up to the resume step
            for _ in range(self.args.resume_step):
                for _ in range(self.args.grad_accum_steps):
                    _ = next(data_iterator, None)
            self.cur_step = self.args.resume_step
        else:
            self.cur_step = 0

        self.accelerator.wait_for_everyone()

    def count_tokens(self, inputs: dict):
        assert self.main_input_name in inputs
        n_tokens = inputs[self.main_input_name].numel()
        n_tokens = torch.tensor(n_tokens, device=self.accelerator.device, dtype=torch.int64)
        self.n_input_tokens_seen += torch.sum(self.accelerator.gather(n_tokens)).cpu().item()

    def clip_grad_norm(self):
        # Clip the gradient, DeepSpeed will do gradient clipping internally.
        if self.accelerator.distributed_type == DistributedType.DEEPSPEED:
            grad_norm = self.model.get_global_grad_norm()
            # In some cases the grad norm may not return a float
            if hasattr(grad_norm, "item"):
                self.grad_norm = grad_norm.item()
        elif self.args.grad_clip > 0.0 and self.accelerator.sync_gradients:
            self.grad_norm: float = self.accelerator.clip_grad_norm_(
                self.model.parameters(), self.args.grad_clip
            )  # type: ignore
            self.grad_norm: float = self.grad_norm.item()  # type: ignore
        else:
            self.grad_norm = None  # type: ignore

    def process_batch(self, data_iterator):
        '''
        Process a batch of data, and update the model and LR.
        '''
        # The average loss over the gradient accumulation steps
        batch_loss = torch.tensor(0.0).to(self.accelerator.device)
        for cur_micro_step in range(self.args.grad_accum_steps):
            # Fetch data
            start_time = time.time()
            inputs: dict | None = next(data_iterator, None)
            self.data_loading_time += time.time() - start_time

            # TODO: Handle data exhaustion here
            if self.state_passing:
                initial_state = self.history_states.get(cur_micro_step, None)
                inputs['use_cache'] = True
                if random.random() > 0.1: inputs['past_key_values'] = initial_state
                inputs['state_passing'] = True

            with self.accelerator.accumulate(self.model):
                self.model.train()
                start_time = time.time()
                outputs = self.model(**inputs)  # type: ignore
                if isinstance(outputs, CausalLMOutputWithPast):
                    loss: Tensor = outputs.loss  # type: ignore
                    if self.state_passing:
                        self.history_states[cur_micro_step] = outputs.past_key_values
                else:
                    loss: Tensor = outputs[0]
                    if self.state_passing:
                        raise NotImplementedError(
                            "State passing is not implemented yet."
                        )

                self.fwd_time += time.time() - start_time

                start_time = time.time()
                self.accelerator.backward(loss)
                self.bwd_time += time.time() - start_time

                batch_loss += loss.item() / self.args.grad_accum_steps

                if self.include_num_input_tokens_seen:
                    self.count_tokens(inputs)

        # # Check for NaN gradients before clipping
        # self.accelerator.print("="*100)
        # self.accelerator.print(f"batch_loss: {batch_loss}, type: {type(batch_loss)}")
        # self.accelerator.print("Checking for NaN gradients before clipping")
        # for name, param in self.model.named_parameters():
        #     if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
        #         self.accelerator.print(f"Warning: NaN/Inf gradients detected in {name}")
        #         param.grad.data.zero_()

        self.clip_grad_norm()

        # Note that the model parameters are only updated when we are
        # at the end of an accumulation cycle.
        self.optimizer.step()
        self.lr_scheduler.step()  # Scheduler stepping is not controlled by the accelerator.
        self.optimizer.zero_grad()

        # self.accelerator.print("Checking for NaN in params after step")
        # for name, param in self.model.named_parameters():
        #     if torch.isnan(param).any() or torch.isinf(param).any():
        #         self.accelerator.print(f"Param {name} contains NaN or Inf after step!")

        # exit()

        return batch_loss

    def train_loop(
        self,
        resume_step: Optional[int] = None,
    ):
        data_iterator = iter(self.train_loader)
        if resume_step is not None:
            data_iterator = islice(data_iterator, resume_step * self.args.grad_accum_steps, None)

        # Load training states from checkpoint
        # self.handle_resumption(data_iterator)

        # For tracking the training efficiency
        self.running_mfu = -1.0
        self.data_loading_time = 0
        self.fwd_time = 0
        self.bwd_time = 0
        self.last_log_time = time.time()
        self.model.zero_grad()

        # At this point, `self.model` is the wrapped model.
        # `self.raw_model` is the original model.

        while self.cur_step < self.args.n_train_steps:
            # TODO: Add evaluation code
            if self.val_loader is not None and self.cur_step % self.args.eval_interval == 0:
                val_loss = self.get_validation_loss(
                    accelerator=self.accelerator,
                    model=self.model,
                    data_loader=self.val_loader,
                    n_batches=self.args.n_eval_batches,
                )
                self.accelerator.log({"loss/val": val_loss}, step=self.cur_step)

            self.cur_step += 1
            self.train_loss = self.process_batch(data_iterator)
            self.train_log()

            # Checkpointing
            if self.cur_step % self.args.save_interval == 0 or self.cur_step == 14602:
                self.save_ckpt()

    def train_log(self):
        '''
        Log to experiment tracker (e.g., tensorboard) and stdout, then
        save a training checkpoint if needed.

        Most of this function will only be run every `log_interval` steps.
        '''
        # Time of this batch
        cur_time = time.time()
        iter_time = cur_time - self.last_log_time
        self.last_log_time = cur_time

        # Logging
        if self.cur_step % self.args.log_interval == 0:

            cur_lr = self.lr_scheduler.get_last_lr()[0]
            self.accelerator.log({"loss/train": self.train_loss}, step=self.cur_step)
            self.accelerator.log({"optim/lr": cur_lr}, step=self.cur_step)
            self.accelerator.log({"efficiency/iter_time": iter_time}, step=self.cur_step)
            self.accelerator.log({"efficiency/data_time": self.data_loading_time}, step=self.cur_step)
            self.accelerator.log({"efficiency/fwd_time": self.fwd_time}, step=self.cur_step)
            self.accelerator.log({"efficiency/bwd_time": self.bwd_time}, step=self.cur_step)
            self.accelerator.log({"misc/cur_step": self.cur_step}, step=self.cur_step)

            if self.grad_norm is not None:
                self.accelerator.log({"optim/grad_norm": self.grad_norm}, step=self.cur_step)

            if self.include_num_input_tokens_seen:
                self.accelerator.log({"train/n_input_tokens_seen": self.n_input_tokens_seen}, step=self.cur_step)

            self.data_loading_time /= self.args.log_interval
            self.fwd_time /= self.args.log_interval
            self.bwd_time /= self.args.log_interval

            log_str = f"[{self.cur_step}/{self.args.n_train_steps}] loss {self.train_loss:.4f}"
            log_str += f" | time {iter_time * 1000:.1f}ms"
            log_str += f" | t_data {self.data_loading_time * 1000:.1f}ms"
            log_str += f" | t_fwd {self.fwd_time * 1000:.1f}ms"
            log_str += f" | t_bwd {self.bwd_time * 1000:.1f}ms"
            log_str += f" | lr {cur_lr:.3e}"

            if self.do_estimate_mfu and self.cur_step >= 5:  # let the training loop settle a bit
                self.running_mfu = self.estimate_mfu(iter_time)
                self.accelerator.log({"efficiency/mfu": self.running_mfu}, step=self.cur_step)
                log_str += f" | mfu {self.running_mfu * 100:.1f}%"

            if self.grad_norm is not None:
                log_str += f" | grad_norm {self.grad_norm:.2e}"

            if self.include_num_input_tokens_seen:
                log_str += f" | tokens {self.n_input_tokens_seen:,}"

            self.accelerator.print(log_str)

            # Reset timers
            self.data_loading_time = 0
            self.fwd_time = 0
            self.bwd_time = 0

    def save_ckpt(self):
        self.accelerator.wait_for_everyone()
        ckpt_dir = self.output_dir / f"ckpt_{self.cur_step}"
        self.accelerator.print(f"Saving checkpoint to {ckpt_dir}")
        self.accelerator.save_state(str(ckpt_dir))

    def load_pretrained_ckpt(self):
        assert self.args.pretrained_path is not None
        assert Path(self.args.pretrained_path).exists(), f"{self.args.pretrained_path} does not exist."
        print(f"Loading pretrained model from {self.args.pretrained_path}")

        ckpt = torch.load(self.args.pretrained_path, map_location="cpu")
        self.model.load_state_dict(ckpt["model"])


    def load_ckpt_state(self):
        self.accelerator.print(f"Resuming from {self.args.resume_path} at step {self.args.resume_step}")
        self.accelerator.wait_for_everyone()
        self.accelerator.load_state(self.args.resume_path)
        self.accelerator.wait_for_everyone()
        self.cur_step = self.args.resume_step
    
    def train(self):
        self.is_training = True

        total_batch_size = self.args.batch_size * self.accelerator.num_processes * self.args.grad_accum_steps
        # Training loop
        self.accelerator.print("===== Start training =====")
        self.accelerator.print(f"Grad accum: {self.args.grad_accum_steps}")
        self.accelerator.print(f"Micro batch size (per device, per forward): {self.args.batch_size}")
        self.accelerator.print(f"Total train batch size (w. parallel, distributed & accumulation): {total_batch_size}")
        self.accelerator.print(f"# train iters: {self.args.n_train_steps}")
        self.accelerator.print(f"# warmup iters: {self.args.n_warmup_steps}")
        self.accelerator.print(f"# drop iters: {self.args.n_drop_steps}")
        self.accelerator.print(f"Eval interval: {self.args.eval_interval}")
        self.accelerator.print(f"Log interval: {self.args.log_interval}")

        # TODO: Handle model loading from pre-trained checkpoint.
        # NOTE: This should not load training states (optimizer, LR scheduler...)
        # if self.args.pretrained_path is not None:
        #     self.load_pretrained_ckpt()

        #     if self.args.target_head_num is not None:
        #         print('Change head number to {}'.format(self.args.target_head_num))
        #         self.model.module.merge_head(self.args.target_head_num)
        #         self.rebuild_optimizer_and_lr()

        # Wrap the model with accelerator classes
        self.model, self.optimizer, self.lr_scheduler, self.train_loader = self.accelerator.prepare(
            self.model, self.optimizer, self.lr_scheduler, self.train_loader)

        if self.args.resume_path is not None and self.args.resume_step is not None:
            self.load_ckpt_state()
        else:
            self.cur_step = 0

        # Cannot both exists resume_step and skip_step
        assert not (self.args.resume_step is not None and self.args.skip_step is not None), (
            "You cannot specify both resume_step and skip_step. "
            "Please use one of them to control the training resumption."
        )

        skip_step = self.args.resume_step if self.args.skip_step is None else self.args.skip_step
        self.accelerator.print(f"Starting from step {self.cur_step}, skipped {skip_step} steps.")
        self.train_loop(resume_step=skip_step)

        self.accelerator.end_training()  # Some experiment trackers need this
        self.accelerator.print("TRAINING DONE")
