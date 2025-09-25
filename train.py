from datetime import datetime
import json
from pathlib import Path
from typing import Tuple

from accelerate import Accelerator
from accelerate.utils import set_seed
from datasets.distributed import split_dataset_by_node
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from accelerate import DistributedDataParallelKwargs

from arguments import Args
from transformers import get_scheduler
from optim.lr_scheduler import WSDScheduler
from optim.optimizer import get_optimizer
from data import get_data
from trainer.trainer import LMTrainer
from utils import get_num_params

from safetensors.torch import load_file
from transformers import default_data_collator

def load_train_config(args: Args):
    '''
    Load from `args.train_config` if it exists, and use the values
    when the argument is not provided.
    '''
    if args.train_config is not None and Path(args.train_config).exists():
        config = json.load(open(args.train_config, "r"))
        for k, v in config.items():
            if getattr(args, k) is None:
                setattr(args, k, v)
    else:
        print(f"WARNING: train config {args.train_config} does not exist.")
        print("It is highly recommended to provide a train config.")


def get_model(
    accelerator: Accelerator,
    args: Args,
) -> nn.Module:
    if args.model_name == 'gpt':
        from modeling import gpt
        model = gpt.get_model(config_path=args.model_config, args=args)
    elif args.model_name == 'llama3':
        from modeling import llama3
        model = llama3.get_model(config_path=args.model_config, args=args)
    elif args.model_name == 'gated-deltanet':
        from modeling import gated_deltanet
        model = gated_deltanet.get_model(config_path=args.model_config, args=args)
    elif args.model_name == 'gla':
        from modeling import gla
        model = gla.get_model(config_path=args.model_config, args=args)
    elif args.model_name == 'mamba2':
        from modeling import mamba2
        model = mamba2.get_model(config_path=args.model_config, args=args)
    elif args.model_name == 'transformer':
        from modeling import transformer
        model = transformer.get_model(config_path=args.model_config, args=args)
    else:
        raise ValueError

    if args.target_head_num is not None:
        model.merge_head(args.target_head_num, is_init=not hasattr(args, "init_from"))
    if args.drop_prop is not None:
        model.drop_layers(args.drop_prop, args.drop_mode, args.freeze_rest)

    n_non_embed_param = get_num_params(model, non_embedding=True)
    n_param = get_num_params(model, non_embedding=False)
    accelerator.print("=========================================")
    accelerator.print(f"# parameters: {n_param:,}")
    accelerator.print(f"# parameters (non-embed): {n_non_embed_param:,}")
    accelerator.print("=========================================")

    model.to(accelerator.device, dtype=torch.bfloat16)
    return model


def get_accelerator(args: Args) -> Accelerator:
    assert args.grad_accum_steps is not None
    assert args.run_name is not None
    assert args.proj_name is not None
    # ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        project_dir=f"./accel_logs/{args.proj_name}",
        log_with=args.report_to,
        gradient_accumulation_steps=args.grad_accum_steps,
        # kwargs_handlers=[ddp_kwargs],
        # This argument allows us to step the LR scheduler manually.
        # This is needed because internally, accelerator expects the
        # LR scheduler to step more frequently when using multiple
        # processes or when using gradient accumulation (e.g.,
        # the warmup steps should be scaled by
        # num_processes * gradient_accumulation).
        step_scheduler_with_optimizer=False,
    )
    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y%m%d-%H%M%S")
    hps = args.as_dict()
    run_name = f"{args.run_name}_{formatted_time}"
    accelerator.init_trackers(
        project_name=args.proj_name,
        config=hps,
        init_kwargs={
            "wandb": {
                "name": run_name,
            },
            "swanlab": {
                "experiment_name": run_name,
            }
        },
    )
    return accelerator

from datasets import IterableDataset
from accelerate.data_loader import IterableDatasetShard

def get_dataloaders(
    args: Args, tokenizer: PreTrainedTokenizerBase, accelerator: Accelerator
) -> Tuple[DataLoader, DataLoader | None]:
    assert args.data_name is not None
    assert args.data_path is not None
    assert args.max_len is not None
    assert args.batch_size is not None

    train_ds = get_data(
        tokenizer=tokenizer,
        data_name=args.data_name,
        data_path=args.data_path,
        max_len=args.max_len,
        shift=False,
    )  # type: ignore

    # For distributed training, I think this makes sure different threads
    # fetch from different file.
    accelerator.print(f"Number of shards: {train_ds.n_shards}")  # type: ignore

    train_loader = DataLoader(
        train_ds,  # type: ignore
        batch_size=args.batch_size,
        collate_fn=default_data_collator,
        num_workers=6,
        pin_memory=True,
    )
    if args.validation_data_path is not None and args.validation_data_name is not None:
        val_ds = get_data(
            tokenizer=tokenizer,
            data_name=args.validation_data_name,
            data_path=args.validation_data_path,
            max_len=args.max_len,
            is_main_process=accelerator.is_main_process,
        )  # type: ignore
        # val_ds = split_dataset_by_node(val_ds, rank=int(os.environ["RANK"]), world_size=int(os.environ["WORLD_SIZE"]))
        val_loader = DataLoader(
            val_ds,  # type: ignore
            batch_size=args.batch_size * 4,
        )
    else:
        val_loader = None

    return train_loader, val_loader


def get_args() -> Args:
    args = Args().parse_args()
    if args.model_name in ['gated_deltanet', 'gated-deltanet', 'gla']:
        if args.compile == 1:
            print("Gated-DeltaNet and GLA does not support torch.compile, setting compile to 0...")
            args.compile = 0

    if args.grad_ckpt == 1 and args.compile == 1:
        print("Gradient Accumulation does not support torch.compile, setting compile to 0...")
        args.compile = 0
    return args


def main():
    # torch.set_float32_matmul_precision('high')  # wtf is this?
    torch.set_default_dtype(torch.bfloat16)
    args = get_args()
    load_train_config(args)
    set_seed(args.seed)
    accelerator = get_accelerator(args)

    accelerator.print("================ args ================")
    accelerator.print(args)
    accelerator.print("======================================")

    # Make output dir and dump args.
    output_dir = Path(args.output_dir, args.proj_name, args.run_name)
    if accelerator.is_main_process:
        output_dir.mkdir(exist_ok=True, parents=True)
        args.save(str(output_dir / "args.json"))

    # This is the actual batch size
    tokens_per_iter = (
        accelerator.num_processes
        * args.grad_accum_steps
        * args.batch_size
        * args.max_len
    )
    accelerator.print(f"Tokens per batch: {tokens_per_iter:,}")
    accelerator.print(f"Process: {accelerator.num_processes}")
    accelerator.print(f"Grad accum: {args.grad_accum_steps}")
    accelerator.print(f"Batch size: {args.batch_size}")
    accelerator.print(f"Max len: {args.max_len:,}")

    model: nn.Module = get_model(accelerator, args)
    tokenizer = AutoTokenizer.from_pretrained(args.tok_path)

    # check vocab size
    if model.config.vocab_size < len(tokenizer):
        accelerator.print(
            f"WARNING: vocab size of model ({model.config.vocab_size}) != vocab size of tokenizer ({len(tokenizer)}), adapting the model vocab size."
        )
        model.set_vocab_size(len(tokenizer))

    accelerator.print("================ model ================")
    accelerator.print(model)
    accelerator.print("======================================")

    # # Test generation
    # prompt = 'My name is'
    # with torch.no_grad():
    #     print("Testing generation...")
    #     print(f"Prompt: {prompt}")
    #     inputs = tokenizer(prompt, return_tensors="pt").to(accelerator.device)
    #     outputs = model.generate(**inputs, max_new_tokens=20)
    #     print(tokenizer.decode(outputs[0], skip_special_tokens=True))

    if accelerator.is_main_process:
        # save model config
        model_config = model.config
        model_config_path = output_dir / "model_config.json"
        with open(model_config_path, "w") as f:
            json.dump(model_config.to_dict(), f, indent=4, sort_keys=True)

    accelerator.print("Preparing optimizers...")
    optimizer = get_optimizer(
        model=model,
        lr=args.lr,
        weight_decay=args.weight_decay,
        beta1=args.beta1,
        beta2=args.beta2,
        verbose=accelerator.is_main_process,
    )

    # LR scheduler, we use a function instead of a class.
    args.lr_scheduler = "cosine" if args.lr_scheduler is None else args.lr_scheduler
    if args.lr_scheduler == "cosine":
        lr_scheduler = get_scheduler(
            name="cosine_with_min_lr",
            optimizer=optimizer,
            num_warmup_steps=args.n_warmup_steps,
            num_training_steps=args.n_train_steps,
            scheduler_specific_kwargs={"min_lr": args.min_lr if args.min_lr is not None else 0.0},
        )
    elif args.lr_scheduler == "constant":
        lr_scheduler = get_scheduler(
            name="constant",
            optimizer=optimizer,
            num_training_steps=args.n_train_steps,
        )
    elif args.lr_scheduler == "cosine_cycle":
        lr_scheduler = get_scheduler(
            name="cosine_with_restarts",
            optimizer=optimizer,
            num_warmup_steps=args.n_warmup_steps,
            num_training_steps=args.n_train_steps,
            scheduler_specific_kwargs={"num_cycles": args.num_cycles}  # 👈 设置周期数
        )
    else:
        raise ValueError(f"Unknown lr_scheduler: {args.lr_scheduler}")

    # Compile with PyTorch 2.0, very powerful
    if bool(args.compile):
        assert args.device != "mps", "torch.compile not supported on MPS"
        accelerator.print("compiling the model... (takes a ~minute)")
        # unoptimized_model = model
        model = torch.compile(model)  # requires PyTorch 2.0  # type: ignore

    accelerator.print("Preparing dataloaders...")
    train_loader, val_loader = get_dataloaders(
        args=args,
        tokenizer=tokenizer,
        accelerator=accelerator,
    )

    accelerator.print("Preparing LMTrainer...")
    trainer = LMTrainer(
        args=args,
        output_dir=output_dir,
        accelerator=accelerator,
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        state_passing=args.state_passing if args.state_passing is not None else False,
    )

    accelerator.print("===== Start training =====")
    trainer.train()


if __name__ == '__main__':
    main()
