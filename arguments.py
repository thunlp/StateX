
from tap import Tap


class Args(Tap):
    # pretrained_path: str | None = None  # Will load model weights from this path.
    resume_path: str | None = None
    '''
    When specified, will load training states (model weights, optimizer states, etc.) from this path.
    '''
    resume_step: int | None = None
    skip_step: int | None = None
    state_passing: bool | None = None

    extend_dim: int | None = None
    target_head_num: int | None = None
    merged_head_num: int | None = None
    drop_prop: int | None = None
    drop_mode: str | None = None
    droped_prop: int | None = None
    freeze_rest: bool | None = False

    model_name: str = 'gpt'
    model_config: str | None = None
    train_config: str | None = None

    # Tokenizer
    tok_path: str = "./tokenizer/llama-tokenizer"

    output_dir: str = "output"
    eval_interval: int = None
    log_interval: int = 1
    save_interval: int = None
    eval_iters: int = None
    eval_only: int = 0  # if not 0, script exits right after the first eval
    always_save_checkpoint: int = 1  # if not 0, always save a checkpoint after each eval
    init_from: str = 'scratch'  # 'scratch' or 'resume' or 'gpt2*'

    # wandb logging
    report_to: str = "swanlab"  # disabled by default
    proj_name: str = "my-project"
    run_name: str = "test"
    '''The code will append str(time.time()) to the run name.'''

    # Data
    n_workers: int = 1  # For DataLoader
    data_name: str = None
    data_path: str = None
    validation_data_path: str = None
    validation_data_name: str = None
    n_eval_batches: int = None
    grad_accum_steps: int = None
    '''per-gpu gradient accumulation steps'''
    batch_size: int = None
    '''If grad_accum_steps > 1, this is the per-gpu batch size.'''
    max_len: int = None

    # model
    vocab_size: int = None
    tie_emb: int = None
    '''
    Should be multiple of 64 for better speed.
    '''
    n_layer: int = None
    n_head: int = None
    d_model: int = None
    dropout: float = None
    """for pretraining 0 is good, for finetuning try 0.1+"""

    bias: int = None
    """0 to turn off bias. Do we use bias inside LayerNorm and Linear layers?"""

    # Attention
    n_head: int = None
    n_kv_head: int = None
    dim_k: int = None
    dim_v: int = None
    att_q_norm: int = None
    att_k_norm: int = None
    att_tie_kv: int = None
    head_mixing: int = 1
    attn_impl: str = None

    # RoPE
    use_rope: int = None
    rope_theta: float = None

    # FFN
    ffn_per_layer: int = None
    ffn_act_fn: str = None
    ffn_tie_kv: int = None
    ffn_is_gated: int = None
    ffn_d_mid: int = None
    mhf_q_norm: int = None
    mhf_output_norm: int = None
    use_mhf: int = None
    mhf_use_q_proj: int = None
    mhf_use_o_proj: int = None
    mhf_n_heads: int = None
    mhf_dim_k: int = None
    mhf_dim_v: int = None
    mhf_head_mixing: int = None

    # adamw optimizer
    lr_scheduler: str | None = None
    lr: float = None  # max learning rate
    weight_decay: float = None
    beta1: float = None
    beta2: float = None
    grad_clip: float = None  # clip gradients at this value, or disable if == 0.0
    min_lr: float = None  # minimum learning rate, should be ~= lr/10 according to MiniCPM
    num_cycles: int = None

    # learning rate decay settings
    n_warmup_steps: int = None
    n_train_steps: int = None
    n_drop_steps: int = None
    '''
    Should be 10% of n_train_iters according to MiniCPM, but we use 20% by default from:
    https://arxiv.org/pdf/2405.18392
    '''

    # DDP settings (only used by train_torch.py)
    backend: str = "nccl"  # 'nccl', 'gloo', etc.

    # system
    device: str = "cuda"
    """examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks"""
    machine_rank: int | None = None
    n_machines: int | None = None
    master_ip: str | None = None
    master_port: int | None = None
    use_deepspeed: int = 0
    n_gpus: int | None = None

    dtype: str = None
    """
    float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler.
    """

    compile: int = 1
    '''
    If 1, the code will use PyTorch 2.0 to compile the model to be faster.
    '''
    seed: int = 0
    grad_ckpt: int = 0  # Whether to use gradient checkpointing
    use_cce_loss: int = 0  # Whether to use Cut-CE loss (to reduce logits memory cost)
    cce_loss_impl: str = 'torch_compile'

    exp_group: str = None
    comment: str = None

    trust_remote_code: bool = False


if __name__ == "__main__":
    args = Args().parse_args()
    print(args)
