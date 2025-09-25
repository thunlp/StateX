# -*- coding: utf-8 -*-

from arguments import Args

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from .config import Mamba2Config
from .model import Mamba2ForCausalLM, Mamba2Model

AutoConfig.register(Mamba2Config.model_type, Mamba2Config, exist_ok=True)
AutoModel.register(Mamba2Config, Mamba2Model, exist_ok=True)
AutoModelForCausalLM.register(Mamba2Config, Mamba2ForCausalLM, exist_ok=True)

from safetensors.torch import load_file

__all__ = ['Mamba2Config', 'Mamba2ForCausalLM', 'Mamba2Model']

def get_config(
    path: str | None = None,
    args: Args | None = None,
) -> Mamba2Config:
    """
    Load from `args.model_config` if it exists, then override the
    attributes with command line arguments.
    """
    if path is not None:
        model_config = Mamba2Config.from_json_file(path)
    else:
        model_config = Mamba2Config()  # use default values

    # Override values in the config from the command line arguments
    if args is not None:
        for k, v in args.as_dict().items():
            if hasattr(model_config, k) and v is not None:
                setattr(model_config, k, v)
    return model_config

def get_model(config_path:str, args: Args) -> Mamba2ForCausalLM:
    model_config = get_config(
        path=config_path,
        args=args,
    )
    model = Mamba2ForCausalLM(model_config)
    if args.init_from is not None and args.init_from != 'scratch':
        if args.merged_head_num is not None:
            print(f'loading model with merged {args.merged_head_num} heads')
            model.merge_head(args.merged_head_num)
        if args.droped_prop is not None:
            print(f'loading model with droped prop {args.droped_prop}')
            model.drop_layers(args.droped_prop, args.drop_mode)

        print(f"Loading model parameters from: {args.init_from}")
        state_dict = load_file(args.init_from)
        state_dict = {key.replace('_orig_mod.', ''): val for key, val in state_dict.items()}
        # print(list(state_dict.keys()), file=sys.stderr)
        
        if 'lm_head.weight' not in state_dict:
            state_dict['lm_head.weight'] = state_dict['backbone.embeddings.weight']

        if model_config.layer_sharing is not None:
            model.model.layers = model.model.layers

        # Convert params dtype
        for key, value in model.named_parameters():
            if key in state_dict and value.dtype != state_dict[key].dtype:
                value.data = value.data.to(state_dict[key].dtype)
        model.load_state_dict(state_dict, strict=True)

        if model_config.layer_sharing is not None:
            if model_config.layer_sharing == 'mirror':
                print("Using mirror layer sharing")
                for i in range(model_config.num_hidden_layers):
                    model.model.layers[-i] = model.model.layers[i]
            elif model_config.layer_sharing == 'double':
                print("Using double layer sharing")
                for i in range(model_config.num_hidden_layers):
                    model.model.layers[2 * i + 1] = model.model.layers[2 * i]
            else:
                assert False, f"Unknown layer sharing type: {config.layer_sharing}"

    if args.grad_ckpt:
        model.gradient_checkpointing_enable()

    if args.use_cce_loss:
        model.config.fuse_cross_entropy = True
        if args.cce_loss_impl is not None:
            model.config.cce_loss_impl = args.cce_loss_impl

    return model
