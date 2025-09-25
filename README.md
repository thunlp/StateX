# StateX

This is a simple implementation of the StateX. Run the scripts to reproduce the results in the paper.

## Requirements

`pip install -r requirements.txt` and 8 GPUs with at least 80GB memory each.

Download the required pretrained models (GLA-1.3B, Mamba2-1.3B) by running

```bash
cd huggingface
python download.py
```

You should download the dataset `AlppAI/SlimPajama-chunked` to the directory `/home/data/slimpj-chunked` or modify the path in `data/slimpj.py` and `configs/training/drop/10b_bsz-512k_64k_cos_3e-4_slimpj.json`.

## Training

To train a 1.3B model with StateX, run `bash scripts/gla_StateX.sh` and `bash scripts/mamba2_StateX.sh`. To train the LPT version, run `bash scripts/gla_LPT.sh` and `bash scripts/mamba2_LPT.sh`.

The checkpoints and logs will be saved in `./output/{proj_name}/{model_name}/{run_name}`. The `args.json` and `model_config.json` files are saved in the same folder. The checkpoint will be saved in `ckpt_{step}` folder in safetensors format. You can use safetensors to load the model.