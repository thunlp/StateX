from pathlib import Path
from typing import Union, Optional, List, Dict

from torch import Tensor
import torch
from datasets import load_dataset


def build_dataset(
    tokenizer,
    data_dir: Union[str, Path] = "/home/test/test07/data/slimpj-chunked",
    streaming: bool = True,
    n_workers: int = 8,
    overwrite_cache: bool = False,
    token_ids_only: bool = True,
    max_len: int = 512,
    eos_token_id: Optional[int] = None,
    do_log: bool = False,
    shift: bool = True,
    **kwargs,
):
    """
    Returns an iterable of batches of token IDs.

    This will use `load_dataset` from the HuggingFace Datasets library to load the
    data from `data_dir`, tokenize each example, concatenate the input IDs, add an
    EOS token ID at the end of each sequence, then split into chunks of `max_len`
    tokens, and return a tensor of (batch_size, max_len).
    """
    if eos_token_id is None:
        eos_token_id = tokenizer.eos_token_id

    # Get all data_files
    data_dir = Path(data_dir).absolute()
    all_data_files = []
    for chunk_dir in sorted(data_dir.glob('chunk*')):
        all_data_files += sorted(chunk_dir.glob("*.jsonl"))
    all_data_files = [str(x) for x in all_data_files]

    # print(all_data_files)
    if do_log:
        print(f">> Loading data from {str(data_dir)}, {streaming = }")
    raw_dataset = load_dataset(
        str(data_dir),
        streaming=streaming,
        split='train',
    )

    text_column_name = 'text'
    col_names = ['text']

    # Tokenize in streaming mode
    def process_fn(examples: dict) -> Dict[str, Tensor]:
        '''
        A process function to use with `Dataset.map`. It tokenizes
        texts in the batch, concatenate them, and split into chunks
        with `max_len` tokens (discarding the last chunk if
        incomplete).
        '''
        texts: List[str] = examples[text_column_name]
        encodings = tokenizer(texts, max_length=10 ** 6, truncation=True)
        batch_ids: List[List[int]] = encodings["input_ids"]
        for ids in batch_ids:
            ids += [eos_token_id]
        concat_ids = sum(batch_ids, [])  # Concatenate into one long ids
        total_len = len(concat_ids)

        input_ids: List[List[int]] = []
        labels: List[List[int]] = []
        chunk_len = max_len

        # Rounded down to multiple of chunk_len.
        # So the last remainder chunk is discarded.
        total_len = total_len // chunk_len * chunk_len

        for i in range(0, total_len, chunk_len):
            this_chunk: List[int] = concat_ids[i : i + chunk_len]
            if shift:
                # Next token prediction
                input_ids.append(this_chunk[:-1])
                labels.append(this_chunk[1:])
            else:
                # input IDs == labels
                input_ids.append(this_chunk)
                labels.append(this_chunk)

        input_ids_t = torch.tensor(input_ids, dtype=torch.long)
        labels_t = torch.tensor(labels, dtype=torch.long)

        batch = {
            "input_ids": input_ids_t,
            "labels": labels_t,
        }
        return batch

    if streaming:
        # Why can't we load the dataset with multiple workers when
        # in streaming mode?
        tokenized_dataset = raw_dataset.map(
            process_fn,
            batched=True,
            remove_columns=col_names if token_ids_only else [],
        )
    else:
        tokenized_dataset = raw_dataset.map(
            process_fn,
            batched=True,
            num_proc=n_workers,  # type: ignore
            remove_columns=col_names if token_ids_only else [],
            load_from_cache_file=not overwrite_cache,  # type: ignore
            desc="Running tokenizer on dataset",  # type: ignore
        )
    return tokenized_dataset
