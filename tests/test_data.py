'''
This will build the dataset instance, then load the first batch and print it out.
'''
import sys
from transformers import AutoTokenizer
from datasets import Dataset

sys.path.append("../")

from data import get_data


tokenizer = AutoTokenizer.from_pretrained("../tokenizer/gpt2-tokenizer")
data_name = "slimpj_200k"
data_path = "../data/slimpj-200k/data"


tokenized_data = get_data(
    tokenizer=tokenizer,
    data_name=data_name,
    data_path=data_path,
)

# training loop
train_ds: Dataset = tokenized_data["train"]

for eg in train_ds:
    print(eg)
    break
# val_ds: Dataset = tokenized_data["validation"]
