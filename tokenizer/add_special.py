from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("./llama-tokenizer")
special_tokens = {
    "additional_special_tokens": ["<|PRE|>", "<|MID|>", "<|SUF|>"]
}
tokenizer.add_special_tokens(special_tokens)
tokenizer.save_pretrained("./llama-tokenizer-FIM")
