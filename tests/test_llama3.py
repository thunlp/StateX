import sys
import torch
from transformers import AutoTokenizer

sys.path.append("..")
from modeling.llama3 import LlamaForCausalLM


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pretrained_path = '/home/test/test07/ckpts/llama3.2/llama3.2-1b-instruct'

    print(f"Loading model and tokenizer from {pretrained_path}")
    model = LlamaForCausalLM.from_pretrained(pretrained_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_path)
    print(model.config)

    prompt = '''input: France
output: Paris

input: Germany
output: Berlin

input: China
output: Beijing

input: Norway
output:'''

    ctx = '''Read this and answer the question.

Yingfa Chen is a 1st year PhD at Tsinghua University, current doing research in natural language processing. He was born in Norway but he is ethnically Chinese and his mother tongue is Cantonese Chinese. His hometown is called Lillesand, a small coastal town in the southern part of Norway, it has a population around 10,000 people. He came to Beijing in 2018 for a Bachelor's degree at Tsinghua, and stayed here until now.'''

    query = 'Question: It is now October, 2024, how many years has Yingfa been in Beijing?\nAnswer:'
    prompt = f'{ctx}\n\n{query}'
    print("===== prompt =====")
    print(prompt)
    print('==================')
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    # print("Input IDs:", input_ids)
    with torch.no_grad():
        output = model.generate(input_ids, max_new_tokens=40)
    # print("Output:", output)
    output_text = tokenizer.decode(output[0], skip_special_tokens=True)[len(prompt):]
    print("Output text:", output_text)


if __name__ == "__main__":
    main()
