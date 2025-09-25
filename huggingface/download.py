import os

download_list = [
    "fla-hub/gla-1.3B-100B",
    "rokyang/mamba2-1.3b-hf",
]

for model_name in download_list:
    print(f"Downloading {model_name}...")
    os.system(f'./hfd.sh {model_name}')
