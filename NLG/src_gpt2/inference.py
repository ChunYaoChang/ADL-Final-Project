from argparse import ArgumentParser
from pathlib import Path
from util import (
    load_data,
    preprocess_data,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
import torch





def main(
    data_dir_path: Path, 
    model_path: Path,
    model_checkpoint: str,
    num_sen_per_input: int,
    batch_size: int,
):
    if(num_sen_per_input % 2 == 0):
        print('num_sen_per_input should be an odd number.')
        exit(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    train_raw_data = load_data(data_dir_path/'train')
    dev_raw_data = load_data(data_dir_path/'dev')
    train_data = preprocess_data(train_raw_data, tokenizer, num_sen_per_input)
    dev_data = preprocess_data(dev_raw_data, tokenizer, num_sen_per_input)
    model = AutoModelForCausalLM.from_config(torch.load(model_path))
    print(len(train_data), len(dev_data))
    





if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('data_dir_path', type=Path)
    parser.add_argument('model_path', type=Path)
    parser.add_argument('--model_checkpoint', type=str, default='microsoft/DialoGPT-small')
    ## num_sen_per_input should be an odd number
    parser.add_argument('--num_sen_per_input', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=4)
    args = parser.parse_args()
    main(**vars(args))