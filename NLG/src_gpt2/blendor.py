from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration
from argparse import ArgumentParser
from pathlib import Path
from util import (
    load_data,
)
import torch

    
def construct_conv(row, tokenizer, eos = True):
    flatten = lambda l: [item for sublist in l for item in sublist]
    
    conv = flatten(conv)
    return conv

def main(
    data_dir_path: Path, 
    model_checkpoint: str,
    num_sen_per_input: int,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    train_raw_data = load_data(data_dir_path/'train')
    model = BlenderbotForConditionalGeneration.from_pretrained(model_checkpoint)
    tokenizer = BlenderbotTokenizer.from_pretrained(model_checkpoint)
    utterances =  [[sentence['utterance'] for sentence in data['turns']] for data in train_raw_data][0]
    first_data = utterances[0:4]
    first_input = tokenizer(first_data)
    first_input_ids = [input_id for input_ids in first_input['input_ids'] for input_id in input_ids]
    first_attention_masks = [att_mask  for attention_masks in first_input['attention_mask'] for att_mask in attention_masks] 
    inputs = {'input_ids': torch.LongTensor(first_input_ids).unsqueeze(0), 'attention_mask': torch.LongTensor(first_attention_masks).unsqueeze(0)}
    reply_ids = model.generate(**inputs)
    print('User: ', first_data)
    print('Chit-Chat Response: ',tokenizer.batch_decode(reply_ids))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('data_dir_path', type=Path)
    parser.add_argument('--model_checkpoint', type=str, default='facebook/blenderbot-400M-distill')
    parser.add_argument('--num_sen_per_input', type=int, default=5)
    args = parser.parse_args()
    main(**vars(args))