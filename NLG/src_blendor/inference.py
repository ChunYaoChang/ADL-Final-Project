from argparse import ArgumentParser
from pathlib import Path
from util import (
    load_data,
    construct_conv,
)
from transformers import (
    BlenderbotTokenizer, 
    BlenderbotForConditionalGeneration,
    AutoTokenizer,
)
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import torch
import json
import random
import torch.nn.functional as F

def main(
    data_dir_path: Path, 
    begin_model_path: Path,
    end_model_path: Path,
    output_file_path: Path,
    model_checkpoint: str,
    cls_model_checkpoint: str,
    batch_size: int,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    dev_raw_data = load_data(data_dir_path)
    model = BlenderbotForConditionalGeneration.from_pretrained(model_checkpoint).to(device)
    tokenizer = BlenderbotTokenizer.from_pretrained(model_checkpoint)
    begin_model = torch.load(begin_model_path).to(device)
    end_model = torch.load(end_model_path).to(device)
    cls_tokenizer = AutoTokenizer.from_pretrained(cls_model_checkpoint)
    utterances =  [[(data['turns'][idx]['utterance'], data['turns'][idx+1]['utterance']) for idx in range(0,len(data['turns']),2)] for data in dev_raw_data]
    dialogues_id = [data['dialogue_id'] for data in dev_raw_data]
    result = []
    count = 0
    total = 0
    # Enable these patterm to appear in the chit-chat response
    enable_patterm = ['you are welcome', 'you are very welcome', 'you\'re welcome']
    # Ban probability in percentage
    ban_probability = 95
    # Generate reponse from BlendorBot
    for i, (uttr, dia_id) in enumerate(zip(utterances, dialogues_id)):
        dia_result = {}
        dia_result['dialogue_id'] = dia_id
        turns = []
        for (user_uttr, sys_uttr) in tqdm(uttr, ncols=0, desc=f'Dialogue {len(dialogues_id)}/{i+1}:'):
            posible_ans = []
            # Generate chit-chat response at the beginning 
            inputs = tokenizer(user_uttr)
            inputs = {'input_ids': torch.LongTensor(inputs['input_ids']).unsqueeze(0).to(device), 'attention_mask': torch.LongTensor(inputs['attention_mask']).unsqueeze(0).to(device)}
            reply_ids = model.generate(
                **inputs,
                max_length=20,
                min_length=10,
                num_beams=20,
                no_repeat_ngram_size=2,
                num_return_sequences=20, 
            )
            candidates = tokenizer.batch_decode(reply_ids, skip_special_tokens=True)
            sys_chitchat = [x.strip() for x in candidates]
            # Enable some patterms to appear
            candidates = [] 
            for chitchat in sys_chitchat:
                flag = True
                for patterm in enable_patterm:
                    if chitchat.lower().find(patterm) >= 0:
                        if random.randint(1,100) <= ban_probability:
                            flag = False
                            break 
                if flag:
                    candidates.append(chitchat)
            input_ids, att_masks = [],[]
            for x in candidates:
                input_id, att_mask = construct_conv([user_uttr,x,sys_uttr] ,cls_tokenizer)
                input_ids.append(torch.LongTensor(input_id))
                att_masks.append(torch.LongTensor(att_mask))
            if len(input_ids) != 0:
                input_ids = pad_sequence(input_ids, batch_first=True, padding_value=cls_tokenizer.pad_token_id)
                att_masks = pad_sequence(att_masks, batch_first=True, padding_value=0)
                inputs = {'input_ids': input_ids.to(device), 'attention_mask': att_masks.to(device)}
                output = begin_model(**inputs)
                prob = F.softmax(output['logits'], dim=1)
                winner_index = torch.argmax(prob[:,1])
                if prob[winner_index,1] > 0.5:
                    posible_ans.append({'chit-chat': candidates[winner_index], 'posibility': prob[winner_index,1], 'type': 'beginning'})
            # Generate chit-chat response at the end 
            inputs = tokenizer([user_uttr, sys_uttr])
            input_ids = [x for input_id in inputs['input_ids'] for x in input_id]
            att_masks = [x for att_mask in inputs['attention_mask'] for x in att_mask]
            inputs = {'input_ids': torch.LongTensor(input_ids).unsqueeze(0).to(device), 'attention_mask': torch.LongTensor(att_masks).unsqueeze(0).to(device)}
            reply_ids = model.generate(
                **inputs,
                max_length=20,
                min_length=10,
                num_beams=20,
                no_repeat_ngram_size=2,
                num_return_sequences=20, 
            )
            candidates = tokenizer.batch_decode(reply_ids, skip_special_tokens=True)
            sys_chitchat = [x.strip() for x in candidates]
            # Enable some patterms to appear
            candidates = [] 
            for chitchat in sys_chitchat:
                flag = True
                for patterm in enable_patterm:
                    if chitchat.lower().find(patterm) >= 0:
                        if random.randint(1,100) <= ban_probability:
                            flag = False
                            break 
                if flag:
                    candidates.append(chitchat)
            input_ids, att_masks = [],[]
            for x in candidates:
                input_id, att_mask = construct_conv([user_uttr,sys_uttr,x] ,cls_tokenizer)
                input_ids.append(torch.LongTensor(input_id))
                att_masks.append(torch.LongTensor(att_mask))
            if len(input_ids) != 0:
                input_ids = pad_sequence(input_ids, batch_first=True, padding_value=cls_tokenizer.pad_token_id)
                att_masks = pad_sequence(att_masks, batch_first=True, padding_value=0)
                inputs = {'input_ids': input_ids.to(device), 'attention_mask': att_masks.to(device)}
                output = end_model(**inputs)
                prob = F.softmax(output['logits'], dim=1)
                winner_index = torch.argmax(prob[:,1])
                if prob[winner_index,1] > 0.5:
                    posible_ans.append({'chit-chat': candidates[winner_index], 'posibility': prob[winner_index,1].item(), 'type': 'end'})
            if len(posible_ans) != 0:
                posible_ans.sort(key = lambda x: x['posibility'])
                best_ans = posible_ans[0]
                if best_ans['type'] == 'beginning':
                    turns.append({'speaker': 'USER', 'utterance': user_uttr})
                    turns.append({'speaker': 'SYSTEM', 'utterance': best_ans['chit-chat']+'. '+sys_uttr, 'modified': True, 'additional_chitchat': best_ans['chit-chat']})
                    #print({'speaker': 'USER', 'utterance': user_uttr})
                    #print({'speaker': 'SYSTEM', 'utterance': best_ans['chit-chat']+' '+sys_uttr, 'modified': True, 'additional_chitchat': best_ans['chit-chat']})
                    #print('----------------------------------------------------------------')
                elif best_ans['type'] == 'end':
                    turns.append({'speaker': 'USER', 'utterance': user_uttr})
                    turns.append({'speaker': 'SYSTEM', 'utterance': sys_uttr+'. '+best_ans['chit-chat'], 'modified': True, 'additional_chitchat': best_ans['chit-chat']})
                    #print({'speaker': 'USER', 'utterance': user_uttr})
                    #print({'speaker': 'SYSTEM', 'utterance': sys_uttr+' '+best_ans['chit-chat'], 'modified': True, 'additional_chitchat': best_ans['chit-chat']})
                    #print('----------------------------------------------------------------')
                count += 1
            elif len(posible_ans) == 0:
                turns.append({'speaker': 'USER', 'utterance': user_uttr})
                turns.append({'speaker': 'SYSTEM', 'utterance': sys_uttr, 'modified': False})
            total += 1
        dia_result['turns'] = turns
        result.append(dia_result)
    print(f'Succesful rate: {count/total}')
    with open(output_file_path, 'w') as fp:
        json.dump(result, fp, indent=6)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('data_dir_path', type=Path)
    parser.add_argument('begin_model_path', type=Path)
    parser.add_argument('end_model_path', type=Path)
    parser.add_argument('output_file_path', type=Path)
    parser.add_argument('--model_checkpoint', type=str, default='facebook/blenderbot-400M-distill')
    parser.add_argument('--cls_model_checkpoint', type=str, default='distilbert-base-uncased')
    parser.add_argument('--batch_size', type=int, default=4)
    args = parser.parse_args()
    main(**vars(args))