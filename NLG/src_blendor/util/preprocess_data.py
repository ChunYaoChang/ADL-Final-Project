import random

def construct_conv(row, tokenizer, eos = True):
    conv = tokenizer(row)
    # Delete the sentence start token from second and third sentence input_ids
    conv['input_ids'][1], conv['input_ids'][2] = conv['input_ids'][1][1:], conv['input_ids'][2][1:]
    conv['attention_mask'][1], conv['attention_mask'][2] = conv['attention_mask'][1][1:], conv['attention_mask'][2][1:]
    input_ids = [ids for input_id in conv['input_ids'] for ids in input_id]
    attention_masks = [att_mask for att_masks in conv['attention_mask'] for att_mask in att_masks]
    return input_ids, attention_masks

def split_data(utter, train_type):
    split_utter_list, labels = [], []
    for i in range(1,len(utter),2):
        candidate = utter[i-1:i+1]
        result, label = add_chitchat(candidate, train_type)
        if len(result) != 0:
            split_utter_list += result
            labels += label
    return split_utter_list, labels

def add_chitchat(candidates, train_type):
    assert candidates[0]['speaker'] == 'USER' and candidates[1]['speaker'] == 'SYSTEM'
    results, labels = [], []
    user_uttr = candidates[0]['utterance']
    system_res = candidates[1]
    system_uttr = system_res['utterance'] 
    if 'beginning' not in system_res.keys() or 'end' not in system_res.keys():
        return ([], None)
    if train_type == 'beginning':
        begin_chitchat = [(begin['candidate'], begin['label']) for begin in system_res['beginning']]
        for (chat_uttr, label) in begin_chitchat:
            results.append([user_uttr, chat_uttr, system_uttr])
            labels.append(0 if label=='bad' else 1)
    elif train_type == 'end':
        end_chitchat = [(end['candidate'], end['label']) for end in system_res['end']]
        for (chat_uttr, label) in end_chitchat:
            results.append([user_uttr, system_uttr, chat_uttr])
            labels.append(0 if label=='bad' else 1)
    return results, labels

def preprocess_data(data_list, tokenizer, train_type):
    utterances =  [[sentence for sentence in data['turns']] for data in data_list]
    splited_utterances, labels = [], []
    for utter in utterances:
        result, label = split_data(utter, train_type)
        splited_utterances += result 
        labels += label 
    input_ids, att_masks = [], []
    for utter in splited_utterances:
        input_id, att_mask = construct_conv(utter, tokenizer)
        input_ids.append(input_id)
        att_masks.append(att_mask)
    print(len(input_ids))
    return {'input_ids': input_ids, 'attention_mask': att_masks, 'labels' :labels}