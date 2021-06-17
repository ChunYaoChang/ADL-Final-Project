import random

def construct_conv(row, tokenizer, eos = True):
    flatten = lambda l: [item for sublist in l for item in sublist]
    conv = list(reversed([tokenizer.encode(x) + [tokenizer.eos_token_id] for x in row]))
    conv = flatten(conv)
    return conv

def split_data(utter, n):
    split_utter_list = []
    for i in range(n,len(utter),2):
        split_utter_list.append(utter[i-n:i+1])
    return split_utter_list

def add_chitchat(sentence):
    utter = sentence['utterance']
    if sentence['speaker'] == 'USER':
        return utter
    if 'beginning' not in sentence.keys() or 'end' not in sentence.keys():
        return utter 
    begin_chitchat = [candidate['candidate'] for candidate in sentence['beginning'] if candidate['label'] == 'good']
    end_chitchat = [candidate['candidate'] for candidate in sentence['end'] if candidate['label'] == 'good']
    if(len(begin_chitchat) != 0 and len(end_chitchat) != 0):
        begin_or_end = random.randint(0,1)
        # 0 for end
        if(begin_or_end == 0):
            index = random.randint(0,len(end_chitchat)-1)
            return utter+' '+end_chitchat[index]
        # 1 for begin
        else:
            index = random.randint(0,len(begin_chitchat)-1)
            return begin_chitchat[index]+' '+utter
    elif(len(begin_chitchat) != 0):
        index = random.randint(0,len(begin_chitchat)-1)
        return begin_chitchat[index]+' '+utter
    elif(len(end_chitchat) != 0):
        index = random.randint(0,len(end_chitchat)-1)
        return utter+' '+end_chitchat[index]
    else: return utter  

def preprocess_data(data_list, tokenizer, n):
    utterances =  [[add_chitchat(sentence) for sentence in data['turns']] for data in data_list]
    splited_utterances = []
    for utter in utterances:
        splited_utterances += split_data(utter, n)
    return [construct_conv(utter, tokenizer) for utter in splited_utterances]