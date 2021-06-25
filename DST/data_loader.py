# Copyright (c) Facebook, Inc. and its affiliates

import json
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
import ast
from tqdm import tqdm
import os
import random
from functools import partial
from utils.fix_label import fix_general_label_error
from collections import OrderedDict
EXPERIMENT_DOMAINS = ["Banks_1", "Buses_1", "Buses_2", "Calendar_1", "Events_1", "Events_2", "Flights_1", "Flights_2", "Homes_1", "Hotels_1", "Hotels_2", "Hotels_3", "Media_1", "Movies_1", "Music_1", "Music_2", "RentalCars_1", "RentalCars_2", "Restaurants_1", "RideSharing_1", "RideSharing_2", "Services_1", "Services_2", "Services_3", "Travel_1", "Weather_1", "Alarm_1", "Banks_2", "Flights_3", "Hotels_4", "Media_2", "Movies_2", "Restaurants_2", "Services_4", "Buses_3", "Events_3", "Flights_4", "Homes_2", "Media_3", "Messaging_1", "Movies_3", "Music_3", "Payment_1", "RentalCars_3", "Trains_1", "hotel", "train", "restaurant", "attraction", "taxi", 'hospital', 'bus', 'police']
random.seed(577)
EXPERIMENT_DOMAINS_lower = [domain.lower() for domain in EXPERIMENT_DOMAINS]
HISTORY_MAX_LEN = 450
GPT_MAX_LEN = 1024

class DSTDataset(Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""
    def __init__(self, data, args):
        """Reads source and target sequences from txt files."""
        self.data = data
        self.args = args

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        item_info = self.data[index]
        if self.args["slot_lang"] == "value":
            random.shuffle(item_info["value_list"])
            item_info["intput_text"] += " is " + " or ".join(item_info["value_list"]) + " or none?"
        return item_info

    def __len__(self):
        return len(self.data)



def read_data(args, path_name, SLOTS, tokenizer, description, dataset=None):
    slot_lang_list = ["description_human", "rule_description", "value_description", "rule2", "rule3"]
    print(("Reading all files from {}".format(path_name)))
    data = []
    domain_counter = {}
    # read files
    with open(path_name) as f:
        dials = json.load(f)

        if dataset=="train" and args["fewshot"]>0:
            random.Random(args["seed"]).shuffle(dials)
            dials = dials[:int(len(dials)*args["fewshot"])]
        #elif dataset=="dev":
        #    random.Random(args["seed"]).shuffle(dials)
        #    dials = dials[:int(len(dials)*0.3)]

        for dial_dict in dials:
            dialog_history = ""

            # Counting domains
            for domain in dial_dict["domains"]:
                if domain not in EXPERIMENT_DOMAINS_lower:
                    continue
                if domain not in domain_counter.keys():
                    domain_counter[domain] = 0
                domain_counter[domain] += 1

            # Unseen domain setting
            if args["only_domain"] != "none" and args["only_domain"] not in dial_dict["domains"]:
                continue
            if (args["except_domain"] != "none" and dataset == "test" and args["except_domain"] not in dial_dict["domains"]) or \
            (args["except_domain"] != "none" and dataset != "test" and [args["except_domain"]] == dial_dict["domains"]):
                continue

            # Reading data
            for ti, turn in enumerate(dial_dict["turns"]):
                turn_id = ti

                # accumulate dialogue utterances
                dialog_history +=  (" System: " + turn["system"] + " User: " + turn["user"])

                if args["fix_label"]:
                    slot_values = fix_general_label_error(turn["state"]["slot_values"],SLOTS)
                else:
                    slot_values = turn["state"]["slot_values"]
                # input: dialogue history + slot
                # output: value

                # Generate domain-dependent slot list
                slot_temp = SLOTS
                if dataset == "train" or dataset == "dev":
                    if args["except_domain"] != "none":
                        slot_temp = [k for k in SLOTS if args["except_domain"] not in k]
                        slot_values = OrderedDict([(k, v) for k, v in slot_values.items() if args["except_domain"] not in k])
                    elif args["only_domain"] != "none":
                        slot_temp = [k for k in SLOTS if args["only_domain"] in k]
                        slot_values = OrderedDict([(k, v) for k, v in slot_values.items() if args["only_domain"] in k])
                else:
                    if args["except_domain"] != "none":
                        slot_temp = [k for k in SLOTS if args["except_domain"] in k]
                        slot_values = OrderedDict([(k, v) for k, v in slot_values.items() if args["except_domain"] in k])
                    elif args["only_domain"] != "none":
                        slot_temp = [k for k in SLOTS if args["only_domain"] in k]
                        slot_values = OrderedDict([(k, v) for k, v in slot_values.items() if args["only_domain"] in k])


                turn_belief_list = [str(k)+'-'+str(v) for k,v in slot_values.items()]

                # baseline gpt have different preprocessing, e.g., output: (slot1-value1, slot2-value2, slot3-value3, ...)
                if "gpt" in args["model_name"]:
                    turn_slots = []
                    turn_slot_values = []
                    if len(dialog_history.split())>800:
                        continue
                    for slot in slot_temp:
                        # skip unrelevant slots for out of domain setting
                        if args["except_domain"] != "none" and dataset !="test":
                            if slot.split("-")[0] not in dial_dict["domains"]:
                                continue
                        input_text = dialog_history + f" {tokenizer.sep_token} {slot}" + " " + tokenizer.bos_token
                        output_text = input_text+ " " + turn["state"]["slot_values"].get(slot, 'none').strip() + " " + tokenizer.eos_token
                        slot_text = slot
                        value_text = turn["state"]["slot_values"].get(slot, 'none').strip()

                        data_detail = {
                            "ID":dial_dict["dial_id"],
                            "domains":dial_dict["domains"],
                            "turn_id":turn_id,
                            "dialog_history":dialog_history,
                            "turn_belief":turn_belief_list,
                            "intput_text":input_text,
                            "output_text":output_text,
                            "slot_text":slot_text,
                            "value_text":value_text
                            }
                        data.append(data_detail)

                else:
                    for slot in slot_temp:
                        if slot.split("-")[0] not in dial_dict["domains"]:
                            continue
                        # skip unrelevant slots for out of domain setting
                        if args["except_domain"] != "none" and dataset !="test":
                            if slot.split("-")[0] not in dial_dict["domains"]:
                                continue

                        output_text = slot_values.get(slot, 'none').strip() + f" {tokenizer.eos_token}"
                        slot_text = slot
                        value_text = slot_values.get(slot, 'none').strip()

                        if args["slot_lang"]=="human":
                            slot_lang = description[slot]["description_human"]
                            input_text = dialog_history + f" {tokenizer.sep_token} {slot_lang}?"
                        elif args["slot_lang"]=="naive":
                            slot_lang = description[slot]["naive"]
                            input_text = dialog_history + f" {tokenizer.sep_token} {slot_lang}?"
                        elif args["slot_lang"]=="value":
                            slot_lang = description[slot]["naive"]
                            input_text = dialog_history + f" {tokenizer.sep_token} {slot_lang}"
                        elif args["slot_lang"]=="question":
                            slot_lang = description[slot]["question"]
                            input_text = dialog_history + f" {tokenizer.sep_token} {slot_lang}"
                        elif args["slot_lang"]=="slottype":
                            slot_lang = description[slot]["slottype"]
                            input_text = dialog_history + f" {tokenizer.sep_token} {slot_lang}?"
                        else:
                            input_text = dialog_history + f" {tokenizer.sep_token} {slot}"
                        
                        data_detail = {
                            "ID":dial_dict["dial_id"],
                            "domains":dial_dict["domains"],
                            "turn_id":turn_id,
                            "dialog_history":dialog_history,
                            "turn_belief":turn_belief_list,
                            "intput_text":input_text,
                            "output_text":output_text,
                            "slot_text":slot_text,
                            "value_text":value_text,
                            "value_list":description[slot]["values"]
                            }
                        data.append(data_detail)
    # print(len(data))
    #for idx in range(10):
    #    print(data[idx])
    print("domain_counter", domain_counter)
    return data, slot_temp



def get_slot_information(ontology):
    
    #ontology_domains = dict([(k, v) for k, v in ontology.items()])
    ontology_domains = dict([(k, v) for k, v in ontology.items() if k.split("-")[0] in EXPERIMENT_DOMAINS])
    SLOTS = [k.replace(" ","").lower() for k in ontology_domains.keys()]

    return SLOTS


def gpt_collate_fn(data,tokenizer):
    batch_data = {}
    for key in data[0]:
        batch_data[key] = [d[key] for d in data]

    output_batch = tokenizer(batch_data["output_text"], padding=True, return_tensors="pt", add_special_tokens=False, return_attention_mask=False, truncation=True, max_length=1000)
    batch_data["input_ids"] = output_batch['input_ids']
    return batch_data


def collate_fn(data, tokenizer):
    batch_data = {}
    for key in data[0]:
        batch_data[key] = [d[key] for d in data]

    input_batch = tokenizer(batch_data["intput_text"], padding=True, return_tensors="pt", add_special_tokens=False, verbose=False)
    batch_data["encoder_input"] = input_batch["input_ids"]
    batch_data["attention_mask"] = input_batch["attention_mask"]
    output_batch = tokenizer(batch_data["output_text"], padding=True, return_tensors="pt", add_special_tokens=False, return_attention_mask=False)
    # replace the padding id to -100 for cross-entropy
    output_batch['input_ids'].masked_fill_(output_batch['input_ids']==tokenizer.pad_token_id, -100)
    batch_data["decoder_output"] = output_batch['input_ids']

    return batch_data


def prepare_data(args, tokenizer):
    pre = 'preprocessed_data' 
    post = 'all'

    path_train = f'{pre}/train_{post}.json'
    path_dev = f'{pre}/dev_{post}.json'
    if args["test_type"] == "seen":
        path_test = f'{pre}/test_seen_{post}.json'
    else:
        path_test = f'{pre}/test_unseen_{post}.json'

    pre = 'ontology'
    ontology = json.load(open(f'{pre}/ontology_{post}.json', 'r'))
    ALL_SLOTS = get_slot_information(ontology)

    pre = 'slot_description'
    description = json.load(open(f'{pre}/slot_description_{post}_wo_domain_desc.json', 'r'))
    
    data_train = None
    data_dev = None
    data_test = None
    if args["mode"] != "test":
        data_train, _ = read_data(args, path_train, ALL_SLOTS, tokenizer, description, "train")
        data_dev, _ = read_data(args, path_dev, ALL_SLOTS, tokenizer, description, "dev")
    else:    
        data_test, _ = read_data(args, path_test, ALL_SLOTS, tokenizer, description, "test")
        #data_test, ALL_SLOTS = read_data(args, path_test, ALL_SLOTS, tokenizer, description, "test") # This for few shot


    train_dataset = DSTDataset(data_train, args)
    dev_dataset = DSTDataset(data_dev, args)
    test_dataset = DSTDataset(data_test, args)
    
    train_loader = None
    dev_loader = None
    test_loader = None

    if "gpt" in args["model_name"]:
        train_loader = DataLoader(train_dataset, batch_size=args["train_batch_size"], shuffle=True, collate_fn=partial(gpt_collate_fn, tokenizer=tokenizer), num_workers=16)
        test_loader = DataLoader(test_dataset, batch_size=args["test_batch_size"], shuffle=False, collate_fn=partial(gpt_collate_fn, tokenizer=tokenizer), num_workers=16)
        dev_loader = DataLoader(dev_dataset, batch_size=args["dev_batch_size"], shuffle=False, collate_fn=partial(gpt_collate_fn, tokenizer=tokenizer), num_workers=16)
    else:
        if args["mode"] != "test":
            train_loader = DataLoader(train_dataset, batch_size=args["train_batch_size"], shuffle=True, collate_fn=partial(collate_fn, tokenizer=tokenizer), num_workers=16)
            dev_loader = DataLoader(dev_dataset, batch_size=args["dev_batch_size"], shuffle=False, collate_fn=partial(collate_fn, tokenizer=tokenizer), num_workers=16)
        else:    
            test_loader = DataLoader(test_dataset, batch_size=args["test_batch_size"], shuffle=False, collate_fn=partial(collate_fn, tokenizer=tokenizer), num_workers=16)
    #test_loader = None
    fewshot_loader_dev=None
    fewshot_loader_test=None
    return train_loader, dev_loader, test_loader, ALL_SLOTS, fewshot_loader_dev, fewshot_loader_test
