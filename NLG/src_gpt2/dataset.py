import os
import sys
import json

import torch
from torch.utils.data import Dataset

class DialogueDataset(Dataset):
    def __init__(self, path, tokenizer):
        files = sorted([os.path.join(path, file) for file in os.listdir(path)])
        self.data = []
        self.label = []
        self.id = []
        self.tokenizer = tokenizer
        for file in files:
            print(file)
            with open(file, "r") as f:
                dialogues = json.load(f)
                for dialogue in dialogues:
                    dialogue_data, dialogue_label, dialogue_id = self.parse_dialogue(dialogue)
                    self.data.extend(dialogue_data)
                    self.label.extend(dialogue_label)
                    self.id.extend(dialogue_id)
        assert(len(self.data) == len(self.label) == len(self.id))
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.label[index], self.id[index]

    def parse_dialogue(self, dialogue):
        dialogue_id, services, turns = dialogue["dialogue_id"], dialogue["services"], dialogue["turns"]
        dialogue_history = ""
        dialogue_data, dialogue_label, id = [], [], []
        for turn in turns:
            speaker, utterance = turn["speaker"], turn["utterance"]
            if speaker == "SYSTEM":
                try:
                    for begin in turn["beginning"]:
                        candidate, label = begin["candidate"], begin["label"]
                        input_ids = self.tokenizer(
                            dialogue_history + "<sys>" + candidate + "<sys>" + utterance + self.tokenizer.eos_token,
                            return_tensors="pt"
                        )["input_ids"]
                        dialogue_data.append(input_ids)
                        dialogue_label.append(label)
                        id.append(dialogue_id)
                    for end in turn["end"]:
                        candidate, label = end["candidate"], end["label"]
                        input_ids = self.tokenizer(
                            dialogue_history + "<sys>" + utterance + "<sys>" + candidate + self.tokenizer.eos_token,
                            return_tensors="pt"
                        )["input_ids"]
                        dialogue_data.append(input_ids)
                        dialogue_label.append(label)
                        id.append(dialogue_id)
                except KeyError:
                    pass
                finally:
                    dialogue_history += "<sys>" + utterance
            elif speaker == "USER":
                dialogue_history += "<usr>" + utterance
        return dialogue_data, dialogue_label, id

    def collate_fn(self, samples):
        input_ids = [sample[0] for sample in samples]
        labels = [sample[1] for sample in samples]
        ids = [sample[2] for sample in samples]
        max_len = max(len(input_id[0]) for input_id in input_ids)
        for i in range(len(input_ids)):
            if len(input_ids[i][0]) == max_len:
                continue
            padding_tensor = torch.LongTensor(
                [tokenizer.pad_token_id if tokenizer._pad_token is not None else tokenizer.eos_token_id] * \
                (max_len - len(input_ids[i][0]))
            ).unsqueeze(0)
            # print("before:", input_ids[i].shape)
            input_ids[i] = torch.cat((input_ids[i], padding_tensor), dim=1)
            # print("after:", input_ids[i].shape)
        # for input_id in input_ids:
            # print(input_id.shape)

        return torch.stack(input_ids), labels, ids

class DialogueTestDataset(Dataset):
    def __init__(self, path, tokenizer):
        files = sorted([os.path.join(path, file) for file in os.listdir(path)])
        self.data = []
        self.id = []
        self.tokenizer = tokenizer
        for file in files:
            print(file)
            with open(file, "r") as f:
                dialogues = json.load(f)
                for dialogue in dialogues:
                    dialogue_data, dialogue_id = self.parse_dialogue(dialogue)
                    self.data.extend(dialogue_data)
                    self.id.extend(dialogue_id)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.id[index]

    def parse_dialogue(self, dialogue):
        dialogue_id, services, turns = dialogue["dialogue_id"], dialogue["services"], dialogue["turns"]
        dialogue_history = ""
        dialogue_data, id = [], []
        for turn in turns:
            speaker, utterance = turn["speaker"], turn["utterance"]
            if speaker == "SYSTEM":
                dialogue_history += "<sys>" + utterance
                input_ids = self.tokenizer(
                    dialogue_history + "<sys>",
                    return_tensors="pt",
                )["input_ids"]
                dialogue_data.append(input_ids)
                id.append(dialogue_id)
            elif speaker == "USER":
                dialogue_history += "<usr>" + utterance
                input_ids = self.tokenizer(
                    dialogue_history + "<sys>",
                    return_tensors="pt",
                )["input_ids"]
                dialogue_data.append(input_ids)
                id.append(dialogue_id)
        return dialogue_data, id

    def collate_fn(self, samples):
        input_ids = [sample[0] for sample in samples]
        ids = [sample[1] for sample in samples]
        max_len = max(len(input_id[0]) for input_id in input_ids)
        for i in range(len(input_ids)):
            if len(input_ids[i][0]) == max_len:
                continue
            padding_tensor = torch.LongTensor(
                [tokenizer.pad_token_id if tokenizer._pad_token is not None else tokenizer.eos_token_id] * \
                (max_len - len(input_ids[i][0]))
            ).unsqueeze(0)
            input_ids[i] = torch.cat((input_ids[i], padding_tensor), dim=1)

        return torch.stack(input_ids), ids
