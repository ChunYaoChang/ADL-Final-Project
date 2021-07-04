import os
import sys
import json
import time
import logging

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, DistilBertForSequenceClassification
from accelerate import Accelerator

from dataset import DialogueTestDataset
from utils import parse_schema, add_tokens

def score(uttr1, uttr2, uttr3, model, tokenizer, device):
    model.eval()
    with torch.no_grad():
        # print(uttr1)
        # print(uttr2)
        # print(uttr3)
        uttr1_input_ids = tokenizer(uttr1, return_tensors="pt")["input_ids"].squeeze(0).to(device)
        uttr2_input_ids = tokenizer(uttr2, return_tensors="pt")["input_ids"].squeeze(0).to(device)
        uttr3_input_ids = tokenizer(uttr3, return_tensors="pt")["input_ids"].squeeze(0).to(device)
        input_ids = torch.cat((uttr1_input_ids[:-1], uttr2_input_ids[1:], uttr3_input_ids[1:])).unsqueeze(0)
        outputs = model(input_ids=input_ids)
        logits = outputs["logits"]
        return torch.nn.functional.softmax(logits, dim=1).squeeze(0).detach().cpu()

def generate_chit_chat(history, model, model_cls, tokenizer, tokenizer_cls, device, next_turn=None):
    # print(history)
    model.eval()
    with torch.no_grad():
        sys_token_id = tokenizer.convert_tokens_to_ids("<sys>")
        usr_token_id = tokenizer.convert_tokens_to_ids("<usr>")
        input_ids = tokenizer(
            "".join(history) + "<sys>",
            return_tensors="pt"
        )["input_ids"].to(device)
        dialogue_token_length = len(input_ids[0])
        outputs = model.generate(
            input_ids=input_ids,
            do_sample=True,
            num_beams=5,
            min_length=dialogue_token_length+10,
            max_length=dialogue_token_length+30,
            num_return_sequences=5,
            pad_token_id=tokenizer.eos_token_id
        )
        outputs = outputs[:, dialogue_token_length:]
        scores, responses = [], []
        # print(history[-1])
        # print(history[-1][5:])
        """
        model.to("cpu")
        model_cls.to(device)
        """
        for i, output in enumerate(outputs):
            end_index = len(output)
            for j, char_id in enumerate(output):
                if char_id == usr_token_id or \
                   char_id == sys_token_id or \
                   char_id == tokenizer.eos_token_id:
                    end_index = j
                    break
            generated_utterance_id = output[:end_index]
            generated_utterance = "".join(tokenizer.batch_decode(generated_utterance_id))
            chat_score = score(
                history[-1][5:] if next_turn is not None else history[-2][5:],
                generated_utterance if next_turn is not None else history[-1][5:],
                next_turn if next_turn is not None else generated_utterance,
                model_cls,
                tokenizer_cls,
                device
            )
            scores.append(chat_score[1].item())
            responses.append(generated_utterance)
        """
        model_cls.to("cpu")
        model.to(device)
        """
        max_score = max(scores)
        response_with_max_score = responses[scores.index(max_score)]
    return response_with_max_score if max_score > 0.5 else ""

def generate_dialogue(dialogue, model, begin_cls, end_cls, tokenizer, tokenizer_cls, device):
    generated_dialogue = {dialogue["dialogue_id"]: {}}
    this_id = dialogue["dialogue_id"]
    logging.info(f"dialogue_id: {this_id}")
    history = []
    turns = dialogue["turns"]
    for i, turn in enumerate(turns):
        turn_id, speaker, utterance = str(turn["turn_id"]), turn["speaker"], turn["utterance"]
        history.append("<usr>" + utterance if speaker == "USER" else "<sys>" + utterance)
        if speaker == "USER":
            continue
        # Cut memory usage
        while len(history) > 10:
            del history[0]
        # print(history)
        start = generate_chit_chat(history[:-1], model, begin_cls, tokenizer, tokenizer_cls, device, next_turn=utterance)
        end = generate_chit_chat(history, model, end_cls, tokenizer, tokenizer_cls, device)
        generated_turn = {"start": start, "mod": "", "end": end}
        generated_dialogue[dialogue["dialogue_id"]][turn_id] = generated_turn
    return generated_dialogue

def generate(model, path, begin_cls, end_cls, tokenizer, tokenizer_cls, device):
    files = [os.path.join(path, file) for file in sorted(os.listdir(path))]
    output_file = "./generated_test_seen.json"
    generated_dialogues = {}
    for i, file in enumerate(files):
        logging.info(file)
        start_time = time.time()
        with open(file, "r") as f:
            dialogues = json.load(f)
            for dialogue in dialogues:
                generated_dialogue = generate_dialogue(dialogue, model, begin_cls, end_cls, tokenizer, tokenizer_cls, device)
                generated_dialogues.update(generated_dialogue)
        with open(output_file, "w") as f:
            json.dump(generated_dialogues, f, indent=2)
        logging.info(f"Elapsed time for a file: {time.time() - start_time}")

def main():
    domain_to_slots = parse_schema()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Prepare tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    add_tokens(tokenizer)
    model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
    model.resize_token_embeddings(len(tokenizer))
    model.load_state_dict(torch.load("./checkpoints/gpt2.ckpt"))
    begin_cls = torch.load("./checkpoints/BeginClassifier.mdl").to(device)
    end_cls = torch.load("./checkpoints/EndClassifier.mdl").to(device)
    tokenizer_cls = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    dev_path = "./test_seen/"
    generate(model, dev_path, begin_cls, end_cls, tokenizer, tokenizer_cls, device)

if __name__ == "__main__":
    logging.basicConfig(
        filename="./log/generate_seen.txt",
        encoding="utf-8",
        filemode="a",
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S",
        level=logging.INFO
    )
    main()
