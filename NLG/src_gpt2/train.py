import os
import sys
import time
import json
import logging

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, AdamW, get_constant_schedule_with_warmup
from accelerate import Accelerator

from dataset import DialogueDataset
from utils import parse_schema, add_tokens

def train(epochs, model, optimizer, scheduler, train_dataloader, accelerator, checkpoint_path, saving_step, logging_step, gradient_clip_value, gradient_accumulation_step):
    step = 0
    for epoch in range(epochs):
        epoch_loss, step_time = [], []
        optimizer.zero_grad()
        epoch_start_time = time.time()
        for i, (input_ids, labels, ids) in enumerate(train_dataloader):
            start_time = time.time()
            model.train()
            outputs = model(input_ids, labels=torch.clone(input_ids))
            loss = outputs[0]
            if labels[0] == "good":
                loss = torch.mul(loss, 5)
            epoch_loss.append(loss.item())
            accelerator.backward(loss)
            if step > 0 and step % gradient_accumulation_step == 0:
                accelerator.clip_grad_norm_(model.parameters(), gradient_clip_value)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            if step > 0 and step % (logging_step * gradient_accumulation_step) == 0:
                logging.info(f"Current step: {step // gradient_accumulation_step}")
                logging.info(f"Current average training loss: {sum(epoch_loss) / len(epoch_loss)}")
                logging.info(f"Average time per step: {sum(step_time) / len(step_time)}")
                if len(step_time) > 1e5:
                    step_time = step_time[len(step_time) // 2:]
            if step > 0 and step % (saving_step * gradient_accumulation_step) == 0:
                unwrapped_model = accelerator.unwrap_model(model)
                accelerator.save(unwrapped_model.state_dict(),  os.path.join(checkpoint_path, f"checkpoint-{step}.ckpt"))
                logging.info(f"Saving model at step {step}")
            step += 1
            step_time.append(time.time() - start_time)
        logging.info(f"Epoch {epoch+1}, elapsed time: {time.time() - epoch_start_time}, training loss: {sum(epoch_loss) / len(epoch_loss)}")
        unwrapped_model = accelerator.unwrap_model(model)
        accelerator.save(unwrapped_model.state_dict(), os.path.join(checkpoint_path, f"checkpoint-epoch-{epoch+1}.ckpt"))
        logging.info(f"Saving model at epoch {epoch+1}")

def main():
    domain_to_slots = parse_schema()

    # Prepare tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    add_tokens(tokenizer)

    # Prepare datasets
    train_path = "./data-0614/train/"
    dev_path = "./data-0614/dev/"
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    model.resize_token_embeddings(len(tokenizer))
    train_data = DialogueDataset(train_path, tokenizer)
    
    # Prepare hyperparameters, optimizer, scheduler, dataloader
    lr = 1e-5
    epochs = 5
    batch_size = 1
    saving_step = 400
    logging_step = 100
    weight_decay = 1e-3
    gradient_clip_value = 2.0
    gradient_accumulation_step = 128
    checkpoint_path = "./checkpoints/"

    no_decay = ["bias", "LayerNorm.weight"]
    param_groups = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(param_groups, lr=lr)
    scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=100)

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=train_data.collate_fn)
    accelerator = Accelerator(fp16=True)
    model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)
    train(
        epochs, model, optimizer, scheduler, train_dataloader, accelerator, checkpoint_path,
        saving_step, logging_step, gradient_clip_value, gradient_accumulation_step
    )

if __name__ == "__main__":
    logging.basicConfig(
        filename="./log/training_log.txt",
        encoding="utf-8",
        filemode="a",
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S",
        level=logging.INFO
    )
    main()
