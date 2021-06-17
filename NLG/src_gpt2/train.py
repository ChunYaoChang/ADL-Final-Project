from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AdamW,
    get_linear_schedule_with_warmup,
)
from torch.nn.utils.rnn import pad_sequence
from argparse import ArgumentParser
from pathlib import Path
from util import (
    load_data,
    preprocess_data,
)
from dataset import (
    NLPDataset,
)
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch

def main(
    data_dir_path: Path, 
    model_checkpoint: str,
    num_sen_per_input: int,
    batch_size: int,
    num_epoch: int,
    warmup_steps: int,
    weight_decay: float, 
    lr: float,
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
    print(len(train_data), len(dev_data))
    train_dataset = NLPDataset(train_data)
    dev_dataset = NLPDataset(dev_data)
    # Padding to equal length
    def collate(data):
        if tokenizer._pad_token is None:
            return pad_sequence(data, batch_first=True)
        return pad_sequence(data, batch_first=True, padding_value=tokenizer.pad_token_id)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate, drop_last = True)
    dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size,shuffle=True, collate_fn=collate, drop_last = True)
    model = AutoModelForCausalLM.from_pretrained(model_checkpoint).to(device)
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_epoch*len(train_dataloader)
    )
    model.zero_grad()
    for epoch in range(num_epoch):
        train_epoch_loss, eval_epoch_loss = 0,0
        model.train()
        for i,item in enumerate(tqdm(train_dataloader, ncols=0, desc=f'Epoch {epoch} Training:')):
            inputs, labels = item.to(device), item.to(device)
            outputs = model(inputs, labels=labels)
            loss = outputs[0]
            loss.backward()
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            train_epoch_loss += loss.item()
        model.eval()
        for item in tqdm(dev_dataloader, ncols=0, desc=f'Epoch {epoch} Evaluation:'):
            inputs, labels = item.to(device), item.to(device) 
            outputs = model(inputs, labels=labels)
            loss = outputs[0]
            eval_epoch_loss += loss.item()
        print(f'Epoch {epoch}: Training loss = {train_epoch_loss/len(train_dataloader)}, Evaluation loss = {eval_epoch_loss/len(dev_dataloader)}')
        if epoch == 0:
            torch.save(model, './BestModel.mdl')
            best_eval_loss = eval_epoch_loss/len(dev_dataloader)
        else:
            if(best_eval_loss > eval_epoch_loss/len(dev_dataloader)):
                torch.save(model, './BestModel.mdl')
                best_eval_loss = eval_epoch_loss/len(dev_dataloader)
        
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('data_dir_path', type=Path)
    parser.add_argument('--model_checkpoint', type=str, default='microsoft/DialoGPT-small')
    ## num_sen_per_input should be an odd number
    parser.add_argument('--num_sen_per_input', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_epoch', type=int, default=2)
    parser.add_argument('--warmup_steps', type=int, default=0)
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=5e-5)
    args = parser.parse_args()
    main(**vars(args))