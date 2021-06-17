from transformers import (
    AutoModelForSequenceClassification,
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
    train_type: str,
    model_checkpoint: str,
    batch_size: int,
    num_epoch: int,
    warmup_steps: int,
    weight_decay: float, 
    lr: float,
):
    assert train_type == 'beginning' or train_type == 'end'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    train_raw_data = load_data(data_dir_path/'train')
    dev_raw_data = load_data(data_dir_path/'dev')
    train_data = preprocess_data(train_raw_data, tokenizer, train_type)
    dev_data = preprocess_data(dev_raw_data, tokenizer, train_type)
    train_dataset = NLPDataset(train_data)
    dev_dataset = NLPDataset(dev_data)
    # Padding to equal length
    def collate(data):
        input_ids = [x['input_ids'] for x in data]
        att_masks = [x['attention_mask'] for x in data]
        labels = torch.LongTensor([x['labels'].item() for x in data])
        if tokenizer._pad_token is None:
            input_ids = pad_sequence(input_ids, batch_first=True)
            att_masks = pad_sequence(att_masks, batch_first=True)
            
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
        att_masks = pad_sequence(att_masks, batch_first=True, padding_value=0)
        return {'input_ids': input_ids, 'attention_mask': att_masks, 'labels': labels}

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate, drop_last = True)
    dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size,shuffle=True, collate_fn=collate, drop_last = True)
    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=2).to(device)
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
        train_epoch_acc, eval_epoch_acc = 0,0
        model.train()
        for i,item in enumerate(tqdm(train_dataloader, ncols=0, desc=f'Epoch {epoch} Training:')):
            inputs = {key: value.to(device) for (key, value) in item.items()}
            outputs = model(**inputs)
            loss, logits = outputs[0], outputs[1]
            loss.backward()
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            predict_ans = torch.argmax(logits, dim=1)
            labels = inputs['labels']
            train_epoch_acc += torch.sum(predict_ans == labels).item()/len(predict_ans)
            train_epoch_loss += loss.item()
        model.eval()
        for item in tqdm(dev_dataloader, ncols=0, desc=f'Epoch {epoch} Evaluation:'):
            inputs = {key: value.to(device) for (key, value) in item.items()}
            outputs = model(**inputs)
            loss, logits = outputs[0], outputs[1]
            predict_ans = torch.argmax(logits, dim=1)
            labels = inputs['labels']
            eval_epoch_acc += torch.sum(predict_ans == labels).item()/len(predict_ans)
            eval_epoch_loss += loss.item()
        print(f'Epoch {epoch}: Training loss = {train_epoch_loss/len(train_dataloader)}, Evaluation loss = {eval_epoch_loss/len(dev_dataloader)}')
        print(f'Epoch {epoch}: Training acc = {train_epoch_acc/len(train_dataloader)}, Evaluation acc = {eval_epoch_acc/len(dev_dataloader)}')
        if epoch == 0:
            torch.save(model, './BeginClassifier.mdl' if train_type == 'beginning' else './EndClassifier.mdl')
            best_eval_loss = eval_epoch_loss/len(dev_dataloader)
        else:
            if(best_eval_loss > eval_epoch_loss/len(dev_dataloader)):
                torch.save(model, './BeginClassifier.mdl' if train_type == 'beginning' else './EndClassifier.mdl')
                best_eval_loss = eval_epoch_loss/len(dev_dataloader)
        
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('data_dir_path', type=Path)
    parser.add_argument('--train_type', type=str, help='Either end or beginning"')
    parser.add_argument('--model_checkpoint', type=str, default="distilbert-base-uncased")
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_epoch', type=int, default=3)
    parser.add_argument('--warmup_steps', type=int, default=0)
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=5e-5)
    args = parser.parse_args()
    main(**vars(args))