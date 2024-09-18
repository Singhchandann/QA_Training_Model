import torch
from torch.utils.data import DataLoader, Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, AdamW, get_linear_schedule_with_warmup
import pandas as pd
from tqdm import tqdm

class QADataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.texts = dataframe['text']
        self.summaries = dataframe['summary']
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = str(self.texts[index])
        summary = str(self.summaries[index])
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        target_encoding = self.tokenizer(
            summary,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': target_encoding['input_ids'].flatten()
        }

def load_data(file_path, tokenizer, max_len, batch_size):
    df = pd.read_csv(file_path)
    dataset = QADataset(df, tokenizer, max_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader

def train_epoch(model, data_loader, optimizer, scheduler, device):
    model.train()
    losses = []
    for batch in tqdm(data_loader):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())
    return sum(losses) / len(losses)

def eval_model(model, data_loader, device):
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            losses.append(loss.item())
    return sum(losses) / len(losses)

def train_model(train_loader, val_loader, model, device, learning_rate, epochs, total_steps, warmup_steps):
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        print(f"Train loss: {train_loss}")
        val_loss = eval_model(model, val_loader, device)
        print(f"Val loss: {val_loss}")

if __name__ == "__main__":
    MAX_LEN = 512
    BATCH_SIZE = 4
    EPOCHS = 3
    LEARNING_RATE = 5e-5

    tokenizer = T5Tokenizer.from_pretrained('t5-base')
    model = T5ForConditionalGeneration.from_pretrained('t5-base')

    train_loader = load_data('train_data.csv', tokenizer, MAX_LEN, BATCH_SIZE)
    val_loader = load_data('val_data.csv', tokenizer, MAX_LEN, BATCH_SIZE)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    total_steps = len(train_loader) * EPOCHS
    warmup_steps = 0

    train_model(train_loader, val_loader, model, device, LEARNING_RATE, EPOCHS, total_steps, warmup_steps)
