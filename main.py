!pip install transformers --quietimport torch
import torch.nn as nn
from torch.optim import Adam
from transformers import AutoTokenizer , AutoModel
from torch.utils.data import Dataset , DataLoader
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from transformers import get_scheduler
from sklearn.metrics import accuracy_score
from tqdm import tqdm


import kagglehub
path = kagglehub.dataset_download("rmisra/news-headlines-dataset-for-sarcasm-detection")

print("Path to dataset files:", path)

# load the dataset

import pandas as pd

df = pd.read_json("/kaggle/input/news-headlines-dataset-for-sarcasm-detection/Sarcasm_Headlines_Dataset.json", lines=True)

# Check the data
print(df.head())
print(df.columns)


train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['headline'].tolist(),
    df['is_sarcastic'].tolist(),
    test_size=0.2,
    random_state=42
)

##tokenizing using Autotokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=64)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=64)


train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
model.to(device)

optimizer = AdamW(model.parameters(), lr=2e-5)

num_training_steps = len(train_loader) * 3  # 3 epochs
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps
)

##training model

from tqdm.auto import tqdm

model.train()
for epoch in range(3):
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    for batch in loop:
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        loop.set_postfix(loss=loss.item())


###model evaluation from here

model.eval()

predictions = []
true_labels = []

with torch.no_grad():
    for batch in val_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        labels = batch['labels'].cpu().numpy()

        predictions.extend(preds)
        true_labels.extend(labels)

acc = accuracy_score(true_labels, predictions)
print(f"Validation Accuracy: {acc:.4f}")
print("Sarcasm detection using BERT tokenizer")




