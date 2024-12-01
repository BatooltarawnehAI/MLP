import gc
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from transformers import AutoModel
import pandas as pd
from Pre_Process import load_and_process

# Define BERT_CNN model compatible with bert-large-uncased
class BERT_CNN(nn.Module):
    def __init__(self, bert):
        super(BERT_CNN, self).__init__()
        self.bert = bert
        self.conv = nn.Conv2d(in_channels=13, out_channels=13, kernel_size=(3, 1024), padding=(1, 0))
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=(3, 1), stride=(1, 1))
        self.dropout = nn.Dropout(0.1)
        self.flat = nn.Flatten()
        self.fc = nn.Linear(468, 3)  # Adjusted for max_length=36
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_ids, attention_mask):
        # Get hidden states from BERT
        outputs = self.bert(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = torch.stack(outputs.hidden_states, dim=0)  # Shape: (13, batch_size, seq_length, hidden_size)

        # Reshape for Conv2D: (batch_size, 13, seq_length, hidden_size)
        x = hidden_states.permute(1, 0, 2, 3)

        # Apply Conv2D, pooling, and classification layers
        x = self.pool(self.relu(self.conv(self.dropout(x))))
        x = self.flat(self.dropout(x))
        x = self.fc(self.dropout(x))
        return self.softmax(x)

# Function to train the model
def train():
    model.train()
    total_loss = 0
    total_preds = []
    total = len(train_dataloader)

    for step, batch in enumerate(train_dataloader):
        batch = [r.to(device) for r in batch]
        input_ids, attention_mask, labels = batch
        del batch
        gc.collect()
        torch.cuda.empty_cache()

        model.zero_grad()
        preds = model(input_ids, attention_mask)
        loss = cross_entropy(preds, labels)
        total_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_preds.append(preds.detach().cpu().numpy())

    avg_loss = total_loss / total
    total_preds = np.concatenate(total_preds, axis=0)
    return avg_loss, total_preds

# Function to evaluate the model
def evaluate():
    model.eval()
    total_loss = 0
    total_preds = []
    total = len(val_dataloader)

    for step, batch in enumerate(val_dataloader):
        batch = [t.to(device) for t in batch]
        input_ids, attention_mask, labels = batch
        del batch
        gc.collect()
        torch.cuda.empty_cache()

        with torch.no_grad():
            preds = model(input_ids, attention_mask)
            loss = cross_entropy(preds, labels)
            total_loss += loss.item()
            total_preds.append(preds.detach().cpu().numpy())

    avg_loss = total_loss / total
    total_preds = np.concatenate(total_preds, axis=0)
    return avg_loss, total_preds

# Specify GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load Dataset
input_ids, attention_masks, labels = load_and_process()
df = pd.DataFrame(list(zip(input_ids, attention_masks)), columns=['input_ids', 'attention_masks'])

# Split Dataset
train_text, temp_text, train_labels, temp_labels = train_test_split(df, labels, test_size=0.2, stratify=labels, random_state=42)
val_text, test_text, val_labels, test_labels = train_test_split(temp_text, temp_labels, test_size=0.5, stratify=temp_labels, random_state=42)

train_seq = torch.tensor(train_text['input_ids'].tolist())
train_mask = torch.tensor(train_text['attention_masks'].tolist())
train_y = torch.tensor(train_labels.tolist())

val_seq = torch.tensor(val_text['input_ids'].tolist())
val_mask = torch.tensor(val_text['attention_masks'].tolist())
val_y = torch.tensor(val_labels.tolist())

test_seq = torch.tensor(test_text['input_ids'].tolist())
test_mask = torch.tensor(test_text['attention_masks'].tolist())
test_y = torch.tensor(test_labels.tolist())

# Create DataLoaders
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
batch_size = 16
train_data = TensorDataset(train_seq, train_mask, train_y)
train_dataloader = DataLoader(train_data, sampler=RandomSampler(train_data), batch_size=batch_size)

val_data = TensorDataset(val_seq, val_mask, val_y)
val_dataloader = DataLoader(val_data, sampler=SequentialSampler(val_data), batch_size=batch_size)

# Load BERT-large model
bert = AutoModel.from_pretrained('bert-large-uncased', output_hidden_states=True)
model = BERT_CNN(bert).to(device)

# Optimizer and Loss
from transformers import AdamW
optimizer = AdamW(model.parameters(), lr=2e-5)
cross_entropy = nn.NLLLoss()

# Training Loop
epochs = 3
best_valid_loss = float('inf')
for epoch in range(epochs):
    print(f'\nEpoch {epoch + 1}/{epochs}')
    train_loss, _ = train()
    valid_loss, _ = evaluate()
    print(f'\nTraining Loss: {train_loss:.3f}')
    print(f'Validation Loss: {valid_loss:.3f}')
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss

# Test Evaluation
model.eval()
with torch.no_grad():
    preds = model(test_seq.to(device), test_mask.to(device))
    preds = preds.detach().cpu().numpy()

preds = np.argmax(preds, axis=1)
print("\nPerformance:")
print(classification_report(test_y, preds))
print(f"Accuracy: {accuracy_score(test_y, preds):.4f}")
