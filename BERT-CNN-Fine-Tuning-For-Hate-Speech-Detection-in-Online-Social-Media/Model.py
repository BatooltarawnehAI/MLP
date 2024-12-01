import torch
import torch.nn as nn

class BERT_CNN(nn.Module):
    def __init__(self, bert):
        super(BERT_CNN, self).__init__()
        self.bert = bert  # Accept the BERT model as an argument
        self.conv = nn.Conv2d(in_channels=13, out_channels=13, kernel_size=(3, 1024), padding=(1, 0))  # Adjusted for 1024 hidden size
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=(3, 1), stride=(1, 1))
        self.dropout = nn.Dropout(0.1)
        self.flat = nn.Flatten()
        self.fc = nn.Linear(442, 3)  # Adjusted for flattened size
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, sent_id, mask):
        # Get all hidden states from the BERT model
        outputs = self.bert(sent_id, attention_mask=mask, output_hidden_states=True)
        all_layers = outputs.hidden_states  # Extract hidden states

        # Stack all layers and process them with CNN
        x = torch.stack(all_layers, dim=0)  # Shape: (13, batch_size, seq_length, hidden_size)
        x = x.permute(1, 0, 2, 3)  # Permute to shape: (batch_size, 13, seq_length, hidden_size)

        # Apply convolutional, pooling, and dropout layers
        x = self.pool(self.dropout(self.relu(self.conv(self.dropout(x)))))
        x = self.flat(self.dropout(x))
        x = self.fc(self.dropout(x))
        return self.softmax(x)
