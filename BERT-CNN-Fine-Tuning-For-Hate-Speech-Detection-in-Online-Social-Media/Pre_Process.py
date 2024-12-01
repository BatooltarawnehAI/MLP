# Load the data
input_ids, attention_masks, labels = load_and_process()

# Convert to PyTorch tensors
import torch
input_ids = torch.tensor(input_ids)
attention_masks = torch.tensor(attention_masks)
labels = torch.tensor(labels)

# Use the tensors in DataLoaders
from torch.utils.data import TensorDataset, DataLoader
dataset = TensorDataset(input_ids, attention_masks, labels)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
