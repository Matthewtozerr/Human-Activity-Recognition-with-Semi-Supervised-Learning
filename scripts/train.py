from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from model import ActivityCNN
import torch.nn as nn

preprocessed_data_path = Path(__file__).parent.parent / "data" / "ProcessedData"
output_path = Path(__file__).parent.parent / "outputs"
output_path.mkdir(parents=True, exist_ok=True)

#load the preprocessed data
X_train = np.load(preprocessed_data_path / "X_train.npy")
X_val = np.load(preprocessed_data_path / "X_val.npy")
X_test = np.load(preprocessed_data_path / "X_test.npy")

y_train = np.load(preprocessed_data_path / "y_train.npy")
y_val = np.load(preprocessed_data_path / "y_val.npy")
y_test = np.load(preprocessed_data_path / "y_test.npy")

print("Train:", X_train.shape, y_train.shape)
print("Validation:", X_val.shape, y_val.shape)
print("Test:", X_test.shape, y_test.shape)

#need to convert shapes from (N, 128, 6) to (N, 6, 128) for pytorch
#nn.Conv1D expects (batch, channels, time)
X_train = torch.tensor(X_train.transpose(0, 2, 1).copy(), dtype=torch.float32)
X_val = torch.tensor(X_val.transpose(0, 2, 1).copy(), dtype=torch.float32)
X_test = torch.tensor(X_test.transpose(0, 2, 1).copy(), dtype=torch.float32)

#convert labels to torch tensors (0 indexed)
classes = np.unique(np.concatenate([y_train, y_val, y_test]))
num_classes = len(classes)
label_to_index = {label: index for index, label in enumerate(classes)}

def encode_labels(labels):
    return torch.tensor([label_to_index[label] for label in labels], dtype=torch.long)

y_train = encode_labels(y_train)
y_val = encode_labels(y_val)
y_test = encode_labels(y_test)

#create DataLoaders for training, validation, and testing (batches)
batch_size = 64

train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = TensorDataset(X_val, y_val)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

torch.manual_seed(42)
np.random.seed(42)

if torch.cuda.is_available():
    device = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("Using Device:", device)

model = ActivityCNN(num_classes=num_classes)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

epochs = 30

print(model)

num_parameters = sum(p.numel() for p in model.num_parameters())
print(f"Model parameters: {num_parameters:,}")

def train_epoch(model, loader, criterion, optimizer, device):

    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for X_batch, y_batch in loader:

        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()

        logits = model(X_batch)

        loss = criterion(logits, y_batch)
        loss.backward()

        optimizer.step()

        batch_size = X_batch.size(0)
        
        total_loss += loss.item() * batch_size

        predictions = logits.argmax(dim=1) #take activity with max score 

        correct += (predictions == y_batch).sum().item()
        total += batch_size

    return total_loss / total, correct / total
    
def evaluate(model, loader, criterion, device):

    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for X_batch, y_batch in loader:

            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(X_batch)

            loss = criterion(logits, y_batch)

            batch_size = X_batch.size(0)

            total_loss += loss.item() * batch_size

            predictions = logits.argmax(dim=1)

            correct += (predictions == y_batch).sum().item()
            total += batch_size

    return total_loss / total, correct / total



