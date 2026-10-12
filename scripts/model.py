import torch 
import torch.nn as nn

class ActivityCNN(nn.Module):

    def __init__(self, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            #block 1 - input: (batch, 6, 128)
            nn.Conv1d(in_channels=6, out_channels=32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2), #halfs time by taking max of each pair of neighbours

            #block 2 - input: (batch, 32, 64)
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2), 

            #block 3 - input: (batch, 64, 32)
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1) #one averaged time position per feature channel
        )
        #output: (B, 128, 1)

        #recieves 128 summary features and produces 1 score per class (activity)
        self.classifier = nn.Linear(in_features=128, out_features=num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.squeeze(-1) #(batch, 128, 1) -> (batch, 128) - removing last dimesion (=1)
        logits = self.classifier(x) #calls the activity scores for each batch
        
        return logits