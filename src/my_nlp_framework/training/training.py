# src/my_nlp_framework/training/train.py
import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from my_nlp_framework.evaluation.metrics import accuracy

def train_model(model, train_loader, val_loader, epochs, learning_rate, device):
    model.to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    criterion = CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            tokens = batch.to(device)
            outputs = model(tokens)
            loss = criterion(outputs, tokens)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        val_accuracy = 0
        with torch.no_grad():
            for batch in val_loader:
                tokens = batch.to(device)
                outputs = model(tokens)
                val_accuracy += accuracy(outputs, tokens)
        
        print(f"Epoch {epoch+1}, Loss: {loss.item()}, Validation Accuracy: {val_accuracy/len(val_loader)}")
