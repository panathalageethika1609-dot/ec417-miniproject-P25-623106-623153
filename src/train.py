import torch
import torch.nn as nn
import torch.optim as optim
from src.data import get_dataloaders
from src.models import get_baseline_model, get_main_model
import os

def train_model(model_name="resnet18", epochs=5, lr=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training {model_name} on {device}...")
    
    train_loader, val_loader, _ = get_dataloaders()
    
    if model_name == "resnet18":
        model = get_baseline_model().to(device)
    else:
        model = get_main_model().to(device)
        
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    os.makedirs("results", exist_ok=True)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}")
        
    torch.save(model.state_dict(), f"results/{model_name}_best.pth")
    print(f"Model saved to results/{model_name}_best.pth")

if __name__ == "__main__":
    train_model("vit", epochs=3, lr=5e-5)