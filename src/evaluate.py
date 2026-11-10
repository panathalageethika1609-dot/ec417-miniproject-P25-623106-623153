import torch
from src.data import get_dataloaders
from src.models import get_baseline_model, get_main_model
import os
import pandas as pd

def evaluate_model(model_name="resnet18"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating {model_name} on {device}...")
    
    _, _, test_loader = get_dataloaders()
    
    if model_name == "resnet18":
        model = get_baseline_model().to(device)
    else:
        model = get_main_model().to(device)
        
    model.load_state_dict(torch.load(f"results/{model_name}_best.pth", map_location=device))
    model.eval()
    
    top1_correct = 0
    top5_correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            
            # Compute Top-1 and Top-5 accuracy
            _, pred = outputs.topk(5, 1, True, True)
            pred = pred.t()
            correct = pred.eq(labels.view(1, -1).expand_as(pred))
            
            top1_correct += correct[:1].reshape(-1).float().sum().item()
            top5_correct += correct[:5].reshape(-1).float().sum().item()
            total += labels.size(0)
            
    top1_acc = 100.0 * top1_correct / total
    top5_acc = 100.0 * top5_correct / total
    
    print(f"Test Top-1 Accuracy: {top1_acc:.2f}%")
    print(f"Test Top-5 Accuracy: {top5_acc:.2f}%")
    
    # Save metrics to CSV file required by submission guidelines
    os.makedirs("results", exist_ok=True)
    csv_path = "results/metrics.csv"
    df = pd.DataFrame([{
        "Model": model_name,
        "Top-1 Accuracy (%)": f"{top1_acc:.2f}",
        "Top-5 Accuracy (%)": f"{top5_acc:.2f}"
    }])
    
    if os.path.exists(csv_path):
        df.to_csv(csv_path, mode='a', header=False, index=False)
    else:
        df.to_csv(csv_path, index=False)
        
    print(f"Metrics saved to {csv_path}")

if __name__ == "__main__":
    evaluate_model("vit")