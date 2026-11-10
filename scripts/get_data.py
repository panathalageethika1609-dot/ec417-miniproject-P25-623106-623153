import torchvision.datasets as datasets
import os

os.makedirs("data", exist_ok=True)
train_set = datasets.Flowers102("data", split="train", download=True)
val_set   = datasets.Flowers102("data", split="val", download=True)
test_set  = datasets.Flowers102("data", split="test", download=True)
print("Flowers-102 downloaded successfully.")