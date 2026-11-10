import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
import timm

def get_baseline_model():
    # Baseline: Pretrained ResNet-18
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 102)
    return model

def get_main_model():
    # Main Model: Pretrained Vision Transformer (ViT-B/16)
    model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=102)
    return model