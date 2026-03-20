import torchvision.models as models
import torch.nn as nn

def get_model():
    model = models.mobilenet_v3_small(pretrained=True)
    
    for param in model.parameters():
        param.requires_grad = False

    model.classifier[3] = nn.Linear(1024, 1)
    return model