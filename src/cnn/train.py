import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from .model import get_model

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

def train(data_dir):
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    model = get_model()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=0.001)

    model.train()
    for epoch in range(3):
        for images, labels in loader:
            labels = labels.float().unsqueeze(1)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

    return model