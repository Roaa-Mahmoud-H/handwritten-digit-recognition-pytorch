"""Train MLP and CNN models and evaluate their test accuracy."""
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.data import get_dataloaders
from src.model import CNN, MLP


def accuracy_for(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            correct += (model(images).argmax(1) == labels).sum().item()
            total += labels.size(0)
    return correct / total



# Main kept straightforward for easy editing by learners.
def main():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    train_loader, val_loader, test_loader = get_dataloaders()
    criterion = nn.CrossEntropyLoss()
    results = []

    for name, model in (("MLP", MLP()), ("CNN", CNN())):
        print(f"\nTraining {name}")
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
        best_val_acc = 0.0
        best_path = Path(f"{name.lower()}_best.pth")
        for epoch in range(10):
            model.train()
            running_loss = correct = total = 0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * labels.size(0)
                correct += (outputs.argmax(1) == labels).sum().item()
                total += labels.size(0)
            val_acc = accuracy_for(model, val_loader, device)
            scheduler.step()
            print(f"Epoch {epoch + 1:02d}/10 | loss {running_loss / total:.4f} | "
                  f"train acc {correct / total:.4%} | val acc {val_acc:.4%}")
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), best_path)
        model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
        test_acc = accuracy_for(model, test_loader, device)
        results.append((name, best_val_acc, test_acc, sum(p.numel() for p in model.parameters())))
        if name == "CNN":
            torch.save(model.state_dict(), "mnist_cnn.pth")

    print("\nFinal results")
    print(f"{'Model':<8} {'Best val accuracy':>18} {'Test accuracy':>15} {'Parameters':>13}")
    for name, val_acc, test_acc, params in results:
        print(f"{name:<8} {val_acc:>17.2%} {test_acc:>14.2%} {params:>13,}")
    print("\nSaved CNN weights to mnist_cnn.pth")


if __name__ == "__main__":
    main()
