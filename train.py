import argparse
import torch
from torch.utils.data import DataLoader
from src.loaders import load_elliptic, make_loader

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for batch in loader:
        x, y = batch.x.to(device), batch.y.to(device)
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(loader)

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            x, y = batch.x.to(device), batch.y.to(device)
            outputs = model(x)
            loss = criterion(outputs, y)
            running_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    avg_loss = running_loss / len(loader)
    accuracy = correct / total
    return avg_loss, accuracy

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the Elliptic dataset
    data = load_elliptic()
    # Split nodes for train/val (example: use masks if available)
    train_mask = getattr(data, "train_mask", None)
    val_mask = getattr(data, "val_mask", None)

    train_loader = make_loader(data, batch_size=args.batch_size, shuffle=True, input_nodes=train_mask)
    val_loader = make_loader(data, batch_size=args.batch_size, shuffle=False, input_nodes=val_mask)

    # Example model (replace with your actual model)
    from torch.nn import Linear, CrossEntropyLoss
    class SimpleMLP(torch.nn.Module):
        def __init__(self, in_dim, out_dim):
            super().__init__()
            self.fc = Linear(in_dim, out_dim)
        def forward(self, x):
            return self.fc(x)

    model = SimpleMLP(data.num_features, int(data.y.max().item()) + 1).to(device)
    criterion = CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(f"Epoch {epoch:02d} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc:.4f}")

    torch.save(model.state_dict(), "model.pth")

if __name__ == "__main__":
    main()