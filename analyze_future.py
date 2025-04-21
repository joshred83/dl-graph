import torch
from loaders import load_elliptic
from main import Net, train, test

def main():
    # load_elliptic yields a DataLoader for each timestep
    timesteps = list(load_elliptic())

    # define loss
    criterion = torch.nn.CrossEntropyLoss()

    for t in range(len(timesteps) - 1):
        train_loader = timesteps[t]
        eval_loader  = timesteps[t + 1]

        # instantiate a fresh model & optimizer
        model = Net()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # train on timestep t
        train(model, train_loader, optimizer, criterion)

        # evaluate on timestep t+1
        eval_loss, eval_acc = test(model, eval_loader, criterion)

        print(f"Timestep {t} → {t+1}: Loss={eval_loss:.4f}, Acc={eval_acc:.4f}")

if __name__ == "__main__":
    main()