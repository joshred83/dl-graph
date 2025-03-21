import torch
import torch.nn.functional as F
from torch_geometric.datasets import EllipticBitcoinDataset
from torch_geometric.nn import GCNConv
from sklearn.metrics import classification_report

# Load the Elliptic dataset
dataset = EllipticBitcoinDataset(root='elliptic_dataset')
data = dataset[0]

# Train/test split based on provided masks
train_mask = data.train_mask
test_mask = data.test_mask

# Model definition
class GCN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes):
        super(GCN, self).__init__()
        
        # Two Graph Convolutional layers
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# Initialize model
model = GCN(
    num_features=data.num_node_features,
    hidden_channels=64,
    num_classes=dataset.num_classes
)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training function
def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.nll_loss(out[train_mask], data.y[train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

# Testing function
@torch.no_grad()
def test():
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out[test_mask].argmax(dim=1)
    y_true = data.y[test_mask].cpu().numpy()
    y_pred = pred.cpu().numpy()
    report = classification_report(y_true, y_pred, target_names=['Licit', 'Illicit'])
    print(report)

# Training loop
for epoch in range(1, 51):
    loss = train()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

# Evaluation
test()