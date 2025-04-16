import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GATv2Conv


class HybridGCNGATModel(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        num_layers,
        out_channels=None,
        dropout=0.0,
        act=F.relu,
        heads=8,
        concat=True,
        v2=False,
        **kwargs
    ):
        super(HybridGCNGATModel, self).__init__()

        if out_channels is None:
            out_channels = hidden_channels
            # strictly, the value of out_channels does not have to be equal to hidden_channels
            # but it is a reasonable practice to set them equal for simplicity

        # Store attributes that match expected interface
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.heads = heads
        self.concat = concat  # Whether to concatenate attention heads, note this is forced to be True for compatibility with our loss function.
        self.v2 = v2  # Whether to use GATv2 or GAT, note this is not followed in the code but putting here for completeness

        if v2:
            self.gat_layer = GATv2Conv
        else:
            self.gat_layer = GATConv

        # Create ModuleList named 'convs' to match expected interface
        # namely in test_model
        self.convs = nn.ModuleList()  # building up our model layer by layer
        self.act = act  # i.e., reLU

        # First layer: GCN for initial representation
        first_conv = GCNConv(
            in_channels, hidden_channels
        )  
        self.convs.append(first_conv)

        # Middle layers: Alternating GCN and GAT
        hidden_dim = (
            hidden_channels  # just keep them the same to not lose dimensionality
        )
        for i in range(1, num_layers - 1):
            if i % 2 == 0:
                # GCN layer
                layer = GCNConv(hidden_dim, hidden_channels)
                self.convs.append(layer)
                hidden_dim = hidden_channels
            else:
                # GAT layer
                # https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GATConv.html
                # from the “Graph Attention Networks” paper.
                # https://arxiv.org/abs/2105.14491 is the "How Attentive are Graph Attention Networks?" paper
                # which proposes GATv2
                # If concat=True (default), output will be heads * out_channels
                # If concat=False, output will be just out_channels but with multiple heads
                # Again, we are forcing concat=True for compatibility with our loss function.
                if concat:
                    gat_hidden = (
                        hidden_channels // heads
                    )  # Divide to maintain same overall dimension
                    layer = self.gat_layer(hidden_dim, gat_hidden, heads=heads, concat=True)
                    hidden_dim = gat_hidden * heads
                else:  # not followed but putting here for completeness
                    layer = self.gat_layer(
                        hidden_dim, hidden_channels, heads=heads, concat=False
                    )
                    hidden_dim = hidden_channels
                self.convs.append(layer)



        self.convs.append(GCNConv(hidden_dim, out_channels))

        # With our default number of layers = 4, we would have:
        # GCN -> GAT -> GCN -> GCN
        # so from a theoretical perspective it might be better to have five layers, i.e
        # GCN -> GAT -> GCN -> GAT -> GCN

        """
        https://www.sciencedirect.com/science/article/abs/pii/S1389128625000507
        Proposes a inputs -> GCN -> reLU -> GAT -> dropout -> GAT -> outputs approach

        . In the hybrid approach, GCN layers are used first to capture broader and more general 
        connectivity patterns in the graph. 
        These layers provide a foundation for the basic interactions between nodes and edges. 
        Then, GAT layers are integrated to refine the learned representations by focusing on the 
        most critical relationships in the data, and attention weights are assigned 
        to highlight important connections. 
        This two-stage process enables the model to learn both the general structure 
        (via GCN) and the specific fine-grained relationships 
        (via GAT) that are critical for anomaly detection

        https://arxiv.org/html/2503.00961v1
        also proposes a "Fusion" of Graph-Attention

        """

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)

            # Only reshape if using GAT with concat=False (which gives multi-dimensional output)
            if isinstance(conv, self.gat_layer) and not self.concat and x.dim() > 2:
                x = x.mean(dim=1)  # Average the heads instead of concatenating

            if i < len(self.convs) - 1:
                x = self.act(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        return x

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()


class HybridGCNGATBackbone(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        num_layers,
        out_channels=None,
        dropout=0.0,
        act=F.relu,
        **kwargs
    ):
        super(HybridGCNGATBackbone, self).__init__()
        self.hybrid_model = HybridGCNGATModel(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            out_channels=out_channels or hidden_channels,
            dropout=dropout,
            act=act,
            heads=kwargs.get("heads", 8),
            concat=True,  # force concat=True for compatibility
            v2 = kwargs.get("v2", False),  # whether to use GATv2
        )
        self.in_channels = in_channels  # Expose this for compatibility
        self.convs = self.hybrid_model.convs

    def forward(self, x, edge_index):
        return self.hybrid_model(x, edge_index)

    def reset_parameters(self):
        self.hybrid_model.reset_parameters()
