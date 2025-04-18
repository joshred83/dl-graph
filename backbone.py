import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GATv2Conv, LayerNorm


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
        v2=True,
        layer_norm=True,
        residual=True,
        last_layer='GAT',
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
        self.v2 = v2  # Whether to use GATv2 or GAT

        self.layer_norm = layer_norm  # Whether to use layer normalization
        self.residual = residual

        if v2:
            self.gat_layer = GATv2Conv
        else:
            self.gat_layer = GATConv

        # Create ModuleList named 'convs' to match expected interface
        # namely in test_model
        self.convs = nn.ModuleList()  # building up our model layer by layer

        if self.layer_norm:
            self.norms = nn.ModuleList()
        self.act = act  # i.e., reLU

        # First layer: GCN for initial representation
        first_conv = GCNConv(
            in_channels, hidden_channels
        )  
        self.convs.append(first_conv)

        if self.layer_norm:
            self.norms.append(LayerNorm(hidden_channels))


        # Middle layers: Alternating GCN and GAT
        hidden_dim = (
            hidden_channels  # just keep them the same to not lose dimensionality
        )
        for i in range(1, num_layers - 1):
            if i % 2 == 0:
                # GCN layer
                layer = GCNConv(hidden_dim, hidden_channels)
                self.convs.append(layer)
                if self.layer_norm:
                    self.norms.append(LayerNorm(hidden_channels))
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
                    layer = self.gat_layer(hidden_dim, gat_hidden, heads=heads, concat=True, add_self_loops=True)
                    if self.layer_norm:
                        self.norms.append(LayerNorm(gat_hidden * heads))

                    hidden_dim = gat_hidden * heads
                else:  # not followed but putting here for completeness
                    layer = self.gat_layer(
                        hidden_dim, hidden_channels, heads=heads, concat=False, add_self_loops=True
                    )
                    if self.layer_norm:
                        self.norms.append(LayerNorm(hidden_channels))
                    hidden_dim = hidden_channels
                self.convs.append(layer)


        if last_layer == 'GAT':
            self.convs.append(GATv2Conv(
            in_channels=hidden_channels, 
            out_channels=out_channels, 
            heads=1, 
            dropout=0.0,  # No dropout in final layer
            add_self_loops=True,
            concat=False  # Average multiple heads over one head, this works out dimension wise
        ))
        else:
            self.convs.append(GCNConv(hidden_dim, out_channels))

        if self.layer_norm:
            self.norms.append(LayerNorm(out_channels))

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
            h = conv(x, edge_index)

            # apply layer normalization
            if self.layer_norm and i < len(self.norms):
                h = self.norms[i](h)

            # Only reshape if using GAT with concat=False (which gives multi-dimensional output)
            if isinstance(conv, self.gat_layer) and not self.concat and x.dim() > 2:
                h = h.mean(dim=1)  # Average the heads instead of concatenating

            if i < len(self.convs) - 1:
                h = self.act(h)
                h = F.dropout(h, p=self.dropout, training=self.training)

            x = h

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
            v2 = kwargs.get("v2", True),  # whether to use GATv2
        )
        self.in_channels = in_channels  # Expose this for compatibility
        self.convs = self.hybrid_model.convs

    def forward(self, x, edge_index):
        return self.hybrid_model(x, edge_index)

    def reset_parameters(self):
        self.hybrid_model.reset_parameters()


class GATBackbone(nn.Module):
    def __init__(
        self, 
        in_channels, 
        hidden_channels, 
        out_channels, 
        num_layers, 
        heads=4,  # Reduced from 8 to 4 for stability
        dropout=0.1,  
        act=F.relu,  
        use_norm=True,
        residual=True,  # Added residual connections
        eps=1e-5,  # Small epsilon to prevent division by zero
        **kwargs
    ):
        super(GATBackbone, self).__init__()
        
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList() if use_norm else None
        self.act = act 
        self.dropout = dropout
        self.out_channels = out_channels
        self.heads = heads
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.use_norm = use_norm
        self.residual = residual
        self.eps = eps
        
        # Skip connections for residual if dimensions match
        self.use_first_res = (in_channels == hidden_channels * heads)
        
        # While this paper uses "MGAT" instead of the conventional "GAT"
        # it does use multiple GATs https://www.mdpi.com/2227-7390/12/2/293 albeit in parallel.

        # Input layer
        self.convs.append(GATv2Conv(
            in_channels=in_channels, 
            out_channels=hidden_channels, 
            heads=heads, 
            dropout=dropout, 
            add_self_loops=True,  # Explicitly add self-loops
            concat=True
        ))
        
        if use_norm:
            self.norms.append(LayerNorm(hidden_channels * heads))
        
        # Hidden layers
        for i in range(num_layers - 2):
            self.convs.append(GATv2Conv(
                in_channels=hidden_channels * heads, 
                out_channels=hidden_channels, 
                heads=heads, 
                dropout=dropout,
                add_self_loops=True,
                concat=True
            ))
            
            if use_norm:
                self.norms.append(LayerNorm(hidden_channels * heads))
        
        # Output layer with careful initialization
        self.final_conv = GATv2Conv(
            in_channels=hidden_channels * heads, 
            out_channels=out_channels, 
            heads=1, 
            dropout=0.0,  # No dropout in final layer
            add_self_loops=True,
            concat=False  # Average multiple heads over one head, this works out dimension wise
        )
        
        # Initialize with small weights to prevent exploding gradients at start
        self._init_parameters()
    
    def _init_parameters(self):
        """Custom initialization to help with stability"""
        
        # higher gain in the earlier layers to have stronger initial activations
        for conv in self.convs:
            if hasattr(conv, 'lin'): # not in GATv2Conv
                nn.init.xavier_normal_(conv.lin.weight, gain=0.5)
                if conv.lin.bias is not None:
                    nn.init.zeros_(conv.lin.bias)
            if hasattr(conv, 'lin_l'):
                nn.init.xavier_normal_(conv.lin_l.weight, gain=0.5)
                if conv.lin_l.bias is not None:
                    nn.init.zeros_(conv.lin_l.bias)
            if hasattr(conv, 'lin_r'):
                nn.init.xavier_normal_(conv.lin_r.weight, gain=0.5)
                if conv.lin_r.bias is not None:
                    nn.init.zeros_(conv.lin_r.bias)
                
            if hasattr(conv, 'att'):
                nn.init.xavier_normal_(conv.att, gain=0.5)
        
        # Initialize final layer conservatively - lower gain
        # means less aggressive initialization, avoiding extreme initial values
        # because attention can "amplify" the gradients
        if hasattr(self.final_conv, 'lin'):
            nn.init.xavier_normal_(self.final_conv.lin.weight, gain=0.1)
            if self.final_conv.lin.bias is not None:
                nn.init.zeros_(self.final_conv.lin.bias)
        
        if hasattr(self.final_conv, 'lin_l'):
            nn.init.xavier_normal_(self.final_conv.lin_l.weight, gain=0.1)
            if self.final_conv.lin_l.bias is not None:
                nn.init.zeros_(conv.lin_l.bias)

        if hasattr(self.final_conv, 'lin_r'):
            nn.init.xavier_normal_(self.final_conv.lin_r.weight, gain=0.1)
            if self.final_conv.lin_r.bias is not None:
                nn.init.zeros_(conv.lin_r.bias)
                
        if hasattr(self.final_conv, 'att'):
            nn.init.xavier_normal_(self.final_conv.att, gain=0.1)
    
    def forward(self, x, edge_index):
        h = x  # Original input for potential residual connections
        
        # Process through hidden layers with careful handling of NaNs and Infs
        for i, conv in enumerate(self.convs):
            # Apply the GAT convolution
            x_new = conv(x, edge_index)
            
            # Check and fix any numerical issues immediately
            x_new = torch.nan_to_num(x_new, nan=0.0, posinf=self.eps, neginf=-self.eps)
            
            # Apply normalization if enabled
            if self.use_norm and i < len(self.norms):
                x_new = self.norms[i](x_new)
            
            # Apply activation function
            x_new = self.act(x_new)
            
            # Apply dropout except in the final layer
            x_new = F.dropout(x_new, p=self.dropout, training=self.training)
            
            # Add residual connection if dimensions match and not the first layer
            # (unless use_first_res is True)
            if self.residual and (i > 0 or self.use_first_res) and x.size() == x_new.size():
                x = x_new + x
            else:
                x = x_new
            
            # Safe check for NaNs - helps with debugging
            if torch.isnan(x).any():
                print(f"NaN values detected in layer {i} after processing")
                x = torch.nan_to_num(x_new, nan=0.0, posinf=self.eps, neginf=-self.eps)
        
        # Final layer (output) handling
        x = self.final_conv(x, edge_index)
        if torch.isnan(x).any():
            print("NaN values detected in final layer")
            x = torch.nan_to_num(x_new, nan=0.0, posinf=self.eps, neginf=-self.eps)
        
        
        return x
    
    def reset_parameters(self):
        """Reset parameters of all layers"""
        for conv in self.convs:
            conv.reset_parameters()
        
        self.final_conv.reset_parameters()
        
        if self.use_norm:
            for norm in self.norms:
                norm.reset_parameters()
        
        # Re-apply our custom initialization
        self._init_parameters()