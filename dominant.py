import math
import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple
from torch_geometric.nn import GCN
from torch_geometric.utils import to_dense_adj

from pygod.nn.decoder import DotProductDecoder
from pygod.nn.functional import double_recon_loss


class DOMINANTAugmented(nn.Module):
    """
    Deep Anomaly Detection on Attributed Networks with built-in data augmentation

    DOMINANT is an anomaly detector consisting of a shared graph
    convolutional encoder, a structure reconstruction decoder, and an
    attribute reconstruction decoder. The reconstruction mean squared
    error of the decoders are defined as structure anomaly score and
    attribute anomaly score, respectively.

    This implementation includes augmentation techniques:
    - Feature interpolation: Interpolate node features with neighbor features
    - Noise perturbation: Add random noise to features and structure
    - Adaptive alpha: Gradually adjust the balance parameter during training

    See :cite:`ding2019deep` for details.

    Parameters
    ----------
    in_dim : int
        Input dimension of model.
    hid_dim :  int
       Hidden dimension of model. Default: ``64``.
    num_layers : int, optional
       Total number of layers in model. A half (floor) of the layers
       are for the encoder, the other half (ceil) of the layers are
       for decoders. Default: ``4``.
    dropout : float, optional
       Dropout rate. Default: ``0.``.
    act : callable activation function or None, optional
       Activation function if not None.
       Default: ``torch.nn.functional.relu``.
    sigmoid_s : bool, optional
        Whether to apply sigmoid to the structure reconstruction.
        Default: ``False``.
    backbone : torch.nn.Module, optional
        The backbone of the deep detector implemented in PyG.
        Default: ``torch_geometric.nn.GCN``.
    use_interpolation : bool, optional
        Whether to use feature interpolation. Default: ``False``.
    interpolation_rate : float, optional
        Feature interpolation rate. Default: ``0.2``.
    use_perturbation : bool, optional
        Whether to use noise perturbation. Default: ``False``.
    feature_noise : float, optional
        Feature noise level. Default: ``0.1``.
    structure_noise : float, optional
        Structure noise level. Default: ``0.1``.
    use_adaptive_alpha : bool, optional
        Whether to use adaptive alpha. Default: ``False``.
    alpha : float, optional
        Initial alpha value. Default: ``0.5``.
    end_alpha : float, optional
        Final alpha value. Default: ``0.5``.
    pos_weight_a : float, optional
        Positive weight for feature reconstruction loss. Default: ``0.5``.
    pos_weight_s : float, optional
        Positive weight for structure reconstruction loss. Default: ``0.5``.
    bce_s : bool, optional 
        Whether to use binary cross entropy for structure reconstruction loss. Default: ``False``.
    **kwargs : optional
        Additional arguments for the backbone.
    """

    def __init__(self,
                 in_dim,
                 hid_dim=64,
                 num_layers=4,
                 dropout=0.,
                 act=torch.nn.functional.relu,
                 sigmoid_s=False,
                 backbone=GCN,
                 use_interpolation=False,
                 interpolation_rate=0.2,
                 use_perturbation=False,
                 feature_noise=0.1,
                 structure_noise=0.1,
                 use_adaptive_alpha=False,
                 alpha=0.5,
                 end_alpha=0.5,
                 pos_weight_a=0.5, # params for double_recon_loss
                 pos_weight_s=0.5, # params for double_recon_loss
                 bce_s=False, # params for double_recon_loss
                 **kwargs):
 
        super(DOMINANTAugmented, self).__init__()

        # split the number of layers for the encoder and decoders
        assert num_layers >= 2, \
            "Number of layers must be greater than or equal to 2."
        encoder_layers = math.floor(num_layers / 2)
        decoder_layers = math.ceil(num_layers / 2)

        self.shared_encoder = backbone(in_channels=in_dim,
                                       hidden_channels=hid_dim,
                                       num_layers=encoder_layers,
                                       out_channels=hid_dim,
                                       dropout=dropout,
                                       act=act,
                                       **kwargs)

        self.attr_decoder = backbone(in_channels=hid_dim,
                                     hidden_channels=hid_dim,
                                     num_layers=decoder_layers,
                                     out_channels=in_dim,
                                     dropout=dropout,
                                     act=act,
                                     **kwargs)

        self.struct_decoder = DotProductDecoder(in_dim=hid_dim,
                                                hid_dim=hid_dim,
                                                num_layers=decoder_layers - 1,
                                                dropout=dropout,
                                                act=act,
                                                sigmoid_s=sigmoid_s,
                                                backbone=backbone,
                                                **kwargs)

        self.loss_func = double_recon_loss
        self.emb = None
        
        # Data augmentation settings
        self.use_interpolation = use_interpolation
        self.interpolation_rate = interpolation_rate
        self.use_perturbation = use_perturbation
        self.feature_noise = feature_noise
        self.structure_noise = structure_noise
        self.use_adaptive_alpha = use_adaptive_alpha
        self.current_alpha = alpha
        self.start_alpha = alpha
        self.end_alpha = end_alpha
        self.pos_weight_a = pos_weight_a
        self.pos_weight_s = pos_weight_s
        self.bce_s = bce_s

    def feature_interpolation(self, x: Tensor, adj: Tensor) -> Tensor:
        """
        Interpolate node features based on their neighbors for directed graphs
        
        Parameters
        ----------
        x : torch.Tensor
            Node feature matrix
        adj : torch.Tensor
            Adjacency matrix (directed)
            
        Returns
        -------
        torch.Tensor
            Interpolated features
        """
        x_copy = x.clone()
        
        # Option 1: Using outgoing neighbors (rows of adjacency matrix)
        out_degree = torch.sum(adj, dim=1)
        out_degree = torch.clamp(out_degree, min=1e-10)
        # Create adjacency matrix
        out_degree_inv = 1.0 / out_degree
        out_degree_inv = torch.diag(out_degree_inv)
        row_normalized_adj = torch.mm(out_degree_inv, adj)
        
        # Use row-normalized adjacency to aggregate features from outgoing neighbors
        out_neighbor_features = torch.mm(row_normalized_adj, x_copy)
        
        # Using incoming neighbors (columns of adjacency matrix)
        in_degree = torch.sum(adj, dim=0)
        in_degree = torch.clamp(in_degree, min=1e-10)
        in_degree_inv = 1.0 / in_degree
        in_degree_inv = torch.diag(in_degree_inv)
        col_normalized_adj = torch.mm(adj, in_degree_inv)
        
        # Use column-normalized adjacency to aggregate features from incoming neighbors
        in_neighbor_features = torch.mm(col_normalized_adj.t(), x_copy)
        
        # interpolated_features = (1 - self.interpolation_rate) * x_copy + self.interpolation_rate * out_neighbor_features
        
        # interpolated_features = (1 - self.interpolation_rate) * x_copy + self.interpolation_rate * in_neighbor_features
        
        # 3. Use both in- and out-neighbors with equal weight
        combined_neighbor_features = (out_neighbor_features + in_neighbor_features) / 2.0
        interpolated_features = (1 - self.interpolation_rate) * x_copy + self.interpolation_rate * combined_neighbor_features
        
        return interpolated_features

    def add_noise_perturbation(self, x: Tensor, adj: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Add random noise to both features and structure
        
        Parameters
        ----------
        x : torch.Tensor
            Node feature matrix
        adj : torch.Tensor
            Adjacency matrix
            
        Returns
        -------
        tuple
            Perturbed features and adjacency matrix
        """
        # Create copies to avoid modifying the originals
        x_copy = x.clone()
        adj_copy = adj.clone()
        
        # Feature perturbation with Gaussian noise
        if self.feature_noise > 0:
            noise = torch.normal(0, self.feature_noise, size=x_copy.shape).to(x_copy.device)
            perturbed_x = x_copy + noise
        else:
            perturbed_x = x_copy
        
        # Structure perturbation with random edge dropout
        if self.structure_noise > 0:
                # Only apply perturbation to existing edges (where adj_copy > 0)
                # This ensures we only drop existing edges, not add new ones in random directions
                edge_mask = (adj_copy > 0).float()
                
                # Create random dropout mask, but only for existing edges
                dropout_mask = (torch.rand_like(adj_copy) > self.structure_noise).float() * edge_mask
                
                # Apply the dropout mask to existing edges
                perturbed_adj = adj_copy * dropout_mask
        else:
            perturbed_adj = adj_copy
        
        return perturbed_x, perturbed_adj

    def augment_data(self, x: Tensor, adj: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Combine feature interpolation and noise perturbation
        
        Parameters
        ----------
        x : torch.Tensor
            Node feature matrix
        adj : torch.Tensor
            Adjacency matrix
            
        Returns
        -------
        tuple
            Augmented features and adjacency matrix
        """
        augmented_x = x
        augmented_adj = adj
        
        # Apply feature interpolation if enabled
        if self.use_interpolation:
            augmented_x = self.feature_interpolation(augmented_x, adj)
        
        # Apply noise perturbation if enabled
        if self.use_perturbation:
            augmented_x, augmented_adj = self.add_noise_perturbation(augmented_x, augmented_adj)
        
        return augmented_x, augmented_adj

    def forward(self, x: Tensor, edge_index: Tensor, apply_augmentation: bool = True) -> Tuple[Tensor, Tensor]:
        """
        Forward computation with optional data augmentation.

        Parameters
        ----------
        x : torch.Tensor
            Input attribute embeddings.
        edge_index : torch.Tensor
            Edge index.
        apply_augmentation : bool
            Whether to apply data augmentation. Default: ``True``.

        Returns
        -------
        x_ : torch.Tensor
            Reconstructed attribute embeddings.
        s_ : torch.Tensor
            Reconstructed adjacency matrix.
        """
        # Convert edge_index to dense adjacency matrix for augmentation
        dense_adj = to_dense_adj(edge_index)[0]
        
        # Track if we're using the original or augmented graph structure
        using_original_structure = True
        
        # Apply data augmentation if enabled
        if apply_augmentation and (self.use_interpolation or self.use_perturbation):
            # Apply feature and/or structure augmentation
            augmented_x, augmented_dense_adj = self.augment_data(x, dense_adj)
            
            # If structure was perturbed, convert back to edge_index and use throughout
            if self.use_perturbation:
                using_original_structure = False
                # Convert to edge_index format, handling potential weights
                # This is more precise than the original implementation
                augmented_edge_index = torch.nonzero(augmented_dense_adj > 0).t()
                # If the adjacency matrix has weights, extract them
                edge_weights = augmented_dense_adj[augmented_edge_index[0], augmented_edge_index[1]]
                
                # Use augmented graph structure for encoder
                self.emb = self.shared_encoder(augmented_x, augmented_edge_index)
                
                # Use the same augmented graph structure for decoders
                x_ = self.attr_decoder(self.emb, augmented_edge_index)
                s_ = self.struct_decoder(self.emb, augmented_edge_index)
            else:
                # Only features were augmented, not structure
                # Use original edge_index but augmented features
                self.emb = self.shared_encoder(augmented_x, edge_index)
                x_ = self.attr_decoder(self.emb, edge_index)
                s_ = self.struct_decoder(self.emb, edge_index)
        else:
            # No augmentation, use original data
            self.emb = self.shared_encoder(x, edge_index)
            x_ = self.attr_decoder(self.emb, edge_index)
            s_ = self.struct_decoder(self.emb, edge_index)

        # Store which structure was used for reference or analysis
        self.using_original_structure = using_original_structure
        
        return x_, s_
        

    @staticmethod
    def process_graph(data):
        """
        Obtain the dense adjacency matrix of the graph.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Input graph.
        """
        data.s = to_dense_adj(data.edge_index)[0]

    def update_alpha(self, epoch: int, total_epochs: int) -> float:
        """
        Update the alpha parameter based on current epoch
        
        Parameters
        ----------
        epoch : int
            Current epoch
        total_epochs : int
            Total number of epochs
        
        Returns
        -------
        float
            Updated alpha value
        """
        if self.use_adaptive_alpha:
            self.current_alpha = self.start_alpha - (self.start_alpha - self.end_alpha) * (epoch / total_epochs)
        return self.current_alpha
    
    def get_alpha(self) -> float:
        """
        Get the current alpha value
        
        Returns
        -------
        float
            Current alpha value
        """
        return self.current_alpha
    
    def compute_loss(self, x: Tensor, x_: Tensor, s: Tensor, s_: Tensor) -> Tensor:
            """
            Compute loss using double_recon_loss with current alpha value
            
            Parameters
            ----------
            x : torch.Tensor
                Ground truth node feature
            x_ : torch.Tensor
                Reconstructed node feature
            s : torch.Tensor
                Ground truth node structure
            s_ : torch.Tensor
                Reconstructed node structure
                
            Returns
            -------
            score : torch.tensor
                Outlier scores with gradients
            """
            return self.loss_func(x, x_, s, s_, 
                                weight=self.current_alpha,
                                pos_weight_a=self.pos_weight_a,
                                pos_weight_s=self.pos_weight_s,
                                bce_s=self.bce_s)