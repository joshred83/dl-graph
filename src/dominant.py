import math
import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple
from torch_geometric.nn import GCN
from torch_geometric.utils import to_dense_adj

from pygod.nn.decoder import DotProductDecoder
from pygod.nn.functional import double_recon_loss

from src.backbone import HybridGCNGATBackbone, GATBackbone


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
        - optional: ``HybridGCNGATBackbone``.
    apply_augmentation : bool, optional
        Whether to apply data augmentation. Default: ``True``.
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
    use_aggregation : bool, optional
        Whether to use feature aggregation. Default: ``False``.
    aggregation_mean : bool, optional
        Whether to use mean aggregation. Default: ``False``.
    aggregation_max : bool, optional
        Whether to use max aggregation. Default: ``False``.
    pos_weight_a : float, optional
        Positive weight for feature reconstruction loss. Default: ``0.5``.
    pos_weight_s : float, optional
        Positive weight for structure reconstruction loss. Default: ``0.5``.
    bce_s : bool, optional
        Whether to use binary cross entropy for structure reconstruction loss. Default: ``False``.
    **kwargs : optional
        Additional arguments for the backbone.
            if using HybridGCNGATBackbone, the following additional arguments are available:
            - heads (int): Number of attention heads for GAT layers, default value is 8 .
            - v2 (bool): Whether to use GATv2; if False, GAT is used. Default: ``True``.
    """

    def __init__(
        self,
        in_dim,
        hid_dim=64,
        num_layers=3,
        dropout=0.0,
        act=torch.nn.functional.relu,
        sigmoid_s=False,
        backbone='gcn',
        apply_augmentation=True,
        use_interpolation=False,
        interpolation_rate=0.2,
        use_perturbation=False,
        feature_noise=0.1,
        structure_noise=0.1,
        use_adaptive_alpha=False,
        alpha=0.5,
        end_alpha=0.5,
        use_aggregation=False,
        aggregation_mean=False,
        aggregation_max=False,
        pos_weight_a=0.5,  # params for double_recon_loss
        pos_weight_s=0.5,  # params for double_recon_loss
        bce_s=False,  # params for double_recon_loss
        **kwargs,
    ):



        super(DOMINANTAugmented, self).__init__()

        assert backbone in [
            'gcn',
            'gat',
            'hybrid',
        ], "Backbone must be one of ['gcn', 'gat', 'hybrid']"

        match backbone:
            case 'gcn':
                backbone = GCN
            case 'gat':
                backbone = GATBackbone
            case 'hybrid':
                backbone = HybridGCNGATBackbone

        # split the number of layers for the encoder and decoders
        assert num_layers >= 2, "Number of layers must be greater than or equal to 2."
        encoder_layers = math.floor(num_layers / 2)
        decoder_layers = math.ceil(num_layers / 2)

        self.attr_decoder = backbone(
            in_channels=hid_dim,
            hidden_channels=hid_dim,
            num_layers=decoder_layers,
            out_channels=in_dim,
            dropout=dropout,
            act=act,
            **kwargs,
        )

        self.struct_decoder = DotProductDecoder(
            in_dim=hid_dim,
            hid_dim=hid_dim,
            num_layers=decoder_layers - 1,
            dropout=dropout,
            act=act,
            sigmoid_s=sigmoid_s,
            backbone=backbone,
            **kwargs,
        )

        self.loss_func = double_recon_loss
        self.emb = None

        # Data augmentation settings
        self.apply_augmentation = apply_augmentation
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

        # Feature aggregation settings
        self.use_aggregation = use_aggregation
        self.aggregation_mean = aggregation_mean
        self.aggregation_max = aggregation_max

        self.original_in_dim = in_dim
        encoder_in_dim = in_dim

        if use_aggregation and not apply_augmentation:
            print("Warning: use_aggregation=True but apply_augmentation=False.")
            print(
                "Setting apply_augmentation=True to maintain dimensional consistency."
            )
            self.apply_augmentation = True

        assert self.use_aggregation == (
            self.aggregation_mean or self.aggregation_max
        ), "Feature aggregation must be enabled if mean or max aggregation is used."

        if self.use_aggregation:
            if self.aggregation_mean:
                print("Using mean aggregation.")
                encoder_in_dim += in_dim  # Add another in_dim features
            if self.aggregation_max:
                print("Using max aggregation.")
                encoder_in_dim += in_dim  # Add another in_dim features
        if self.use_interpolation:
            print("Using feature interpolation.")

        if self.use_perturbation:
            print("Using noise perturbation.")
        if self.use_adaptive_alpha:
            print("Using adaptive alpha.")

        print(f"Using {backbone.__name__} as backbone, with {num_layers} layers.")

        # print(f"Input features dimension: {in_dim}")
        # print(f"Encoder input dimension: {encoder_in_dim}")

        # this is essential to set the in_channels of the shared_encoder correctly with `encoder_in_dim`
        # otherwise, the shared_encoder will not work properly and you will have matrix-multiplication issues
        self.shared_encoder = backbone(
            in_channels=encoder_in_dim,
            hidden_channels=hid_dim,
            num_layers=encoder_layers,
            out_channels=hid_dim,
            dropout=dropout,
            act=act,
            **kwargs,
        )

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

        References
        - https://arxiv.org/pdf/2009.11746
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
        combined_neighbor_features = (
            out_neighbor_features + in_neighbor_features
        ) / 2.0
        interpolated_features = (
            1 - self.interpolation_rate
        ) * x_copy + self.interpolation_rate * combined_neighbor_features

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
            noise = torch.normal(0, self.feature_noise, size=x_copy.shape).to(
                x_copy.device
            )
            perturbed_x = x_copy + noise
        else:
            perturbed_x = x_copy

        # Structure perturbation with random edge dropout
        if self.structure_noise > 0:
            # Only apply perturbation to existing edges (where adj_copy > 0)
            # This ensures we only drop existing edges, not add new ones in random directions
            edge_mask = (adj_copy > 0).float()

            # Create random dropout mask, but only for existing edges
            dropout_mask = (
                torch.rand_like(adj_copy) > self.structure_noise
            ).float() * edge_mask

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
            augmented_x, augmented_adj = self.add_noise_perturbation(
                augmented_x, augmented_adj
            )

        return augmented_x, augmented_adj

    def forward(
        self, x: Tensor, edge_index: Tensor, apply_augmentation: bool = None
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward computation with optional data augmentation.

        Parameters
        ----------
        x : torch.Tensor
            Input attribute embeddings.
        edge_index : torch.Tensor
            Edge index.
        apply_augmentation : bool
            Whether to apply data augmentation. If None, use default from initialization.

        Returns
        -------
        x_ : torch.Tensor
            Reconstructed attribute embeddings.
        s_ : torch.Tensor
            Reconstructed adjacency matrix.
        """

        if apply_augmentation is None:
            apply_augmentation = self.apply_augmentation

        # Convert edge_index to dense adjacency matrix for augmentation
        dense_adj = to_dense_adj(edge_index)[0]

        # Track if we're using the original or augmented graph structure
        using_original_structure = True

        current_x = x
        current_adj = dense_adj
        current_edge_index = edge_index
        # print(f"Original x shape: {x.shape}")

        if apply_augmentation:
            # First apply operations that don't change feature dimensions
            if self.use_interpolation or self.use_perturbation:
                current_x, augmented_adj = self.augment_data(current_x, dense_adj)

                if self.use_perturbation:
                    using_original_structure = False
                    current_adj = augmented_adj
                    # Convert to edge_index format
                    current_edge_index = torch.nonzero(augmented_adj > 0).t()

            # Apply feature aggregation last as it changes input dimensions
            if self.use_aggregation:
                current_x = self.feature_aggregation(
                    current_x, current_adj, current_edge_index
                )

        # print(f"Encoder expected input channels: {self.shared_encoder.in_channels}")
        # print(f"attr_decoder expected input channels: {self.attr_decoder.in_channels}")

        # Store which structure was used for reference or analysis
        self.using_original_structure = using_original_structure

        # Pass through encoder and decoders
        self.emb = self.shared_encoder(current_x, current_edge_index)
        x_ = self.attr_decoder(self.emb, current_edge_index)
        s_ = self.struct_decoder(self.emb, current_edge_index)

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
            self.current_alpha = self.start_alpha - (
                self.start_alpha - self.end_alpha
            ) * (epoch / total_epochs)
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
        return self.loss_func(
            x,
            x_,
            s,
            s_,
            weight=self.current_alpha,
            pos_weight_a=self.pos_weight_a,
            pos_weight_s=self.pos_weight_s,
            bce_s=self.bce_s,
        )

    def feature_aggregation(
        self, x: Tensor, adj: Tensor, edge_index: Tensor = None
    ) -> Tensor:
        """
        Aggregate node features based on neighborhood statistics to obtain more
        discriminative normality/abnormality patterns.

        Parameters
        ----------
        x : torch.Tensor
            Node feature matrix
        adj : torch.Tensor
            Adjacency matrix (dense format)
        edge_index : torch.Tensor, optional
            Edge indices (sparse format, can be used instead of adj)
            not currently implemented (:

        Returns
        -------
        torch.Tensor
            Aggregated features with mean and/or max
            Note that the output dimension may not match the input dimension, if we did augment.
            This is because we concatenate the original features with the aggregated ones, across
            dim=1, i.e. the feature dimension.

            If we did not do aggregation, the output dimension will be original_in_dim
            If we did aggregation, the output dimension will be original_in_dim + in_dim
            (potetnially + in_dim again if we implement multiple aggregations)
        """
        x_copy = x.clone()
        result = x_copy  # Start with original features

        aggregations = []
        if self.aggregation_mean:
            # Compute the mean of the features of the neighbors
            out_degree = torch.sum(adj, dim=1)
            out_degree = torch.clamp(out_degree, min=1e-10)
            out_degree_inv = 1.0 / out_degree
            # we want to use the inverse of the out-degree, because
            # we need each row to sum to 1, so that the neighbors are weighted appropriately.
            out_degree_inv = torch.diag(out_degree_inv)
            row_normalized_adj = torch.mm(out_degree_inv, adj)
            # Use row-normalized adjacency to aggregate features from neighbors
            mean_neighbor_features = torch.mm(row_normalized_adj, x_copy)
            aggregations.append(mean_neighbor_features)

        if self.aggregation_max:
            # Softmax-based approximation of max
            temperature = 0.1
            # because max is not differentiable
            # by dividing adj by temperature, we push the highest values to be very high
            # and the others to be very low
            # so that when we apply softmax, the highest value will dominate,
            # thus serving to  approximate the max operation
            softmax_weights = torch.nn.functional.softmax(adj / temperature, dim=1)
            max_neighbor_features = torch.mm(softmax_weights, x_copy)
            aggregations.append(max_neighbor_features)

        # Concatenate all features
        if aggregations:
            concatenated = torch.cat([result] + aggregations, dim=1)
            return concatenated
        else:
            return result
