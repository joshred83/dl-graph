
from pygod.detector import DOMINANT as PyGDOMINANT
#from pygod.detector import DeepDetector
#from pygod.nn import DOMINANTBase
#import torch.nn.functional as F
import time
from inspect import signature
from abc import ABC, abstractmethod
from typing import Callable
from typing import Optional  

import torch
import numpy as np
#from scipy.stats import binom
#from scipy.special import erf

#from torch_geometric.nn import GIN
from torch_geometric import compile
from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import to_dense_adj


from functools import partial



from .metrics import (
    eval_roc_auc,
    eval_average_precision,
    eval_f1,
    eval_precision_at_k,  
      eval_recall_at_k,
      eval_precision,
      eval_recall,
      eval_classification_report)


class DOMINANT(PyGDOMINANT):
    """
    Deep Anomaly Detection on Attributed Networks

    Patched Dominant for better analysis of loss. 
    See :cite:`ding2019deep` for details.

    Parameters
    ----------
    hid_dim :  int, optional
        Hidden dimension of model. Default: ``64``.
    num_layers : int, optional
       Total number of layers in model. A half (floor) of the layers
       are for the encoder, the other half (ceil) of the layers are
       for decoders. Default: ``4``.
    dropout : float, optional
        Dropout rate. Default: ``0.``.
    weight_decay : float, optional
        Weight decay (L2 penalty). Default: ``0.``.
    act : callable activation function or None, optional
        Activation function if not None.
        Default: ``torch.nn.functional.relu``.
    sigmoid_s : bool, optional
        Whether to use sigmoid function to scale the reconstructed
        structure. Default: ``False``.
    backbone : torch.nn.Module, optional
        The backbone of the deep detector implemented in PyG.
        Default: ``torch_geometric.nn.GCN``.
    contamination : float, optional
        The amount of contamination of the dataset in (0., 0.5], i.e.,
        the proportion of outliers in the dataset. Used when fitting to
        define the threshold on the decision function. Default: ``0.1``.
    lr : float, optional
        Learning rate. Default: ``0.004``.
    epoch : int, optional
        Maximum number of training epoch. Default: ``100``.
    gpu : int
        GPU Index, -1 for using CPU. Default: ``-1``.
    batch_size : int, optional
        Minibatch size, 0 for full batch training. Default: ``0``.
    num_neigh : int, optional
        Number of neighbors in sampling, -1 for all neighbors.
        Default: ``-1``.
    weight : float, optional
        Weight between reconstruction of node feature and structure.
        Default: ``0.5``.
    verbose : int, optional
        Verbosity mode. Range in [0, 3]. Larger value for printing out
        more log information. Default: ``0``.
    save_emb : bool, optional
        Whether to save the embedding. Default: ``False``.
    compile_model : bool, optional
        Whether to compile the model with ``torch_geometric.compile``.
        Default: ``False``.
    **kwargs : optional
        Additional arguments for the backbone.

    Attributes
    ----------
    decision_score_ : torch.Tensor
        The outlier scores of the training data. Outliers tend to have
        higher scores. This value is available once the detector is
        fitted.
    threshold_ : float
        The threshold is based on ``contamination``. It is the
        :math:`N \\times` ``contamination`` most abnormal samples in
        ``decision_score_``. The threshold is calculated for generating
        binary outlier labels.
    label_ : torch.Tensor
        The binary labels of the training data. 0 stands for inliers
        and 1 for outliers. It is generated by applying
        ``threshold_`` on ``decision_score_``.
    emb : torch.Tensor or tuple of torch.Tensor or None
        The learned node hidden embeddings of shape
        :math:`N \\times` ``hid_dim``. Only available when ``save_emb``
        is ``True``. When the detector has not been fitted, ``emb`` is
        ``None``. When the detector has multiple embeddings,
        ``emb`` is a tuple of torch.Tensor.
    """
    def __init__(self,*args,  loader_fn: Callable = None,  **kwds):
        #loader factory: Callable = None should function that generates
        #  a  DataLoader object, using only data as an argument. 
   

        super().__init__(*args, **kwds)


        if loader_fn is None:
            # If no loader factory is provided, use the default NeighborLoader
            # self.batch_size  and self.num_neigh are set in the super class
            self.loader_fn = partial(NeighborLoader, batch_size=self.batch_size, num_neighbors=self.num_neigh)
        else:
            self.loader_fn = loader_fn
        
        self.history_ = {
            "train": {"total": [], "attr": [], "struct": [],
                      'auc': [], 'precision': [], 'recall': [],
              'ap': [], 'f1': []},
            'eval':  {'total': [], 'attr': [], 'struct': [],
              'auc': [], 'precision': [], 'recall': [],
              'ap': [], 'f1': []}
        }
        # Track per-node anomaly scores for train/eval
        self.scores_ = {
            "train": None,
            "eval": None
        }
        print("DOMINANT initialized")
        print("kwargs: ", kwds)
        print("args: ", args)
        
    @staticmethod
    def process_graph(data, batch_mode=False):
        """
        Obtain the dense adjacency matrix of the graph.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Input graph.
        """
        if batch_mode:
            assert hasattr(data, "s"), "Batch mode requires dense adjacency matrix."
            assert hasattr(data, "n_id"), "Batch mode requires dense adjacency matrix."
            data.s_batch = data.s[:, data.n_id]
        else:
            data.s = to_dense_adj(data.edge_index)[0]
            data.s_batch = data.s
    def forward_model(self, data):
        data.to(self.device)
        self.process_graph(data)

        batch_size = data.batch_size

        x = data.x
        s = data.s_batch
        edge_index = data.edge_index
        x_hat, s_hat = self.model(x, edge_index)

        x_seeds     = x[:batch_size]                        # [batch_size × in_dim]
        xhat_seeds  = x_hat[:batch_size]                    # [batch_size × in_dim]
        s_seeds     = s[:batch_size][:, :batch_size]              # [batch_size × batch_size]
        shat_seeds  = s_hat[:batch_size][:, :batch_size]            # [batch_size × batch_size]]
        score, attr_score, struct_score = double_recon_loss(
                                     x_seeds,
                                     xhat_seeds,
                                     s_seeds,
                                     shat_seeds,
                                     self.weight)
        
        loss = torch.mean(score)
        attr_loss = torch.mean(attr_score)
        struct_loss = torch.mean(struct_score)
    
        return loss, score.detach().cpu(), attr_loss, struct_loss
    @staticmethod
    def label_mask(y_true, mask_value=2):  
        """
        Create a mask for the labels.
        """
        return y_true != mask_value
 
    def fit(self, data, label =None, eval_data = None, eval_label = None):

        ### loss tracking patch ###


        if label is None and hasattr(data, "y"):
            label = data.y

        self.process_graph(data)
        self.eval_data = eval_data
        if self.eval_data is not None:
            self.process_graph(self.eval_data)

        self.num_nodes, self.in_dim = data.x.shape
        if self.batch_size == 0:
            self.batch_size = data.x.shape[0]


        self.model = self.init_model(**self.kwargs)
        if self.compile_model:
            self.model = compile(self.model)
        if not self.gan:
            optimizer = torch.optim.Adam(self.model.parameters(),
                                         lr=self.lr,
                                         weight_decay=self.weight_decay)
        else:
            self.opt_in = torch.optim.Adam(self.model.inner.parameters(),
                                           lr=self.lr,
                                           weight_decay=self.weight_decay)
            optimizer = torch.optim.Adam(self.model.outer.parameters(),
                                         lr=self.lr,
                                         weight_decay=self.weight_decay)

        self.model.train()
        self.decision_score_ = torch.zeros(data.x.shape[0])
        loader = self.loader_fn(data=data, input_nodes=None)
        for epoch in range(self.epoch):
            self.epoch=epoch
            start_time = time.time()
            epoch_loss = 0
            attr_loss = 0
            struct_loss = 0
            if self.gan:
                self.epoch_loss_in = 0
            for batch, sampled_data in enumerate(loader):
                self.batch = batch
                batch_size = sampled_data.batch_size
                node_idx = sampled_data.n_id
                self.process_graph(sampled_data, batch_mode=True)
                loss, raw_score, batch_attr_loss, batch_struct_loss = self.forward_model(sampled_data)
                epoch_loss += loss.item() * batch_size
                attr_loss += batch_attr_loss.item() * batch_size
                struct_loss += batch_struct_loss.item() * batch_size
                self.decision_score_[node_idx[:batch_size]] = raw_score
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Save per-epoch mean losses for training
            loss_value = epoch_loss / data.x.shape[0]
            attr_losses = attr_loss / data.x.shape[0]
            struct_losses = struct_loss / data.x.shape[0]
            self.history_["train"]["total"].append(float(loss_value))
            self.history_["train"]["attr"].append(float(attr_losses))
            self.history_["train"]["struct"].append(float(struct_losses))

            # Save per-epoch mean losses for evaluation
            self.epoch_threshold_ = np.percentile(self.decision_score_, 100 * (1 - self.contamination))
            mask = self.label_mask(label)
            y_train_pred = (self.decision_score_[mask] > self.epoch_threshold_).long()
            
            train_label = label[mask].cpu().numpy()
            score = self.decision_score_[mask].cpu().numpy()
            # compute train metrics
            auc_train  = eval_roc_auc(train_label, score)
            prec_train = eval_precision(train_label, y_train_pred)
            rec_train  = eval_recall(  train_label, y_train_pred)
            ap_train   = eval_average_precision(train_label, score)
            f1_train   = eval_f1(train_label, y_train_pred)

            # store into history_
            self.history_['train']['auc'].append(auc_train)
            self.history_['train']['precision'].append(prec_train)
            self.history_['train']['recall'].append(rec_train)
            self.history_['train']['ap'].append(ap_train)
            self.history_['train']['f1'].append(f1_train)

# and log it
            logger(epoch=epoch,
                loss=loss_value,
                struct=struct_losses,
                attr=attr_losses,
                auc=auc_train,
                precision=prec_train,
                recall=rec_train,
                ap=ap_train,
                f1=f1_train,    
                time=time.time()-start_time,
                verbose=self.verbose,
                train=True)
            if self.gan:
                loss_value = (self.epoch_loss_in / data.x.shape[0], loss_value)
           


            
        
            if self.eval_data is not None:
                self.evaluate(epoch=epoch)
        self.decision_score_ = self.decision_score_.detach().cpu()
        self._process_decision_score()
        self.scores_["train"] = self.decision_score_.numpy()
        return self

    def decision_function(self, data, label=None):

        self.process_graph(data)
        loader = self.loader_fn(data=data, input_nodes=None)

        self.model.eval()
        outlier_score = torch.zeros(data.x.shape[0])
        if self.save_emb:
            if type(self.hid_dim) is tuple:
                self.emb = (torch.zeros(data.x.shape[0], self.hid_dim[0]),
                            torch.zeros(data.x.shape[0], self.hid_dim[1]))
            else:
                self.emb = torch.zeros(data.x.shape[0], self.hid_dim)
        start_time = time.time()
        test_loss = 0
        for sampled_data in loader:
            loss, score, *_ = self.forward_model(sampled_data)
            batch_size = sampled_data.batch_size
            node_idx = sampled_data.n_id
            if self.save_emb:
                if type(self.hid_dim) is tuple:
                    self.emb[0][node_idx[:batch_size]] = \
                        self.model.emb[0][:batch_size].cpu()
                    self.emb[1][node_idx[:batch_size]] = \
                        self.model.emb[1][:batch_size].cpu()
                else:
                    self.emb[node_idx[:batch_size]] = \
                        self.model.emb[:batch_size].cpu()

            test_loss = loss.item() * batch_size
            outlier_score[node_idx[:batch_size]] = score

        loss_value = test_loss / data.x.shape[0]
        if self.gan:
            loss_value = (self.epoch_loss_in / data.x.shape[0], loss_value)

        return outlier_score
    
    def evaluate(self, epoch):
        """
        Evaluate the model on the evaluation data.
        """
        self.model.eval()
        data = self.eval_data.to(self.device)
        start_time = time.time()
        loader = self.loader_fn(data=data)

        # Initialize accumulators
        total_loss = 0.0
        total_attr = 0.0
        total_struct = 0.0
        num_nodes = data.x.shape[0]
        eval_score = torch.zeros(num_nodes, device=self.device)

        with torch.no_grad():
            for batch_idx, batch_data in enumerate(loader):
                batch_size = batch_data.batch_size
                node_idx = batch_data.n_id
                self.process_graph(batch_data, batch_mode=True)
                loss, scores, attr_l, struct_l = self.forward_model(batch_data)

                # Accumulate batch results
                total_loss += loss.item() * batch_size
                total_attr += attr_l.item() * batch_size
                total_struct += struct_l.item() * batch_size
                eval_score[node_idx[:batch_size]] = scores.to(self.device)

        # Compute mean losses
        loss_value = total_loss / num_nodes
        attr_value = total_attr / num_nodes
        struct_value = total_struct / num_nodes

        # Record loss history
        self.history_["eval"]["total"].append(loss_value)
        self.history_["eval"]["attr"].append(attr_value)
        self.history_["eval"]["struct"].append(struct_value)

        # Prepare labels and predictions
        mask = self.label_mask(data.y)
        preds = (eval_score > self.epoch_threshold_).long()[mask].cpu().numpy()
        true_labels = data.y[mask].cpu().numpy()
        scores = eval_score[mask].cpu().numpy()

        # Compute evaluation metrics
        auc_eval = eval_roc_auc(true_labels, scores)
        prec_eval = eval_precision(true_labels, preds)
        rec_eval = eval_recall(true_labels, preds)
        ap_eval = eval_average_precision(true_labels, scores)
        f1_eval = eval_f1(true_labels, preds)
        # class_report = eval_classification_report(true_labels, preds)

        # Record metric history
        self.history_["eval"]["auc"].append(auc_eval)
        self.history_["eval"]["precision"].append(prec_eval)
        self.history_["eval"]["recall"].append(rec_eval)
        self.history_["eval"]["ap"].append(ap_eval)
        self.history_["eval"]["f1"].append(f1_eval)
        # self.history_["eval"]["classification_report"].append(class_report)

        # Store raw scores
        self.scores_["eval"] = eval_score.cpu().numpy()

        # Log results
        logger(
            epoch=epoch,
            loss=loss_value,
            struct=struct_value,
            attr=attr_value,
            auc=auc_eval,
            precision=prec_eval,
            recall=rec_eval,
            ap=ap_eval,
            f1=f1_eval,
         
            time=time.time() - start_time,
            verbose=self.verbose,
            score=self.scores_["eval"],
            target=data.y.cpu().numpy(),
            train=False
        )



def double_recon_loss(x, x_hat, s, s_hat, weight=0.5):
    """
    Simplified double reconstruction loss.
    Returns a tuple of (combined score, feature error, structure error),
    where the combined score is the weighted sum of the other errors. 

    Parameters
    ----------
    x       : torch.Tensor  # ground‐truth features
    x_hat   : torch.Tensor  # reconstructed features
    s       : torch.Tensor  # ground‐truth structure
    s_hat   : torch.Tensor  # reconstructed structure
    weight  : float         # balance between feature and structure losses

    Returns
    -------
    score       : torch.Tensor  # weight * feature_error + (1-weight) * structure_error
    attr_err : torch.Tensor  # sqrt of summed squared feature differences
    struct_err  : torch.Tensor  # sqrt of summed squared structure differences
    """
    # feature reconstruction error
    diff_attr    = (x - x_hat).pow(2)
    attr_err  = torch.sqrt(diff_attr.sum(dim=1))

    # structure reconstruction error
    diff_stru    = (s - s_hat).pow(2)
    struct_err   = torch.sqrt(diff_stru.sum(dim=1))

    # combined outlier score per node
    score = weight * attr_err + (1 - weight) * struct_err

    return score, attr_err, struct_err

def section(s):
    print("="*100)
    print(s)
    print("="*100)

def logger(epoch=0,
           loss=0,
           struct=0,
           attr=0,
           auc=None,
           precision=None,
           recall=None,
           ap=None,
           f1=None,
           classification_report=None,
           score=None,
           
           target=None,
           time=None,
           verbose=0,
           train=True,
           deep=True):
    """
    Logger for detector.

    Parameters
    ----------
    epoch : int, optional
        The current epoch.
    loss : float, optional
        The current epoch loss value.
    score : torch.Tensor, optional
        The current outlier scores.
    target : torch.Tensor, optional
        The ground truth labels.
    time : float, optional
        The current epoch time.
    verbose : int, optional
        Verbosity mode. Range in [0, 3]. Larger value for printing out
        more log information. Default: ``0``.
    train : bool, optional
        Whether the logger is used for training.
    deep : bool, optional
        Whether the logger is used for deep detector.
    """
    if verbose > 0:
        if deep:
            if train:
                print()
                section(f"Epoch {epoch}")
                print("Train: ", end='')
            else:
                print("Test:  ", end='')

    
            print("Loss   {:8.4f}".format(loss), end=' | ')
            print("Struct {:8.4f}".format(struct), end=' | ')
            print("Attr   {:8.4f}".format(attr), end=' | ')

        if verbose > 1:
            print("AUC {:.4f}".format(auc), end=' | ')
            print("Precision {:.4f}".format(precision), end=' | ')
            print("Recall {:.4f}".format(recall), end=' | ')
            print("F1 {:.4f}".format(f1), end=' | ')
            print("AP {:.4f}".format(ap), end=' | ')
            print("Time {:.2f}".format(time))


if __name__ == "__main__":
    # Simple test for DOMINANT loss and score tracking
    import torch
    
    from src.loaders import load_elliptic
    t_data = load_elliptic(root="../data/elliptic", force_reload=False, local=True, use_temporal=True, t=1)
    t2_data = load_elliptic(root="../data/elliptic", force_reload=False, local=True, use_temporal=True, t=2)
    model = DOMINANT(verbose=3, epoch=30)
    model.fit(t_data, eval_data=t2_data)


