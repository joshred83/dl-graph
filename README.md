ejl - please see somewhat YOLO'd update of dominant in `dominant.py`

calling convention is basically the same with new options/params.

usage:
- update the config dict in `main.py`
- run `main.py` which implements training and testing loops.
- loss plots, train/test metrics, and model params are saved in the `saved_models` directory


### TODO 4-14
* Define whether we want to perturb the entire dataset or some of it
* "Something about the math" to justify our additional hacks
* Transfer Learning: does it work better than a classifier directly?
  * i.e. first train a model on the entire dataset, then freeze the weights and train a classifier on top of it
  * We wouldn't discard the unlabeled data, in order to learn the encoder to create a low-D representation, we should use everything available, conceivably.
  * Then given the low-D representation, use e.g. a FC network to classify off of the "backbone" learned by DOMINANT
  * We can think of this is as DOMINANT functioning as an Encoder
* Baseline the performance on Elliptic dataset (and others, as appropriate) against "traditional" ML models
* Consider additional datasets built into pytorch-geometric e.g. Amazon
* Additional architectural considerations, beyond the stuff we already did (i.e. feature aggregation, feature interpolation, and noise perturbations)
  * Do we want to discard the feature aggregation at all? Don't need it for Elliptic? But use it on other data sets as a point of comparison?
    * Or even better discard the "synthetic" aggregated features from the Elliptic dataset and "DIY it"
  * Consider using RNN - EvolveGCN
  * Consider using Attention
  * Static vs. Dynamic options as with EvolveGCN
  * Consider a different loss function? 
  * Skip-layer connections?
* Metrics: loss vs AUC/accuracy metrics
  
Prioritization:
* 1. Additional architectural considerations
* 2. Transfer Learning

just start your own notebook
try to make code modular (: