"""Learning paradigm implementations.

Each paradigm is a subclass of WindowedDataset that implements
paradigm-specific logic for how data is returned.

Available Paradigms
-------------------
SupervisedDataset : Supervised learning (inputs → targets)
    Returns (inputs_dict, targets_dict) for regression/classification.

Future Paradigms (not yet implemented)
--------------------------------------
ContrastiveDataset : Self-supervised contrastive learning
    Returns (view1, view2) augmented pairs for SimCLR-style training.

MaskedDataset : Masked autoencoding
    Returns (masked_input, targets) for MAE-style training.
"""

from myotorch.datasets.paradigms.supervised import SupervisedDataset

__all__ = ["SupervisedDataset"]
