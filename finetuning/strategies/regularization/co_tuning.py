"""
Modified from https://github.com/thuml/Transfer-Learning-Library/blob/master/tllib/regularization/co_tuning.py
"""

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from timm import utils


class Relationship:
    """Learns the category relationship p(y_s|y_t) between source dataset and target dataset.

    Args:
        data_loader (torch.utils.data.DataLoader): A data loader of target dataset.
        classifier (torch.nn.Module): A classifier for Co-Tuning.
        device (torch.nn.Module): The device to run classifier.
        cache (str, optional): Path to find and save the relationship file.

    """

    def __init__(self, data_loader, classifier, device, args, cache=None):
        super().__init__()
        self.data_loader = data_loader
        self.classifier = classifier
        self.args = args
        self.device = device
        if cache is None or not os.path.exists(cache):
            source_predictions, target_labels = self.collect_labels()
            self.relationship = self.get_category_relationship(source_predictions, target_labels)
            if cache is not None and utils.is_primary(args):
                np.save(cache, self.relationship)
        else:
            self.relationship = np.load(cache)

    def __getitem__(self, category):
        return self.relationship[category]

    def collect_labels(self):
        """
        Collects predictions of target dataset by source model and corresponding ground truth class labels.

        Returns:
            - source_probabilities, [N, N_p], where N_p is the number of classes in source dataset
            - target_labels, [N], where 0 <= each number < N_t, and N_t is the number of classes in target dataset
        """

        print("Collecting labels to calculate relationship")
        source_predictions = []
        target_labels = []

        self.classifier.to(self.device)
        self.classifier.eval()
        with torch.no_grad():
            for x, label in tqdm.tqdm(self.data_loader):
                x = x.to(self.device)
                feat = self.classifier.forward_features(x)
                y_s = self.classifier.forward_head(feat, pre_logits=True)

                if self.args.distributed:
                    gather_tensors = [torch.zeros_like(y_s) for _ in range(self.args.world_size)]
                    torch.dist.all_gather(gather_tensors, y_s)
                    y_s = torch.cat(gather_tensors, dim=0)
                y_s = (
                    y_s - y_s.max(dim=1, keepdim=True).values
                )  # original implementation: https://github.com/thuml/CoTuning/blob/fc3eb54c9b44251c7cba557f9b90bdee5eca6ec3/module/relationship_learning.py#L41
                source_predictions.append(F.softmax(y_s, dim=1).detach().cpu().numpy())
                target_labels.append(label.cpu().numpy())

        return np.concatenate(source_predictions, 0), np.concatenate(target_labels, 0)

    def get_category_relationship(self, source_probabilities, target_labels):
        """
        The direct approach of learning category relationship p(y_s | y_t).

        Args:
            source_probabilities (numpy.array): [N, N_p], where N_p is the number of classes in source dataset
            target_labels (numpy.array): [N], where 0 <= each number < N_t, and N_t is the number of classes in target dataset

        Returns:
            Conditional probability, [N_c, N_p] matrix representing the conditional probability p(pre-trained class | target_class)
        """
        N_t = np.max(target_labels) + 1  # the number of target classes
        conditional = []
        for i in range(N_t):
            this_class = source_probabilities[target_labels == i]
            average = np.mean(this_class, axis=0, keepdims=True)
            conditional.append(average)
        return np.concatenate(conditional)


class CoTuningLoss(nn.Module):
    """
    The Co-Tuning loss in `Co-Tuning for Transfer Learning (NIPS 2020)
    <http://ise.thss.tsinghua.edu.cn/~mlong/doc/co-tuning-for-transfer-learning-nips20.pdf>`_.

    Inputs:
        - input: p(y_s) predicted by source classifier.
        - target: p(y_s|y_t), where y_t is the ground truth class label in target dataset.

    Shape:
        - input:  (b, N_p), where b is the batch size and N_p is the number of classes in source dataset
        - target: (b, N_p), where b is the batch size and N_p is the number of classes in source dataset
        - Outputs: scalar.
    """

    def __init__(self, relationship: Relationship, weight: float):
        super().__init__()
        self.relationship = relationship
        self.weight = weight

    def forward(self, feature, target, **kwargs):
        if target.ndim == 2:  # if using mixup
            target = target.argmax(dim=-1)
        y_s = self.relationship[target.cpu().numpy()]
        y_s_tensor = torch.from_numpy(y_s).to(feature.device).float()
        y_t = feature - feature.max(dim=1, keepdim=True).values
        y = -y_s_tensor * F.log_softmax(y_t, dim=-1)
        y = torch.mean(torch.sum(y, dim=-1))
        return self.weight * y
