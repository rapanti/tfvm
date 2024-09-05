"""
Modified from https://github.com/thuml/Transfer-Learning-Library/blob/master/tllib/regularization/co_tuning.py
"""

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm


class Relationship(object):
    """Learns the category relationship p(y_s|y_t) between source dataset and target dataset.

    Args:
        data_loader (torch.utils.data.DataLoader): A data loader of target dataset.
        classifier (torch.nn.Module): A classifier for Co-Tuning.
        device (torch.nn.Module): The device to run classifier.
        cache (str, optional): Path to find and save the relationship file.

    """

    def __init__(self, data_loader, classifier, device, cache=None):
        super(Relationship, self).__init__()
        self.data_loader = data_loader
        self.classifier = classifier
        self.device = device
        if cache is None or not os.path.exists(cache):
            source_predictions, target_labels = self.collect_labels()
            self.relationship = self.get_category_relationship(source_predictions, target_labels)
            if cache is not None:
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
                y_s = self.classifier.forward_features(x)

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
        source_target = self.relationship[target.cpu().numpy()]
        source_target_tensor = torch.from_numpy(source_target).to(feature.device).float()
        y = -source_target_tensor * F.log_softmax(feature, dim=-1)
        y = torch.mean(torch.sum(y, dim=-1))
        return self.weight * y
