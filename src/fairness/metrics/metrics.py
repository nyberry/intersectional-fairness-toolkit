import numpy as np
from itertools import product


def group_accuracy(group_label, subject_labels, predictions, true_statuses):
    n_samples = len(predictions)
    accurate_or_not = [pred == truth
                       for pred, truth
                       in zip(predictions, true_statuses)]
    mask = [True] * n_samples
    for observation in range(n_samples):
        if subject_labels[observation] != group_label:
            mask[observation] = False
    group_results = [acc for acc, include
                     in zip(accurate_or_not, mask)
                     if include is True]
    if len(group_results) > 0:
        accuracy = sum(group_results) / len(group_results)
    else:
        accuracy = np.nan
    return accuracy


def accuracy_diff(
    group_a_label, group_b_label, subject_labels, predictions, true_statuses
):
    pass


def accuracy_ratio():
    pass


def accuracy_ratio_logged():
    pass


def intersect_accuracy_diff():
    pass


def intersect_accuracy_ratio():
    pass


def intersect_accuracy_ratio_logged():
    pass


def group_fnr():
    pass


def group_fpr():
    pass


def group_ppv():
    pass


def group_npv():
    pass