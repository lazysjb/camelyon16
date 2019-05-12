import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, auc, confusion_matrix, log_loss,
    precision_score, recall_score, roc_curve)

from params import args
from utils.config import NON_GRAY_RATIO_THRESHOLD
from utils.slide_utils import get_inference_file_name


def get_evaluation_metrics(inference_model_name,
                           partition_option,
                           data_split_type,
                           plot_roc_curve=True,
                           plot_confusion_matrix=True):
    """Evaluate model based on inference output dataframe"""
    inference_file_name = get_inference_file_name(inference_model_name,
                                                  partition_option,
                                                  data_split_type)
    inference_dir = os.path.join(args.output_data_dir, 'inference')
    inference_df = pd.read_pickle(os.path.join(inference_dir,
                                               inference_file_name))

    # Evaluate within the tissue region
    filter_mask = (inference_df['is_roi'] == 1) & \
                  (inference_df['non_gray_ratio'] > NON_GRAY_RATIO_THRESHOLD)
    inference_df = inference_df[filter_mask].copy()

    y_truth = inference_df['label'].astype(int).values
    y_pred_prob = inference_df['y_pred_prob'].values
    y_pred = (y_pred_prob > 0.5).astype(int)

    accuracy = accuracy_score(y_truth, y_pred)
    precision = precision_score(y_truth, y_pred)
    recall = recall_score(y_truth, y_pred)

    loss = log_loss(y_truth, y_pred_prob)
    fpr, tpr, threshold = roc_curve(y_truth, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    # accuracy if we predict everything as negative
    dummy_accuracy = 1 - y_truth.mean()

    output = {
        'model_name': inference_model_name,
        'loss': loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'auc': roc_auc,
        'dummy_accuracy': dummy_accuracy,
    }

    if plot_roc_curve:
        plt.figure()

        lw = 2
        plt.plot(fpr, tpr, color='orange',
                 lw=lw,
                 label='{} (area = {:.2f})'.format(
                     inference_model_name, roc_auc))
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc='lower right')

    if plot_confusion_matrix:
        cm = confusion_matrix(y_truth, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, ax=ax, fmt='g')

        # labels, title and ticks
        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels')
        ax.set_title('Confusion Matrix')
        ax.xaxis.set_ticklabels([0, 1])
        ax.yaxis.set_ticklabels([0, 1])

    return output
