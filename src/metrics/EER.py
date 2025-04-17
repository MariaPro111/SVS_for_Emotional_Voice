from sklearn import metrics
import numpy as np
import torch

from src.metrics.base_metric import BaseMetric

class EERMetric(BaseMetric):
    def __init__(self, device, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

    def __call__(self, scores: torch.Tensor, labels: torch.Tensor, **kwargs):
        """
        Вычисляет Equal Error Rate (EER) для бинарной классификации.
        
        Args:
            logits (Tensor): raw output scores.
            labels (Tensor): бинарные метки (0 или 1).
        Returns:
            eer (float): EER в процентах.
        """  
        scores = scores.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()

        fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=1)
        fnr = 1 - tpr
        idx = np.nanargmin(np.abs(fnr - fpr))
        idx = np.nanargmin(np.abs(fnr - fpr))
        eer = max(fpr[idx], fnr[idx]) * 100
        best_threshold = thresholds[idx]

        return {
            "eer": eer,
            "threshold": best_threshold
        }
    