import torch

from src.metrics.base_metric import BaseMetric


class ClassificationMetric(BaseMetric):
    def __init__(self, metric, device, class_type="speakers", *args, **kwargs):
        """
        Example of a nested metric class. Applies metric function
        object (for example, from TorchMetrics) on tensors.

        Notice that you can define your own metric calculation functions
        inside the '__call__' method.

        Args:
            metric (Callable): function to calculate metrics.
            device (str): device for the metric calculation (and tensors).
        """
        super().__init__(*args, **kwargs)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.metric = metric.to(device)
        self.class_type = class_type

    def __call__(self, logits, labels, emo_logits=None, emo_labels=None, **kwargs):
        """
        Metric calculation logic.

        Args:
            logits (Tensor): model output predictions.
            labels (Tensor): ground-truth labels.
        Returns:
            metric (float): calculated metric.
        """
        if self.class_type == "emotions":
            classes = emo_logits.argmax(dim=-1)
            return self.metric(classes, emo_labels)
        classes = logits.argmax(dim=-1)
        return self.metric(classes, labels)