from torch import nn
from transformers import AutoFeatureExtractor


class FeatureExtractor(nn.Module):
    def __init__(self, name="microsoft/wavlm-base-plus-sv"):
        super().__init__()
        self.name = name
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(name)

    def forward(self, x):
        inputs = self.feature_extractor(x, sampling_rate=16000, return_tensors="pt")
        return inputs['input_values'].squeeze(0)
