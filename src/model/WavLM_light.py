from transformers import WavLMForXVector, WavLMConfig
import torch.nn as nn


class WavLMWrapper(nn.Module):
    def __new__(cls, config_name="microsoft/wavlm-base-plus-sv", model_name="wavlm"):
        config = WavLMConfig.from_pretrained(config_name)
        model = WavLMForXVector(config)
        model.model_name = model_name
        return model
    
    