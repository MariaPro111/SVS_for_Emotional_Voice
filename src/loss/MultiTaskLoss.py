import torch, math
import torch.nn as nn
import torch.nn.functional as F
from src.loss.AAMsoftmax import AAMsoftmax


class MultiTaskLoss(nn.Module):
    def __init__(self, n_speakers, n_emotions, margin, scale, embedding_size=192, speaker_coef=1, emotion_coef=1): 
        super().__init__()
        self.speaker_coef = speaker_coef
        self.emotion_coef = emotion_coef
        self.speaker_criterion = AAMsoftmax(n_speakers, margin, scale, embedding_size)
        
        self.emotion_head = nn.Linear(embedding_size, n_emotions)
        self.emotion_criterion = nn.CrossEntropyLoss()

    def forward(self, embeddings, labels, emo_labels, **kwargs): 
        speaker_results = self.speaker_criterion(embeddings, labels)
        
        emotion_logits = self.emotion_head(embeddings)
        emotion_loss = self.emotion_criterion(emotion_logits, emo_labels)
        total_loss = self.speaker_coef * speaker_results["loss"] + self.emotion_coef * emotion_loss
        
        return {
            "loss": total_loss, 
            "speaker_loss": speaker_results["loss"],
            "emotion_loss": emotion_loss,
            "logits": speaker_results["logits"],
            "emo_logits": emotion_logits
        }
