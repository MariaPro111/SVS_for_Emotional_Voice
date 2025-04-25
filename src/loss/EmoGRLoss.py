import torch, math
import torch.nn as nn
import torch.nn.functional as F
from src.loss.AAMsoftmax import AAMsoftmax
from src.loss.functional import revgrad


class EmotionGRLoss(nn.Module):
    def __init__(self, n_speakers, n_emotions, margin, scale, embedding_size=192, speaker_coef=1, emotion_coef=1, alpha=0.4): 
        super().__init__()
        self.speaker_coef = speaker_coef
        self.emotion_coef = emotion_coef
        self.alpha = torch.tensor(alpha, requires_grad=False)

        self.speaker_criterion = AAMsoftmax(n_speakers, margin, scale, embedding_size)     
        self.emotion_head = nn.Linear(embedding_size, n_emotions)
        self.emotion_criterion = nn.CrossEntropyLoss()

    def forward(self, embeddings, labels, emo_labels, **kwargs): 
        speaker_results = self.speaker_criterion(embeddings, labels)
        
        reversed_embeddings = revgrad(embeddings, self.alpha)
        emotion_logits = self.emotion_head(reversed_embeddings)
        emotion_loss = self.emotion_criterion(emotion_logits, emo_labels)

        total_loss = self.speaker_coef * speaker_results["loss"] + self.emotion_coef * emotion_loss
        
        return {
            "loss": total_loss, 
            "speaker_loss": speaker_results["loss"],
            "emotion_loss": emotion_loss,
            "logits": speaker_results["logits"],
            "emo_logits": emotion_logits
        }