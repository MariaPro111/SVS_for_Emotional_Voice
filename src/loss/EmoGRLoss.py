import torch, math
import torch.nn as nn
import torch.nn.functional as F
from src.loss.AAMsoftmax import AAMsoftmax
from src.loss.functional import revgrad


class EmotionGRLoss(nn.Module):
    def __init__(self, n_speakers, n_emotions, margin, scale, embedding_size=192, speaker_coef=1, emotion_coef=1, max_alpha=0.4, n_epochs=20, epoch_len=200): 
        super().__init__()
        self.speaker_coef = speaker_coef
        self.emotion_coef = emotion_coef
        self.max_alpha = max_alpha
        self.total_steps = n_epochs * epoch_len
        self.register_buffer('current_step', torch.zeros(1, dtype=torch.long))

        self.speaker_criterion = AAMsoftmax(n_speakers, margin, scale, embedding_size)     
        # self.emotion_head = nn.Linear(embedding_size, n_emotions)
        self.emotion_head = nn.Sequential(
            nn.Linear(embedding_size, 128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, n_emotions)
        )
        self.emotion_criterion = nn.CrossEntropyLoss()

    def get_alpha(self):
        p = self.current_step.item() / self.total_steps
        return self.max_alpha * (2.0 / (1.0 + math.exp(-10 * p)) - 1)

    def forward(self, embeddings, labels, emo_labels, **kwargs): 
        self.current_step += 1

        speaker_results = self.speaker_criterion(embeddings, labels)
        
        alpha = self.get_alpha()
        reversed_embeddings = revgrad(embeddings, torch.tensor(alpha, device=embeddings.device))
        emotion_logits = self.emotion_head(reversed_embeddings)
        emotion_loss = self.emotion_criterion(emotion_logits, emo_labels)

        total_loss = self.speaker_coef * speaker_results["loss"] + self.emotion_coef * emotion_loss
        
        return {
            "loss": total_loss, 
            "speaker_loss": speaker_results["loss"],
            "emotion_loss": emotion_loss,
            "logits": speaker_results["logits"],
            "emo_logits": emotion_logits,
            "alpha": alpha
        }