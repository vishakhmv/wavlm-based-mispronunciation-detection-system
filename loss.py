import torch
import torch.nn as nn
import torch.nn.functional as F

class SCTCSBLoss(nn.Module):
    def __init__(self, num_features=35):
        super(SCTCSBLoss, self).__init__()
        self.num_features = num_features
        self.blank_idx = 2
        self.ctc_loss = nn.CTCLoss(blank=self.blank_idx, reduction='mean', zero_infinity=True)

    def forward(self, logits, targets, input_lengths, target_lengths):
        """
        logits: (Batch, SeqLen, 71)
        targets: List of 35 tensors, each (Batch, MaxTargetLen) - Values are 1 (presence) and 0 (absence)
        input_lengths: (Batch,)
        target_lengths: (Batch,)
        """
        batch_size, seq_len, _ = logits.size()
        
        logits = logits.transpose(0, 1)
        
        total_loss = 0.0
        
        for i in range(self.num_features):
            feature_logits = torch.stack([
                logits[:, :, i], 
                logits[:, :, i + self.num_features], 
                logits[:, :, -1]
            ], dim=-1)
            
            log_probs = F.log_softmax(feature_logits, dim=-1)
            
            feature_targets = 1 - targets[i] 
            
            loss = self.ctc_loss(log_probs, feature_targets, input_lengths, target_lengths)
            total_loss += loss
            
        return total_loss
