import torch
import torch.nn as nn
from transformers import WavLMModel

from config import config

class WavLMMDD(nn.Module):
    def __init__(self, model_name=config.MODEL_NAME, num_features=config.NUM_FEATURES):
        super(WavLMMDD, self).__init__()
        
        self.wavlm = WavLMModel.from_pretrained(
            model_name,
            apply_spec_augment=True,
            mask_time_prob=0.05,
            mask_feature_prob=0.05
        )
        
        # Verify configuration overrides took effect
        print(f"SpecAugment Enabled: {self.wavlm.config.apply_spec_augment}")
        print(f"Time Mask Prob: {self.wavlm.config.mask_time_prob}")
        print(f"Feature Mask Prob: {self.wavlm.config.mask_feature_prob}")
        
        self.wavlm.freeze_feature_encoder()
        
        dropout_prob = (
            self.wavlm.config.final_dropout 
            if hasattr(self.wavlm.config, "final_dropout") 
            else self.wavlm.config.hidden_dropout
        )
        self.dropout = nn.Dropout(dropout_prob)
        
        self.num_features = num_features
        self.num_classes = num_features * 2 + 1 
        
        self.classifier = nn.Linear(self.wavlm.config.hidden_size, self.num_classes)
        
    def forward(self, input_values, attention_mask=None):

        outputs = self.wavlm(input_values, attention_mask=attention_mask)
        
        hidden_states = outputs.last_hidden_state
        
        hidden_states = self.dropout(hidden_states)
        
        logits = self.classifier(hidden_states)
        
        return logits

    def get_feat_extract_output_lengths(self, input_lengths):
        """
        Safely computes the downsampled sequence lengths from the CNN feature extractor.
        Isolates the private Hugging Face method.
        """
        return self.wavlm._get_feat_extract_output_lengths(input_lengths)
