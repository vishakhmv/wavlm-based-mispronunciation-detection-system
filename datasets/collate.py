import torch
from torch.nn.utils.rnn import pad_sequence
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from phoneme_features import PHONOLOGICAL_FEATURES

assert len(PHONOLOGICAL_FEATURES) == 35, "PHONOLOGICAL_FEATURES list is broken."

def mdd_collate_fn(batch):
    """
    Collate function for MDD datasets.
    Pads variable-length audio and 35 independent feature targets.
    """
    waveforms = []
    input_lengths = []
    
    all_feature_targets = [[] for _ in range(35)]
    target_lengths = []
    
    valid_batch = []
    for item in batch:
        waveform = item.get("waveform")

        if waveform is None or waveform.numel() == 0:
            continue
            
        if item.get("feature_targets") and len(item["feature_targets"]) == 35:

            if all(t.size(0) > 0 for t in item["feature_targets"]):
                valid_batch.append(item)
            
    if not valid_batch:
        raise RuntimeError("Empty batch after filtering invalid items. Dataset preprocessing may have failed or no valid samples are in this batch.")

    audio_paths = []
    phoneme_sequences = []

    for item in valid_batch:
        waveforms.append(item["waveform"])
        input_lengths.append(item["waveform"].size(0))
        
        assert len(item["feature_targets"]) == 35, f"Expected 35 feature targets, got {len(item['feature_targets'])}"
        
        for i in range(35):
            all_feature_targets[i].append(item["feature_targets"][i])
            
        target_lengths.append(item["feature_targets"][0].size(0))
        
        audio_paths.append(item.get("audio_path", ""))
        phoneme_sequences.append(item.get("phoneme_sequence", ""))

    waveforms_padded = pad_sequence(waveforms, batch_first=True, padding_value=0.0)
    input_lengths = torch.tensor(input_lengths, dtype=torch.long)
    
    padded_feature_targets = []
    for i in range(35):
        padded_seq = pad_sequence(all_feature_targets[i], batch_first=True, padding_value=0)
        padded_feature_targets.append(padded_seq)
        
    target_lengths = torch.tensor(target_lengths, dtype=torch.long)

    return {
        "waveforms": waveforms_padded,
        "input_lengths": input_lengths,
        "feature_targets": padded_feature_targets,
        "target_lengths": target_lengths,
        "audio_paths": audio_paths,
        "phoneme_sequences": phoneme_sequences
    }
