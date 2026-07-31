import torch
import torchaudio
from torch.utils.data import Dataset

class BaseSpeechDataset(Dataset):
    """
    Base class for speech datasets. 
    Handles common operations like loading, mono conversion, and cached resampling.
    """
    def __init__(self, sample_rate=16000):
        super().__init__()
        self.sample_rate = sample_rate
        self.resamplers = {}
        
    def load_audio(self, audio_path: str) -> torch.Tensor:
        """
        Loads an audio file, converts to mono, resamples if necessary,
        and returns a 1D float32 tensor.
        Returns None if loading fails.
        """
        try:
            waveform, sr = torchaudio.load(audio_path)
            
            if waveform.size(0) > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
                
            if sr != self.sample_rate:
                if sr not in self.resamplers:
                    self.resamplers[sr] = torchaudio.transforms.Resample(sr, self.sample_rate)
                waveform = self.resamplers[sr](waveform)
                
            waveform = waveform.squeeze(0)
            return waveform
            
        except Exception as e:
            raise IOError(f"Error loading audio {audio_path}: {e}")

    def phonemes_to_feature_targets(self, phonemes):
        from phoneme_features import phoneme_to_features, PHONOLOGICAL_FEATURES
        features_dict = phoneme_to_features(phonemes)
        feature_targets = []
        for feat_name in PHONOLOGICAL_FEATURES:
            feat_seq = features_dict[feat_name]
            feature_targets.append(torch.tensor(feat_seq, dtype=torch.long))
        return feature_targets
