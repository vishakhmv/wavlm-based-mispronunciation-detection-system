import os
import torch
from datasets.base_dataset import BaseSpeechDataset
import sys

import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from phoneme_features import phoneme_to_features, PHONOLOGICAL_FEATURES
from config import config

logger = logging.getLogger(__name__)

class L2ArcticDataset(BaseSpeechDataset):
    """
    Dataset loader for L2-ARCTIC.
    Parses the centralized actual_phonemes_with_annotation.txt file.
    """
    def __init__(self, data_dir, split="train", sample_rate=16000):
        super().__init__(sample_rate=sample_rate)
        
        if "l2arctic" not in data_dir.lower():
            data_dir = os.path.join(data_dir, "l2arctic_release_v5.0")
            
        self.data_dir = data_dir
        self.split = split
        self.samples = []
        
        if split == "train":
            self.target_speakers = config.TRAIN_SPEAKERS
        elif split == "test":
            self.target_speakers = config.TEST_SPEAKERS
        else:
            self.target_speakers = ()
            
        annotation_file = os.path.join(data_dir, "actual_phonemes_with_annotation.txt")
        self._load_annotations(annotation_file)
        self._print_stats()

    def _load_annotations(self, annotation_file):
        if not os.path.exists(annotation_file):
            raise FileNotFoundError(f"Annotation file not found at {annotation_file}")
            
        with open(annotation_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    file_id = parts[0]
                    phonemes = parts[1:]
                    
                    if '/' in file_id:
                        speaker, audio_name = file_id.split('/', 1)
                        if audio_name.endswith('.wav'):
                            audio_name = audio_name[:-4]
                    elif '\\' in file_id:
                        speaker, audio_name = file_id.split('\\', 1)
                        if audio_name.endswith('.wav'):
                            audio_name = audio_name[:-4]
                    elif ',' in file_id:
                        speaker, audio_name = file_id.split(',', 1)
                    elif '_' in file_id:
                        speaker, audio_name = file_id.split('_', 1)
                        speaker = speaker.upper()
                    else:
                        import re
                        match = re.match(r"([A-Za-z]+)(.*)", file_id)
                        if match:
                            speaker = match.group(1).upper()
                            audio_name = file_id
                        else:
                            continue
                            
                    if speaker not in self.target_speakers:
                        continue
                        
                    audio_path = os.path.join(self.data_dir, speaker, "wav", f"{audio_name}.wav")
                    
                    if os.path.exists(audio_path):
                        self.samples.append({
                            "audio_path": audio_path,
                            "phonemes": phonemes,
                            "feature_targets": self.phonemes_to_feature_targets(phonemes)
                        })
                    else:
                        logger.warning(f"Missing audio: {audio_path}")

    def _print_stats(self):
        logger.info(f"--- L2-ARCTIC Dataset ({self.split}) Validation ---")
        logger.info(f"Total samples: {len(self.samples)}")
        if self.samples:
            avg_len = sum(len(s['phonemes']) for s in self.samples) / len(self.samples)
            max_len = max(len(s['phonemes']) for s in self.samples)
            logger.info(f"Average phoneme length: {avg_len:.2f}")
            logger.info(f"Maximum phoneme length: {max_len}")
        logger.info("------------------------------------------")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        item = self.samples[idx]
        
        waveform = self.load_audio(item["audio_path"])
        
        phonemes = item["phonemes"]
        phoneme_sequence = " ".join(phonemes)
            
        return {
            "waveform": waveform,
            "feature_targets": item["feature_targets"],
            "audio_path": item["audio_path"],
            "phonemes": phonemes,
            "phoneme_sequence": phoneme_sequence
        }
