import os
import torch
from datasets.base_dataset import BaseSpeechDataset
import sys

import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from phoneme_features import phoneme_to_features, PHONOLOGICAL_FEATURES, TIMIT_61_TO_39

logger = logging.getLogger(__name__)

class TIMITDataset(BaseSpeechDataset):
    """
    Dataset loader for TIMIT.
    Scans TIMIT and TIMIT_WAV for .WAV and .PHN files and maps 61 phonemes to 39.
    """
    def __init__(self, data_dir, split="train", sample_rate=16000):
        super().__init__(sample_rate=sample_rate)
        if "TIMIT_LDC93S1" not in data_dir:
            data_dir = os.path.join(data_dir, "TIMIT_LDC93S1")
            
        self.data_dir = data_dir
        self.split = split
        self.samples = []
        self.skipped_samples = 0
        
        self._scan_dataset(split)
        self._print_stats()

    def _scan_dataset(self, split):
        search_dirs = [
            os.path.join(self.data_dir, "TIMIT", split.upper()),
            os.path.join(self.data_dir, "TIMIT_WAV", split.upper()),
            os.path.join(self.data_dir, "TIMIT", split.lower()),
            os.path.join(self.data_dir, "TIMIT_WAV", split.lower())
        ]
        
        valid_dirs = [d for d in search_dirs if os.path.exists(d)]
        if not valid_dirs:
            raise FileNotFoundError(f"No valid TIMIT split directory found for {split} in {self.data_dir}")
            
        for scan_dir in valid_dirs:
            for root, _, files in os.walk(scan_dir):
                for file in files:
                    if file.upper().endswith(".WAV"):
                        audio_path = os.path.join(root, file)
                        phn_path = audio_path[:-4] + ".PHN"
                        
                        if not os.path.exists(phn_path):
                            phn_path = audio_path[:-4] + ".phn"
                            
                        if not os.path.exists(phn_path) and "TIMIT_WAV" in phn_path:
                            alt_phn_path = phn_path.replace("TIMIT_WAV", "TIMIT")
                            if os.path.exists(alt_phn_path):
                                phn_path = alt_phn_path
                                
                        if os.path.exists(phn_path):
                            phonemes = self._parse_phn_file(phn_path)
                            self.samples.append({
                                "audio_path": audio_path,
                                "phonemes": phonemes,
                                "feature_targets": self.phonemes_to_feature_targets(phonemes)
                            })
                        else:
                            self.skipped_samples += 1

    def _parse_phn_file(self, phn_path):
        phonemes = []
        with open(phn_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    phn = parts[2].lower()
                    mapped_phn = TIMIT_61_TO_39.get(phn, None)
                    if mapped_phn and mapped_phn != 'sil':
                        phonemes.append(mapped_phn)
        return phonemes

    def _print_stats(self):
        logger.info(f"--- TIMIT Dataset ({self.split}) Validation ---")
        logger.info(f"Total valid samples: {len(self.samples)}")
        logger.info(f"Skipped samples (missing annotations): {self.skipped_samples}")
        if self.samples:
            valid_lens = [len(s['phonemes']) for s in self.samples if len(s['phonemes']) > 0]
            if valid_lens:
                avg_len = sum(valid_lens) / len(valid_lens)
                max_len = max(valid_lens)
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
