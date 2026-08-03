import os
import torch
from datasets.base_dataset import BaseSpeechDataset
import sys

import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from phoneme_features import phoneme_to_features, PHONOLOGICAL_FEATURES
from config import config

logger = logging.getLogger(__name__)

class LibriSpeechDataset(BaseSpeechDataset):
    """
    Dataset loader for LibriSpeech.
    Parses the transcript.txt and uses Libri_lexicon.txt to get phonemes.
    """
    def __init__(self, data_dir, split="train-clean-100", sample_rate=16000):
        super().__init__(sample_rate=sample_rate)
        
        if "LibriSpeech" not in data_dir:
            data_dir = os.path.join(data_dir, "LibriSpeech")
            
        self.data_dir = os.path.join(data_dir, split)
        self.split = split
        self.samples = []
        self.skipped_samples = 0
        self.oov_sentences = 0
        
        lexicon_path = os.path.join(data_dir, "Libri_lexicon.txt")
        self.lexicon = self._load_lexicon(lexicon_path)
        
        transcript_path = os.path.join(data_dir, f"{split}.txt")
        if not os.path.exists(transcript_path):
            transcript_path = os.path.join(data_dir, f"{split}_transcript.txt")
        if not os.path.exists(transcript_path):
            transcript_path = os.path.join(self.data_dir, f"{split}.txt")
        if not os.path.exists(transcript_path):
            transcript_path = os.path.join(self.data_dir, f"{split}_transcript.txt")
            
        self._load_transcripts(transcript_path)
        self._print_stats()

    def _load_lexicon(self, lexicon_path):
        lexicon = {}
        if os.path.exists(lexicon_path):
            with open(lexicon_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        word = parts[0]
                        phonemes = parts[1:]
                        lexicon[word] = phonemes
        return lexicon

    def _text_to_phonemes(self, text):
        words = text.split()
        phonemes = []
        has_oov = False
        for word in words:
            word = word.upper().strip(".,?!'\"")
            if word in self.lexicon:
                phonemes.extend(self.lexicon[word])
            else:
                has_oov = True
                
        if has_oov and config.SKIP_OOV_SENTENCES:
            return None
        return phonemes

    def _load_transcripts(self, transcript_path):
        if not os.path.exists(transcript_path):
            raise FileNotFoundError(f"Transcript not found at {transcript_path}")
            
        with open(transcript_path, 'r') as f:
            for line in f:
                parts = line.strip().split(maxsplit=1)
                if len(parts) == 2:
                    file_id = parts[0]
                    text = parts[1]
                    
                    speaker_id, chapter_id, _ = file_id.split('-')
                    audio_path = os.path.join(self.data_dir, speaker_id, chapter_id, f"{file_id}.flac")
                    
                    if not os.path.exists(audio_path):
                        audio_path = audio_path.replace('.flac', '.wav')
                        
                    if os.path.exists(audio_path):
                        phonemes = self._text_to_phonemes(text)
                        if phonemes is None:
                            self.oov_sentences += 1
                            self.skipped_samples += 1
                        else:
                            self.samples.append({
                                "audio_path": audio_path,
                                "phonemes": phonemes,
                                "feature_targets": self.phonemes_to_feature_targets(phonemes)
                            })
                    else:
                        self.skipped_samples += 1

    def _print_stats(self):
        logger.info(f"--- LibriSpeech Dataset ({self.split}) Validation ---")
        logger.info(f"Total valid samples: {len(self.samples)}")
        logger.info(f"Skipped samples: {self.skipped_samples}")
        logger.info(f"OOV sentences dropped: {self.oov_sentences}")
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
