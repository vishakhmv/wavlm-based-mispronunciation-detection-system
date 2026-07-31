import os
import torch
from dataclasses import dataclass

@dataclass
class Config:
    BASE_DIR: str = os.environ.get("BASE_DIR", "/sd1/meenakshi")
    
    PROJECT_DIR: str = os.path.join(BASE_DIR, "wavlm-based-mispronunciation-detection-system")
    DATASET_DIR: str = os.path.join(BASE_DIR, "Dataset")
    
    OUTPUT_DIR: str = os.path.join(PROJECT_DIR, "outputs")
    CHECKPOINT_DIR: str = os.path.join(OUTPUT_DIR, "checkpoints")
    LOG_DIR: str = os.path.join(OUTPUT_DIR, "logs")
    RESULTS_DIR: str = os.path.join(OUTPUT_DIR, "results")
    PREDICTIONS_DIR: str = os.path.join(OUTPUT_DIR, "predictions")
    
    MODEL_NAME: str = "microsoft/wavlm-large"
    NUM_FEATURES: int = 35
    NUM_OUTPUTS: int = 71
    
    BATCH_SIZE: int = 8
    EPOCHS: int = 30
    LEARNING_RATE: float = 1e-4
    WEIGHT_DECAY: float = 0.005
    WARMUP_RATIO: float = 0.1
    SEED: int = 42
    
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    SKIP_OOV_SENTENCES: bool = True
    
    ALL_SPEAKERS: tuple = (
        'ABA', 'ASI', 'BWC', 'LJS', 'NCC', 'NJS', 
        'SVBI', 'TNI', 'TXHC', 'YDCK', 'YKWK', 'HJK', 
        'RRBI', 'ERRO', 'EBVS', 'SCA', 'THV', 'HKK', 
        'YBAA', 'MBOS', 'HQTV', 'PNV', 'ZHAA', 'RMS'
    )
    TEST_SPEAKERS: tuple = ('HKK', 'YBAA', 'MBOS', 'HQTV', 'PNV', 'ZHAA')
    TRAIN_SPEAKERS: tuple = tuple(sorted(set(ALL_SPEAKERS) - set(TEST_SPEAKERS)))
    
    NUM_WORKERS: int = 4
    PIN_MEMORY: bool = True
    
    SAMPLE_RATE: int = 16000
    
config = Config()
