import os
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
import json
import datetime
import subprocess
import argparse

from config import config
from datasets.collate import mdd_collate_fn
from datasets.timit_dataset import TIMITDataset
from datasets.librispeech_dataset import LibriSpeechDataset
from datasets.l2arctic_dataset import L2ArcticDataset
from torch.utils.data import DataLoader
from model import WavLMMDD
from loss import SCTCSBLoss

def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )
    return LambdaLR(optimizer, lr_lambda, last_epoch)

def get_git_revision_hash() -> str:
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    except Exception:
        return "Unknown"

def train(dataset_name: str, resume_path: str = None):
    import random
    import numpy as np
    
    torch.manual_seed(config.SEED)
    torch.cuda.manual_seed_all(config.SEED)
    np.random.seed(config.SEED)
    random.seed(config.SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    device = torch.device(config.DEVICE)
    print(f"Using device: {device}")
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    experiment_dir = os.path.join(config.OUTPUT_DIR, f"experiment_{dataset_name}_{timestamp}")
    os.makedirs(os.path.join(experiment_dir, "checkpoints"), exist_ok=True)
    
    metadata = {
        "dataset_name": dataset_name,
        "model_name": config.MODEL_NAME,
        "batch_size": config.BATCH_SIZE,
        "epochs": config.EPOCHS,
        "learning_rate": config.LEARNING_RATE,
        "weight_decay": config.WEIGHT_DECAY,
        "warmup_ratio": config.WARMUP_RATIO,
        "git_commit": get_git_revision_hash(),
        "timestamp": timestamp,
        "skip_oov_sentences": config.SKIP_OOV_SENTENCES,
        "train_speakers": config.TRAIN_SPEAKERS,
        "test_speakers": config.TEST_SPEAKERS
    }
    with open(os.path.join(experiment_dir, "config.json"), 'w') as f:
        json.dump(metadata, f, indent=4)
    log_file = open(os.path.join(experiment_dir, "train.log"), 'w')
    def log_print(msg):
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()
        
    log_print(f"--- Experiment started: {timestamp} ---")
    log_print(f"Dataset selected: {dataset_name}")
    
    if dataset_name.lower() == "l2arctic":
        train_dataset = L2ArcticDataset(config.DATASET_DIR, split="train")
        val_dataset = L2ArcticDataset(config.DATASET_DIR, split="test")
    elif dataset_name.lower() == "timit":
        train_dataset = TIMITDataset(config.DATASET_DIR, split="train")
        val_dataset = TIMITDataset(config.DATASET_DIR, split="test")
    elif dataset_name.lower() == "librispeech":
        train_dataset = LibriSpeechDataset(config.DATASET_DIR, split="train-clean-100")
        val_dataset = LibriSpeechDataset(config.DATASET_DIR, split="dev-clean")
    elif dataset_name.lower() == "timit_l2":
        from torch.utils.data import ConcatDataset
        train_dataset = ConcatDataset([
            TIMITDataset(config.DATASET_DIR, split="train"),
            L2ArcticDataset(config.DATASET_DIR, split="train")
        ])
        val_dataset = ConcatDataset([
            TIMITDataset(config.DATASET_DIR, split="test"),
            L2ArcticDataset(config.DATASET_DIR, split="test")
        ])
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    log_print("Validating datasets before training...")
    if len(train_dataset) == 0:
        raise RuntimeError("Train dataset is empty! Check preprocessing or paths.")
    if len(val_dataset) == 0:
        raise RuntimeError("Validation dataset is empty! Check preprocessing or paths.")
    
    log_print(f"Train Dataset: {len(train_dataset)} samples")
    log_print(f"Val Dataset: {len(val_dataset)} samples")
    
    sample = train_dataset[0]
    if not sample["waveform"].size(0) > 0:
        raise RuntimeError("Dataset generated empty audio waveform.")
    if len(sample["feature_targets"]) != config.NUM_FEATURES:
        raise RuntimeError(f"Expected {config.NUM_FEATURES} feature targets, got {len(sample['feature_targets'])}.")
    if not all(t.size(0) > 0 for t in sample["feature_targets"]):
        raise RuntimeError("Dataset generated empty target features.")
    
    log_print("✓ Dataset validation passed.\n")
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True, 
        collate_fn=mdd_collate_fn,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        collate_fn=mdd_collate_fn,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )
    
    model = WavLMMDD(model_name=config.MODEL_NAME, num_features=config.NUM_FEATURES)
    model.to(device)
    
    criterion = SCTCSBLoss(num_features=config.NUM_FEATURES)
    
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    
    total_steps = len(train_loader) * config.EPOCHS
    warmup_steps = int(total_steps * config.WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    best_val_loss = float('inf')
    start_epoch = 1
    
    history = []
    
    if resume_path and os.path.exists(resume_path):
        log_print(f"Resuming from checkpoint: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if scaler and checkpoint.get('scaler_state_dict') is not None:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_val_loss = checkpoint.get('val_loss', float('inf'))
        log_print(f"Successfully loaded checkpoint. Resuming at epoch {start_epoch}...")
    
    import atexit
    atexit.register(log_file.close)
    
    log_print("Starting training loop...")
    try:
        for epoch in range(start_epoch, config.EPOCHS + 1):
            model.train()
            epoch_loss = 0.0
            
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{config.EPOCHS} [Train]")
            for batch in progress_bar:
                waveforms = batch["waveforms"].to(device)
                input_lengths = batch["input_lengths"]
                target_lengths = batch["target_lengths"]
                
                targets = [t.to(device) for t in batch["feature_targets"]]
                
                optimizer.zero_grad(set_to_none=True)
                
                attention_mask = torch.zeros_like(waveforms, dtype=torch.long, device=device)
                for b, length in enumerate(input_lengths):
                    attention_mask[b, :length] = 1
                    
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == 'cuda')):
                    logits = model(waveforms, attention_mask=attention_mask)
                    downsampled_input_lengths = model.get_feat_extract_output_lengths(input_lengths)
                    loss = criterion(logits, targets, downsampled_input_lengths, target_lengths)
                
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    scale_before = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    scale_after = scaler.get_scale()
                    
                    if scale_before <= scale_after:
                        scheduler.step()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step()
                
                epoch_loss += loss.item()
                current_lr = scheduler.get_last_lr()[0]
                progress_bar.set_postfix({'loss': loss.item(), 'lr': f"{current_lr:.2e}"})
                
            avg_train_loss = epoch_loss / len(train_loader)
            
            model.eval()
            val_loss = 0.0
            val_bar = tqdm(val_loader, desc=f"Epoch {epoch}/{config.EPOCHS} [Val]")
            with torch.no_grad():
                for batch in val_bar:
                    waveforms = batch["waveforms"].to(device)
                    input_lengths = batch["input_lengths"]
                    target_lengths = batch["target_lengths"]
                    targets = [t.to(device) for t in batch["feature_targets"]]
                    
                    attention_mask = torch.zeros_like(waveforms, dtype=torch.long, device=device)
                    for b, length in enumerate(input_lengths):
                        attention_mask[b, :length] = 1
                        
                    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == 'cuda')):
                        logits = model(waveforms, attention_mask=attention_mask)
                        downsampled_input_lengths = model.get_feat_extract_output_lengths(input_lengths)
                        loss = criterion(logits, targets, downsampled_input_lengths, target_lengths)
                        
                    val_loss += loss.item()
                    val_bar.set_postfix({'loss': loss.item()})
                    
            avg_val_loss = val_loss / len(val_loader)
            
            log_print(f"Epoch {epoch} Summary - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {current_lr:.2e}")
            
            history.append({
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "lr": current_lr
            })
            with open(os.path.join(experiment_dir, "history.json"), 'w') as f:
                json.dump(history, f, indent=4)
            
            checkpoint_state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict() if scaler else None,
                'val_loss': avg_val_loss,
            }
            
            checkpoint_path = os.path.join(experiment_dir, "checkpoints", "latest.pt")
            torch.save(checkpoint_state, checkpoint_path)
            log_print(f"Saved latest checkpoint to {checkpoint_path}")
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_path = os.path.join(experiment_dir, "checkpoints", "best_model.pt")
                torch.save(checkpoint_state, best_model_path)
                log_print(f"*** New best model saved! (Val Loss: {best_val_loss:.4f}) ***")
                
    except KeyboardInterrupt:
        log_print("\nTraining interrupted by user. Saving latest checkpoint...")
        if 'checkpoint_state' in locals():
            interrupt_path = os.path.join(experiment_dir, "checkpoints", "interrupted.pt")
            torch.save(checkpoint_state, interrupt_path)
            log_print(f"Saved interrupted checkpoint to {interrupt_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train WavLM-MDD model")
    parser.add_argument("--dataset", type=str, default="l2arctic", choices=["l2arctic", "timit", "librispeech", "timit_l2"], help="Dataset to train on")
    parser.add_argument("--resume", type=str, default=None, help="Path to a checkpoint (.pt) to resume training from")
    
    args = parser.parse_args()
    train(args.dataset, args.resume)
