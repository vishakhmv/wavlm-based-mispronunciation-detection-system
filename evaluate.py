import os
import torch
import Levenshtein
from tqdm import tqdm
from typing import List
import json
import argparse
import datetime
import matplotlib.pyplot as plt
import numpy as np

from config import config
from datasets.collate import mdd_collate_fn
from datasets.timit_dataset import TIMITDataset
from datasets.librispeech_dataset import LibriSpeechDataset
from datasets.l2arctic_dataset import L2ArcticDataset
from torch.utils.data import DataLoader
from model import WavLMMDD
from phoneme_features import PHONOLOGICAL_FEATURES

def plot_learning_curve(history_path, output_dir):
    if not os.path.exists(history_path):
        return
    with open(history_path, 'r') as f:
        history = json.load(f)
    if not history:
        return
    epochs = [x['epoch'] for x in history]
    train_loss = [x.get('train_loss', 0) for x in history]
    val_loss = [x.get('val_loss', 0) for x in history]
    
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, label='Train Loss', marker='o')
    plt.plot(epochs, val_loss, label='Validation Loss', marker='o')
    plt.title('Learning Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'learning_curve.png'))
    plt.close()

def plot_confusion_matrix(tp, fp, fn, tn, output_dir):
    matrix = np.array([[tn, fp], [fn, tp]])
    fig, ax = plt.subplots(figsize=(5, 5))
    cax = ax.matshow(matrix, cmap='Blues')
    
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(matrix[i, j]), va='center', ha='center',
                    color='white' if matrix[i, j] > matrix.max() / 2. else 'black')
            
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Negative', 'Positive'])
    ax.set_yticklabels(['Negative', 'Positive'])
    plt.title('Global Confusion Matrix', pad=20)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    ax.xaxis.set_ticks_position('bottom')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()

def decode_predictions(logits, input_lengths, model):
    """
    Decodes the model logits into binary feature sequences for each of the 35 features.
    logits: (Batch, SeqLen, 71)
    Returns a list of batch decoded sequences.
    """
    batch_size, seq_len, _ = logits.size()
    downsampled_lengths = model.get_feat_extract_output_lengths(input_lengths)
    
    batch_decoded = []
    
    for b in range(batch_size):
        length = downsampled_lengths[b].item()
        utterance_logits = logits[b, :length, :]
        
        feature_sequences = []
        for i in range(config.NUM_FEATURES):

            feat_logits = torch.stack([
                utterance_logits[:, i], 
                utterance_logits[:, i + config.NUM_FEATURES], 
                utterance_logits[:, -1]
            ], dim=-1)
            
            preds = torch.argmax(feat_logits, dim=-1)
            
            seq = []
            prev = -1
            for p in preds:
                p_item = p.item()
                if p_item != prev and p_item != 2:
                    val = 1 if p_item == 0 else 0
                    seq.append(val)
                prev = p_item
                
            feature_sequences.append(seq)
        batch_decoded.append(feature_sequences)
        
    return batch_decoded

def calculate_edits(predicted: List[int], target: List[int]) -> dict:
    """
    Calculate raw counts of edits for binary sequences using Levenshtein alignment.
    """
    pred_str = "".join([str(x) for x in predicted])
    tgt_str = "".join([str(x) for x in target])
    
    if len(tgt_str) == 0:
        return {"FA": 0, "FR": 0, "total_0": 0, "total_1": 0, "edits": 0, "N": 0}
        
    distance = Levenshtein.distance(pred_str, tgt_str)
    
    ops = Levenshtein.editops(pred_str, tgt_str)
    FA = 0
    FR = 0
    
    for op, p_idx, t_idx in ops:
        if op == 'replace':
            if pred_str[p_idx] == '1' and tgt_str[t_idx] == '0':
                FA += 1
            elif pred_str[p_idx] == '0' and tgt_str[t_idx] == '1':
                FR += 1
        elif op == 'insert':
            if tgt_str[t_idx] == '1':
                FR += 1
            else:
                FA += 1
        elif op == 'delete':
            if pred_str[p_idx] == '1':
                FA += 1
            else:
                FR += 1
                
    total_1 = tgt_str.count('1')
    total_0 = tgt_str.count('0')
    
    return {
        "FA": FA,
        "FR": FR,
        "total_0": total_0,
        "total_1": total_1,
        "edits": distance,
        "N": len(tgt_str)
    }

def evaluate(checkpoint_path: str, dataset_name: str):
    device = torch.device(config.DEVICE)
    print(f"Using device: {device}")
    
    if dataset_name.lower() == "l2arctic":
        dataset = L2ArcticDataset(config.DATASET_DIR, split="test")
    elif dataset_name.lower() == "timit":
        dataset = TIMITDataset(config.DATASET_DIR, split="test")
    elif dataset_name.lower() == "librispeech":
        dataset = LibriSpeechDataset(config.DATASET_DIR, split="test-clean")
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    test_loader = DataLoader(
        dataset, 
        batch_size=config.BATCH_SIZE,
        shuffle=False, 
        collate_fn=mdd_collate_fn,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )
    
    if len(dataset) == 0:
        raise RuntimeError(f"Evaluation dataset '{dataset_name}' is empty.")
    
    # Load Model
    model = WavLMMDD(model_name=config.MODEL_NAME)
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
        
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' not in checkpoint:
        raise KeyError(f"Checkpoint at {checkpoint_path} is missing 'model_state_dict'.")
        
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    feature_stats = {feat: {"FA": 0, "FR": 0, "total_0": 0, "total_1": 0, "edits": 0, "N": 0} for feat in PHONOLOGICAL_FEATURES}
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            waveforms = batch["waveforms"].to(device)
            input_lengths = batch["input_lengths"]
            target_lengths = batch["target_lengths"]
            targets = [t.to(device) for t in batch["feature_targets"]]
            
            attention_mask = torch.zeros_like(waveforms, dtype=torch.long, device=device)
            for b, length in enumerate(input_lengths):
                attention_mask[b, :length] = 1
                
            logits = model(waveforms, attention_mask=attention_mask)
            
            decoded_batch = decode_predictions(logits, input_lengths, model)
            
            for b in range(len(decoded_batch)):
                decoded_seqs = decoded_batch[b] 
                target_len = target_lengths[b].item()
                
                if target_len == 0:

                    continue 
                
                for i, feat_name in enumerate(PHONOLOGICAL_FEATURES):
                    pred_seq = decoded_seqs[i]
                    tgt_seq = targets[i][b, :target_len].cpu().tolist()
                    
                    stats = calculate_edits(pred_seq, tgt_seq)
                    for k in feature_stats[feat_name]:
                        feature_stats[feat_name][k] += stats[k]
                    
    aggregated_results = {}
    print("\n--- Final Evaluation Results ---")
    
    global_stats = {"FA": 0, "FR": 0, "total_0": 0, "total_1": 0, "edits": 0, "N": 0}
    
    for feat_name, stats in feature_stats.items():
        FA = stats["FA"]
        FR = stats["FR"]
        total_0 = stats["total_0"]
        total_1 = stats["total_1"]
        edits = stats["edits"]
        N = stats["N"]
        
        fer = edits / N if N > 0 else 0.0
        far = FA / total_0 if total_0 > 0 else 0.0
        frr = FR / total_1 if total_1 > 0 else 0.0
        
        TP = total_1 - FR
        if TP < 0: TP = 0
        FP = FA
        FN = FR
        TN = total_0 - FA
        if TN < 0: TN = 0
        
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / total_1 if total_1 > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        aggregated_results[feat_name] = {
            "fer": fer, "far": far, "frr": frr,
            "precision": precision, "recall": recall, "f1": f1,
            "confusion_matrix": {"TP": TP, "FP": FP, "FN": FN, "TN": TN}
        }
        
        for k in global_stats:
            global_stats[k] += stats[k]
            
    g_FA = global_stats["FA"]
    g_FR = global_stats["FR"]
    g_total_0 = global_stats["total_0"]
    g_total_1 = global_stats["total_1"]
    g_edits = global_stats["edits"]
    g_N = global_stats["N"]
    
    g_fer = g_edits / g_N if g_N > 0 else 0.0
    g_far = g_FA / g_total_0 if g_total_0 > 0 else 0.0
    g_frr = g_FR / g_total_1 if g_total_1 > 0 else 0.0
    
    g_TP = g_total_1 - g_FR
    if g_TP < 0: g_TP = 0
    g_FP = g_FA
    g_FN = g_FR
    g_TN = g_total_0 - g_FA
    if g_TN < 0: g_TN = 0
    
    g_precision = g_TP / (g_TP + g_FP) if (g_TP + g_FP) > 0 else 0.0
    g_recall = g_TP / g_total_1 if g_total_1 > 0 else 0.0
    g_f1 = 2 * (g_precision * g_recall) / (g_precision + g_recall) if (g_precision + g_recall) > 0 else 0.0
    
    aggregated_results["global_metrics"] = {
        "FER": g_fer, "FAR": g_far, "FRR": g_frr,
        "precision": g_precision, "recall": g_recall, "f1": g_f1,
        "confusion_matrix": {"TP": g_TP, "FP": g_FP, "FN": g_FN, "TN": g_TN}
    }
    
    final_output = {
        "global_metrics": aggregated_results["global_metrics"],
        "per_feature_metrics": {k: v for k, v in aggregated_results.items() if k != "global_metrics"}
    }
    
    print(f"FER:       {g_fer * 100:.2f}%")
    print(f"FAR:       {g_far * 100:.2f}%")
    print(f"FRR:       {g_frr * 100:.2f}%")
    print(f"Precision: {g_precision * 100:.2f}%")
    print(f"Recall:    {g_recall * 100:.2f}%")
    print(f"F1 Score:  {g_f1 * 100:.2f}%")
    
    train_dataset = "unknown"
    if "experiment_" in checkpoint_path:
        try:
            parts = checkpoint_path.split("experiment_")[1].split(os.sep)[0].split("_")
            train_dataset = "_".join(parts[:-2])
        except:
            pass
            
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    eval_folder_name = f"eval_{train_dataset}_model_on_{dataset_name}_{timestamp}"
    eval_out_dir = os.path.join(config.RESULTS_DIR, eval_folder_name)
    os.makedirs(eval_out_dir, exist_ok=True)
    
    out_file = os.path.join(eval_out_dir, "evaluation_report.json")
    
    with open(out_file, 'w') as f:
        json.dump(final_output, f, indent=4)
        
    print(f"\nDetailed per-feature results saved to: {out_file}")
    
    plot_confusion_matrix(g_TP, g_FP, g_FN, g_TN, eval_out_dir)
    history_path = os.path.join(os.path.dirname(os.path.dirname(checkpoint_path)), "history.json")
    plot_learning_curve(history_path, eval_out_dir)
    print(f"Visualizations saved to: {eval_out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate WavLM-MDD model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the model checkpoint (.pt)")
    parser.add_argument("--dataset", type=str, default="l2arctic", choices=["l2arctic", "timit", "librispeech"], help="Dataset to evaluate on")
    
    args = parser.parse_args()
    evaluate(args.checkpoint, args.dataset)
