#!/usr/bin/env python3
"""
lstm_text_generator_pytorch.py
Production-ready, character-level LSTM text generator using PyTorch.

I write in 1st person style: I build systems that are reproducible,
well-structured and ready for extension (logging, callbacks, etc.).
"""
from __future__ import annotations

import argparse
import math
import os
import random
import re
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

# ----------------------------
# Config / hyperparameters
# ----------------------------
@dataclass
class Config:
    data_file: str = "shakespeare.txt"
    seq_length: int = 100
    embedding_dim: int = 256
    lstm_units: int = 512
    num_layers: int = 2
    dropout: float = 0.2
    batch_size: int = 128
    epochs: int = 0  # set >0 to train
    lr: float = 1e-3
    val_split: float = 0.1
    checkpoint_path: str = "lstm_pytorch_best.pt"
    patience: int = 5  # early stopping
    grad_clip: float = 5.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    max_train_samples: Optional[int] = None  # set to int to limit data for quick runs
    deterministic: bool = True  # if True, try to make inference deterministic

cfg = Config()

# ----------------------------
# Utilities
# ----------------------------
def set_seed(seed: int, deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        # request deterministic behavior (may slow training/inference)
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def clean_text(text: str) -> str:
    # Lowercase and keep only alphanum, spaces and newlines
    # If you want punctuation preserved, expand allowed chars here.
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s\n]", "", text)
    return text


# ----------------------------
# Dataset
# ----------------------------
class CharDataset(Dataset):
    def __init__(self, text: str, seq_len: int, char_to_int: Dict[str, int]):
        self.text = text
        self.seq_len = seq_len
        self.char_to_int = char_to_int
        self.vocab_size = len(char_to_int)
        # map text to indices
        self.data_idxs = [char_to_int[c] for c in self.text]
        self.n_samples = max(0, len(self.data_idxs) - seq_len)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = idx
        seq_in = self.data_idxs[start : start + self.seq_len]
        seq_out = self.data_idxs[start + self.seq_len]
        x = torch.tensor(seq_in, dtype=torch.long)
        y = torch.tensor(seq_out, dtype=torch.long)
        return x, y


# ----------------------------
# Model
# ----------------------------
class CharLSTM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 256,
        hidden_size: int = 512,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x: torch.Tensor, hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        emb = self.embedding(x)  # (batch, seq_len, emb)
        out, hidden = self.lstm(emb, hidden)  # out: (batch, seq_len, hidden)
        out = self.dropout(out)
        last = out[:, -1, :]  # (batch, hidden)
        logits = self.fc(last)  # (batch, vocab_size)
        return logits, hidden

    def init_hidden(self, batch_size: int, device: str):
        weight = next(self.parameters()).data
        num_layers = self.lstm.num_layers
        hidden_size = self.lstm.hidden_size
        h0 = weight.new_zeros(num_layers, batch_size, hidden_size).to(device)
        c0 = weight.new_zeros(num_layers, batch_size, hidden_size).to(device)
        return (h0, c0)


# ----------------------------
# Training & validation loops
# ----------------------------
def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
    clip: float,
):
    model.train()
    total_loss = 0.0
    total_count = 0
    for x_batch, y_batch in tqdm(dataloader, desc="train", leave=False):
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        batch_size = x_batch.size(0)
        hidden = model.init_hidden(batch_size, device)
        optimizer.zero_grad()
        logits, _ = model(x_batch, hidden)
        loss = criterion(logits, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        total_loss += loss.item() * batch_size
        total_count += batch_size
    avg_loss = total_loss / total_count if total_count > 0 else float("inf")
    return avg_loss


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str,
):
    model.eval()
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for x_batch, y_batch in tqdm(dataloader, desc="val", leave=False):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            batch_size = x_batch.size(0)
            hidden = model.init_hidden(batch_size, device)
            logits, _ = model(x_batch, hidden)
            loss = criterion(logits, y_batch)
            total_loss += loss.item() * batch_size
            total_count += batch_size
    avg_loss = total_loss / total_count if total_count > 0 else float("inf")
    return avg_loss


# ----------------------------
# Sampling helpers
# ----------------------------
def sample_from_logits(logits: np.ndarray, temperature: float = 1.0) -> int:
    if temperature <= 0:
        return int(np.argmax(logits))
    scaled = logits.astype(np.float64) / float(temperature)
    scaled = scaled - np.max(scaled)
    exp = np.exp(scaled)
    probs = exp / (exp.sum() + 1e-20)
    return int(np.random.choice(len(probs), p=probs))


# ----------------------------
# Text generation (modes)
# ----------------------------
def generate_text_greedy(
    model: nn.Module,
    seed_text: str,
    char_to_int: Dict[str, int],
    int_to_char: Dict[int, str],
    length: int = 500,
    device: str = "cpu",
    seq_len: int = 100,
):
    model.eval()
    if len(seed_text) != seq_len:
        raise ValueError(f"Seed length must equal seq_len ({seq_len}). Got {len(seed_text)}.")
    pattern = [char_to_int[c] for c in seed_text]
    generated = seed_text
    with torch.no_grad():
        for _ in range(length):
            x = torch.tensor([pattern], dtype=torch.long).to(device)
            logits, _ = model(x)
            logits = logits.squeeze(0)
            idx = int(torch.argmax(logits).item())
            ch = int_to_char[idx]
            generated += ch
            pattern.append(idx)
            pattern = pattern[1:]
    return generated


def generate_text_temperature(
    model: nn.Module,
    seed_text: str,
    char_to_int: Dict[str, int],
    int_to_char: Dict[int, str],
    length: int = 500,
    temperature: float = 1.0,
    device: str = "cpu",
    seq_len: int = 100,
):
    model.eval()
    if len(seed_text) != seq_len:
        raise ValueError(f"Seed length must equal seq_len ({seq_len}). Got {len(seed_text)}.")
    pattern = [char_to_int[c] for c in seed_text]
    generated = seed_text
    with torch.no_grad():
        for _ in range(length):
            x = torch.tensor([pattern], dtype=torch.long).to(device)
            logits, _ = model(x)
            logits = logits.squeeze(0).cpu().numpy()
            idx = sample_from_logits(logits, temperature)
            ch = int_to_char[idx]
            generated += ch
            pattern.append(idx)
            pattern = pattern[1:]
    return generated


def generate_text_beam(
    model: nn.Module,
    seed_text: str,
    char_to_int: Dict[str, int],
    int_to_char: Dict[int, str],
    length: int = 200,
    beam_width: int = 5,
    device: str = "cpu",
    seq_len: int = 100,
):
    model.eval()
    if len(seed_text) != seq_len:
        raise ValueError(f"Seed length must equal seq_len ({seq_len}). Got {len(seed_text)}.")
    init_pattern = [char_to_int[c] for c in seed_text]
    beams: List[Tuple[List[int], float, str]] = [(init_pattern[:], 0.0, seed_text)]
    with torch.no_grad():
        for _ in range(length):
            candidates = []
            for pattern, logp, gen_text in beams:
                x = torch.tensor([pattern], dtype=torch.long).to(device)
                logits, _ = model(x)
                logits = logits.squeeze(0).cpu().float().numpy()
                logits = logits - np.max(logits)
                probs = np.exp(logits)
                probs = probs / (probs.sum() + 1e-20)
                top_k = min(beam_width, len(probs))
                idxs = np.argpartition(-probs, top_k - 1)[:top_k]
                for idx in idxs:
                    p = probs[idx]
                    if p <= 0:
                        continue
                    new_logp = logp + float(np.log(p + 1e-20))
                    new_pattern = pattern[:] + [int(idx)]
                    new_pattern = new_pattern[1:]
                    new_text = gen_text + int_to_char[int(idx)]
                    candidates.append((new_pattern, new_logp, new_text))
            candidates.sort(key=lambda t: t[1], reverse=True)
            beams = candidates[:beam_width]
    best = max(beams, key=lambda t: t[1])
    return best[2]


# ----------------------------
# Main
# ----------------------------
def main(config: Config, args):
    set_seed(config.seed, deterministic=config.deterministic)
    device = config.device
    print(f"Running on device: {device}")

    # 1) Load data
    if not os.path.exists(config.data_file):
        print(f"Data file not found: {config.data_file}", file=sys.stderr)
        sys.exit(1)

    with open(config.data_file, "r", encoding="utf-8") as f:
        raw = f.read()

    raw = raw[: config.max_train_samples] if config.max_train_samples else raw
    raw = clean_text(raw)
    print(f"Loaded {len(raw)} characters after cleaning.")

    # 2) build vocab from raw
    chars = sorted(list(set(raw)))
    char_to_int = {c: i for i, c in enumerate(chars)}
    int_to_char = {i: c for c, i in char_to_int.items()}
    vocab_size = len(chars)
    print(f"Vocab size (from raw): {vocab_size} unique characters.")

    # If checkpoint exists, prefer checkpoint vocab (to avoid mismatch)
    if os.path.exists(config.checkpoint_path):
        try:
            print(f"Found checkpoint {config.checkpoint_path}; inspecting vocab...")
            ckpt = torch.load(config.checkpoint_path, map_location=device)
            if "vocab" in ckpt and "char_to_int" in ckpt["vocab"]:
                ckpt_vocab = ckpt["vocab"]["char_to_int"]
                if isinstance(ckpt_vocab, dict):
                    print(f"Checkpoint vocab size: {len(ckpt_vocab)}")
                    char_to_int = ckpt_vocab
                    int_to_char = {i: c for c, i in char_to_int.items()}
                    vocab_size = len(char_to_int)
        except Exception as e:
            print(f"Warning: could not inspect checkpoint vocab: {e}")

    # 3) dataset / dataloader
    dataset = CharDataset(raw, config.seq_length, char_to_int)
    if len(dataset) == 0:
        print("Dataset is empty. Check sequence length / input file.", file=sys.stderr)
        sys.exit(1)

    val_len = max(1, int(len(dataset) * config.val_split))
    train_len = len(dataset) - val_len
    train_set, val_set = random_split(dataset, [train_len, val_len], generator=torch.Generator().manual_seed(config.seed))
    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=config.batch_size, shuffle=False, drop_last=False)
    print(f"Train samples: {len(train_set)}, Val samples: {len(val_set)}")

    # 4) model (create after deciding vocab_size)
    model = CharLSTM(
        vocab_size=vocab_size,
        embedding_dim=config.embedding_dim,
        hidden_size=config.lstm_units,
        num_layers=config.num_layers,
        dropout=config.dropout,
    ).to(device)

    # If checkpoint exists, load weights
    if os.path.exists(config.checkpoint_path):
        try:
            print(f"Loading checkpoint weights from {config.checkpoint_path}")
            ckpt = torch.load(config.checkpoint_path, map_location=device)
            try:
                model.load_state_dict(ckpt["model_state"])
                print("Checkpoint loaded (strict=True).")
            except RuntimeError:
                model.load_state_dict(ckpt["model_state"], strict=False)
                print("Checkpoint loaded with strict=False (architectural mismatch warnings may apply).")
        except Exception as e:
            print(f"Warning: failed to load checkpoint weights: {e}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    best_val_loss = math.inf
    epochs_since_improve = 0

    # Optionally train
    if config.epochs > 0 and not args.generate_only:
        for epoch in range(1, config.epochs + 1):
            print(f"\nEpoch {epoch}/{config.epochs}")
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, config.grad_clip)
            val_loss = evaluate(model, val_loader, criterion, device)
            print(f"Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_since_improve = 0
                try:
                    torch.save(
                        {
                            "model_state": model.state_dict(),
                            "config": vars(config),
                            "vocab": {"char_to_int": char_to_int, "int_to_char": int_to_char},
                            "epoch": epoch,
                            "val_loss": val_loss,
                        },
                        config.checkpoint_path,
                    )
                    print(f"Saved improved checkpoint to {config.checkpoint_path}")
                except Exception as e:
                    print(f"Warning: failed to save checkpoint: {e}")
            else:
                epochs_since_improve += 1
                print(f"No improvement for {epochs_since_improve} epoch(s).")

            if epochs_since_improve >= config.patience:
                print("Early stopping triggered.")
                break
    else:
        if args.generate_only:
            print("Generate-only mode: skipping training and using checkpoint if available.")
        else:
            print("Skipping training (epochs=0). If you want to train, set epochs>0 in config.")

    # 5) Generation utilities and samples
    def clean_seed(seed: str, length: int) -> str:
        s = clean_text(seed)
        if len(s) >= length:
            return s[:length]
        return s.ljust(length)

    print("\n" + "=" * 50)
    print("GENERATED TEXT SAMPLES")
    print("=" * 50)

    start_idx = random.randint(0, max(0, len(raw) - config.seq_length - 1))
    random_seed = raw[start_idx : start_idx + config.seq_length]
    specific_seed = "to be or not to be that is the question whether tis nobler in the mind to suffer the slings and arrows"
    specific_seed = clean_seed(specific_seed, config.seq_length)

    mode = args.mode.lower()
    gen_len = args.gen_len

    if mode == "greedy":
        gen_fn = generate_text_greedy
    elif mode == "temp":
        gen_fn = lambda *a, **k: generate_text_temperature(*a, temperature=args.temperature, **k)
    elif mode == "beam":
        gen_fn = lambda *a, **k: generate_text_beam(*a, beam_width=args.beam_width, **k)
    else:
        raise ValueError("Unsupported mode. Choose from greedy,temp,beam")

    print(f"\n--- Sample 1: Random Seed, mode={mode}, temp={args.temperature} ---")
    print(gen_fn(model, random_seed, char_to_int, int_to_char, length=gen_len, device=device, seq_len=config.seq_length))

    print(f"\n--- Sample 2: Specific Seed, mode={mode}, temp={args.temperature} ---")
    print(gen_fn(model, specific_seed, char_to_int, int_to_char, length=gen_len, device=device, seq_len=config.seq_length))

    print(f"\n--- Sample 3: Random Seed, mode={mode}, temp={args.temperature} ---")
    print(gen_fn(model, random_seed, char_to_int, int_to_char, length=gen_len, device=device, seq_len=config.seq_length))

    print("\n" + "=" * 50)
    print("MODEL REPORT")
    print("=" * 50)
    print(f"Architecture: Embedding({config.embedding_dim}) -> LSTM({config.lstm_units}) x {config.num_layers} -> Dense({vocab_size})")
    print(f"Sequence length: {config.seq_length}, Batch size: {config.batch_size}, Dropout: {config.dropout}")
    if config.epochs > 0 and not args.generate_only:
        print(f"Best validation loss: {best_val_loss:.4f}")
    print("Notes: Increase dataset size, tune embedding/lstm sizes, and use mixed precision + multi-GPU for large training runs.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Char-level LSTM text generator (PyTorch)")
    parser.add_argument("--data", type=str, default=cfg.data_file, help="Path to text file")
    parser.add_argument("--epochs", type=int, default=cfg.epochs, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=cfg.batch_size, help="Batch size")
    parser.add_argument("--seq_len", type=int, default=cfg.seq_length, help="Sequence length")
    parser.add_argument("--checkpoint", type=str, default=cfg.checkpoint_path, help="Checkpoint path")
    parser.add_argument("--max_train_chars", type=int, default=cfg.max_train_samples or 0, help="Limit chars (0 = all)")
    parser.add_argument("--generate_only", action="store_true", help="Skip training and only generate using checkpoint (if present)")
    parser.add_argument("--mode", type=str, default="greedy", choices=["greedy", "temp", "beam"], help="Generation mode: greedy/temp/beam")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature (temp mode)")
    parser.add_argument("--beam_width", type=int, default=5, help="Beam width (beam mode)")
    parser.add_argument("--gen_len", type=int, default=500, help="Generated sequence length")
    args = parser.parse_args()

    # override config from CLI
    if args.data:
        cfg.data_file = args.data
    cfg.epochs = args.epochs
    cfg.batch_size = args.batch_size
    cfg.seq_length = args.seq_len
    cfg.checkpoint_path = args.checkpoint
    if args.max_train_chars and args.max_train_chars > 0:
        cfg.max_train_samples = args.max_train_chars

    if args.generate_only:
        cfg.epochs = 0

    main(cfg, args)
