import random

import numpy as np
import torch
from torch import nn


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_epoch(model, data_loader, loss_fn, optimizer, device, log_every=50):
    model.train()
    total_loss = 0.0
    for batch_idx, batch in enumerate(data_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        targets = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if log_every and batch_idx % log_every == 0:
            print(f"Batch {batch_idx}/{len(data_loader)}, Loss: {loss.item():.4f}")

    return total_loss / len(data_loader)


def eval_epoch(model, data_loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs, targets)
            total_loss += loss.item()
    return total_loss / len(data_loader)


def run_training(model, train_loader, val_loader, optimizer, device, epochs,
                  patience, model_save_path, loss_log_paths=None):
    """Runs the train/val loop with early stopping. Saves the checkpoint
    with the lowest validation loss to model_save_path. If loss_log_paths
    is given as (train_path, val_path), appends "epoch,loss" per epoch."""
    loss_fn = nn.SmoothL1Loss()
    best_val_loss = float("inf")
    patience_counter = 0

    if loss_log_paths:
        open(loss_log_paths[0], "w").close()
        open(loss_log_paths[1], "w").close()

    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch + 1}/{epochs} ---")
        train_loss = train_epoch(model, train_loader, loss_fn, optimizer, device)
        val_loss = eval_epoch(model, val_loader, loss_fn, device)
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if loss_log_paths:
            with open(loss_log_paths[0], "a") as f:
                f.write(f"{epoch + 1},{train_loss:.4f}\n")
            with open(loss_log_paths[1], "a") as f:
                f.write(f"{epoch + 1},{val_loss:.4f}\n")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_save_path)
            print("Best model saved (lowest validation loss).")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    print("\nTraining complete.")
