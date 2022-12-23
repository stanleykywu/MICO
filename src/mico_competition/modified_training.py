import os
import argparse
import warnings
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torchcsprng import create_mt19937_generator, create_random_device_generator

from torch.utils.data import DataLoader

from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from opacus.utils.batch_memory_manager import BatchMemoryManager

from prv_accountant.dpsgd import find_noise_multiplier

from .accountant import PRVAccountant

from mico_competition import ChallengeDataset, MLP, load_purchase100

from tqdm import tqdm, trange

from datetime import datetime

from typing import Callable, Optional


def accuracy(preds: torch.Tensor, labels: torch.Tensor) -> float:
    return (preds == labels).mean()

def train_purchase(
    model: nn.Module,
    device: torch.device,
    train_loader: DataLoader,
    criterion,
    optimizer: optim.Optimizer,
    epoch: int,
    batch_size: int,
    max_physical_batch_size: int,
    dp: bool,
    compute_epsilon: Optional[Callable[[int], float]] = None,
    logging_steps: int = 10,
):
    model.train()

    losses = []
    top1_acc = []

    with BatchMemoryManager(
        data_loader=train_loader,
        max_physical_batch_size=max_physical_batch_size,
        optimizer=optimizer
    ) as memory_safe_data_loader:

        if not dp:
            data_loader = train_loader
        else:
            data_loader = memory_safe_data_loader

        # BatchSplittingSampler.__len__() approximates (badly) the length in physical batches
        # See https://github.com/pytorch/opacus/issues/516
        # We instead heuristically keep track of logical batches processed
        pbar = tqdm(data_loader, desc="Batch", unit="batch", position=1, leave=True, total=len(train_loader), disable=None)
        logical_batch_len = 0
        for i, (inputs, target) in enumerate(data_loader):
            inputs = inputs.to(device)
            target = target.to(device)

            logical_batch_len += len(target)
            if logical_batch_len >= batch_size:
                pbar.update(1)
                logical_batch_len = logical_batch_len % max_physical_batch_size

            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output, target)

            preds = np.argmax(output.detach().cpu().numpy(), axis=1)
            labels = target.detach().cpu().numpy()
            acc = accuracy(preds, labels)

            losses.append(loss.item())
            top1_acc.append(acc)

            loss.backward()
            optimizer.step()

            if (pbar.n + 1) % logging_steps == 0 or (pbar.n + 1) == pbar.total:
                if dp:
                    epsilon = compute_epsilon(delta=target_delta)
                    pbar.set_postfix(
                        epoch=f"{epoch:02}",
                        train_loss=f"{np.mean(losses):.3f}",
                        accuracy=f"{np.mean(top1_acc) * 100:.3f}",
                        dp=f"(ε={epsilon:.2f}, δ={target_delta})"
                    )
                else:
                    pbar.set_postfix(
                        epoch=f"{epoch:02}",
                        train_loss=f"{np.mean(losses):.3f}",
                        accuracy=f"{np.mean(top1_acc) * 100:.3f}",
                        dp="(ε = ∞, δ = 0)"
                    )

        pbar.update(pbar.total - pbar.n)
        
        
def train_cifar(
    model: nn.Module,
    device: torch.device,
    train_loader: DataLoader,
    criterion,
    optimizer: optim.Optimizer,
    epoch: int,
    batch_size: int,
    max_physical_batch_size: int,
    dp: bool,
    compute_epsilon: Optional[Callable[[int], float]] = None,
    logging_steps: int = 10,
):
    model.train()

    losses = []
    top1_acc = []

    with BatchMemoryManager(
        data_loader=train_loader,
        max_physical_batch_size=max_physical_batch_size,
        optimizer=optimizer
    ) as memory_safe_data_loader:

        if not dp:
            data_loader = train_loader
        else:
            data_loader = memory_safe_data_loader

        # BatchSplittingSampler.__len__() approximates (badly) the length in physical batches
        # See https://github.com/pytorch/opacus/issues/516
        # We instead heuristically keep track of logical batches processed
        pbar = tqdm(data_loader, desc="Batch", unit="batch", position=1, leave=True, total=len(train_loader), disable=None)
        logical_batch_len = 0
        for i, (inputs, target) in enumerate(data_loader):
            inputs = inputs.to(device)
            target = target.to(device)

            logical_batch_len += len(target)
            if logical_batch_len >= batch_size:
                pbar.update(1)
                logical_batch_len = logical_batch_len % max_physical_batch_size

            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output, target)

            preds = np.argmax(output.detach().cpu().numpy(), axis=1)
            labels = target.detach().cpu().numpy()
            acc = accuracy(preds, labels)

            losses.append(loss.item())
            top1_acc.append(acc)

            loss.backward()
            optimizer.step()

            if (pbar.n + 1) % logging_steps == 0 or (pbar.n + 1) == pbar.total:
                if dp:
                    epsilon = compute_epsilon(delta=target_delta)
                    pbar.set_postfix(
                        epoch=f"{epoch:02}",
                        train_loss=f"{np.mean(losses):.3f}",
                        accuracy=f"{np.mean(top1_acc) * 100:.3f}",
                        dp=f"(ε={epsilon:.2f}, δ={target_delta})"
                    )
                else:
                    pbar.set_postfix(
                        epoch=f"{epoch:02}",
                        train_loss=f"{np.mean(losses):.3f}",
                        accuracy=f"{np.mean(top1_acc) * 100:.3f}",
                        dp="(ε = ∞, δ = 0)"
                    )

        pbar.update(pbar.total - pbar.n)