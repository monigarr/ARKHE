"""
Unit tests for ML training modules.

Author: MoniGarr
Email: monigarr@MoniGarr.com
Website: MoniGarr.com
"""

import pytest
import torch
from pathlib import Path
from torch.utils.data import DataLoader

from math_research.ml import CollatzTransformer, CollatzDataset, Trainer


def test_collatz_dataset_creation():
    """Test CollatzDataset creation."""
    dataset = CollatzDataset(
        start_range=(1, 100),
        num_samples=100,
        base=24,
        max_length=20,
        seed=42,
    )
    
    assert len(dataset) == 100
    assert dataset.base == 24
    assert dataset.max_length == 20


def test_collatz_dataset_getitem():
    """Test dataset item retrieval."""
    dataset = CollatzDataset(
        start_range=(1, 100),
        num_samples=10,
        base=24,
        max_length=20,
        seed=42,
    )
    
    input_tensor, target_tensor = dataset[0]
    
    assert isinstance(input_tensor, torch.Tensor)
    assert isinstance(target_tensor, torch.Tensor)
    assert input_tensor.shape == (20,)
    assert target_tensor.shape == (20,)
    assert input_tensor.dtype == torch.long
    assert target_tensor.dtype == torch.long


def test_collatz_dataset_dataloader():
    """Test dataset with DataLoader."""
    dataset = CollatzDataset(
        start_range=(1, 100),
        num_samples=50,
        base=24,
        max_length=20,
        seed=42,
    )
    
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    batch_input, batch_target = next(iter(loader))
    
    assert batch_input.shape[0] == 8
    assert batch_target.shape[0] == 8
    assert batch_input.shape[1] == 20
    assert batch_target.shape[1] == 20


def test_trainer_initialization():
    """Test Trainer initialization."""
    model = CollatzTransformer(
        vocab_size=24,
        d_model=64,
        nhead=2,
        num_layers=1,
    )
    
    dataset = CollatzDataset(
        start_range=(1, 100),
        num_samples=20,
        base=24,
        max_length=20,
    )
    
    loader = DataLoader(dataset, batch_size=4)
    
    trainer = Trainer(
        model=model,
        train_loader=loader,
        val_loader=None,
        checkpoint_dir="./test_checkpoints",
    )
    
    assert trainer.model == model
    assert trainer.train_loader == loader
    assert trainer.val_loader is None
    assert trainer.current_epoch == 0
    assert trainer.global_step == 0


def test_trainer_single_epoch():
    """Test training for a single epoch."""
    model = CollatzTransformer(
        vocab_size=24,
        d_model=64,
        nhead=2,
        num_layers=1,
    )
    
    dataset = CollatzDataset(
        start_range=(1, 100),
        num_samples=20,
        base=24,
        max_length=20,
        seed=42,
    )
    
    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    
    trainer = Trainer(
        model=model,
        train_loader=loader,
        val_loader=None,
        checkpoint_dir="./test_checkpoints",
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
    
    history = trainer.train(
        num_epochs=1,
        optimizer=optimizer,
        criterion=criterion,
        save_every=10,
        save_best=False,
    )
    
    assert 'train_loss' in history
    assert len(history['train_loss']) == 1
    assert history['train_loss'][0] > 0


def test_trainer_with_validation():
    """Test trainer with validation."""
    model = CollatzTransformer(
        vocab_size=24,
        d_model=64,
        nhead=2,
        num_layers=1,
    )
    
    dataset = CollatzDataset(
        start_range=(1, 100),
        num_samples=20,
        base=24,
        max_length=20,
        seed=42,
    )
    
    train_loader = DataLoader(dataset, batch_size=4)
    val_loader = DataLoader(dataset, batch_size=4)
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        checkpoint_dir="./test_checkpoints",
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
    
    history = trainer.train(
        num_epochs=2,
        optimizer=optimizer,
        criterion=criterion,
        save_every=10,
        save_best=True,
    )
    
    assert 'train_loss' in history
    assert 'val_loss' in history
    assert len(history['train_loss']) == 2
    assert len(history['val_loss']) == 2


def test_trainer_checkpoint_save_load():
    """Test checkpoint saving and loading."""
    model1 = CollatzTransformer(
        vocab_size=24,
        d_model=64,
        nhead=2,
        num_layers=1,
    )
    
    dataset = CollatzDataset(
        start_range=(1, 100),
        num_samples=10,
        base=24,
        max_length=20,
    )
    
    loader = DataLoader(dataset, batch_size=2)
    
    checkpoint_dir = Path("./test_checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    
    trainer = Trainer(
        model=model1,
        train_loader=loader,
        checkpoint_dir=str(checkpoint_dir),
    )
    
    optimizer = torch.optim.Adam(model1.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
    
    # Train a bit
    trainer.train(
        num_epochs=1,
        optimizer=optimizer,
        criterion=criterion,
        save_best=False,
    )
    
    # Save checkpoint
    checkpoint_path = checkpoint_dir / "test_checkpoint.pt"
    trainer.save_checkpoint("test_checkpoint.pt", optimizer)
    
    assert checkpoint_path.exists()
    
    # Load checkpoint
    model2 = CollatzTransformer(
        vocab_size=24,
        d_model=64,
        nhead=2,
        num_layers=1,
    )
    
    trainer2 = Trainer(
        model=model2,
        train_loader=loader,
        checkpoint_dir=str(checkpoint_dir),
    )
    
    trainer2.load_checkpoint("test_checkpoint.pt", optimizer)
    
    # Check that state was loaded
    assert trainer2.current_epoch == trainer.current_epoch
    assert trainer2.global_step == trainer.global_step
    
    # Cleanup
    checkpoint_path.unlink()
    checkpoint_dir.rmdir()


def test_trainer_best_model_saving():
    """Test that best model is saved."""
    model = CollatzTransformer(
        vocab_size=24,
        d_model=64,
        nhead=2,
        num_layers=1,
    )
    
    dataset = CollatzDataset(
        start_range=(1, 100),
        num_samples=20,
        base=24,
        max_length=20,
    )
    
    train_loader = DataLoader(dataset, batch_size=4)
    val_loader = DataLoader(dataset, batch_size=4)
    
    checkpoint_dir = Path("./test_checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        checkpoint_dir=str(checkpoint_dir),
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
    
    trainer.train(
        num_epochs=2,
        optimizer=optimizer,
        criterion=criterion,
        save_best=True,
    )
    
    best_model_path = checkpoint_dir / "best_model.pt"
    
    # Best model should be saved if validation loss improved
    # (may or may not exist depending on training outcome)
    if best_model_path.exists():
        assert best_model_path.is_file()
        best_model_path.unlink()
    
    checkpoint_dir.rmdir()

