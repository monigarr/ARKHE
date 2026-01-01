"""
Integration tests for complete training pipeline.

Author: MoniGarr
Email: monigarr@MoniGarr.com
Website: MoniGarr.com
"""

import pytest
import torch
from pathlib import Path
from torch.utils.data import DataLoader, random_split

from math_research.ml import (
    CollatzTransformer,
    CollatzDataset,
    Trainer,
    MultiBaseEncoder,
    compute_accuracy,
    compute_exact_match,
)


@pytest.fixture
def temp_checkpoint_dir(tmp_path):
    """Create temporary checkpoint directory."""
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    return str(checkpoint_dir)


def test_end_to_end_training_pipeline(temp_checkpoint_dir):
    """Test complete training pipeline from dataset to evaluation."""
    # Configuration
    config = {
        "data_range": (1, 1000),
        "num_samples": 100,
        "base": 24,
        "max_length": 20,
        "batch_size": 8,
        "vocab_size": 24,
        "d_model": 64,
        "nhead": 2,
        "num_layers": 1,
        "num_epochs": 2,
        "learning_rate": 0.001,
        "validation_split": 0.2,
        "seed": 42,
    }
    
    # 1. Create dataset
    dataset = CollatzDataset(
        start_range=config["data_range"],
        num_samples=config["num_samples"],
        base=config["base"],
        max_length=config["max_length"],
        seed=config["seed"],
    )
    
    assert len(dataset) == config["num_samples"]
    
    # 2. Split dataset
    val_size = int(len(dataset) * config["validation_split"])
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config["seed"])
    )
    
    # 3. Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
    )
    
    # 4. Create model
    model = CollatzTransformer(
        vocab_size=config["vocab_size"],
        d_model=config["d_model"],
        nhead=config["nhead"],
        num_layers=config["num_layers"],
    )
    
    device = torch.device("cpu")
    model.to(device)
    
    # 5. Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        checkpoint_dir=temp_checkpoint_dir,
    )
    
    # 6. Setup training
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
    
    # 7. Train
    history = trainer.train(
        num_epochs=config["num_epochs"],
        optimizer=optimizer,
        criterion=criterion,
        save_every=10,
        save_best=True,
    )
    
    # Verify training history
    assert 'train_loss' in history
    assert 'val_loss' in history
    assert len(history['train_loss']) == config["num_epochs"]
    assert len(history['val_loss']) == config["num_epochs"]
    
    # Loss should be positive
    assert all(loss > 0 for loss in history['train_loss'])
    assert all(loss > 0 for loss in history['val_loss'])
    
    # 8. Evaluate
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = model(inputs)
            predictions = torch.argmax(outputs, dim=-1)
            
            all_predictions.append(predictions.cpu())
            all_targets.append(targets.cpu())
    
    predictions = torch.cat(all_predictions, dim=0)
    targets = torch.cat(all_targets, dim=0)
    
    # 9. Compute metrics
    accuracy = compute_accuracy(predictions, targets)
    exact_match = compute_exact_match(predictions, targets)
    
    assert 0 <= accuracy <= 1
    assert 0 <= exact_match <= 1


def test_model_checkpoint_save_and_load(temp_checkpoint_dir):
    """Test saving and loading model checkpoints."""
    # Create and train model
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
    
    loader = DataLoader(dataset, batch_size=4)
    
    trainer = Trainer(
        model=model,
        train_loader=loader,
        checkpoint_dir=temp_checkpoint_dir,
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
    
    trainer.train(
        num_epochs=1,
        optimizer=optimizer,
        criterion=criterion,
    )
    
    # Save checkpoint
    checkpoint_path = Path(temp_checkpoint_dir) / "test_model.pt"
    trainer.save_checkpoint("test_model.pt", optimizer)
    
    assert checkpoint_path.exists()
    
    # Load checkpoint and verify
    loaded_checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    assert 'model_state_dict' in loaded_checkpoint
    assert 'epoch' in loaded_checkpoint
    assert 'global_step' in loaded_checkpoint
    
    # Create new model and load weights
    new_model = CollatzTransformer(
        vocab_size=24,
        d_model=64,
        nhead=2,
        num_layers=1,
    )
    
    new_model.load_state_dict(loaded_checkpoint['model_state_dict'])
    
    # Verify models produce same output
    test_input = torch.randint(0, 24, (1, 10))
    
    model.eval()
    new_model.eval()
    
    with torch.no_grad():
        output1 = model(test_input)
        output2 = new_model(test_input)
    
    assert torch.allclose(output1, output2, atol=1e-5)


def test_encoding_and_decoding_pipeline():
    """Test encoding/decoding pipeline with actual Collatz values."""
    encoder = MultiBaseEncoder(base=24, max_length=20)
    collatz = CollatzSequence(start=1)
    
    test_values = [27, 127, 255, 511, 1023]
    
    for value in test_values:
        # Get actual Collatz long step
        long_step = collatz.compute_long_step(value)
        actual_result = long_step['result']
        
        # Encode input
        encoded_input = encoder.encode(value)
        assert len(encoded_input) <= 20
        
        # Encode output
        encoded_output = encoder.encode(actual_result)
        
        # Decode to verify
        decoded_input = encoder.decode(encoded_input)
        decoded_output = encoder.decode(encoded_output)
        
        assert decoded_input == value
        assert decoded_output == actual_result


def test_minimal_training_run():
    """Test minimal training run to verify pipeline works."""
    # Very small configuration for quick test
    dataset = CollatzDataset(
        start_range=(1, 100),
        num_samples=10,
        base=24,
        max_length=10,
        seed=42,
    )
    
    loader = DataLoader(dataset, batch_size=2)
    
    model = CollatzTransformer(
        vocab_size=24,
        d_model=32,
        nhead=2,
        num_layers=1,
    )
    
    trainer = Trainer(
        model=model,
        train_loader=loader,
        checkpoint_dir="./test_checkpoints_minimal",
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
    
    # Train for just 1 epoch
    history = trainer.train(
        num_epochs=1,
        optimizer=optimizer,
        criterion=criterion,
        save_best=False,
    )
    
    assert len(history['train_loss']) == 1
    assert history['train_loss'][0] > 0
    
    # Cleanup
    import shutil
    shutil.rmtree("./test_checkpoints_minimal", ignore_errors=True)

