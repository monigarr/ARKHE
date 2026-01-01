"""
Module: trainer
Package: math_research.ml.training

Description:
    Training pipeline for transformer models. Provides comprehensive training
    loop with checkpointing, logging, and evaluation capabilities.

Author: MoniGarr
Author Email: monigarr@MoniGarr.com
Author Website: MoniGarr.com

Author Research Interests:
    - AI/ML Research and Development
    - Extended Reality (XR) Applications
    - 3D Graphics and Visualization
    - Robotics and Autonomous Systems
    - Computer Vision
    - Navigation Systems
    - Natural Language Processing (NLP)
    - Low Resource Languages (spoken in English communities)

Usage:
    >>> from math_research.ml.training import Trainer
    >>> from math_research.ml.models import CollatzTransformer
    >>> 
    >>> model = CollatzTransformer(vocab_size=24)
    >>> trainer = Trainer(model, train_loader, val_loader, config)
    >>> trainer.train(num_epochs=100)

Dependencies:
    - torch>=2.0.0
    - tqdm>=4.65.0
    - pathlib (standard library)
    - typing (standard library)

Version: 0.1.0
Last Modified: 2025-01-09
License: MIT

Notes:
    Trainer supports checkpointing, early stopping, and experiment tracking.
    Can be extended to support wandb, mlflow, or tensorboard integration.
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from math_research.utils.logging import get_logger

logger = get_logger(__name__)


class Trainer:
    """
    Training pipeline for transformer models.
    
    Provides comprehensive training loop with checkpointing, validation,
    and experiment tracking capabilities.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        device: Optional[torch.device] = None,
        checkpoint_dir: str = "./checkpoints",
        log_interval: int = 100,
    ):
        """
        Initialize trainer.
        
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            device: Device to use (auto-detected if None)
            checkpoint_dir: Directory for saving checkpoints
            log_interval: Logging interval (number of batches)
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_interval = log_interval
        
        # Device setup
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        self.model.to(self.device)
        logger.info(f"Using device: {self.device}")
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
    
    def train(
        self,
        num_epochs: int,
        optimizer: Optional[torch.optim.Optimizer] = None,
        criterion: Optional[nn.Module] = None,
        lr_scheduler: Optional[Any] = None,
        save_every: int = 10,
        save_best: bool = True,
    ) -> Dict[str, list]:
        """
        Train the model.
        
        Args:
            num_epochs: Number of training epochs
            optimizer: Optimizer (default: Adam)
            criterion: Loss function (default: CrossEntropyLoss)
            lr_scheduler: Learning rate scheduler (optional)
            save_every: Save checkpoint every N epochs
            save_best: Save checkpoint when validation loss improves
            
        Returns:
            Dictionary with training history
        """
        if optimizer is None:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)
        
        if criterion is None:
            criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
        
        history = {
            'train_loss': [],
            'val_loss': [],
        }
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Training phase
            train_loss = self._train_epoch(optimizer, criterion)
            history['train_loss'].append(train_loss)
            
            logger.info(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}")
            
            # Validation phase
            if self.val_loader:
                val_loss = self._validate_epoch(criterion)
                history['val_loss'].append(val_loss)
                logger.info(f"Epoch {epoch+1}/{num_epochs} - Val Loss: {val_loss:.4f}")
                
                # Save best model
                if save_best and val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint("best_model.pt", optimizer, lr_scheduler)
                    logger.info("Saved best model")
            else:
                val_loss = None
            
            # Learning rate scheduling
            if lr_scheduler:
                if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    lr_scheduler.step(val_loss if val_loss else train_loss)
                else:
                    lr_scheduler.step()
            
            # Periodic checkpointing
            if (epoch + 1) % save_every == 0:
                checkpoint_name = f"checkpoint_epoch_{epoch+1}.pt"
                self.save_checkpoint(checkpoint_name, optimizer, lr_scheduler)
        
        return history
    
    def _train_epoch(self, optimizer: torch.optim.Optimizer, criterion: nn.Module) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Training Epoch {self.current_epoch+1}")
        
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = self.model(inputs)
            
            # Reshape for loss computation
            outputs = outputs.view(-1, outputs.size(-1))
            targets = targets.view(-1)
            
            loss = criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})
            
            # Logging
            if batch_idx % self.log_interval == 0:
                logger.debug(
                    f"Step {self.global_step} - Loss: {loss.item():.4f}"
                )
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def _validate_epoch(self, criterion: nn.Module) -> float:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for inputs, targets in tqdm(self.val_loader, desc="Validation"):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(inputs)
                outputs = outputs.view(-1, outputs.size(-1))
                targets = targets.view(-1)
                
                loss = criterion(outputs, targets)
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def save_checkpoint(
        self,
        filename: str,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr_scheduler: Optional[Any] = None,
    ) -> None:
        """
        Save model checkpoint.
        
        Args:
            filename: Checkpoint filename
            optimizer: Optimizer state (optional)
            lr_scheduler: LR scheduler state (optional)
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'best_val_loss': self.best_val_loss,
        }
        
        if optimizer:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        if lr_scheduler:
            checkpoint['lr_scheduler_state_dict'] = lr_scheduler.state_dict()
        
        filepath = self.checkpoint_dir / filename
        torch.save(checkpoint, filepath)
        logger.info(f"Saved checkpoint to {filepath}")
    
    def load_checkpoint(
        self,
        filename: str,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr_scheduler: Optional[Any] = None,
    ) -> None:
        """
        Load model checkpoint.
        
        Args:
            filename: Checkpoint filename
            optimizer: Optimizer to load state into (optional)
            lr_scheduler: LR scheduler to load state into (optional)
        """
        filepath = self.checkpoint_dir / filename
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.current_epoch = checkpoint.get('epoch', 0)
        self.global_step = checkpoint.get('global_step', 0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if lr_scheduler and 'lr_scheduler_state_dict' in checkpoint:
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        
        logger.info(f"Loaded checkpoint from {filepath}")

