"""
Train command - Train transformer models.

Author: MoniGarr
Email: monigarr@MoniGarr.com
Website: MoniGarr.com
"""

import sys
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader, random_split

from math_research.ml import CollatzTransformer, CollatzDataset, Trainer
from math_research.utils import get_logger, load_config

logger = get_logger(__name__)


def add_arguments(parser):
    """Add arguments for the train command."""
    parser.add_argument(
        "--config",
        type=str,
        help="Path to training configuration file (YAML)",
    )
    
    parser.add_argument(
        "--data-range",
        nargs=2,
        type=int,
        metavar=("MIN", "MAX"),
        default=[1, 10000],
        help="Range of input values for training data (default: 1 10000)",
    )
    
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10000,
        help="Number of training samples (default: 10000)",
    )
    
    parser.add_argument(
        "--base",
        type=int,
        default=24,
        help="Base for encoding (default: 24)",
    )
    
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=24,
        help="Vocabulary size / base for encoding (default: 24)",
    )
    
    parser.add_argument(
        "--d-model",
        type=int,
        default=512,
        help="Model dimension (default: 512)",
    )
    
    parser.add_argument(
        "--nhead",
        type=int,
        default=8,
        help="Number of attention heads (default: 8)",
    )
    
    parser.add_argument(
        "--num-layers",
        type=int,
        default=6,
        help="Number of transformer layers (default: 6)",
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size (default: 32)",
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs (default: 10)",
    )
    
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.0001,
        help="Learning rate (default: 0.0001)",
    )
    
    parser.add_argument(
        "--validation-split",
        type=float,
        default=0.2,
        help="Validation set split ratio (default: 0.2)",
    )
    
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="./checkpoints",
        help="Directory for saving checkpoints (default: ./checkpoints)",
    )
    
    parser.add_argument(
        "--save-every",
        type=int,
        default=5,
        help="Save checkpoint every N epochs (default: 5)",
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducibility",
    )


def execute(args) -> int:
    """Execute the train command."""
    try:
        # Load config if provided
        config = {}
        if args.config:
            config_path = Path(args.config)
            if not config_path.exists():
                print(f"Error: Config file not found: {args.config}", file=sys.stderr)
                return 1
            config = load_config(str(config_path))
            logger.info(f"Loaded configuration from {args.config}")
        
        # Override config with command line arguments
        data_range = tuple(args.data_range)
        num_samples = config.get("data", {}).get("num_samples", args.num_samples)
        base = config.get("encoding", {}).get("base", args.base)
        vocab_size = config.get("model", {}).get("vocab_size", args.vocab_size)
        d_model = config.get("model", {}).get("d_model", args.d_model)
        nhead = config.get("model", {}).get("nhead", args.nhead)
        num_layers = config.get("model", {}).get("num_layers", args.num_layers)
        batch_size = config.get("training", {}).get("batch_size", args.batch_size)
        epochs = config.get("training", {}).get("epochs", args.epochs)
        learning_rate = config.get("training", {}).get("learning_rate", args.learning_rate)
        validation_split = config.get("training", {}).get("validation_split", args.validation_split)
        checkpoint_dir = config.get("training", {}).get("checkpoint_dir", args.checkpoint_dir)
        save_every = config.get("training", {}).get("save_every", args.save_every)
        seed = args.seed or config.get("training", {}).get("seed")
        
        # Set random seed
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        
        logger.info("Creating dataset...")
        # Create dataset
        dataset = CollatzDataset(
            start_range=data_range,
            num_samples=num_samples,
            base=base,
            seed=seed,
        )
        
        # Split into train and validation
        val_size = int(len(dataset) * validation_split)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
        )
        
        logger.info(f"Dataset created: {train_size} train, {val_size} validation samples")
        
        # Create model
        logger.info("Creating model...")
        model = CollatzTransformer(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
        )
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Model created: {total_params:,} total parameters, {trainable_params:,} trainable")
        
        # Create trainer
        logger.info("Starting training...")
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            checkpoint_dir=checkpoint_dir,
        )
        
        # Create optimizer and loss
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
        
        # Train
        history = trainer.train(
            num_epochs=epochs,
            optimizer=optimizer,
            criterion=criterion,
            save_every=save_every,
            save_best=True,
        )
        
        logger.info("Training completed!")
        logger.info(f"Best validation loss: {trainer.best_val_loss:.4f}")
        logger.info(f"Checkpoints saved to: {checkpoint_dir}")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Training error: {e}", exc_info=True)
        print(f"Error: {e}", file=sys.stderr)
        return 1

