"""
Main CLI entry point for ARKHĒ Framework.

Author: MoniGarr
Email: monigarr@MoniGarr.com
Website: MoniGarr.com

Usage:
    python -m src.apps.cli.main generate --start 27
    python -m src.apps.cli.main train --config configs/training/collatz_transformer.yaml
    python -m src.apps.cli.main evaluate --checkpoint checkpoints/best_model.pt
    python -m src.apps.cli.main analyze --start 1 --end 100
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.apps.cli import __version__
from src.apps.cli.commands import generate, train, evaluate, analyze


def create_parser() -> argparse.ArgumentParser:
    """Create the main argument parser."""
    parser = argparse.ArgumentParser(
        prog="arkhe",
        description="ARKHE Framework - Mathematical Sequence Research CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate a Collatz sequence
  arkhe generate --start 27 --output sequence.txt

  # Train a transformer model
  arkhe train --config configs/training/collatz_transformer.yaml

  # Evaluate a trained model
  arkhe evaluate --checkpoint checkpoints/best_model.pt --test-size 1000

  # Analyze sequence patterns
  arkhe analyze --start 1 --end 1000 --output analysis.json
        """,
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version=f"ARKHE {__version__}",
    )
    
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output",
    )
    
    # Subcommands
    subparsers = parser.add_subparsers(
        dest="command",
        help="Available commands",
        metavar="COMMAND",
    )
    
    # Generate command
    generate_parser = subparsers.add_parser(
        "generate",
        help="Generate Collatz sequences",
        description="Generate and save Collatz sequences starting from specified values.",
    )
    generate.add_arguments(generate_parser)
    
    # Train command
    train_parser = subparsers.add_parser(
        "train",
        help="Train transformer models",
        description="Train transformer models on Collatz sequence data.",
    )
    train.add_arguments(train_parser)
    
    # Evaluate command
    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate trained models",
        description="Evaluate trained transformer models on test data.",
    )
    evaluate.add_arguments(evaluate_parser)
    
    # Analyze command
    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Analyze sequence patterns",
        description="Analyze Collatz sequences for patterns and statistics.",
    )
    analyze.add_arguments(analyze_parser)
    
    return parser


def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return 1
    
    # Setup logging if verbose
    if args.verbose:
        from math_research.utils import setup_logging
        setup_logging(level="DEBUG")
    
    # Execute command
    try:
        if args.command == "generate":
            return generate.execute(args)
        elif args.command == "train":
            return train.execute(args)
        elif args.command == "evaluate":
            return evaluate.execute(args)
        elif args.command == "analyze":
            return analyze.execute(args)
        else:
            parser.print_help()
            return 1
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
        return 130
    except Exception as e:
        if args.verbose:
            import traceback
            traceback.print_exc()
        else:
            print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

