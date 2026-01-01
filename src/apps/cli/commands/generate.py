"""
Generate command - Generate Collatz sequences.

Author: MoniGarr
Email: monigarr@MoniGarr.com
Website: MoniGarr.com
"""

import json
import sys
from pathlib import Path
from typing import Optional

from math_research.sequences import CollatzSequence
from math_research.utils import get_logger

logger = get_logger(__name__)


def add_arguments(parser):
    """Add arguments for the generate command."""
    parser.add_argument(
        "--start",
        type=int,
        required=True,
        help="Starting value for the Collatz sequence",
    )
    
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=1000000,
        help="Maximum number of iterations (default: 1000000)",
    )
    
    parser.add_argument(
        "--use-long-step",
        action="store_true",
        default=True,
        help="Use long step optimization (default: True)",
    )
    
    parser.add_argument(
        "--no-long-step",
        dest="use_long_step",
        action="store_false",
        help="Disable long step optimization",
    )
    
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output file path (JSON or TXT). If not specified, prints to stdout",
    )
    
    parser.add_argument(
        "--format",
        choices=["json", "txt", "csv"],
        default="txt",
        help="Output format (default: txt)",
    )
    
    parser.add_argument(
        "--show-stats",
        action="store_true",
        help="Show sequence statistics",
    )


def execute(args) -> int:
    """Execute the generate command."""
    try:
        # Validate start value
        if args.start <= 0:
            print(f"Error: Start value must be positive, got {args.start}", file=sys.stderr)
            return 1
        
        # Create sequence generator
        seq = CollatzSequence(
            start=args.start,
            max_iterations=args.max_iterations,
            use_long_step=args.use_long_step,
        )
        
        # Generate sequence
        logger.info(f"Generating Collatz sequence starting from {args.start}...")
        sequence = seq.generate()
        
        # Prepare output
        output_data = {
            "start": args.start,
            "sequence": sequence,
            "length": len(sequence),
            "max_value": seq.get_max_value(),
            "peak_value": seq.get_peak_value()[0],
            "peak_position": seq.get_peak_value()[1],
        }
        
        # Show stats if requested
        if args.show_stats:
            print(f"\nSequence Statistics:")
            print(f"  Start value: {args.start}")
            print(f"  Sequence length: {len(sequence)} steps")
            print(f"  Maximum value: {seq.get_max_value()}")
            print(f"  Peak value: {seq.get_peak_value()[0]} (at position {seq.get_peak_value()[1]})")
            print()
        
        # Format and output
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if args.format == "json":
                with open(output_path, "w") as f:
                    json.dump(output_data, f, indent=2)
                print(f"Sequence saved to {output_path}")
            elif args.format == "csv":
                import csv
                with open(output_path, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["position", "value"])
                    for i, value in enumerate(sequence):
                        writer.writerow([i, value])
                print(f"Sequence saved to {output_path}")
            else:  # txt
                with open(output_path, "w") as f:
                    f.write(f"Collatz Sequence starting from {args.start}\n")
                    f.write(f"Length: {len(sequence)} steps\n")
                    f.write(f"Max value: {seq.get_max_value()}\n\n")
                    f.write("Sequence:\n")
                    f.write(" ".join(map(str, sequence)))
                print(f"Sequence saved to {output_path}")
        else:
            # Print to stdout
            if args.format == "json":
                print(json.dumps(output_data, indent=2))
            elif args.format == "csv":
                import csv
                writer = csv.writer(sys.stdout)
                writer.writerow(["position", "value"])
                for i, value in enumerate(sequence):
                    writer.writerow([i, value])
            else:  # txt
                print(f"Collatz Sequence starting from {args.start}")
                print(f"Length: {len(sequence)} steps")
                print(f"Sequence: {' '.join(map(str, sequence))}")
        
        return 0
        
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        print(f"Error: {e}", file=sys.stderr)
        return 1

