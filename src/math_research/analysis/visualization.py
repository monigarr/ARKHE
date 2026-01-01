"""
Module: visualization
Package: math_research.analysis

Description:
    Visualization tools for mathematical sequences. Provides functions to create
    various plots including sequence plots, histograms, phase diagrams, and
    comparative visualizations.

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
    >>> from math_research.analysis import SequenceVisualizer
    >>> import matplotlib.pyplot as plt
    >>> 
    >>> visualizer = SequenceVisualizer()
    >>> fig, ax = visualizer.plot_sequence([27, 82, 41, ...])
    >>> plt.show()

Dependencies:
    - matplotlib>=3.7.0
    - numpy>=1.24.0
    - seaborn>=0.12.0 (optional)
    - typing (standard library)

Version: 0.1.0
Last Modified: 2025-01-09
License: MIT

Notes:
    All visualization functions return matplotlib figure and axes objects
    for further customization.
"""

from typing import List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


class SequenceVisualizer:
    """
    Visualization tools for mathematical sequences.
    
    Provides various plotting functions for sequence analysis including
    line plots, histograms, phase diagrams, and comparative plots.
    """
    
    def __init__(self, style: str = "default", figsize: Tuple[int, int] = (10, 6)):
        """
        Initialize visualizer.
        
        Args:
            style: Matplotlib style ('default', 'seaborn', 'ggplot', etc.)
            figsize: Default figure size (width, height)
        """
        self.style = style
        self.figsize = figsize
        
        if style == "seaborn" and HAS_SEABORN:
            sns.set_style("whitegrid")
        elif style != "default":
            plt.style.use(style)
    
    def plot_sequence(
        self,
        sequence: List[int],
        title: Optional[str] = None,
        xlabel: str = "Step",
        ylabel: str = "Value",
        show_peaks: bool = False,
        ax: Optional[plt.Axes] = None,
        **kwargs
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot sequence as a line plot.
        
        Args:
            sequence: Sequence values to plot
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            show_peaks: If True, highlight peak values
            ax: Optional matplotlib axes to plot on (if None, creates new figure)
            **kwargs: Additional arguments passed to plt.plot()
            
        Returns:
            Tuple of (figure, axes) objects
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        else:
            fig = ax.figure
        
        x = np.arange(len(sequence))
        ax.plot(x, sequence, **kwargs)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        
        if title:
            ax.set_title(title)
        else:
            ax.set_title("Sequence Plot")
        
        ax.grid(True, alpha=0.3)
        
        if show_peaks:
            # Highlight peaks
            from math_research.analysis.statistics import SequenceStatistics
            stats = SequenceStatistics(sequence)
            peaks = stats.find_peaks()
            for peak_value, peak_idx in peaks:
                ax.plot(peak_idx, peak_value, 'ro', markersize=8)
        
        if ax is None:  # Only call tight_layout if we created the figure
            plt.tight_layout()
        return fig, ax
    
    def plot_histogram(
        self,
        sequence: List[int],
        bins: int = 50,
        title: Optional[str] = None,
        xlabel: str = "Value",
        ylabel: str = "Frequency",
        **kwargs
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot histogram of sequence values.
        
        Args:
            sequence: Sequence values
            bins: Number of histogram bins
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            **kwargs: Additional arguments passed to plt.hist()
            
        Returns:
            Tuple of (figure, axes) objects
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        ax.hist(sequence, bins=bins, **kwargs)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        
        if title:
            ax.set_title(title)
        else:
            ax.set_title("Value Distribution")
        
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        return fig, ax
    
    def plot_log_sequence(
        self,
        sequence: List[int],
        title: Optional[str] = None,
        ax: Optional[plt.Axes] = None,
        **kwargs
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot sequence with logarithmic y-axis.
        
        Args:
            sequence: Sequence values
            title: Plot title
            ax: Optional matplotlib axes to plot on (if None, creates new figure)
            **kwargs: Additional arguments passed to plot_sequence()
            
        Returns:
            Tuple of (figure, axes) objects
        """
        fig, ax = self.plot_sequence(sequence, title=title, ax=ax, **kwargs)
        ax.set_yscale('log')
        return fig, ax
    
    def plot_comparison(
        self,
        sequences: List[Tuple[List[int], str]],
        title: Optional[str] = None,
        **kwargs
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot multiple sequences on the same axes for comparison.
        
        Args:
            sequences: List of tuples (sequence, label)
            title: Plot title
            **kwargs: Additional arguments passed to plt.plot()
            
        Returns:
            Tuple of (figure, axes) objects
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        for sequence, label in sequences:
            x = np.arange(len(sequence))
            ax.plot(x, sequence, label=label, **kwargs)
        
        ax.set_xlabel("Step")
        ax.set_ylabel("Value")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if title:
            ax.set_title(title)
        else:
            ax.set_title("Sequence Comparison")
        
        plt.tight_layout()
        return fig, ax
    
    def plot_phase_diagram(
        self,
        sequence: List[int],
        title: Optional[str] = None,
        **kwargs
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot phase diagram (value vs next value).
        
        Args:
            sequence: Sequence values
            title: Plot title
            **kwargs: Additional arguments passed to plt.scatter()
            
        Returns:
            Tuple of (figure, axes) objects
        """
        if len(sequence) < 2:
            raise ValueError("Sequence must have at least 2 elements")
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        x = sequence[:-1]
        y = sequence[1:]
        
        ax.scatter(x, y, **kwargs)
        ax.plot([min(x + y), max(x + y)], [min(x + y), max(x + y)], 
                'r--', alpha=0.5, label='y=x')
        ax.set_xlabel("Current Value")
        ax.set_ylabel("Next Value")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if title:
            ax.set_title(title)
        else:
            ax.set_title("Phase Diagram")
        
        plt.tight_layout()
        return fig, ax

