"""
Main Streamlit application for ARKHE Framework.

Author: MoniGarr
Email: monigarr@MoniGarr.com
Website: MoniGarr.com

Usage:
    # From project root:
    streamlit run src/apps/streamlit_demo/app.py
    
    # OR use the launcher script:
    python run_streamlit.py
"""

import sys
from pathlib import Path

# Add src directory to path for imports
# Get the absolute path to the src directory (3 levels up from this file)
# app.py is in: src/apps/streamlit_demo/app.py
# So we need: src/ directory
app_file = Path(__file__).resolve()
src_dir = app_file.parent.parent.parent  # Go from app.py -> streamlit_demo -> apps -> src

# Ensure src directory is in path
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# Verify math_research exists and handle import errors
math_research_path = src_dir / "math_research"
if not math_research_path.exists():
    # Fallback: try to find it from current working directory
    import os
    cwd = Path(os.getcwd()).resolve()
    # Try different possible locations
    for possible_src in [cwd / "src", cwd.parent / "src", Path(__file__).resolve().parent / "src"]:
        if (possible_src / "math_research").exists():
            if str(possible_src) not in sys.path:
                sys.path.insert(0, str(possible_src))
            src_dir = possible_src
            break

# Import streamlit first (needed for error display)
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Now verify math_research imports work
try:
    from math_research.sequences import CollatzSequence
    from math_research.analysis import SequenceStatistics, SequenceVisualizer
    from math_research.ml import MultiBaseEncoder
    from math_research.utils.health import HealthChecker, get_health_status
except ImportError as e:
    st.error(f"""
    **Import Error**: Could not import math_research module.
    
    **Error**: {e}
    
    **Debug Info**:
    - App file: {Path(__file__).resolve()}
    - Src directory: {src_dir}
    - Math research path: {math_research_path}
    - Path exists: {math_research_path.exists()}
    - Sys.path entries with 'src': {[p for p in sys.path if 'src' in str(p)]}
    
    **Solution**: Make sure you're running from the project root:
    ```bash
    cd I:/CursorProjects/Research/Mathematics/ARKHE
    streamlit run src/apps/streamlit_demo/app.py
    ```
    
    Or use the launcher script:
    ```bash
    python run_streamlit.py
    ```
    """)
    st.stop()

# Page configuration
st.set_page_config(
    page_title="ARKHE Framework - Collatz Research",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stButton>button {
        width: 100%;
    }
    </style>
""", unsafe_allow_html=True)


def main():
    """Main application entry point."""
    
    # Initialize session state for page navigation
    if "page" not in st.session_state:
        st.session_state.page = "🏠 Home"
    
    # Sidebar navigation
    st.sidebar.title("🔬 ARKHE Framework")
    st.sidebar.markdown("---")
    
    # Page options
    page_options = ["🏠 Home", "📊 Sequence Explorer", "🤖 Model Inference", "📈 Statistical Analysis", "🏥 Health Check", "⚙️ About"]
    
    # Get index of current page from session state
    try:
        current_index = page_options.index(st.session_state.page)
    except ValueError:
        current_index = 0
        st.session_state.page = "🏠 Home"
    
    # Sidebar selectbox - update session state when changed
    page = st.sidebar.selectbox(
        "Navigation",
        page_options,
        index=current_index,
        key="page_selectbox"
    )
    
    # Update session state if selectbox changed
    if page != st.session_state.page:
        st.session_state.page = page
    
    # Display the selected page
    if st.session_state.page == "🏠 Home":
        show_home_page()
    elif st.session_state.page == "📊 Sequence Explorer":
        show_sequence_explorer()
    elif st.session_state.page == "🤖 Model Inference":
        show_model_inference()
    elif st.session_state.page == "📈 Statistical Analysis":
        show_statistical_analysis()
    elif st.session_state.page == "🏥 Health Check":
        show_health_check()
    elif st.session_state.page == "⚙️ About":
        show_about_page()


def show_home_page():
    """Display home page."""
    st.markdown('<div class="main-header">ARKHE Framework</div>', unsafe_allow_html=True)
    st.markdown("### Mathematical Sequence Research Platform")
    
    st.markdown("""
    Welcome to the **ARKHE Framework** - an enterprise-grade Python framework for 
    mathematical sequence research and machine learning experimentation.
    
    This interactive demo provides tools for:
    - 🔢 **Sequence Generation**: Generate and visualize Collatz sequences
    - 📊 **Statistical Analysis**: Analyze patterns and properties
    - 🤖 **Machine Learning**: Train and evaluate transformer models
    - 📈 **Data Visualization**: Interactive charts and graphs
    """)
    
    st.markdown("---")
    
    # Quick start section
    st.markdown("### 🚀 Quick Start")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **Sequence Explorer**
        
        Generate Collatz sequences starting from any number and visualize them interactively.
        """)
        if st.button("Explore Sequences →", key="quick_explore", use_container_width=True):
            st.session_state.page = "📊 Sequence Explorer"
            #st.rerun()
            show_sequence_explorer()
    
    with col2:
        st.markdown("""
        **Statistical Analysis**
        
        Analyze patterns across multiple sequences and discover mathematical properties.
        """)
        if st.button("Analyze Patterns →", key="quick_analyze", use_container_width=True):
            st.session_state.page = "📈 Statistical Analysis"
            #st.rerun()
            show_statistical_analysis()
    
    with col3:
        st.markdown("""
        **Model Inference**
        
        Use trained transformer models to predict Collatz sequence steps.
        """)
        if st.button("Try Models →", key="quick_inference", use_container_width=True):
            st.session_state.page = "🤖 Model Inference"
            #st.rerun()
            show_model_inference()
    
    st.markdown("---")
    
    # Example sequence
    st.markdown("### 📝 Example: Collatz Sequence from 27")
    
    example_seq = CollatzSequence(start=27)
    example_sequence = example_seq.generate()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Start Value", 27)
    with col2:
        st.metric("Sequence Length", len(example_sequence), "steps")
    with col3:
        st.metric("Max Value", example_seq.get_max_value())
    with col4:
        st.metric("Peak Position", example_seq.get_peak_value()[1])
    
    # Quick visualization
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(example_sequence, linewidth=1.5)
    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Value", fontsize=12)
    ax.set_title("Collatz Sequence Starting from 27", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)


def show_sequence_explorer():
    """Display sequence explorer page."""
    st.markdown('<div class="main-header">📊 Sequence Explorer</div>', unsafe_allow_html=True)
    
    st.markdown("Generate and visualize Collatz sequences interactively.")
    st.markdown("---")
    
    # Input controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        start_value = st.number_input(
            "Starting Value",
            min_value=1,
            max_value=1000000,
            value=27,
            step=1,
            help="Enter a positive integer to start the Collatz sequence"
        )
    
    with col2:
        max_iterations = st.number_input(
            "Max Iterations",
            min_value=100,
            max_value=10000000,
            value=1000000,
            step=10000,
            help="Maximum number of iterations before stopping"
        )
    
    with col3:
        use_long_step = st.checkbox(
            "Use Long Step Optimization",
            value=True,
            help="Use optimized long step computation for faster generation"
        )
    
    # Generate button
    if st.button("🚀 Generate Sequence", type="primary", use_container_width=True):
        with st.spinner("Generating sequence..."):
            try:
                seq = CollatzSequence(
                    start=int(start_value),
                    max_iterations=int(max_iterations),
                    use_long_step=use_long_step
                )
                sequence = seq.generate()
                
                # Store in session state
                st.session_state.current_sequence = sequence
                st.session_state.current_seq_obj = seq
                st.session_state.start_value = start_value
                
                st.success(f"✅ Sequence generated! Length: {len(sequence)} steps")
                
            except Exception as e:
                st.error(f"Error generating sequence: {e}")
    
    # Display results if sequence exists
    if "current_sequence" in st.session_state:
        sequence = st.session_state.current_sequence
        seq = st.session_state.current_seq_obj
        start_val = st.session_state.start_value
        
        st.markdown("---")
        st.markdown("### 📊 Sequence Statistics")
        
        # Statistics metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Length", f"{len(sequence)}", "steps")
        with col2:
            st.metric("Start Value", start_val)
        with col3:
            st.metric("Max Value", seq.get_max_value())
        with col4:
            peak_val, peak_idx = seq.get_peak_value()
            st.metric("Peak Value", peak_val, f"at step {peak_idx}")
        with col5:
            st.metric("Final Value", sequence[-1])
        
        # Visualizations
        st.markdown("### 📈 Visualizations")
        
        tab1, tab2, tab3, tab4 = st.tabs(["Line Plot", "Log Scale", "Histogram", "Sequence Data"])
        
        with tab1:
            visualizer = SequenceVisualizer()
            fig, ax = visualizer.plot_sequence(
                sequence,
                title=f"Collatz Sequence Starting from {start_val}",
                show_peaks=True
            )
            st.pyplot(fig)
        
        with tab2:
            fig, ax = visualizer.plot_log_sequence(
                sequence,
                title=f"Collatz Sequence (Log Scale) - Start: {start_val}"
            )
            st.pyplot(fig)
        
        with tab3:
            fig, ax = visualizer.plot_histogram(
                sequence,
                bins=50,
                title=f"Value Distribution - Start: {start_val}"
            )
            st.pyplot(fig)
        
        with tab4:
            # Display sequence data as DataFrame
            df = pd.DataFrame({
                "Step": range(len(sequence)),
                "Value": sequence
            })
            st.dataframe(df, use_container_width=True, height=400)
            
            # Download button
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download as CSV",
                data=csv,
                file_name=f"collatz_sequence_{start_val}.csv",
                mime="text/csv"
            )
        
        # Statistical summary
        st.markdown("### 📋 Detailed Statistics")
        stats = SequenceStatistics(sequence)
        summary = stats.summary()
        
        col1, col2 = st.columns(2)
        with col1:
            st.json({
                "Basic Stats": {
                    "Length": summary.get("length", "N/A"),
                    "Min": summary.get("min", "N/A"),
                    "Max": summary.get("max", "N/A"),
                    "Mean": f"{summary.get('mean', 0):.2f}" if "mean" in summary else "N/A",
                    "Std Dev": f"{summary.get('std', 0):.2f}" if "std" in summary else "N/A",
                }
            })
        
        with col2:
            # Additional stats
            additional_stats = {
                "Properties": {
                    "Has reached 1": sequence[-1] == 1,
                    "Is complete": True,
                    "Peak value": peak_val,
                    "Peak at step": peak_idx,
                }
            }
            st.json(additional_stats)


def show_model_inference():
    """Display model inference page."""
    st.markdown('<div class="main-header">🤖 Model Inference</div>', unsafe_allow_html=True)
    
    st.markdown("Use trained transformer models to predict Collatz sequence steps.")
    st.markdown("---")
    
    st.info("💡 **Note**: Model inference requires a trained checkpoint. Train a model first using the CLI or notebooks.")
    
    # Model selection (placeholder - would load from checkpoints directory)
    checkpoint_dir = Path("./checkpoints")
    available_checkpoints = []
    
    if checkpoint_dir.exists():
        available_checkpoints = list(checkpoint_dir.glob("*.pt"))
    
    if not available_checkpoints:
        st.warning("⚠️ No trained models found. Please train a model first.")
        st.markdown("""
        To train a model, use the CLI:
        ```bash
        python -m src.apps.cli train --num-samples 10000 --epochs 10
        ```
        """)
        return
    
    # Model selection
    selected_checkpoint = st.selectbox(
        "Select Model Checkpoint",
        options=[str(cp) for cp in available_checkpoints],
        help="Choose a trained model checkpoint to use for inference"
    )
    
    if selected_checkpoint:
        st.markdown("### 🔮 Make Predictions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            input_value = st.number_input(
                "Input Value (Odd Integer)",
                min_value=1,
                max_value=100000,
                value=27,
                step=2,
                help="Enter an odd integer to predict its Collatz long step"
            )
        
        with col2:
            encoding_base = st.selectbox(
                "Encoding Base",
                options=[2, 8, 10, 16, 24, 32],
                index=4,  # Default to 24
                help="Base for encoding (should match training base)"
            )
        
        if st.button("🔮 Predict", type="primary", use_container_width=True):
            with st.spinner("Computing prediction..."):
                try:
                    # Encode input
                    encoder = MultiBaseEncoder(base=encoding_base)
                    encoded = encoder.encode(input_value)
                    
                    # Show encoding
                    st.markdown("### 📝 Input Encoding")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.code(f"Input: {input_value}")
                        st.code(f"Encoded: {encoded}")
                    with col2:
                        st.code(f"Base: {encoding_base}")
                        st.code(f"Length: {len(encoded)}")
                    
                    st.info("⚠️ **Model Loading**: Full model inference implementation would require loading the checkpoint and running forward pass. This is a placeholder for the interface.")
                    
                    # Show actual Collatz computation for comparison
                    st.markdown("### ✅ Actual Collatz Long Step")
                    seq = CollatzSequence(start=1)
                    long_step = seq.compute_long_step(input_value)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Input", input_value)
                    with col2:
                        st.metric("Result", long_step['result'])
                    with col3:
                        st.metric("k", long_step['k'])
                    with col4:
                        st.metric("k'", long_step['k_prime'])
                    
                except Exception as e:
                    st.error(f"Error during prediction: {e}")


def show_statistical_analysis():
    """Display statistical analysis page."""
    st.markdown('<div class="main-header">📈 Statistical Analysis</div>', unsafe_allow_html=True)
    
    st.markdown("Analyze patterns and statistics across multiple Collatz sequences.")
    st.markdown("---")
    
    # Analysis parameters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        start_range = st.number_input(
            "Start Range",
            min_value=1,
            max_value=100000,
            value=1,
            step=1
        )
    
    with col2:
        end_range = st.number_input(
            "End Range",
            min_value=1,
            max_value=100000,
            value=100,
            step=1
        )
    
    with col3:
        step_size = st.number_input(
            "Step Size",
            min_value=1,
            max_value=1000,
            value=1,
            step=1
        )
    
    # Warning for large ranges
    num_sequences = (end_range - start_range) // step_size + 1
    if num_sequences > 1000:
        st.warning(f"⚠️ Large range selected ({num_sequences} sequences). This may take a while.")
    
    max_iter = st.slider(
        "Max Iterations per Sequence",
        min_value=1000,
        max_value=1000000,
        value=100000,
        step=10000
    )
    
    if st.button("🔍 Analyze Sequences", type="primary", use_container_width=True):
        if start_range > end_range:
            st.error("Start range must be <= end range")
        else:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            results = []
            sequences_to_analyze = range(int(start_range), int(end_range) + 1, int(step_size))
            total = len(list(sequences_to_analyze))
            
            for i, start_val in enumerate(sequences_to_analyze):
                if i % max(1, total // 100) == 0:
                    progress_bar.progress((i + 1) / total)
                    status_text.text(f"Analyzing sequence {i+1}/{total} (start: {start_val})")
                
                try:
                    seq = CollatzSequence(start=start_val, max_iterations=int(max_iter))
                    sequence = seq.generate()
                    
                    results.append({
                        "Start": start_val,
                        "Length": len(sequence),
                        "Max Value": seq.get_max_value(),
                        "Peak Value": seq.get_peak_value()[0],
                        "Peak Position": seq.get_peak_value()[1],
                    })
                except Exception as e:
                    st.warning(f"Error processing {start_val}: {e}")
            
            progress_bar.empty()
            status_text.empty()
            
            if results:
                st.session_state.analysis_results = results
                st.success(f"✅ Analyzed {len(results)} sequences!")
    
    # Display results if available
    if "analysis_results" in st.session_state:
        results = st.session_state.analysis_results
        df = pd.DataFrame(results)
        
        st.markdown("### 📊 Summary Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Sequences Analyzed", len(results))
        with col2:
            st.metric("Avg Length", f"{float(df['Length'].mean()):.1f}", "steps")
        with col3:
            st.metric("Max Length", int(df['Length'].max()), "steps")
        with col4:
            st.metric("Avg Max Value", f"{float(df['Max Value'].mean()):.0f}")
        
        # Visualizations
        st.markdown("### 📈 Visualizations")
        
        tab1, tab2, tab3, tab4 = st.tabs(["Length Distribution", "Max Value Distribution", "Scatter Plot", "Data Table"])
        
        with tab1:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(df['Length'], bins=30, edgecolor='black', alpha=0.7)
            ax.set_xlabel("Sequence Length", fontsize=12)
            ax.set_ylabel("Frequency", fontsize=12)
            ax.set_title("Distribution of Sequence Lengths", fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        
        with tab2:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(df['Max Value'], bins=30, edgecolor='black', alpha=0.7, color='orange')
            ax.set_xlabel("Maximum Value", fontsize=12)
            ax.set_ylabel("Frequency", fontsize=12)
            ax.set_title("Distribution of Maximum Values", fontsize=14, fontweight='bold')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        
        with tab3:
            fig, ax = plt.subplots(figsize=(10, 6))
            scatter = ax.scatter(df['Start'], df['Length'], c=df['Max Value'], 
                               cmap='viridis', alpha=0.6, s=50)
            ax.set_xlabel("Start Value", fontsize=12)
            ax.set_ylabel("Sequence Length", fontsize=12)
            ax.set_title("Start Value vs Sequence Length", fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=ax, label='Max Value')
            st.pyplot(fig)
        
        with tab4:
            st.dataframe(df, use_container_width=True, height=400)
            
            # Download
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name=f"collatz_analysis_{start_range}_{end_range}.csv",
                mime="text/csv"
            )


def show_health_check():
    """Display health check page."""
    st.markdown('<div class="main-header">🏥 Health Check</div>', unsafe_allow_html=True)
    
    st.markdown("""
    This page provides real-time health monitoring for the ARKHE Framework.
    Use this to verify that all dependencies, modules, and system resources are functioning correctly.
    """)
    
    # Refresh button
    if st.button("🔄 Refresh Health Status", type="primary"):
        st.rerun()
    
    # Run health checks
    with st.spinner("Running health checks..."):
        checker = HealthChecker()
        health_status = checker.run_all_checks()
    
    # Overall status
    overall_status = health_status["status"]
    status_color = "🟢" if overall_status == "healthy" else "🔴"
    st.markdown(f"### {status_color} Overall Status: {overall_status.upper()}")
    st.markdown(f"**Timestamp:** {health_status['timestamp']}")
    
    st.markdown("---")
    
    # Individual checks
    st.markdown("### Detailed Checks")
    
    checks = health_status["checks"]
    
    # PyTorch check
    with st.expander("🔥 PyTorch", expanded=True):
        pytorch = checks["pytorch"]
        if pytorch["status"] == "healthy":
            st.success("✅ PyTorch is installed and working")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Version", pytorch["version"])
                st.metric("CUDA Available", "Yes" if pytorch["cuda_available"] else "No")
            with col2:
                if pytorch["cuda_available"]:
                    st.metric("CUDA Version", pytorch.get("cuda_version", "N/A"))
                    st.metric("GPU Devices", pytorch.get("device_count", 0))
                    if pytorch.get("device_name"):
                        st.info(f"Device: {pytorch['device_name']}")
        else:
            st.error(f"❌ PyTorch check failed: {pytorch.get('message', 'Unknown error')}")
            if "error" in pytorch:
                st.code(pytorch["error"])
    
    # NumPy check
    with st.expander("🔢 NumPy"):
        numpy = checks["numpy"]
        if numpy["status"] == "healthy":
            st.success(f"✅ NumPy version {numpy['version']}")
        else:
            st.error(f"❌ NumPy check failed: {numpy.get('message', 'Unknown error')}")
    
    # Pandas check
    with st.expander("🐼 Pandas"):
        pandas = checks["pandas"]
        if pandas["status"] == "healthy":
            st.success(f"✅ Pandas version {pandas['version']}")
        else:
            st.error(f"❌ Pandas check failed: {pandas.get('message', 'Unknown error')}")
    
    # Filesystem check
    with st.expander("📁 Filesystem"):
        fs = checks["filesystem"]
        if fs["status"] == "healthy":
            st.success("✅ All required directories are accessible")
            for path_name, path_info in fs["paths"].items():
                if path_info["status"] == "healthy":
                    st.text(f"  ✅ {path_name}: writable")
                else:
                    st.error(f"  ❌ {path_name}: {path_info.get('error', 'not accessible')}")
        else:
            st.error("❌ Some directories are not accessible")
            for path_name, path_info in fs["paths"].items():
                if path_info["status"] == "unhealthy":
                    st.error(f"  ❌ {path_name}: {path_info.get('error', 'not accessible')}")
    
    # Imports check
    with st.expander("📦 Module Imports"):
        imports = checks["imports"]
        if imports["status"] == "healthy":
            st.success("✅ All critical modules imported successfully")
            for module_name, module_info in imports["modules"].items():
                st.text(f"  ✅ {module_name}")
        else:
            st.error("❌ Some modules failed to import")
            for module_name, module_info in imports["modules"].items():
                if module_info["status"] == "healthy":
                    st.text(f"  ✅ {module_name}")
                else:
                    st.error(f"  ❌ {module_name}: {module_info.get('error', 'import failed')}")
    
    # System check
    with st.expander("💻 System Information"):
        system = checks["system"]
        if system["status"] == "healthy":
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Platform:**")
                st.text(f"{system['platform']} {system.get('platform_version', '')}")
                st.markdown("**Python:**")
                st.text(system["python_version"].split("\n")[0])
            with col2:
                if "memory" in system:
                    st.markdown("**Memory:**")
                    st.metric("Total", f"{system['memory']['total_gb']:.2f} GB")
                    st.metric("Available", f"{system['memory']['available_gb']:.2f} GB")
                    st.metric("Used", f"{system['memory']['percent_used']:.1f}%")
                if "disk" in system:
                    st.markdown("**Disk:**")
                    st.metric("Total", f"{system['disk']['total_gb']:.2f} GB")
                    st.metric("Free", f"{system['disk']['free_gb']:.2f} GB")
                    st.metric("Used", f"{system['disk']['percent_used']:.1f}%")
                if "note" in system:
                    st.info(system["note"])
    
    # JSON export
    st.markdown("---")
    st.markdown("### Export Health Status")
    st.json(health_status)


def show_about_page():
    """Display about page."""
    st.markdown('<div class="main-header">⚙️ About ARKHE Framework</div>', unsafe_allow_html=True)
    
    st.markdown("""
    ### Overview
    
    **ARKHE Framework** is an enterprise-grade Python framework for mathematical sequence 
    research and machine learning experimentation. This framework provides tools for exploring 
    mathematical sequences (such as Collatz), performing statistical analysis, and training 
    transformer models to understand sequence patterns.
    
    ### Features
    
    - 🔢 **Sequence Generation**: Generate Collatz and other mathematical sequences
    - 📊 **Statistical Analysis**: Comprehensive statistical tools and pattern detection
    - 🤖 **Machine Learning**: Transformer models for sequence prediction
    - 📈 **Visualization**: Rich plotting and analysis capabilities
    - 🔧 **CLI Tools**: Command-line interface for batch processing
    - 📓 **Jupyter Notebooks**: Interactive research environment
    
    ### Author
    
    **MoniGarr**  
    Email: monigarr@MoniGarr.com  
    Website: MoniGarr.com
    
    ### Research Interests
    
    - AI/ML Research and Development
    - Extended Reality (XR) Applications
    - 3D Graphics and Visualization
    - Robotics and Autonomous Systems
    - Computer Vision
    - Navigation Systems
    - Natural Language Processing (NLP)
    - Low Resource Languages
    
    ### Version
    
    Version: 0.1.0  
    License: MIT
    
    ### Links
    
    - GitHub Repository: [Link to repository]
    - Documentation: [Link to docs]
    - CLI Help: `python -m src.apps.cli --help`
    """)


if __name__ == "__main__":
    main()

