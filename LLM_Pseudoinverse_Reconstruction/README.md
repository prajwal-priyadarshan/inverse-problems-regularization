# LLM-Guided Pseudoinverse Reconstruction

A research project exploring LLM-guided regularization selection for ill-posed inverse problems using pseudoinverse reconstruction techniques.

## Overview

This project demonstrates how Large Language Models can assist in selecting appropriate regularization methods and parameters for solving ill-posed inverse problems. The system compares baseline pseudoinverse reconstruction with Tikhonov regularization and Truncated SVD (TSVD) methods, using diagnostic metrics to guide optimal parameter selection.

## Features

- **Multiple Forward Models**: Blur operator, downsample operator, rank-deficient operator
- **Signal Generation**: Sinusoidal, multi-sine, and piecewise signals
- **Noise Sensitivity Analysis**: Low, medium, and high noise levels
- **Regularization Methods**: 
  - Tikhonov (L2) regularization
  - Truncated SVD (TSVD)
  - Baseline pseudoinverse
- **Diagnostic Tools**:
  - SVD analysis and condition number computation
  - Picard plots
  - L-curve analysis
- **LLM Integration**: Automated reasoning for regularization parameter selection using Groq API

## Prerequisites

- Python 3.8 or higher
- Git

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Sem4Workspace/inverse-problems-regularization.git
cd inverse-problems-regularization
```

### 2. Create a Virtual Environment

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

Create a `.env` file in the project root directory:

```bash
# On Windows
type nul > .env

# On macOS/Linux
touch .env
```

Add your Groq API key to the `.env` file:

```
GROQ_API_KEY=your_groq_api_key_here
```

> **Note**: Get your free Groq API key from [https://console.groq.com/](https://console.groq.com/)

## Usage

### Run the Complete Pipeline

Run the full reconstruction pipeline with all noise levels and LLM guidance:

```bash
python run_pipeline.py --noise-level all --enable-llm
```

### Command Line Options

- `--noise-level`: Specify noise level (`low`, `medium`, `high`, or `all`)
- `--enable-llm`: Enable LLM-guided regularization selection

**Examples:**

```bash
# Run with medium noise only
python run_pipeline.py --noise-level medium

# Run with high noise and LLM guidance
python run_pipeline.py --noise-level high --enable-llm

# Run all noise levels without LLM
python run_pipeline.py --noise-level all
```

### Explore Jupyter Notebooks

Interactive experiments are available in the `experiments/` directory:

```bash
jupyter notebook experiments/
```

Available notebooks:
- `exp_01_baseline_failure.ipynb` - Demonstrates pseudoinverse failure
- `exp_02_regularization_comparison.ipynb` - Compare Tikhonov vs TSVD
- `exp_03_llm_guided_selection.ipynb` - LLM-based parameter selection
- `exp_04_noise_sensitivity.ipynb` - Noise sensitivity analysis

## Project Structure

```
LLM_Pseudoinverse_Reconstruction/
├── data/                          # Generated signals, operators, measurements
├── docs/                          # Documentation and methodology
├── experiments/                   # Jupyter notebooks for experiments
├── results/                       # Output figures, metrics, LLM decisions
├── src/                          # Source code
│   ├── diagnostics/              # SVD analysis, Picard plots, L-curve
│   ├── evaluation/               # Error metrics and comparison tools
│   ├── forward_models/           # Operators (blur, downsample, etc.)
│   ├── llm_reasoning/            # LLM interface and prompt templates
│   ├── noise_models/             # Noise generation
│   ├── reconstruction/           # Pseudoinverse, Tikhonov, TSVD
│   ├── signal_generation/        # Signal generators
│   └── utils/                    # Configuration and helpers
├── run_pipeline.py               # Main execution script
└── requirements.txt              # Python dependencies
```

## Results

Results are saved in the `results/` directory:
- **figures/**: Plots of reconstructions, diagnostics, and comparisons
- **metrics/**: CSV and JSON files with MSE, PSNR, and error metrics
- **llm_outputs/**: LLM decisions and explanations

## How It Works

1. **Signal Generation**: Creates synthetic test signals
2. **Forward Model**: Applies degradation operator (blur, downsample, etc.)
3. **Noise Addition**: Adds Gaussian noise at specified levels
4. **Baseline Reconstruction**: Attempts pseudoinverse (often fails)
5. **Diagnostics**: Analyzes condition number, singular values, Picard plot
6. **Regularization Sweep**: Tests multiple regularization parameters
7. **LLM Guidance**: Uses diagnostic data to recommend optimal parameters
8. **Evaluation**: Computes MSE, PSNR, and relative error metrics

## Configuration

Edit `src/utils/config.py` to modify experiment parameters:

```python
@dataclass
class ExperimentConfig:
    n_samples: int = 200              # Number of samples
    gaussian_sigma: float = 2.5       # Blur kernel width
    kernel_radius: int = 10           # Blur kernel radius
    noise_levels: dict = ...          # Noise level settings
```

## Citation

If you use this code in your research, please cite:

```
@software{llm_pseudoinverse_reconstruction,
  title={LLM-Guided Pseudoinverse Reconstruction},
  author={Your Name},
  year={2026},
  url={https://github.com/Sem4Workspace/inverse-problems-regularization}
}
```

## License

MIT License - See LICENSE file for details

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Troubleshooting

**Issue**: `ModuleNotFoundError` when running scripts
- **Solution**: Make sure virtual environment is activated and dependencies are installed

**Issue**: `GROQ_API_KEY not set` error
- **Solution**: Ensure `.env` file exists in project root with valid API key

**Issue**: Import errors from `src/` modules
- **Solution**: Run scripts from the project root directory

## Contact

For questions or issues, please open an issue on GitHub.
