<img src="https://github.com/NsquaredLab/MyoTorch/blob/main/docs/source/_static/myotorch_logo.png?raw=true" height="250">

<a href="https://www.python.org/downloads/release/python-3100/"><img alt="Code style: black" src="https://img.shields.io/badge/python-%3E=3.10,%20%3C=3.13-blue"></a>
<a href="https://www.pytorchlightning.ai/"><img alt="Code style: black" src="https://img.shields.io/badge/uses-pytorch & pytorch lighting-blueviolet"></a>

> [!TIP]
> Dive deeper into our features and usage with the official [documentation](https://nsquaredlab.github.io/MyoTorch/).

# MyoTorch - The AI toolkit for myocontrol research

## What is MyoTorch? 
MyoTorch is your cutting-edge **research** companion for unlocking the secrets hidden within biomechanical data! It's specifically designed for exploring the complex interplay between **electromyography (EMG)** signals, **kinematics** (movement), and **kinetics** (forces).

Leveraging the power of **PyTorch** and **PyTorch Lightning**, MyoTorch provides a comprehensive suite of tools, including:
*   **Data loaders** and **preprocessing filters** tailored for biomechanical signals.
*   Peer-reviewed **AI models** and components for analysis and prediction tasks.
*   Essential **utilities** to streamline the research workflow.

Whether you're predicting movement from muscle activity, analyzing forces during motion, or developing novel AI approaches for biomechanical challenges, MyoTorch aims to accelerate your research journey.

> [!IMPORTANT]  
> MyoTorch is built for **research**. While powerful, it's evolving and may not have the same level of stability as foundational libraries like NumPy. We appreciate your understanding and contributions!

## Installation

MyoTorch automatically installs with the correct PyTorch version for your platform.

### Basic installation:

```bash
# Install from PyPI
pip install myotorch
```

This will automatically:
- On Linux: Install PyTorch and TorchVision from PyPI (with CUDA support)
- On Windows: Install PyTorch and TorchVision with CUDA 12.4 support

## Development

For development, install the dev dependencies:

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/NsquaredLab/MyoTorch.git # Replace with your actual repo URL if different
    cd MyoTorch
    ```
2.  **Install uv:** If you don't have it yet, install `uv`. Follow the instructions on the [uv GitHub page](https://github.com/astral-sh/uv).
3.  **Set up Virtual Environment & Install Dependencies:** Simply run:
    ```bash
    uv sync --group dev
    ```

> [!NOTE]
> The project is configured to automatically install:
> - On Linux: Standard PyTorch with CUDA from PyPI
> - On Windows: PyTorch with CUDA 12.4 support from the PyTorch custom index

## What is what?
This project uses the following structure:
- `myotorch`: This is the main package. It contains:
  - `datasets`: Contains data loaders, dataset creators, and a wide array of filters to preprocess your biomechanical data (e.g., EMG, kinematics).
  - `models`: Contains all AI models and their components, ready for training and evaluation.
  - `utils`: Various utilities to support data handling, model training, and analysis.
- `docs`: Contains the source files for the documentation.
- `examples`: Contains practical examples demonstrating how to use the package, including tutorials (`01_tutorials`) and specific use cases like applying filters (`02_filters`).
- `tests`: Contains tests to ensure package integrity and correctness.
