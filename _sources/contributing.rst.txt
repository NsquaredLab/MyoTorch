Contributing to MyoTorch
=======================

Thank you for your interest in contributing to MyoTorch! This guide will help you get started with the development process.

Setting Up Development Environment
---------------------------------

1. Fork the repository on GitHub
2. Clone your fork locally:

   .. code-block:: bash

      git clone https://github.com/NsquaredLab/MyoTorch.git
      cd MyoTorch

3. Install uv (if you don't have it yet):
   
   Follow the instructions on the `uv GitHub page <https://github.com/astral-sh/uv>`_.

4. Set up Virtual Environment & Install Dependencies:

   .. code-block:: bash

      uv sync --group dev

5. Install PyTorch with GPU:

   * **For Windows:** Visit the `PyTorch installation guide <https://pytorch.org/get-started/locally/>`_ and select the appropriate options. Use uv to run the install command:

     .. code-block:: bash

        # Example for CUDA 12.6 - Get the correct command from PyTorch website!
        uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126 --upgrade

   * **For Linux:** Pytorch GPU is already installed by MyoTorch.

Contribution Workflow
--------------------

1. Create a new branch for your feature or bugfix:

   .. code-block:: bash

      git checkout -b feature/your-feature-name

2. Make your changes and ensure all tests pass:

   .. code-block:: bash

      pytest

3. Add proper documentation for new features
4. Submit a pull request to the main repository

Code Style
---------

- We follow PEP 8 guidelines for Python code
- Use descriptive variable names
- Document functions using NumPy docstring format
- Add unit tests for new functionality

Documentation
------------

When adding new features, please update the documentation:

1. Add docstrings to your functions and classes
2. Update relevant documentation pages
3. Add example usage if applicable
4. Consider adding examples to the `examples` directory

Getting Help
-----------

If you have questions or need assistance:

- Open an issue on GitHub
- Reach out on our community forums
- Contact the maintainers 