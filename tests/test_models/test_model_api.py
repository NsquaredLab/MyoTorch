"""Tests for model API and naming conventions."""

import importlib

import pytest
from pathlib import Path


class TestModelAPI:
    def test_model_names_raul_net(self):
        """Test that RaulNet models follow naming convention."""
        # Get list of all model files in the new structure
        model_dir = Path(__file__).parent.parent.parent / "myotorch" / "models" / "raul_net"
        model_files = [
            p.stem for p in model_dir.glob("*.py")
            if not p.name.startswith("_") and p.name != "__init__.py"
        ]

        # Check if the model names are correct
        for model_name in sorted(model_files):
            try:
                imported_module = importlib.import_module(
                    f"myotorch.models.raul_net.{model_name}"
                )

                # Extract version number from filename (e.g., "v16" -> "16")
                version = model_name.lstrip("v")
                expected_class = f"RaulNetV{version.upper()}" if not version[0].isdigit() else f"RaulNetV{version}"

                assert hasattr(imported_module, f"RaulNetV{version.split('_')[0]}"), (
                    f"Module {model_name} should have class RaulNetV{version.split('_')[0]}"
                )

            except ModuleNotFoundError:
                pytest.fail(f"Could not import myotorch.models.raul_net.{model_name}")

    def test_models_exported_from_package(self):
        """Test that main models are exported from myotorch.models."""
        from myotorch.models import RaulNetV16, RaulNetV17

        assert RaulNetV16 is not None
        assert RaulNetV17 is not None

    def test_components_exported_from_package(self):
        """Test that components are exported from myotorch.models."""
        from myotorch.models import EuclideanDistance, SMU, PSerf

        assert EuclideanDistance is not None
        assert SMU is not None
        assert PSerf is not None
