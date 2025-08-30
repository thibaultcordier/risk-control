import os
import sys
import unittest

# Add the project root to the path to allow direct imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestModuleImports(unittest.TestCase):
    """Test cases for importing mlrisko modules."""

    def test_parameter_module(self) -> None:
        """Test the parameter module directly."""
        # Let's check what's available in parameter module
        import mlrisko.parameter

        self.assertIsNotNone(mlrisko.parameter)

    def test_abstention_module(self) -> None:
        """Test the abstention module directly."""
        # Let's check what's available in abstention module
        import mlrisko.abstention

        self.assertIsNotNone(mlrisko.abstention)

    def test_plot_module(self) -> None:
        """Test the plot module directly."""
        # Let's check what's available in plot module
        import mlrisko.plot

        self.assertIsNotNone(mlrisko.plot)

    def test_risk_module(self) -> None:
        """Test the risk module directly."""
        # Let's check what's available in risk module
        import mlrisko.risk

        self.assertIsNotNone(mlrisko.risk)


if __name__ == "__main__":
    unittest.main()
