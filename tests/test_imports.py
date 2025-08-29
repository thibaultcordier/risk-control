import os
import sys
import unittest

# Add the project root to the path to allow direct imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestModuleImports(unittest.TestCase):
    """Test cases for importing risk_control modules."""

    def test_parameter_module(self) -> None:
        """Test the parameter module directly."""
        # Let's check what's available in parameter module
        import risk_control.parameter

        self.assertIsNotNone(risk_control.parameter)

    def test_abstention_module(self) -> None:
        """Test the abstention module directly."""
        # Let's check what's available in abstention module
        import risk_control.abstention

        self.assertIsNotNone(risk_control.abstention)

    def test_plot_module(self) -> None:
        """Test the plot module directly."""
        # Let's check what's available in plot module
        import risk_control.plot

        self.assertIsNotNone(risk_control.plot)

    def test_risk_module(self) -> None:
        """Test the risk module directly."""
        # Let's check what's available in risk module
        import risk_control.risk

        self.assertIsNotNone(risk_control.risk)


if __name__ == "__main__":
    unittest.main()
