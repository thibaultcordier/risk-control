from unittest import TestCase


class TestMySubModule(TestCase):
    def test_add(self) -> None:
        self.assertEqual(2 + 1, 3)

    def test_sub(self) -> None:
        self.assertEqual(3 - 2, 1)
