import math
from typing import Self


class ABSTAIN(float):
    def __new__(cls) -> Self:
        # Create a new instance of the float type with NaN value
        return super(ABSTAIN, cls).__new__(cls, math.nan)

    def __bool__(self) -> bool:
        # Override the __bool__ method to return False
        return False


# Create an instance of CustomNaN
_abs = ABSTAIN()


if __name__ == "__main__":
    # Test the behavior
    print(bool(_abs))  # Output: False
    print(math.isnan(_abs))  # Output: True

    import numpy as np

    print(np.bool(_abs))  # Output: False
    print(np.isnan(_abs))  # Output: True

    print(_abs == 0)  # Output: False
    print(_abs == 1)  # Output: False
    print(_abs == _abs)  # Output: False
    print(_abs == np.nan)  # Output: False
    print(_abs == math.nan)  # Output: False

    print(np.argmax([False, _abs, np.nan, True]))  # Output: 1
    print(np.nanargmax([False, _abs, np.nan, True]))  # Output: 3

    print(np.argmax([False, np.nan, _abs, True]))  # Output: 1
    print(np.nanargmax([False, np.nan, _abs, True]))  # Output: 3
