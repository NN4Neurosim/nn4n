from .multi_area import MultiArea
from .multi_area_ei import MultiAreaEI
from .random_input import RandomInput
import warnings

warnings.warn("The `structure` module is deprecated and will be removed in future versions. "
              "Use the `mask` module instead, which inherits everything from `structure`.",
              UserWarning, stacklevel=2)

if __name__ == "__main__":
    print(MultiAreaEI)
    print(MultiArea)
    print(RandomInput)
