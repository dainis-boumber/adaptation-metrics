import itertools as it
from typing import Iterable, TypeVar

from typing import TypeVar

Any = TypeVar("Any")


def batch(iterable: Iterable[Any], size: int) -> Iterable[Any]:
    iters = iter(iterable)

    def take():
        while True:
            yield tuple(it.islice(iters, size))

    return iter(take().__next__, tuple())
