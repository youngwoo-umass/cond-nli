from typing import Callable, TypeVar, Iterable, List
from typing import List, Iterable, Callable, Dict, Tuple, Set

A = TypeVar('A')
B = TypeVar('B')
C = TypeVar('C')


def lmap(func: Callable[[A], B],
         iterable_something: Iterable[A]) -> List[B]:
    return list([func(e) for e in iterable_something])


def flatten(z: Iterable[Iterable[A]]) -> Iterable[A]:
    return [y for x in z for y in x]




def index_by_fn(func: Callable[[A], B], z: Iterable[A]) -> Dict[B, A]:
    d_out: Dict[B, A] = {}
    for e in z:
        d_out[func(e)] = e
    return d_out


def list_equal(a: List, b: List):
    if len(a) != len(b):
        return False

    for a_e, b_e in zip(a, b):
        if a_e != b_e:
            return False
    return True


def assert_list_equal(a: List, b: List):
    if not list_equal(a, b):
        print("List does not equal".format(a, b))
        print("a: ", a)
        print("b: ", b)
        raise ValueError()
