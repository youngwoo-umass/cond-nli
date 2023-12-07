import time
from typing import List, Callable, Generic, Iterable
from typing import TypeVar

A = TypeVar('A')
B = TypeVar('B')


class PromiseKeeper:
    def __init__(self,
                 list_fn: Callable[[A], B],
                 time_estimate=None
                 ):
        self.X_list = []
        self.list_fn = list_fn
        self.time_estimate: float = time_estimate

    def do_duty(self, log_size=False, reset=True):
        x_list = list([X.X for X in self.X_list])
        if self.time_estimate is not None:
            estimate = self.time_estimate * len(x_list)
            if estimate > 10:
                print("PromiseKeeper - Expected time: {0:.0f}sec".format(estimate))

        if log_size:
            print("PromiseKeeper - {} items".format(len(x_list)))

        st = time.time()
        y_list = self.list_fn(x_list)
        for X, y in zip(self.X_list, y_list):
            X.future().set_value(y)

        ed = time.time()
        if log_size:
            elapsed = ed - st
            s_per_inst = elapsed / len(x_list) if len(x_list) else 0
            print(f"finished in {elapsed}s. {s_per_inst} per item")

        if reset:
            self.reset()

    def get_future(self, x):
        return MyPromise(x, self).future()

    def reset(self):
        self.X_list = []


T = TypeVar('T')


class MyFuture(Generic[T]):
    def __init__(self):
        self._Y = None
        self.f_ready = False

    def get(self):
        if not self.f_ready:
            raise Exception("Future is not ready.")
        return self._Y

    def set_value(self, v):
        self._Y = v
        self.f_ready = True


class MyPromise:
    def __init__(self, X, promise_keeper: PromiseKeeper):
        self.X = X
        self.Y = MyFuture()
        promise_keeper.X_list.append(self)

    def future(self):
        return self.Y


# get from all element futures in the list
def list_future(futures: Iterable[MyFuture[T]]) -> List[T]:
    return list([f.get() for f in futures])


def unpack_future(item):
    if isinstance(item, MyFuture):
        return item.get()
    elif isinstance(item, tuple):
        return tuple([unpack_future(sub_item) for sub_item in item])
    elif isinstance(item, list):
        return [unpack_future(sub_item) for sub_item in item]
    elif isinstance(item, dict):
        return {k: unpack_future(v) for k, v in item.items()}
    else:
        return item


if __name__ == '__main__':
    def list_fn(l):
        r = []
        for i in l:
            r.append(i * 2)
            time.sleep(1)
        return r

    pk = PromiseKeeper(list_fn)
    X_list = list(range(10))
    y_list = []
    for x in X_list:
        y = MyPromise(x, pk).future()
        y_list.append(y)

    pk.do_duty()

    for e in y_list:
        print(e.Y)
