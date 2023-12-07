from abc import ABC, abstractmethod
from typing import List, Tuple, TypeVar, NamedTuple

from contradiction.medical_claims.token_tagging.problem_loader import CondNLISentPair
from utils.list_lib import lmap
from utils.misc_lib import save_as_jsonl
from utils.promise import PromiseKeeper, MyFuture, list_future


class BatchTokenScoringSolverIF(ABC):
    @abstractmethod
    def solve(self, payload: List[Tuple[List[str], List[str]]]) -> List[Tuple[List[float], List[float]]]:
        pass


NeuralInput = TypeVar('NeuralInput')
NeuralOutput = TypeVar('NeuralOutput')
CondNLIInput = Tuple[List[str], List[str]]
CondNLIOutput = Tuple[List[float], List[float]]


# Batch Solver Adapter
class BSAdapterIF(ABC):
    @abstractmethod
    def neural_worker(self, items: List):
        pass

    @abstractmethod
    def reduce(self, t1, t2, output: List) -> List[float]:
        pass

    @abstractmethod
    def enum_child(self, t1: List[str], t2: List[str]) -> List:
        pass


class BatchSolver(BatchTokenScoringSolverIF):
    def __init__(self, adapter):
        self.adapter: BSAdapterIF = adapter

    def solve(self, payload: List[CondNLIInput]) -> List[CondNLIOutput]:
        pk = PromiseKeeper(self.adapter.neural_worker)

        class Entry(NamedTuple):
            t1: List[str]
            t2: List[str]
            f_list1: List[MyFuture]
            f_list2: List[MyFuture]

        def prepare_pre_problem(input_obj: CondNLIInput) -> Entry:
            t1, t2 = input_obj
            es_list1: List[NeuralInput] = self.adapter.enum_child(t2, t1)
            es_list2: List[NeuralInput] = self.adapter.enum_child(t1, t2)
            future_list1 = lmap(pk.get_future, es_list1)
            future_list2 = lmap(pk.get_future, es_list2)
            return Entry(t1, t2, future_list1, future_list2)

        future_ref = lmap(prepare_pre_problem, payload)
        pk.do_duty()

        def apply_reduce(e: Entry) -> CondNLIOutput:
            def get_scores(t1, t2, fl: List[MyFuture[NeuralOutput]]) -> List[float]:
                fl: List[MyFuture[NeuralOutput]] = fl
                l: List[NeuralOutput] = list_future(fl)
                scores: List[float] = self.adapter.reduce(t1, t2, l)
                return scores

            return get_scores(e.t2, e.t1, e.f_list1), \
                   get_scores(e.t1, e.t2, e.f_list2)

        return lmap(apply_reduce, future_ref)



def score_tokens_with_batch_solver(
        problems: List[CondNLISentPair], save_path, solver: BatchTokenScoringSolverIF):
    payload = []
    for p in problems:
        input_per_problem = p.text1.split(), p.text2.split()
        payload.append(input_per_problem)

    batch_output = solver.solve(payload)
    assert len(problems) == len(batch_output)
    save_j_list = []
    for p, output in zip(problems, batch_output):
        scores1, scores2 = output
        save_entry = {
            'pair_id': p.pair_id,
            "scores1": scores1,
            "scores2": scores2
        }
        save_j_list.append(save_entry)

    save_as_jsonl(save_j_list, save_path)


