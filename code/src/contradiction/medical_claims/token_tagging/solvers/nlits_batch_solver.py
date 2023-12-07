from abc import ABC, abstractmethod
from typing import List, Tuple

from contradiction.medical_claims.token_tagging.batch_solver_common import NeuralOutput, BSAdapterIF
from data_generator.tokenizer_wo_tf import get_tokenizer
from trainer_v2.custom_loop.per_task.nli_ts_util import LocalDecisionNLICore, enum_hypo_token_tuple_from_tokens, \
    EncodedSegmentIF
from utils.list_lib import flatten
from utils.misc_lib import average


class ESTwoPiece(EncodedSegmentIF):
    def __init__(self, input_x,
                 t1,
                 t2,
                 h_tokens_list,
                 st, ed):
        self.input_x = input_x
        self.t1 = t1
        self.t2 = t2
        self.h_tokens_list = h_tokens_list
        self.st = st
        self.ed = ed

    def get_input(self):
        return self.input_x


class ScoreReducerI(ABC):
    @abstractmethod
    def reduce(self, records: List[Tuple[List[float], ESTwoPiece]], text2_tokens) -> List[float]:
        pass


class ReducerCommon(ScoreReducerI):
    def __init__(self, target_label: int, reduce_fn):
        self.target_label = target_label
        self.reduce_fn = reduce_fn

    def reduce(self, records: List[Tuple[List[float], ESTwoPiece]], text2_tokens) -> List[float]:
        scores_building = [list() for _ in text2_tokens]
        for probs, es_item in records:
            s = probs[self.target_label]
            for i in range(es_item.st, es_item.ed):
                if i < len(scores_building):
                    scores_building[i].append(s)

        scores = [self.reduce_fn(scores) for scores in scores_building]
        return scores


class AvgReducer(ReducerCommon):
    def __init__(self, target_label: int):
        self.target_label = target_label
        super(AvgReducer, self).__init__(target_label, average)


class NLITSAdapter(BSAdapterIF):
    def __init__(self,
                 nlits: LocalDecisionNLICore,
                 score_reducer: ScoreReducerI):
        self.nlits: LocalDecisionNLICore = nlits
        self.tokenizer = get_tokenizer()
        self.score_reducer = score_reducer

    def neural_worker(self, items: List[ESTwoPiece]) -> List[Tuple[NeuralOutput, ESTwoPiece]]:
        l_decisions = self.nlits.predict_es(items)
        return list(zip(l_decisions, items))

    def reduce(self, t1, t2, item: List[Tuple[NeuralOutput, ESTwoPiece]]) -> List[float]:
        return self.score_reducer.reduce(item, t2)

    def enum_child(self, t1, t2):
        p_tokens = list(flatten(map(self.tokenizer.tokenize, t1)))
        n_seg = len(t2)
        es_list = []
        for offset in [0, 1, 2]:
            for window_size in [1, 3, 6]:
                if window_size >= n_seg:
                    break
                for h_first, h_second, st, ed in enum_hypo_token_tuple_from_tokens(self.tokenizer,
                                                                                   t2, window_size, offset):
                    x = self.nlits.encode_fn(p_tokens, h_first, h_second)
                    es = ESTwoPiece(x, t1, t2, [h_first, h_second], st, ed)
                    es_list.append(es)
        return es_list
