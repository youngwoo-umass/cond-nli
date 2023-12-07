from typing import List, Dict

from contradiction.medical_claims.token_tagging.path_helper import get_raw_score_save_path, \
    get_binary_prediction_save_path
from contradiction.medical_claims.token_tagging.prec_recall import get_acc_prec_recall, i2b
from contradiction.medical_claims.token_tagging.problem_loader import CondNLISentPairWithLabel, \
    load_bioclaim_problems_w_labels
from utils.list_lib import index_by_fn
from utils.misc_lib import load_jsonl, save_as_jsonl


def apply_threshold(threshold, s_list):
    return [1 if s >= threshold else 0 for s in s_list]


def convert_to_binary(paired_scores: List[Dict],
                      threshold) -> List[Dict]:
    output = []
    for d in paired_scores:
        out_d = {
            'pair_id': d['pair_id'],
            'sent1_pred': apply_threshold(threshold, d['scores1']),
            'sent2_pred': apply_threshold(threshold, d['scores2']),
        }
        output.append(out_d)
    return output


def get_best_threshold(
        dev_scores: List[Dict],
        labels: List[CondNLISentPairWithLabel],
        target_label,
        metric_to_opt):
    print("get_best_threshold")
    scores_flat, labels_flat = join_score_labels(dev_scores, labels, target_label)

    def apply_threshold_eval(t):
        preds: List[int] = apply_threshold(t, scores_flat)
        return get_acc_prec_recall(i2b(preds), i2b(labels_flat))

    search_interval = range(102)

    max_t = None
    max_f1 = -1
    for i in search_interval:
        t = 0.01 * i
        metrics = apply_threshold_eval(t)
        if metrics[metric_to_opt] > max_f1:
            max_f1 = metrics[metric_to_opt]
            max_t = t
        # print(t, metrics[metric_to_opt], metrics['precision'], metrics['recall'])
    print("Maximum {} = {} at t={}".format(metric_to_opt, max_f1, max_t))
    return max_t


def join_score_labels(dev_scores, labels, target_label):
    label_d = index_by_fn(lambda x: x.pair_id, labels)
    scores_flat = []
    labels_flat = []
    for d in dev_scores:
        try:
            pair_id = d['pair_id']
            pair_label = label_d[pair_id]
            for sent_no in ["1", "2"]:
                scores: List[float] = d["scores" + sent_no]
                label_name = f"sent{sent_no}_{target_label}"
                label: List[int] = pair_label.get_label_by_name(label_name)

                if not any(label):
                    continue

                if len(label) != len(scores):
                    print("WARNING number of tokens differ: {} != {}".format(len(labels), len(scores)))

                scores_flat.extend(scores)
                labels_flat.extend(label)
        except KeyError:
            pass
    return scores_flat, labels_flat


def join_pred_labels(paired_preds, labels, target_label):
    label_d = index_by_fn(lambda x: x.pair_id, labels)
    preds_flat = []
    labels_flat = []
    for d in paired_preds:
        try:
            pair_id = d['pair_id']
            pair_label = label_d[pair_id]
            for sent_no in ["1", "2"]:
                pred: List[int] = d[f"sent{sent_no}_pred"]
                label_name = f"sent{sent_no}_{target_label}"
                label: List[int] = pair_label.get_label_by_name(label_name)

                if len(label) != len(pred):
                    print("WARNING number of tokens differ: {} != {}".format(len(labels), len(pred)))

                if not any(label):
                    continue

                preds_flat.extend(pred)
                labels_flat.extend(label)
        except KeyError:
            pass

    print("{} predictions".format(len(preds_flat)))
    return preds_flat, labels_flat


def build_binary_prediction(run_name, target_label, target_metric):
    labels = load_bioclaim_problems_w_labels("val")
    dev_scores = load_jsonl(get_raw_score_save_path("val", run_name, target_label))
    threshold = get_best_threshold(dev_scores, labels, target_label, target_metric)

    test_pred = load_jsonl(get_raw_score_save_path("test", run_name, target_label))
    predictions = convert_to_binary(test_pred, threshold)
    save_as_jsonl(predictions, get_binary_prediction_save_path("test", run_name, target_label))


def run_test_eval(run_name, target_label, target_metric):
    predictions = load_jsonl(get_binary_prediction_save_path("test", run_name, target_label))
    labels = load_bioclaim_problems_w_labels("test")
    preds_flat, labels_flat = join_pred_labels(predictions, labels, target_label)
    score_d = get_acc_prec_recall(i2b(preds_flat), i2b(labels_flat))
    return score_d[target_metric]