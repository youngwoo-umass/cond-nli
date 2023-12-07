import sys

from contradiction.medical_claims.token_tagging.accuracy_eval import build_binary_prediction, run_test_eval


def main():
    run_name = sys.argv[1]
    target_metric = "f1"
    for target_label in ["contradiction", "neutral"]:
        build_binary_prediction(run_name, target_label, target_metric)
        print("{} {}: {}".format(
            target_label, target_metric,
            run_test_eval(run_name, target_label, target_metric)))


if __name__ == "__main__":
    main()
