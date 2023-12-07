import os

from cpath import output_path
from utils.misc_lib import exist_or_mkdir


def get_raw_score_save_path(split, run_name, token_type):
    save_name = f"{split}_{run_name}_{token_type}"
    exist_or_mkdir(os.path.join(output_path, "scores"))
    save_path = os.path.join(output_path, "scores", save_name + ".txt")
    return save_path


def get_binary_prediction_save_path(split, run_name, token_type):
    save_name = f"{split}_{run_name}_{token_type}"
    exist_or_mkdir(os.path.join(output_path, "binary_predicitons"))
    save_path = os.path.join(output_path, "binary_predicitons", save_name + ".txt")
    return save_path
