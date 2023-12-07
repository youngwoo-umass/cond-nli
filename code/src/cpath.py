import os
from os.path import dirname

from utils.misc_lib import exist_or_mkdir

project_root = os.path.abspath(dirname(dirname(os.path.abspath(__file__))))
data_path = os.path.join(project_root, 'data')
output_path = os.path.join(project_root, 'output')

exist_or_mkdir(data_path)
exist_or_mkdir(output_path)

common_model_dir_root = os.path.join(output_path, 'model')


def at_output_dir(folder_name, file_name):
    return os.path.join(output_path, folder_name, file_name)


def get_bert_config_path():
    return os.path.join(data_path, 'bert_config.json')
