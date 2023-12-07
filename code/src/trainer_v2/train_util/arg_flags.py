import argparse


flags_parser = argparse.ArgumentParser(description='')
flags_parser.add_argument("--input_files", )
flags_parser.add_argument("--eval_input_files", )
flags_parser.add_argument("--init_checkpoint", )
flags_parser.add_argument("--config_path", )
flags_parser.add_argument("--output_dir", )
flags_parser.add_argument("--action", default="train")
flags_parser.add_argument("--job_id", type=int, default=-1)
flags_parser.add_argument("--run_name", default=None)
flags_parser.add_argument("--predict_save_path", default="prediction.pickle")
