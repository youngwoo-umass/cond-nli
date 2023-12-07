from typing import List

from contradiction.medical_claims.token_tagging.batch_solver_common import score_tokens_with_batch_solver
from contradiction.medical_claims.token_tagging.path_helper import get_raw_score_save_path
from contradiction.medical_claims.token_tagging.problem_loader import CondNLISentPair, load_bioclaim_problems
from trainer_v2.chair_logging import c_log
from trainer_v2.custom_loop.run_config2 import get_run_config_for_predict


def solve_cond_nli(args, solver_factory, target_label: str, split):
    run_config = get_run_config_for_predict(args)
    run_name = args.run_name
    c_log.info(f"solve_cond_nli({run_name}, {target_label})")
    solver = solver_factory(run_config)
    run_config.print_info()
    save_path = get_raw_score_save_path(split, run_name, target_label)
    problems: List[CondNLISentPair] = load_bioclaim_problems(split)
    c_log.info("Using {} problems".format(len(problems)))
    score_tokens_with_batch_solver(problems, save_path, solver)


