from settings.options import EvalOptions
from statistics.evaluation import eval_utils

args = EvalOptions().parse()

available_evals = {'mAP': eval_utils.mAP_stats, 'struct_segment': eval_utils.segment_eval_struct_output, 'segment': eval_utils.segment}

available_evals[args.eval_protocol](args)
