from eval.mlsfm_policy import HabitatMultiEvaluator
from config import load_eval_config
from eval.actor import MONActor
import random
import numpy as np
import torch

seed = 120
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

def main():
    # Load the evaluation configuration
    eval_config = load_eval_config()
    # Create the HabitatEvaluator object
    evaluator = HabitatMultiEvaluator(eval_config.EvalConf, MONActor(eval_config.EvalConf))
    evaluator.evaluate()

if __name__ == "__main__":
    main()
