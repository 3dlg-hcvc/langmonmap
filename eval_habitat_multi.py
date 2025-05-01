from eval.habitat_multi_evaluator import HabitatMultiEvaluator
from config import load_eval_config
from eval.actor import MONActor

def main():
    # Load the evaluation configuration
    eval_config = load_eval_config()
    # Create the HabitatEvaluator object
    evaluator = HabitatMultiEvaluator(eval_config.EvalConf, MONActor(eval_config.EvalConf))
    evaluator.evaluate()

if __name__ == "__main__":
    main()
