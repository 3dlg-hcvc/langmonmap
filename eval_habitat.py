from eval.habitat_evaluator import HabitatEvaluator
from config import load_eval_config
from eval.actor import MONActor

def main():
    # Load the evaluation configuration
    eval_config = load_eval_config()
    # Create the HabitatEvaluator object
    evaluator = HabitatEvaluator(eval_config.EvalConf, MONActor(eval_config.EvalConf))
    evaluator.evaluate()

if __name__ == "__main__":
    main()
