from eval.habitat_evaluator import HabitatEvaluator
from config import load_eval_config
# from eval import MONActor
from eval.habitat_evaluator import Result


def main():
    # Load the evaluation configuration
    eval_config = load_eval_config()
    # Create the HabitatEvaluator object
    evaluator = HabitatEvaluator(eval_config.EvalConf, None)
    data = evaluator.read_results('results/', Result.SUCCESS.name)
    # print(data[data['state'] == 2].index.tolist())
    # print(data[data['scene'] == 'hm3d/val/00880-Nfvxx8J5NCo/Nfvxx8J5NCo.basis.glb'][data['state'] == 5].index.tolist())
    # print(data[data['state'] == 5].index.tolist())
    # print(len(data[data['obj'] == 'bed'].index.tolist()))


if __name__ == "__main__":
    main()
