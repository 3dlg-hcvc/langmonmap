<p align="center">
  <h1 align="center">LangNavBench: Evaluation of Natural Language Understanding in Semantic Navigation</h1>

This repository contains the code for the paper "LangNavBench: Evaluation of Natural Language
Understanding in Semantic Navigation". We add instructions on how to run the experiments reported in the paper. [[webpage]](https://3dlg-hcvc.github.io/langmonmap/)

## Abstract
Recent progress in large vision–language models has driven improvements in language-based semantic navigation, where an embodied agent must reach a target object described in natural language. Despite these advances, we still lack a clear, language-focused benchmark for testing how well such agents ground the words in their instructions.
We address this gap with LangNav, an open-set dataset specifically created to test an agent’s ability to locate objects described at different levels of detail, from broad category names to fine attributes and object–object relations. Every description in LangNav was manually checked, yielding a lower error rate than existing lifelong- and semantic-navigation datasets. On top of LangNav we build LangNavBench, a benchmark that measures how well current semantic-navigation methods understand and act on these descriptions while moving toward their targets. LangNavBench allows to systematically compare models on their handling of attributes, spatial and relational cues, and category hierarchies, offering the first thorough, language-centred evaluation of embodied navigation systems. We also present  Multi-Layered Feature Map (MLFM), a method that builds a queryable multi-layered semantic map, particularly effective when dealing with small objects or instructions involving spatial relations. MLFM outperforms state-of-the-art mapping-based navigation baselines on the LangNav dataset.

## Code setup

### 1. Clone the repository
```
# https
git clone https://github.com/3dlg-hcvc/langmonmap.git
# or ssh
git clone git@github.com:3dlg-hcvc/langmonmap.git
```
### 2. Install dependencies
Create a conda environment
```
# create conda environment
conda create -n langnav python=3.9 cmake=3.14.0
conda activate langnav
```

Install Habitat-sim v0.2.5, following instructions from [here](https://github.com/facebookresearch/habitat-sim/tree/v0.2.5). In case you encounter issues with Habitat-sim installation, please refer [here](https://github.com/facebookresearch/habitat-sim/tree/v0.2.5?tab=readme-ov-file#common-testing-issues) or [here](https://github.com/facebookresearch/habitat-sim/issues) for common issues, or reach out to the Habitat team.
```
conda install habitat-sim=0.2.5 -c conda-forge -c aihabitat
```

Install Habitat-lab
```
python -m pip install habitat-lab==0.2.520230802
python -m pip install habitat-baselines==0.2.520230802
```

Install dependencies
```
cd langmonmap
python -m pip install -r requirements.txt
python -m pip install --upgrade timm==1.0.15
```

YOLOV7:
```
git clone https://github.com/WongKinYiu/yolov7
```

Build planning utilities:
```
sudo apt-get install libeigen3-dev
python3 -m pip install ./planning_cpp/
```

### 3. Download the model weights
```
mkdir -p weights/
```
Download SED Clip weights, YOLOV7 weights and MobileSAM weights from [OneMap](https://github.com/KTH-RPL/OneMap?tab=readme-ov-file#3-download-the-model-weights) repository and place it under weights/.

### 4. Download scenes data
Follow instructions for Habitat Synthetic Scenes Dataset (HSSD) and download from [here](https://huggingface.co/datasets/hssd/hssd-hab).
Link the scenes in ``datasets/scene_datasets/fphab/''.
```
mkdir -p datasets/scene_datasets
cd datasets/scene_datasets
ln -s <path_to_hssd> fphab
```

### Download LangNav dataset
Follow HuggingFace [LangNav](https://huggingface.co/datasets/3dlg-hcvc/langnav) dataset to download the data splits.
Place inside ``datasets/langnav''. Please refer to the HF [documentation](https://huggingface.co/docs/hub/en/datasets-downloading#using-git) to know more about downloading HF datasets.

## Running the code
### 1. Run evaluation
You can run the evaluation on the test split with:
```
python eval_mlfm.py --config config/lnav/mlfm_conf.yaml
```
The evaluation run will save out the results in the `results/` directory. You can read the results with:
```
python read_results_mlfm.py --config config/lnav/mlfm_conf.yaml
```
#### Running experiments reported in the paper
You can find all the yaml files under ``config/lnav/paper'' for running the experiments reported in the paper.

### Acknowledgements
Our repository is built on top of the open-sourced [OneMap repo](https://github.com/KTH-RPL/OneMap).
We use assets from [HSSD](https://huggingface.co/datasets/hssd/hssd-hab) to build our dataset.

## Citation
If you use this code in your research, please cite our paper:
```
@misc{raychaudhuri2025langnavbench,
      title={LangNavBench: Evaluation of Natural Language Understanding in Semantic Navigation}, 
      author={Sonia Raychaudhuri and Enrico Cancelli and Tommaso Campari and Lamberto Ballan and Manolis Savva and Angel X. Chang},
      year={2025},
      eprint={2507.07299},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2507.07299}, 
}
```
