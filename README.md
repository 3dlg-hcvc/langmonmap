<p align="center">
  <h1 align="center">MLFM: Multi-Layered Feature Maps for Richer Language Understanding in Zero-Shot Semantic Navigation</h1>

This repository contains the code for the paper "MLFM: Multi-Layered Feature Maps for Richer Language Understanding in Zero-Shot Semantic Navigation". We add instructions on how to run the experiments reported in the paper. [[webpage]](https://3dlg-hcvc.github.io/langmonmap/)

## Abstract
Recent progress in large vision-language models has driven improvements in language-based semantic navigation, where an embodied agent must reach a target object described in natural language. Yet we still lack a clear, language-focused evaluation framework to test how well agents ground the words in their instructions. We address this gap by proposing LangNav, an open-vocabulary multi-object navigation dataset with natural language goal descriptions (e.g. 'go to the red short candle on the table') and corresponding fine-grained linguistic annotations (e.g., attributes: color=red, size=short; relations: support=on). These labels enable systematic evaluation of language understanding. To evaluate on this setting, we extend multi-object navigation task setting to Language-guided Multi-Object Navigation (LaMoN), where the agent must find a sequence of goals specified using language. Furthermore, we propose Multi-Layered Feature Map (MLFM), a novel method that builds a queryable, multi-layered semantic map from pretrained vision-language features and proves effective for reasoning over fine-grained attributes and spatial relations in goal descriptions. Experiments on LangNav show that MLFM outperforms state-of-the-art zero-shot mapping-based navigation baselines.

## Code setup
We used Ubuntu 20.04.6 with cuda version 12.2 to install our code.

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
python eval_mlsfm.py --config config/lnav/mlfm_conf.yaml
```
The evaluation run will save out the results in the `results/` directory. You can read the results with:
```
python read_results_mlsfm.py --config config/lnav/mlfm_conf.yaml
```
#### Running experiments reported in the paper
You can find all the yaml files under ``config/lnav/paper'' for running the experiments reported in the paper.

### Acknowledgements
Our repository is built on top of the open-sourced [OneMap repo](https://github.com/KTH-RPL/OneMap).
We use assets from [HSSD](https://huggingface.co/datasets/hssd/hssd-hab) to build our dataset.

## Citation
If you use this code in your research, please cite our paper:
```
@misc{raychaudhuri2025mlfm,
      title={MLFM: Multi-Layered Feature Maps for Richer Language Understanding in Zero-Shot Semantic Navigation}, 
      author={Sonia Raychaudhuri and Enrico Cancelli and Tommaso Campari and Lamberto Ballan and Manolis Savva and Angel X. Chang},
      year={2025},
      eprint={2507.07299},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2507.07299}, 
}
```
