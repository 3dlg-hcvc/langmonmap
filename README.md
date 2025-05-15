<p align="center">
  <h1 align="center">LangNavBench: Evaluation of Natural Language Understanding in Semantic Navigation</h1>

This repository contains the code for the paper "LangNavBench: Evaluation of Natural Language
Understanding in Semantic Navigation". We add instructions on how to run the experiments reported in the paper.

## Abstract
Recent progress in large vision–language models has driven improvements in language-based semantic navigation, where an embodied agent must reach a target object described in natural language. Despite these advances, we still lack a clear, language-focused benchmark for testing how well such agents ground the words in their instructions.
We address this gap with LangNav, an open-set dataset specifically created to test an agent’s ability to locate objects described at different levels of detail, from broad category names to fine attributes and object–object relations. Every description in LangNav was manually checked, yielding a lower error rate than existing lifelong- and semantic-navigation datasets. On top of LangNav we build LangNavBench, a benchmark that measures how well current semantic-navigation methods understand and act on these descriptions while moving toward their targets. LangNavBench allows to systematically compare models on their handling of attributes, spatial and relational cues, and category hierarchies, offering the first thorough, language-centred evaluation of embodied navigation systems. We also present  Multi-Layered Feature Map (MLFM), a method that builds a queryable multi-layered semantic map, particularly effective when dealing with small objects or instructions involving spatial relations. MLFM outperforms state-of-the-art mapping-based navigation baselines on the LangNav dataset.

## Code setup

### 1. Clone the repository
```
# https
git clone 
# or ssh
git clone 
cd langmonmap/
```
### 2. Install dependencies
Create a conda environment and install Habitat-sim v0.2.5
```
# create conda and install habitat
conda create -n langnav python=3.9 cmake=3.14.0 habitat-sim=0.2.5 headless -c conda-forge  -c aihabitat
conda activate langnav
```
Install Habitat-lab v0.2.5
```
python -m pip install habitat-lab==0.2.520230802
python -m pip install habitat-baselines==0.2.520230802
```

```
cd langmonmap
python -m pip install gdown torch torchvision torchaudio meson
python -m pip install -r requirements.txt
```


Manually install newer `timm` version:
```
python3 -m pip install --upgrade timm>=1.0.7
```
YOLOV7:
```
git clone https://github.com/WongKinYiu/yolov7
```
Build planning utilities:
```
python3 -m pip install ./planning_cpp/
```
### 3. Download the model weights
```
mkdir -p weights/
```
SED extracted weights:
```
gdown 1D_RE4lvA-CiwrP75wsL8Iu1a6NrtrP9T -O weights/clip.pth
```
YOLOV7 weights and MobileSAM weights:
```
wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6e.pt -O weights/yolov7-e6e.pt
wget https://github.com/ChaoningZhang/MobileSAM/raw/refs/heads/master/weights/mobile_sam.pt -O weights/mobile_sam.pt
```
### 4. Download the habitat data


## Running the code
### 1. Run the example
You can run the code on an example, visualized in [rerun.io](https://rerun.io/) with:
#### Docker
You will need to have [rerun.io](https://rerun.io/) installed on the host for visualization.
Ensure the docker is running and you are in the container as described in the [Docker setup](#setup-docker). Then launch
the rerun viewer **on the host** (not inside the docker) with:
```
rerun
```
and launch the example in the container with:
``` 
python3 habitat_test.py --config/mon/base_conf_sim.yaml
```
#### Local
Open the rerun viewer and example from the root of the repository with:
```
rerun
python3 habitat_test.py --config/mon/base_conf_sim.yaml
```
### 2. Run the evaluation
You can reproduce the evaluation results from the paper for single- and multi-object navigation.
#### Single-object navigation
```
python3 eval_habitat.py --config config/mon/eval_conf.yaml
```
This will run the evaluation and save the results in the `results/` directory. You can read the results with:
```
python3 read_results.py --config config/mon/eval_conf.yaml
```
#### Multi-object navigation
```
python3 eval_habitat_multi.py --config config/mon/eval_multi_conf.yaml
```
This will run the evaluation and save the results in the `results_multi/` directory. You can read the results with:
```
python3 read_results_multi.py --config config/mon/eval_multi_conf.yaml
```
#### Dataset generation
While we provide the generated dataset for the evaluation of multi-object navigation, we also release the code to
generate the datasets with varying parameters. You can generate the dataset with
```
python3 eval/dataset_utils/gen_multiobject_dataset.py
```
and change the parameters such as number of objects per episode in the corresponding file.

## Citation
If you use this code in your research, please cite our paper:
```
@misc{langnavbenchraychaudhuri,
      title={LangNavBench: Evaluation of Natural Language Understanding in Semantic Navigation},
}
```
