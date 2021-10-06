# Relation Prediction as an Auxiliary Training Objective for Knowledge Base Completion

This repo provides the code for the paper [Relation Prediction as an Auxiliary Training Objective for Improving Multi-Relational Graph Representations](https://openreview.net/pdf?id=Qa3uS3H7-Le). Incorporating relation prediction into the 1vsAll objective leads to a new self-supervised training objective for knowledge base completion (KBC), which brings significant performance improvement with 3-10 lines of code. Unleash the power of your KBC models with relation prediction objective!

![](./doc/img/ssl_rp_repo.png)

## Prepare Datasets
### Download
Download datasets and put them under `src_data`. The folder should look like this TODO: tree command output
```
src_data/FB15K-237/train # Tab separated file
src_data/FB15K-237/valid # Tab separated file
src_data/FB15K-237/test # Tab separated file
```

As an option, you can download together UMLS, Nations, Kinship, FB15K-237, WN18RR from [here](https://github.com/villmow/datasets_knowledge_embedding) and aristo-v4 from [here](https://allenai.org/data/tuple-kb). You can also download some datasets separately on [WN18RR](https://github.com/TimDettmers/ConvE/blob/master/WN18RR.tar.gz) and [FB15K-237](https://www.microsoft.com/en-us/download/details.aspx?id=52312). 

### Preprocessing
```
mkdir data/
python preprocess_datasets.py
```

## Train the model 
Use option `score_rel` to turn on the auxiliary objective of relation prediction. Use option `w_rel` to set the weight of the relation prediction objective.

For example, the following command trains a ComplEx model **with** relation prediction on FB15K-237
```
python main.py --dataset FB15K-237 --score_rel True --model ComplEx --rank 1000 --learning_rate 0.1 --batch_size 1000 --lmbda 0.05 --w_rel 4 --max_epochs 100
```
And the following command trains a ComplEx model **without** relation prediction on FB15K-237
```
python main.py --dataset FB15K-237 --score_rel False --model ComplEx --rank 1000 --learning_rate 0.1 --batch_size 1000 --lmbda 0.05 --w_rel 4 --max_epochs 100
```

## Dependency
- pytorch
- wandb

## Acknowledgement
This repo is based on the repo [kbc](https://github.com/facebookresearch/kbc), which provides efficient implementations of 1vsAll for ComplEx and CP. Our repo also includes implementations for other models: TransE, RESCAL, and TuckER. 

## BibTex
If you find this repo useful, please cite us
```
@inproceedings{
chen2021relation,
title={Relation Prediction as an Auxiliary Training Objective for Improving Multi-Relational Graph Representations},
author={Yihong Chen and Pasquale Minervini and Sebastian Riedel and Pontus Stenetorp},
booktitle={3rd Conference on Automated Knowledge Base Construction},
year={2021},
url={https://openreview.net/forum?id=Qa3uS3H7-Le}
}
```
# License
This repo is CC-BY-NC licensed, as found in the LICENSE file.