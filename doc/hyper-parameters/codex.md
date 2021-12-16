# Hyper-parameters for CoDEx-S, CoDEx-M and CoDEx-L
## Grid-Search
The hyper-parameters were tuned by grid-search on the validation sets. The runs with best validation MRR are selected. The base model is [ComplEx](https://www.jmlr.org/papers/volume18/16-563/16-563.pdf) with [N3 regularizer](https://arxiv.org/pdf/1806.07297.pdf). The rank for ComplEx is 1000. The embedding initialization is 1e-3. The optimizer is [Adagrad](https://pytorch.org/docs/stable/generated/torch.optim.Adagrad.html).

For ComplEx without relation prediction, we searched
```python
learning_rate=[1e-1, 1e-2],
batch_size=[100, 500, 1000],
lmbda=[0.0005, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 0],
w_rel=[0],
score_rel=[False], 
```

For ComplEx with relation prediction, we searched
```python
learning_rate=[1e-1, 1e-2],
batch_size=[100, 500, 1000],
lmbda=[0.0005, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 0],
w_rel=[4, 2, 1, 0.5, 0.25, 0.125], # weighting of the relation prediction objective
score_rel=[True], # turn on the relation prediction objective
```
We used the same grid for all the CoDEx datasets. Better performance can be found by customizing the grid for each dataset.

## Best Run for CoDEx-S
```python
learning_rate=[1e-2],
batch_size=[1000],
lmbda=[0.01],
w_rel=[0.125], # weighting of the relation prediction objective
score_rel=[True], # turn on the relation prediction objective
```
This configuration of hyper-parameters should reproduce [the results](https://github.com/facebookresearch/ssl-relation-prediction#codex-s).
Note that the best `w_rel=0.125` is actually at the boundary of our grid. We guess further lowering down `w_rel` probably will lead to better results. The model was trained for 500 epochs, which took about 0.75 hours.

#### Trained Embeddings for the Best Run on CoDEx-S
You can download the trained embeddings for the best run from [here](https://dl.fbaipublicfiles.com/ssl-relation-prediction/complex/codex-s.zip). Unzip the model by
```
unzip codex-s.zip
```
After unzipping you will get 3 files: `best_valid.model` (Pytorch model file), `ent_id` (Enitity IDs)and `rel_id` (Relation IDs). You can load the embeddings by running the following python script
```
import torch
state_dict = torch.load('best_valid.model', map_location='cpu') # load on cpu
entity_embedding = state_dict[''embeddings.0.weight'] # get entity embeddings
relation_embedding = state_dict[''embeddings.1.weight'] # get relation embeddings
``` 
Or you can load the `ComplEx` model using our codebase by running the following python scripts under `src`
```
from models import ComplEx
model = ComplEx(sizes=[2034,84,2034], rank=1000, init_size=1e-3)
state_dict = torch.load('best_valid.model', map_location='cpu') # load on cpu
model.load_state_dict(state_dict)
```

#### Training Curve for the Best Run on CoDEx-S
![](/doc/img/codex-s.png)

## Best Run for CoDEx-M
```python
learning_rate=[1e-1],
batch_size=[500],
lmbda=[0.01],
w_rel=[0.125], # weighting of the relation prediction objective
score_rel=[True], # turn on the relation prediction objective
```
This configuration of hyper-parameters should reproduce [the results](https://github.com/facebookresearch/ssl-relation-prediction#codex-m).
Note that the best `w_rel=0.125` is actually at the boundary of our grid. We guess further lowering down `w_rel` probably will lead to better results.  The model was trained for 500 epochs, which took about 9.5 hours.

#### Trained Embeddings for the Best Run on CoDEx-M
You can download the trained embeddings for the best run from [here](https://dl.fbaipublicfiles.com/ssl-relation-prediction/complex/codex-m.zip). Unzip the model by
```
unzip codex-m.zip
```
After unzipping you will get 3 files: `best_valid.model` (Pytorch model file), `ent_id` (Enitity IDs)and `rel_id` (Relation IDs). You can load the embeddings by running the following python script
```
import torch
state_dict = torch.load('best_valid.model', map_location='cpu') # load on cpu
entity_embedding = state_dict[''embeddings.0.weight'] # get entity embeddings
relation_embedding = state_dict[''embeddings.1.weight'] # get relation embeddings
``` 
Or you can load the `ComplEx` model using our codebase by running the following python scripts under `src`
```
from models import ComplEx
model = ComplEx(sizes=[17050,102,17050], rank=1000, init_size=1e-3)
state_dict = torch.load('best_valid.model', map_location='cpu') # load on cpu
model.load_state_dict(state_dict)
```

#### Training Curve for the Best Run on CoDEx-M
![](/doc/img/codex-m.png)


## Best Run for CoDEx-L
```python
learning_rate=[1e-1],
batch_size=[1000],
lmbda=[0.05],
w_rel=[0.25], # weighting of the relation prediction objective
score_rel=[True], # turn on the relation prediction objective
```
This configuration of hyper-parameters should reproduce [the results](https://github.com/facebookresearch/ssl-relation-prediction#codex-l).

## Best Run Entity/Relation Embeddings
...
## Summary
Overall, relation prediction doesn't help CoDEx datasets as much as helping Kinship, FB15K-237 or Aristo-v4. Potential reasons might be the small number of predicates or inappropriate hyper-parameter range. Another possible reason is that the construction process of CoDEx filters out "each relation whose entity pair set overlapped with that of another relation more than 50% of the time" while the relation prediction objective is found to be most effective at distinguishing such "convoluted" relations, which share lots of entity pair set overlap.