# Hyper-parameters for WN18RR
## Grid-Search
The hyper-parameters were tuned by grid-search on the validation sets. The runs with best validation MRR are selected. The base model is [ComplEx](https://www.jmlr.org/papers/volume18/16-563/16-563.pdf) with [N3 regularizer](https://arxiv.org/pdf/1806.07297.pdf). The embedding initialization is 1e-3. The optimizer is [Adagrad](https://pytorch.org/docs/stable/generated/torch.optim.Adagrad.html).

For ComplEx without relation prediction, we searched
```python
learning_rate=[1e-1, 1e-2],
rank=[100, 500, 1000],
batch_size=[100, 500, 1000],
lmbda=[0.005, 0.01, 0.05, 0.1, 0.5, 1],
w_rel=[0],
score_rel=[False], 
```

For ComplEx with relation prediction, we searched
```python
learning_rate=[1e-1, 1e-2],
rank=[100, 500, 1000],
batch_size=[100, 500, 1000],
lmbda=[0.005, 0.01, 0.05, 0.1, 0.5, 1],
w_rel= [0.005, 0.001, 0.05, 0.1, 0.5, 1], # weighting of the relation prediction objective
score_rel=[True], # turn on the relation prediction objective
```

## Best Run for WN18RR
```python
rank=[1000],
learning_rate=[1e-1],
batch_size=[100],
lmbda=[0.10],
w_rel=[0.05], # weighting of the relation prediction objective
score_rel=[True], # turn on the relation prediction objective
```
This configuration of hyper-parameters should reproduce [the results](https://github.com/facebookresearch/ssl-relation-prediction#wn18rr). The model was trained for 500 epochs, which took about 7 hours.

## Trained Embeddings for the Best Run
You can download the trained embeddings for the best run from [here](https://dl.fbaipublicfiles.com/ssl-relation-prediction/complex/wn18rr.zip). Unzip the model by
```
unzip wn18rr.zip
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
model = ComplEx(sizes=[40943,22,40943], rank=1000, init_size=1e-3)
state_dict = torch.load('best_valid.model', map_location='cpu') # load on cpu
model.load_state_dict(state_dict)
```

## Training Curve for the Best Run
![](/doc/img/wn18rr.png)