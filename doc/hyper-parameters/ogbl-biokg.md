# Hyper-parameters for ogbl-biokg
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
w_rel=[2, 1, 0.5, 0.25, 0.125], # weighting of the relation prediction objective
score_rel=[True], # turn on the relation prediction objective
```
We used the same grid for all the CoDEx datasets. Better performance can be found by customizing the grid for each dataset.

## Best Run for ogbl-biokg
```python
learning_rate=[1e-1],
batch_size=[500],
lmbda=[0.01],
w_rel=[0.25], # weighting of the relation prediction objective
score_rel=[True], # turn on the relation prediction objective
```
This configuration of hyper-parameters should reproduce [the results](https://github.com/facebookresearch/ssl-relation-prediction#ogbl-biokg).

## Summary
We used the same grid as for CoDEx datasets. Compared to full ranking for conventional KBC datasets, the evaluation in ogbl-biokg uses a fixed sampled set of 500 entities for both (h,r,?) and (?,r,t) queries. 