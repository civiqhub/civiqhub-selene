# The latent block model

Checked against the code of `develop` at commit `97a0f15`.

## In plain words

The latent block model puts each voter in one group and each item in one block, so that the cells of a group on a block, a co-cluster, look alike: the same rate of votes and the same spread of votes over the levels. {py:class}`~civiqhub_selene.blocks.LatentBlockModel` finds such an arrangement by turns: it describes each co-cluster from the current arrangement, then moves each voter to the group that describes its votes best, then each item to the block that describes its votes best, and starts again until nothing improves. Its estimator is registered as `latent_block_model`, version `0.1.0`.

## Model

### How the votes are drawn

With $Q$ groups ({py:attr}`~civiqhub_selene.blocks.base.BlockModel.n_groups`) and $L$ blocks ({py:attr}`~civiqhub_selene.blocks.base.BlockModel.n_blocks`), the model draws

1. the group $R_i$ of each voter, independently, with the probabilities $\pi_1, \dots, \pi_Q$;
2. the block $C_j$ of each item, independently, with the probabilities $\rho_1, \dots, \rho_L$;
3. given the arrangement $(R, C)$, the state of each shown cell, independently, with the probabilities $F_{R_i C_j s}$ of the response model of its co-cluster: see [The response models](response-models.md).

A cell that is not shown carries no information and enters nowhere ([The data](data.md)). This is the draw of {py:func}`~civiqhub_selene.blocks.synthetic.make_votes` with memberships of $0$ and $1$ only.

### The classified log-likelihood

The engine does not sum the groups and the blocks out. It treats the arrangement as a parameter, and scores an arrangement with its parameters by the classified log-likelihood:

$$
\mathcal{L}_{\mathrm{c}}(R, C) = \sum_{i=1}^{M} w_i \Big( \sum_{j=1}^{P} \sum_{s=0}^{K} S_{ijs} \log F_{R_i C_j s} + \log \pi_{R_i} \Big) + \sum_{j=1}^{P} \log \rho_{C_j}.
$$ (eq-lbm-objective)

Without survey weights, $w_i = 1$ for every voter and {eq}`eq-lbm-objective` is the log-probability of the arrangement and of the states of the shown cells. With mean-one survey weights $w_i$ ({py:attr}`~civiqhub_selene.blocks.VoteMatrix.voter_weights`), it is the weighted pseudo-likelihood of Skinner (1989) and Pfeffermann (1993): each voter counts $w_i$ times. The block term is not weighted: with a single block it is $0$, and the parameters of the response model are then those of a matrix where each voter is repeated in proportion to its weight.

In the counts $N_{qls}$ of the arrangement ([The data](data.md)), the first term is

$$
\sum_{i,j,s} w_i \, S_{ijs} \log F_{R_i C_j s} = \sum_{q=1}^{Q} \sum_{l=1}^{L} \sum_{s=0}^{K} N_{qls} \log F_{qls},
$$

the expected log-likelihood that a response model maximises on each co-cluster.

When the items use several scales, a cell reads the probabilities of the scale of its item: $F_{R_i C_j s}$ becomes $F_{R_i C_j g_j s}$ in {eq}`eq-lbm-objective` and in the scores below, and the first term becomes $\sum_{q,l} \sum_{g \in \mathcal{G}} \sum_{s=0}^{K_g} N_{qlgs} \log F_{qlgs}$, with the counts by scale ([The data](data.md)) and the probabilities by scale of the response model {eq}`eq-response-probit-by-scale`. The code scores the items scale by scale; with one scale, it computes exactly what it computed before items could have several.

## Estimation

The model is estimated by classification expectation-maximisation: it maximises {eq}`eq-lbm-objective` over the arrangement and the parameters together, one block of unknowns at a time. It is not a variational method.

### The scores of a voter and of an item

Given the parameters and the blocks, the score of the voter $i$ for the group $q$ is

$$
A_{iq} = \sum_{j=1}^{P} \sum_{s=0}^{K} S_{ijs} \log F_{q C_j s} + \log \pi_q.
$$ (eq-lbm-voter-score)

Given the parameters and the groups, the score of the item $j$ for the block $l$ is

$$
B_{jl} = \sum_{i=1}^{M} w_i \sum_{s=0}^{K} S_{ijs} \log F_{R_i l s} + \log \rho_l.
$$ (eq-lbm-item-score)

### One iteration

From an arrangement $(R, C)$, an iteration does, in this order:

1. **Counts.** The weighted counts $N_{qls}$ of the arrangement.
2. **Parameters.** The response model updates $F$ from these counts ({py:meth}`~civiqhub_selene.blocks.responses.ResponseModel.update`); the shares become $\pi_q = \frac{1}{M} \sum_{i} w_i \, \mathbf{1}\{R_i = q\}$ and $\rho_l = \frac{1}{P} \sum_{j} \mathbf{1}\{C_j = l\}$, the maximisers of the share terms of {eq}`eq-lbm-objective`, since the weights sum to $M$.
3. **Voters.** With the blocks fixed, each voter moves to its best group by the scores {eq}`eq-lbm-voter-score`, through the sweep below.
4. **Items.** With the new groups fixed, each item moves to its best block by the scores {eq}`eq-lbm-item-score`, through the same sweep.
5. **Objective.** {eq}`eq-lbm-objective` at the new arrangement, with the parameters of step 2. It is the value {py:attr}`~civiqhub_selene.blocks.results.Coclustering.history` notes after the iteration.

### The sweep, and why no cluster empties

A voter, or an item, moves only to a strictly better score, and never leaves a cluster it is the last member of. The elements whose best cluster differs from their current one are visited in increasing order of their index, and the sizes of the clusters are updated as they move. No group and no block ever empties: a model of $Q$ groups and $L$ blocks always returns $Q$ groups and $L$ blocks.

### Why the objective does not decrease

{eq}`eq-lbm-objective` separates: given $C$ and the parameters, it is a sum over the voters of $w_i A_{iq}$ at $q = R_i$, plus terms that do not depend on $R$; given $R$ and the parameters, it is a sum over the items of $B_{jl}$ at $l = C_j$, plus terms that do not depend on $C$. Hence:

- a move of step 3 raises the term of one voter and changes no other term, since $w_i > 0$; the same holds for step 4 and the items;
- step 2 does not lower {eq}`eq-lbm-objective` when the response model maximises the expected log-likelihood of its counts, as the vote rate does, as {py:class}`~civiqhub_selene.blocks.responses.Categorical` does with `smoothing=0`, and as the expectation-maximisation steps of {py:class}`~civiqhub_selene.blocks.responses.OrdinalProbit` do, since each of them does not lower that likelihood.

The value noted after an iteration is {eq}`eq-lbm-objective` at the arrangement of step 4 and the parameters of step 2. The next iteration fits the parameters to that arrangement, which does not lower it, then moves elements, which raises it. The sequence of objectives therefore does not decrease, up to two exceptions: the pseudo-counts of {py:class}`~civiqhub_selene.blocks.responses.Categorical` with `smoothing` above $0$, and the floors of the code (below), can lower it by a tiny amount. The loop of a start stops at the first iteration whose gain is under {py:attr}`~civiqhub_selene.blocks.base.BlockModel.tolerance`, a loss included, or after {py:attr}`~civiqhub_selene.blocks.base.BlockModel.max_iterations` iterations; [The estimation](estimation.md) describes this loop, shared by every block model.

What this does not guarantee: the arrangement reached is a local maximum of {eq}`eq-lbm-objective`, which depends on the start. The model runs several starts ({py:attr}`~civiqhub_selene.blocks.base.BlockModel.n_starts`) and keeps the one with the largest final objective.

### The memberships it reports

{py:class}`~civiqhub_selene.blocks.results.SingleCoclustering` carries the final arrangement ({py:attr}`~civiqhub_selene.blocks.results.Coclustering.group_index`, {py:attr}`~civiqhub_selene.blocks.results.Coclustering.block_index`) and, as {py:attr}`~civiqhub_selene.blocks.results.Coclustering.group_memberships` and {py:attr}`~civiqhub_selene.blocks.results.Coclustering.block_memberships`, the scores {eq}`eq-lbm-voter-score` and {eq}`eq-lbm-item-score` at the final arrangement, turned into shares:

$$
\hat{\theta}_{iq} = \frac{\exp A_{iq}}{\sum_{h=1}^{Q} \exp A_{ih}}, \qquad \hat{\phi}_{jl} = \frac{\exp B_{jl}}{\sum_{h=1}^{L} \exp B_{jh}}.
$$ (eq-lbm-memberships)

They are a soft reading of the final arrangement, each computed with the other side fixed. They are not the posterior probabilities of the groups under a model where the arrangement is random: the engine does not compute those. They also leave the survey weights out of the voters' side, as the scores of step 3 do.

### Beside the classified likelihood

The likelihood of the votes alone, where the groups and the blocks are summed out, has no exact expectation step for a block model: given the votes, the group of a voter and the block of an item are not independent, so the posterior of $(R, C)$ does not factor into one law per voter and one per item. The literature approximates it, with variational methods among others. The engine does not: {py:class}`~civiqhub_selene.blocks.LatentBlockModel` maximises the classified likelihood, and the mixed-membership model, where each voter has a profile, computes its expectation step exactly ([The mixed-membership model](mixed-membership.md)).

## An example

Sixty synthetic voters in three planted groups and twelve items in two planted blocks:

```python
from itertools import pairwise

from civiqhub_selene.blocks import LatentBlockModel, Partition
from civiqhub_selene.blocks.synthetic import make_votes, single_memberships

votes, truth = make_votes(single_memberships((20, 20, 20)), single_memberships((6, 6)), seed=3)
result = LatentBlockModel(n_groups=3, n_blocks=2).estimate(votes, seed=0)
coclustering = result.value
print(result.provenance.method)
print(round(coclustering.objective, 2))
print(coclustering.history.n_iterations, coclustering.history.converged)

objectives = coclustering.history.objectives
print(all(later >= earlier for earlier, later in pairwise(objectives)))

planted = Partition.from_voter_labels(votes.voters, truth.voter_memberships.argmax(axis=1))
print(round(coclustering.groups.agreement_with(planted), 3))
```

```text
latent_block_model 0.1.0
-890.06
10 True
True
0.983
```

The objective rose at every iteration of the start that was kept, and the groups found agree with the planted groups for 59 voters out of 60, after the best renumbering.

## What the code does exactly

| Formula or rule | Code |
| --- | --- |
| Model and {eq}`eq-lbm-objective`, weighted form | Docstring, `src/civiqhub_selene/blocks/single.py:104-130` |
| One iteration | `LatentBlockModel._step`, `blocks/single.py:181-215` |
| Weighted counts | `count_states`, `blocks/_numerics.py:47-77`; call at `blocks/single.py:186-188` |
| $\pi_q$, $\rho_l$ | `blocks/single.py:190-192` |
| {eq}`eq-lbm-voter-score` | `_voter_scores`, `blocks/single.py:48-54` |
| {eq}`eq-lbm-item-score` | `_item_scores`, `blocks/single.py:57-71` |
| The sweep | `_sweep`, `blocks/single.py:74-91` |
| Objective after the moves | `blocks/single.py:199-212` |
| {eq}`eq-lbm-memberships` | `_finish` and `_softmax`, `blocks/single.py:94-96`, `:217-248` |
| Start: arrangement of `initialization`, equal shares | `_start`, `blocks/single.py:169-179` |
| Starts, best start, stop rule | `IterativeEstimator._compute`, `blocks/iterative.py:83-98`; `iterate`, `blocks/_numerics.py:199-227` |
| Value | {py:class}`~civiqhub_selene.blocks.results.SingleCoclustering`, `blocks/results.py:398-432` |

The code also does what the formulas do not say:

- **Floors.** A share $\pi_q$ or $\rho_l$ is floored at $10^{-300}$ before its logarithm, and so is each $F_{qls}$ ([The response models](response-models.md)).
- **The table of co-clusters** of the value is computed from counts without weights, and its parameters are those of the last iteration.
- **The objective of a start** is that of its last iteration; the value reports the objective of the start that was kept, and its history.

## Limits

- One group per voter, for every block: a voter who agrees with one group on some items and with another group on others is put in one of them. The mixed-membership model lifts this.
- The result is a local maximum that depends on the start.
- The memberships reported are a reading of the final arrangement, not posterior probabilities; the result carries no uncertainty.
- The weighted pseudo-likelihood keeps the block term unweighted; the integrated likelihood on a weighted objective is discussed with the choice of the shape.

## References

- Govaert, G. and Nadif, M. (2003). Clustering with block mixture models. *Pattern Recognition*, 36(2), 463–473. doi:10.1016/S0031-3203(02)00074-2. The code cites it for the model.
- Corneli, M., Bouveyron, C. and Latouche, P. (2020), for the ordinal response model: see [The response models](response-models.md).
- Skinner, C. J. (1989) and Pfeffermann, D. (1993), for the weighted pseudo-likelihood, as the code cites them.
