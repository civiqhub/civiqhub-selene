# The data

Checked against the code of `develop` at commit `97a0f15`.

## In plain words

Every block model reads one object: a matrix of votes, one row per voter and one column per item. A cell of the matrix is in one of three states: the item was not shown to the voter, it was shown and the voter did not vote, or the voter voted at one level of an ordered scale. The models never read the matrix cell by cell: they read how many cells of each group of voters and each block of items are in each state. These counts are all a model needs to know about the votes.

## Model

### The scale

A score is the answer of a vote, such as $-2$ for "strongly disagree" up to $+2$ for "strongly agree". The scale {py:class}`~civiqhub_selene.blocks.OrdinalScale` lists the scores in order ({py:attr}`~civiqhub_selene.blocks.OrdinalScale.scores`, from $-2$ to $+2$ by default). The level of a score is its rank on the scale, from $1$ to $K$ ({py:attr}`~civiqhub_selene.blocks.OrdinalScale.n_levels`); {py:meth}`~civiqhub_selene.blocks.OrdinalScale.level_of` and {py:meth}`~civiqhub_selene.blocks.OrdinalScale.score_of` convert one into the other. The models read the levels, never the values of the scores.

### Three states per cell

The matrix {py:class}`~civiqhub_selene.blocks.VoteMatrix` of $M$ voters and $P$ items ({py:attr}`~civiqhub_selene.blocks.VoteMatrix.shape`) holds two arrays: the levels ({py:attr}`~civiqhub_selene.blocks.VoteMatrix.levels`) and the cells shown ({py:attr}`~civiqhub_selene.blocks.VoteMatrix.shown`). With $\Omega$ the set of shown cells, a cell $(i, j)$ is

- **not shown** when $(i, j) \notin \Omega$, whatever its level holds;
- **shown without a vote** when $(i, j) \in \Omega$ and its level is $0$: its state is $s_{ij} = 0$;
- **voted** when $(i, j) \in \Omega$ and its level is $k \geq 1$: its state is $s_{ij} = k$.

A vote at the neutral score is a vote, at the middle level. A cell that is not shown is never read as a zero, nor as a neutral vote: it is absent.

The indicator of the states puts these three cases in one array of shape $(M, P, K + 1)$:

$$
S_{ijs} = \mathbf{1}\{(i, j) \in \Omega \text{ and } s_{ij} = s\}, \qquad s = 0, 1, \dots, K.
$$ (eq-data-states)

On a shown cell exactly one $S_{ijs}$ is $1$; on a cell that is not shown, all are $0$. The models read the votes through this array only (`VoteMatrix._states`, internal).

### The counts of a co-cluster

Given a group $R_i$ for each voter and a block $C_j$ for each item, the count of the co-cluster $(q, l)$ in the state $s$ is

$$
N_{qls} = \sum_{i=1}^{M} \sum_{j=1}^{P} w_i \, S_{ijs} \, \mathbf{1}\{R_i = q\} \, \mathbf{1}\{C_j = l\},
$$ (eq-data-counts)

with $w_i$ the survey weight of the voter $i$ ({py:attr}`~civiqhub_selene.blocks.VoteMatrix.voter_weights`), scaled to average $1$, and $w_i = 1$ for every voter when the votes carry no weight. Without weights, $N_{qls}$ is the number of shown cells of the co-cluster in the state $s$, an integer; $N_{ql0}$ counts its abstentions, and $\sum_{k \geq 1} N_{qlk}$ its votes.

These counts are the sufficient statistic of the block models: given the arrangement, the likelihood of the votes depends on them alone. When a voter belongs to several groups with degrees $\theta_{iq}$, and an item to several blocks with degrees $\phi_{jl}$, the counts become expected counts under the model, and they are fractional; the page of the mixed-membership model gives them.

### The cells that are not shown

A cell that is not shown enters no sum: neither the counts {eq}`eq-data-counts`, nor any likelihood of the following pages. An abstention, on the contrary, is a state of the cell, and each co-cluster has its own probability of abstention.

Leaving the cells that are not shown out of the likelihood is what likelihood inference does when the missing cells are *missing at random* (Rubin 1976): it assumes that whether an item was shown to a voter does not depend on the vote the voter would have given. The engine makes this assumption for the cells not shown, and only for them: it does not make it for abstentions, which it models.

### Holding cells out

{py:meth}`~civiqhub_selene.blocks.VoteMatrix.hold_out` returns the same matrix where chosen cells are treated as never shown: they leave $\Omega$, and their levels stay in the matrix, to be predicted. A model estimated on the remaining cells is then scored on the cells held out; the page on choosing the shape uses this.

### When a result exists

A co-clustering of $Q$ groups and $L$ blocks exists on a matrix only if the matrix holds at least one vote, at least $\max(2, Q)$ voters with a vote ({py:attr}`~civiqhub_selene.blocks.VoteMatrix.n_active_voters`) and at least $\max(2, L)$ items with a vote ({py:attr}`~civiqhub_selene.blocks.VoteMatrix.n_active_items`). Otherwise the result carries no value but an explained absence, {py:class}`~civiqhub_selene.types.Unavailable`, which {py:func}`~civiqhub_selene.blocks.base.availability` returns.

## The votes of a group on a block

{py:meth}`~civiqhub_selene.blocks.VoteMatrix.group_by` reads the matrix through a partition of the voters and a partition of the items ({py:class}`~civiqhub_selene.blocks.Partition`), from a model or from anywhere else, and returns {py:class}`~civiqhub_selene.blocks.GroupedVotes`. Its {py:meth}`~civiqhub_selene.blocks.GroupedVotes.counts` are the counts {eq}`eq-data-counts` without weights, and {py:meth}`~civiqhub_selene.blocks.GroupedVotes.vote_counts` gives the votes of one co-cluster, $(N_{ql1}, \dots, N_{qlK})$, as the {py:class}`~civiqhub_selene.ordinal.VoteCounts` of the ordinal family, on a scale of five scores.

The matrix below has three voters and four items. Two cells are not shown: the item `i3` to the voter `v2`, and the item `i2` to the voter `v3`, whose level $2$ is kept in the matrix but is not read. Three shown cells carry no vote.

```python
from civiqhub_selene.blocks import Partition, VoteMatrix

levels = [
    [5, 0, 3, 1],
    [4, 0, 0, 2],
    [1, 2, 0, 5],
]
shown = [
    [True, True, True, True],
    [True, True, False, True],
    [True, False, True, True],
]
votes = VoteMatrix.from_arrays(levels, shown=shown)
print(votes.shape, votes.n_shown, votes.n_votes)

groups = Partition.from_voter_labels(votes.voters, [0, 0, 1])
blocks = Partition.from_item_labels(votes.items, [0, 0, 1, 1])
grouped = votes.group_by(groups, blocks)
counts = grouped.counts()
for q in range(2):
    for l in range(2):
        print(f"N[{q + 1}, {l + 1}] =", counts[q, l].astype(int).tolist())
print(grouped.vote_counts(0, 0).counts)
```

```text
(3, 4) 10 7
N[1, 1] = [2, 0, 0, 0, 1, 1]
N[1, 2] = [0, 1, 1, 1, 0, 0]
N[2, 1] = [0, 1, 0, 0, 0, 0]
N[2, 2] = [1, 0, 0, 0, 0, 1]
(0, 0, 0, 1, 1)
```

Each row of counts reads $(N_{ql0}, N_{ql1}, \dots, N_{ql5})$: the abstentions first, then the votes level by level. The ten shown cells are counted once each; the hidden level of `v3` on `i2` is counted nowhere. Holding out the cell of `v3` on `i4` removes it from $\Omega$ and keeps its level:

```python
import numpy as np

held_out = np.zeros(votes.shape, dtype=bool)
held_out[2, 3] = True
held = votes.hold_out(held_out)
print(held.n_shown, held.n_votes, held.levels[2, 3])
```

```text
9 6 5
```

## A generative process: `make_votes`

{py:func}`~civiqhub_selene.blocks.synthetic.make_votes` draws a matrix from a planted truth: memberships $\theta_{iq}$ of the voters and $\phi_{jl}$ of the items, and for each co-cluster a vote rate $\nu_{ql}$, a mean $\mu_{ql}$ and a standard deviation $\sigma_{ql}$. The draws come from one generator seeded by `seed`, in this order, which is part of its contract:

1. for each voter $i$ and each block $l$, the role $c_l$: the group the voter follows on that block, drawn from $\boldsymbol{\theta}_i$;
2. for each item $j$ and each voter $i$, the block $l$ of the cell $(i, j)$, drawn from $\boldsymbol{\phi}_j$;
3. the co-cluster of the cell is $(c_l, l)$: the role of the voter on the block of the cell, and that block;
4. whether the cell carries a vote, with the probability $\nu_{c_l l}$;
5. a hidden opinion, normal with the mean $\mu_{c_l l}$ and the standard deviation $\sigma_{c_l l}$;
6. the level of the vote: $1$ plus the number of thresholds $t_k = k + \tfrac{1}{2}$, $k = 1, \dots, K - 1$, that lie below the opinion; $0$ when the cell carries no vote.

Every cell is shown: $\Omega$ is the whole matrix, and a cell without a vote is an abstention. With memberships of $0$ and $1$ only ({py:func}`~civiqhub_selene.blocks.synthetic.single_memberships`), each voter has one group and each item one block: the draw is that of the latent block model. With spread memberships, it is that of the mixed-membership model. Steps 4 to 6 are the ordinal probit response model.

## What the code does exactly

| Formula or rule | Code |
| --- | --- |
| Three states, a vote at the neutral score is a vote | `src/civiqhub_selene/blocks/data.py:180-185` |
| Scale, levels from $1$ to $K$, level $0$ internal | `blocks/data.py:43-56`, `:75-100` |
| $S_{ijs}$ {eq}`eq-data-states`, zero on a cell not shown | `blocks/data.py:666-688` |
| $N_{qls}$ {eq}`eq-data-counts` | `count_states`, `blocks/_numerics.py:47-77` (internal) |
| $w_i$ scaled to average $1$ | `blocks/data.py:519-528` |
| `GroupedVotes.counts`, `vote_counts` | `blocks/data.py:753-803` |
| `hold_out` | `blocks/data.py:621-637` |
| Conditions of a result | `blocks/base.py:56-75` |
| `make_votes`, order of the draws | `blocks/synthetic.py:142-155` (contract), `:197-206` (draws) |

The code computes {eq}`eq-data-counts` in one product over all cells, with the indicators of $R$ and $C$ as one-hot matrices. {py:meth}`~civiqhub_selene.blocks.GroupedVotes.counts` and the table of co-clusters of a result use no weights; the latent block model uses the weighted counts to estimate its parameters.

## Limits

- One scale is shared by every item: each shown cell has $K + 1$ possible states. A scale per item is not supported yet.
- The engine does not model why an item is shown or not: it assumes that a cell that is not shown hides nothing about the vote it would have received.
- `vote_counts` needs a scale of five scores, that of the ordinal family.

## References

- Rubin, D. B. (1976). Inference and missing data. *Biometrika*, 63(3), 581–592.
