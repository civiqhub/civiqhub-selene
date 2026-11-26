# Reading the results

Checked against the code of `develop` at commit `c29d115`.

## In plain words

A co-clustering result is a description of a collective: groups of voters, blocks of items, and how each group answers each block. This page says what each quantity of a result means mathematically, and what it does not say. The numbers of the groups are arbitrary, the memberships are not probabilities of error, and no co-clustering result carries an uncertainty.

## Model

### The table of co-clusters

{py:attr}`~civiqhub_selene.blocks.results.Coclustering.coclusters` is a {py:class}`~civiqhub_selene.blocks.CoclusterTable`: for each group $q$ and block $l$,

- {py:attr}`~civiqhub_selene.blocks.CoclusterTable.vote_rate`, $\nu_{ql}$, the share of the shown cells of the co-cluster that carry a vote;
- {py:attr}`~civiqhub_selene.blocks.CoclusterTable.vote_shares`, $p_{ql1}, \dots, p_{qlK}$, the distribution of its votes over the levels;
- {py:attr}`~civiqhub_selene.blocks.CoclusterTable.n_votes`, $\sum_{k \geq 1} N_{qlk}$, its number of votes, counted without survey weights: an integer for a single-membership result, an expected and fractional number for a mixed one;
- under the ordinal probit, {py:attr}`~civiqhub_selene.blocks.CoclusterTable.latent_mean` and {py:attr}`~civiqhub_selene.blocks.CoclusterTable.latent_spread`, $\mu_{ql}$ and $\sigma_{ql}$, on the axis of levels.

These are the parameters of the response model at the end of the estimation ([The response models](response-models.md)): a description of how the group answers the block, not a test of a difference between groups.

When the items use several scales, the co-cluster keeps one vote rate and one opinion, and its distribution of votes is given scale by scale: {py:attr}`~civiqhub_selene.blocks.CoclusterTable.vote_shares_by_scale` holds $p_{qlg1}, \dots, p_{qlgK_g}$ for each scale $g$, and {py:attr}`~civiqhub_selene.blocks.CoclusterTable.thresholds` the grid of each scale under the ordinal probit, whose $\mu_{ql}$ and $\sigma_{ql}$ then read on the unit axis by default. {py:attr}`~civiqhub_selene.blocks.CoclusterTable.has_fixed_spread` marks the co-clusters whose spread was held, their votes all being on scales of two levels. The readings of one scale, such as {py:attr}`~civiqhub_selene.blocks.CoclusterTable.vote_shares`, refuse a table of several. The agreement between two groups below averages the distance of their votes over the scales of the result.

### Memberships and main groups

{py:attr}`~civiqhub_selene.blocks.results.Coclustering.group_index` gives each voter its main group, the largest of its memberships {py:attr}`~civiqhub_selene.blocks.results.Coclustering.group_memberships`; likewise for the items. The memberships mean different things in the two models:

- in a mixed-membership result they are $\theta_{iq}$ and $\phi_{jl}$, parameters of the model, estimated with their priors ([The mixed-membership model](mixed-membership.md)): a voter of memberships $(0.5, 0.5)$ follows each group on about half of the blocks;
- in a single-membership result they are $\hat{\theta}_{iq}$ and $\hat{\phi}_{jl}$, a soft reading of the final arrangement, computed with the other side fixed ([The latent block model](latent-block-model.md)): how much better the voter's group describes its votes than the others, not a probability that the voter belongs to another group.

Neither is an uncertainty of the result.

### Mixedness and changes of role

For a mixed-membership result, {py:attr}`~civiqhub_selene.blocks.results.MixedCoclustering.mixedness` is, for each voter,

$$
1 - \max_{q} \theta_{iq},
$$ (eq-results-mixedness)

$0$ for a voter that follows one group on every block, and up to $1 - 1/Q$. {py:meth}`~civiqhub_selene.blocks.results.MixedCoclustering.role_changes` counts the voters by their most probable role on two blocks: the entry $(q, r)$ is the number of voters whose largest $\kappa_{ilq}$ is at $q$ on the first block and at $r$ on the second. Off the diagonal, a voter changes role.

### How much two groups agree

{py:func}`~civiqhub_selene.viz.graphs.group_graph` gives, for each pair of groups, one minus the total variation distance between their distributions of votes, averaged over the blocks:

$$
\operatorname{agree}(q, r) = \sum_{l=1}^{L} \bar{\rho}_l \cdot \frac{1}{\lvert \mathcal{G} \rvert} \sum_{g \in \mathcal{G}} \Big(1 - \frac{1}{2} \sum_{k=1}^{K_g} \left| p_{qlgk} - p_{rlgk} \right|\Big),
$$ (eq-results-agreement)

where $\bar{\rho}_l$ is the share of the block $l$ in the result: the share of the items whose block is $l$ in a single-membership result, the mean of $\phi_{jl}$ over the items in a mixed one, normalised to sum to $1$. The distance is taken on each scale $g$ of the result, on the shares $p_{qlgk}$ of its $K_g$ levels, and averaged over the scales of $\mathcal{G}$, each counting equally, whether or not the block $l$ holds items of that scale. With one scale, it is $1 - \sum_l \bar{\rho}_l \cdot \frac{1}{2} \sum_{k=1}^{K} \lvert p_{qlk} - p_{rlk} \rvert$. It is $1$ for two groups with the same distributions and $0$ for two groups without any common level. It reads the distributions of the votes only, not the vote rates: two groups that vote alike but not as often agree fully.

### How much two partitions agree

{py:meth}`~civiqhub_selene.blocks.Partition.agreement_with` compares two partitions of the same identifiers, such as the groups of two results, or the groups of a result and a planted truth. With $T$ the table that counts the identifiers in the cluster $q$ of the first partition and the cluster $r$ of the second, it returns

$$
\operatorname{match} = \frac{1}{|\mathcal{I}|} \max_{\chi} \sum_{q} T_{q \chi(q)},
$$ (eq-results-match)

the share of the identifiers $\mathcal{I}$ that the two partitions arrange in the same way, after the best renumbering $\chi$ of the clusters of the second. The maximum is found by trying every renumbering, which the code allows up to $8$ clusters.

## What a result does not say

### The numbers of the groups

The likelihoods of both models do not change when the groups are renumbered, nor when the blocks are: a result with the groups $1, 2, 3$ and the same result with the groups $3, 1, 2$ are the same description. The number of a group carries no meaning, and two computations of different seeds can number the same groups differently. To compare two results, compare their partitions with {eq}`eq-results-match`, never their numbers.

### Uncertainty

A co-clustering result carries no uncertainty: its {py:attr}`~civiqhub_selene.types.Result.uncertainty` is `None`, which means "not defined", never "zero". No standard error, no interval and no measure of stability is part of the engine for these results. The memberships, the agreement between groups and the match between partitions describe one result; they say nothing of how much it would change on other votes.

### Decisions, themes and individuals

A result describes a collective and decides nothing. A block of items is a set of items that the groups answer alike: it is not a theme, and its items need not share a subject. The memberships of an individual voter are a step of the computation; {py:attr}`~civiqhub_selene.blocks.results.MixedCoclustering.mixedness` is published as a histogram over the voters, never as a list by voter.

## An example

Two computations of the same model, on the same votes, with two seeds:

```python
from civiqhub_selene.blocks import LatentBlockModel
from civiqhub_selene.blocks.synthetic import make_votes, single_memberships

votes, _ = make_votes(single_memberships((20, 20, 20)), single_memberships((6, 6)), seed=3)
model = LatentBlockModel(n_groups=3, n_blocks=2)
first = model.estimate(votes, seed=0)
second = model.estimate(votes, seed=5)

print(first.value.group_index[[0, 20, 40]].tolist(), second.value.group_index[[0, 20, 40]].tolist())
print(float((first.value.group_index == second.value.group_index).mean()))
print(first.value.groups.agreement_with(second.value.groups))
print(first.uncertainty)
```

```text
[2, 1, 0] [0, 2, 1]
0.0
1.0
None
```

No voter has the same group number in the two results, and the two results are the same partition of the voters once the groups are renumbered. The result carries no uncertainty.

## What the code does exactly

| Formula or rule | Code |
| --- | --- |
| Table of co-clusters, by scale | `CoclusterTable`, `src/civiqhub_selene/blocks/coclusters.py:235-529`; `Cocluster`, `:47-187` |
| Main group and block | `blocks/single.py:288-301`; `blocks/mixed.py:416-417` |
| {eq}`eq-results-mixedness` | `MixedCoclustering.mixedness`, `blocks/results.py:530-540` |
| `role_changes` | `blocks/results.py:542-560` |
| {eq}`eq-results-agreement`, shares of the blocks, averaged over the scales | `group_graph`, `src/civiqhub_selene/viz/graphs.py:280-316`; `_agreement_by_block`, `:77-86`; `Coclustering.blocks`, `blocks/results.py:301-311`; `MixedCoclustering._shares`, `:523-528` |
| {eq}`eq-results-match`, at most 8 clusters | `Partition.agreement_with`, `blocks/partition.py:330-361` |
| No uncertainty | `Estimator._uncertainty`, `src/civiqhub_selene/estimator.py:223-232`, not redefined by the block models; `Uncertainty`, `src/civiqhub_selene/types.py:298-299` |

## Limits

- The table of co-clusters gives point values, without an interval.
- The match between partitions is defined up to $8$ clusters.
- The agreement between groups ignores the vote rates.

## References

The quantities of this page are defined by the code; the models behind them are referenced on their own pages.
