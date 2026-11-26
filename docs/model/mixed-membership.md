# The mixed-membership model

Checked against the code of `develop` at commit `97a0f15`.

## In plain words

In the latent block model, a voter belongs to one group on every block of items. {py:class}`~civiqhub_selene.blocks.MixedMembershipBlockModel` lets a voter follow one group on some items and another group on others: each voter has a degree of membership in each group, and on each block it takes one role, the group it follows there. Each item, likewise, has a degree of membership in each block. The model computes its expectation step exactly, by going through every choice of a role per block. Its estimator is registered as `mixed_membership_block_model`, version `0.1.0`.

## Model

### How the votes are drawn

With $Q$ groups and $L$ blocks, each voter $i$ has memberships $\boldsymbol{\theta}_i = (\theta_{i1}, \dots, \theta_{iQ})$ and each item $j$ memberships $\boldsymbol{\phi}_j = (\phi_{j1}, \dots, \phi_{jL})$, each summing to $1$. The model draws

1. for each voter $i$ and each block $l$, a role $c_l$, the group the voter follows on that block, with the probabilities $\boldsymbol{\theta}_i$; the voter's profile is $\boldsymbol{c} = (c_1, \dots, c_L)$, one of $Q^L$;
2. for each shown cell $(i, j)$, a block $l$, with the probabilities $\boldsymbol{\phi}_j$;
3. the state of the cell, with the probabilities $F_{c_l l s}$ of the co-cluster $(c_l, l)$: the role of the voter on the block of the cell, and that block.

A cell that is not shown carries no information and enters nowhere. This is the draw of {py:func}`~civiqhub_selene.blocks.synthetic.make_votes`.

### The observed log-likelihood, exactly

Given its profile, the cells of a voter are independent, and the block of a cell is summed out within the cell. The probability of the cell $(i, j)$ under the profile $\boldsymbol{c}$ is therefore

$$
G_{ij\boldsymbol{c}} = \sum_{l=1}^{L} \phi_{jl} \, F_{c_l l s_{ij}},
$$ (eq-mm-cell)

and the profile is summed out per voter. The log-likelihood of the votes is

$$
\mathcal{L} = \sum_{i=1}^{M} w_i \log \sum_{\boldsymbol{c}} \Big( \prod_{l=1}^{L} \theta_{i c_l} \Big) \prod_{j : (i, j) \in \Omega} G_{ij\boldsymbol{c}},
$$ (eq-mm-likelihood)

with $w_i = 1$ without survey weights, and the weighted pseudo-likelihood of Skinner (1989) and Pfeffermann (1993) with them. The sum over the profiles has $Q^L$ terms: it is computed exactly, not approximated.

### The objective

The memberships are regularised by symmetric Dirichlet priors, of concentration $\alpha$ ({py:attr}`~civiqhub_selene.blocks.MixedMembershipBlockModel.voter_concentration`) on each $\boldsymbol{\theta}_i$ and $\beta$ ({py:attr}`~civiqhub_selene.blocks.MixedMembershipBlockModel.item_concentration`) on each $\boldsymbol{\phi}_j$, both at least $1$. The model maximises the log-likelihood plus the log-densities of the priors, up to a constant:

$$
\mathcal{J} = \mathcal{L} + (\alpha - 1) \sum_{i=1}^{M} w_i \sum_{q=1}^{Q} \log \theta_{iq} + (\beta - 1) \sum_{j=1}^{P} \sum_{l=1}^{L} \log \phi_{jl}.
$$ (eq-mm-objective)

$\alpha = 1$ or $\beta = 1$ is no prior on that side. A concentration above $1$ keeps every membership away from $0$. The prior of each voter is weighted by $w_i$, the prior of the items is not.

### The latent block model as a special case

Let every membership be $0$ or $1$: $\theta_{iq} = \mathbf{1}\{q = R_i\}$ and $\phi_{jl} = \mathbf{1}\{l = C_j\}$. Then $\prod_l \theta_{i c_l}$ is $1$ for the one profile $\boldsymbol{c} = (R_i, \dots, R_i)$ and $0$ for every other, and in {eq}`eq-mm-cell` only the term $l = C_j$ is left: $G_{ij\boldsymbol{c}} = F_{R_i C_j s_{ij}}$. {eq}`eq-mm-likelihood` becomes

$$
\mathcal{L} = \sum_{i=1}^{M} w_i \sum_{j : (i, j) \in \Omega} \log F_{R_i C_j s_{ij}} = \sum_{i=1}^{M} w_i \sum_{j=1}^{P} \sum_{s=0}^{K} S_{ijs} \log F_{R_i C_j s},
$$

the log-likelihood of the votes of the latent block model at the arrangement $(R, C)$: the first term of its classified log-likelihood. The draw is the same as well. The mixed-membership model contains the latent block model; what it adds is a role per block for each voter and a block per cell for each item.

## Estimation

The model is estimated by expectation-maximisation (Dempster, Laird and Rubin 1977) on the profiles and the blocks of the cells, for the maximum of {eq}`eq-mm-objective`. The expectation step is exact. It is not a variational method.

### Expectation

At the current memberships and response parameters, the posterior probability of the profile $\boldsymbol{c}$ of the voter $i$ is

$$
\eta_{i\boldsymbol{c}} = \frac{\prod_{l} \theta_{i c_l} \prod_{j : (i, j) \in \Omega} G_{ij\boldsymbol{c}}}{\sum_{\boldsymbol{c}'} \prod_{l} \theta_{i c'_l} \prod_{j : (i, j) \in \Omega} G_{ij\boldsymbol{c}'}},
$$ (eq-mm-profiles)

computed in logarithms, and the denominator gives the term of the voter in {eq}`eq-mm-likelihood`. Under the profile $\boldsymbol{c}$, the shown cell $(i, j)$ took the block $l$ with the probability $\phi_{jl} F_{c_l l s_{ij}} / G_{ij\boldsymbol{c}}$. From these come three expectations:

- the probability that the voter $i$ follows the group $q$ on the block $l$, {py:attr}`~civiqhub_selene.blocks.results.MixedCoclustering.role_by_block`:
  $\kappa_{ilq} = \sum_{\boldsymbol{c}} \eta_{i\boldsymbol{c}} \, \mathbf{1}\{c_l = q\}$;
- the expected number of shown cells of the item $j$ that took the block $l$:
  $H_{jl} = \sum_{i} w_i \sum_{\boldsymbol{c}} \eta_{i\boldsymbol{c}} \, \phi_{jl} F_{c_l l s_{ij}} / G_{ij\boldsymbol{c}}$, over the shown cells;
- the expected counts of each co-cluster in each state, fractional:
  $N_{qls} = \sum_{i} w_i \sum_{j} S_{ijs} \sum_{\boldsymbol{c}} \eta_{i\boldsymbol{c}} \, \mathbf{1}\{c_l = q\} \, \phi_{jl} F_{qls} / G_{ij\boldsymbol{c}}$.

When the items use several scales, a cell reads the probabilities of the scale of its item: $F_{c_l l s_{ij}}$ becomes $F_{c_l l g_j s_{ij}}$ in {eq}`eq-mm-cell` and in the expectations, and the counts are kept by scale, $N_{qlgs}$, the sum over the items $j$ of the scale $g$ only. The code computes the probability of a cell under each profile scale by scale, the voters by chunks as before; with one scale, it computes exactly what it computed before items could have several.

### Maximisation

Given these expectations, {eq}`eq-mm-objective` separates into a part for each $\boldsymbol{\theta}_i$, a part for each $\boldsymbol{\phi}_j$ and a part for the response model. Maximising $\sum_{q} \big(\sum_l \kappa_{ilq} + \alpha - 1\big) \log \theta_{iq}$ under $\sum_q \theta_{iq} = 1$, and the same for the items, gives

$$
\theta_{iq} \propto \sum_{l=1}^{L} \kappa_{ilq} + \alpha - 1, \qquad \phi_{jl} \propto H_{jl} + \beta - 1,
$$ (eq-mm-mstep)

each row normalised to sum to $1$. The weight $w_i$ multiplies the whole part of the voter $i$ and cancels in its normalisation. The response model then updates $F$ from the fractional counts $N_{qls}$ ([The response models](response-models.md)).

With $\alpha = 1.2$, the default, a voter who follows the group $q$ on every block with certainty gets $\theta_{iq} = (L + \alpha - 1) / (L + Q(\alpha - 1))$: $2.2 / 2.4 \approx 0.92$ for $Q = L = 2$.

### Why the objective does not decrease

Each iteration is an expectation step followed by a maximisation step of {eq}`eq-mm-objective`. By the property of expectation-maximisation, the objective does not decrease when every part of the maximisation is a maximum, or does not lower its part: true of {eq}`eq-mm-mstep`, of the vote rate, of {py:class}`~civiqhub_selene.blocks.responses.OrdinalProbit` and of {py:class}`~civiqhub_selene.blocks.responses.Categorical` with `smoothing=0`. With a smoothed {py:class}`~civiqhub_selene.blocks.responses.Categorical`, the pseudo-counts can lower the objective by a tiny amount near convergence, where the loop then stops. The value noted after each iteration is {eq}`eq-mm-objective` at the new parameters, with the expectation step computed there.

### The start

Each start reads the arrangement $(R, C)$ that its {py:attr}`~civiqhub_selene.blocks.base.BlockModel.initialization` gives (random by default, spectral with {py:class}`~civiqhub_selene.blocks.SpectralStart`, or the result of a single-membership model with {py:class}`~civiqhub_selene.blocks.FixedStart`). It turns the arrangement into memberships that put {py:attr}`~civiqhub_selene.blocks.MixedMembershipBlockModel.start_weight` on the group or the block given, and share the rest evenly between the others (a single cluster gets $1$). The response model starts from its initial parameters, updated once on the weighted counts of the arrangement.

### What the value holds

{py:class}`~civiqhub_selene.blocks.results.MixedCoclustering` carries

- $\theta$ and $\phi$ as {py:attr}`~civiqhub_selene.blocks.results.Coclustering.group_memberships` and {py:attr}`~civiqhub_selene.blocks.results.Coclustering.block_memberships`, and the main group of a voter and the main block of an item, their largest membership, as {py:attr}`~civiqhub_selene.blocks.results.Coclustering.group_index` and {py:attr}`~civiqhub_selene.blocks.results.Coclustering.block_index`;
- $\kappa$ as {py:attr}`~civiqhub_selene.blocks.results.MixedCoclustering.role_by_block`, every profile in the order of `itertools.product` as {py:attr}`~civiqhub_selene.blocks.results.MixedCoclustering.profiles`, and $\eta$ as {py:attr}`~civiqhub_selene.blocks.results.MixedCoclustering.profile_probabilities`;
- $\mathcal{L}$ as {py:attr}`~civiqhub_selene.blocks.results.MixedCoclustering.log_likelihood` and $\mathcal{J}$ as {py:attr}`~civiqhub_selene.blocks.results.Coclustering.objective`.

It has no integrated likelihood: its shape is compared by the log-likelihood of held-out cells.

## An example

Sixty synthetic voters: twenty follow the first group on both blocks, twenty the second group, and twenty have the memberships $(0.5, 0.5)$, so that each of them draws its role on each block. Sixteen items form two blocks.

```python
import numpy as np
from civiqhub_selene.blocks import MixedMembershipBlockModel, Partition
from civiqhub_selene.blocks.synthetic import make_votes, single_memberships

voter_memberships = np.vstack([single_memberships((20, 20)), np.full((20, 2), 0.5)])
votes, truth = make_votes(
    voter_memberships,
    single_memberships((8, 8)),
    vote_rate=[[0.9, 0.9], [0.9, 0.9]],
    latent_mean=[[1.5, 4.5], [4.5, 1.5]],
    latent_spread=[[0.6, 0.6], [0.6, 0.6]],
    seed=1,
)
result = MixedMembershipBlockModel(n_groups=2, n_blocks=2).estimate(votes, seed=0)
coclustering = result.value
print(result.provenance.method)

planted_blocks = truth.item_memberships.argmax(axis=1)
blocks = Partition.from_item_labels(votes.items, planted_blocks)
print(coclustering.blocks.agreement_with(blocks))

roles = coclustering.role_by_block.argmax(axis=2)
for planted_block in range(2):
    found_block = np.bincount(coclustering.block_index[planted_blocks == planted_block]).argmax()
    planted = Partition.from_voter_labels(votes.voters, truth.roles[:, planted_block])
    found = Partition.from_voter_labels(votes.voters, roles[:, found_block])
    print(found.agreement_with(planted))

print(np.round(coclustering.group_memberships[[0, 20, 40]], 2).tolist())
```

```text
mixed_membership_block_model 0.1.0
1.0
1.0
1.0
[[0.08, 0.92], [0.92, 0.08], [0.5, 0.5]]
```

The blocks are found, and on each block the role that {py:attr}`~civiqhub_selene.blocks.results.MixedCoclustering.role_by_block` gives each voter is the role it was drawn with, after the best renumbering of the groups. The voters of a single group get $0.92$, the bound that the prior $\alpha = 1.2$ sets; a voter drawn with $(0.5, 0.5)$ keeps about $0.5$.

## What the code does exactly

| Formula or rule | Code |
| --- | --- |
| Model, {eq}`eq-mm-likelihood`, {eq}`eq-mm-objective`, weights | Docstring, `src/civiqhub_selene/blocks/mixed.py:168-206` |
| {eq}`eq-mm-cell`, {eq}`eq-mm-profiles`, $\kappa$, $H$, $N$, $\mathcal{L}$ | `_expect`, `blocks/mixed.py:103-161` |
| {eq}`eq-mm-objective` as computed | `_state`, `blocks/mixed.py:374-402` |
| {eq}`eq-mm-mstep`, response update | `_step`, `blocks/mixed.py:328-336`; `_normalised`, `:95-100` |
| Start | `_start`, `blocks/mixed.py:309-326`; `_smoothed`, `:84-92` |
| Settings and their bounds; `MAX_PROFILES` | `blocks/mixed.py:28-29`, `:276-305` |
| Value | `_finish`, `blocks/mixed.py:338-372`; {py:class}`~civiqhub_selene.blocks.results.MixedCoclustering`, `blocks/results.py:434-475` |

The code also does what the formulas do not say:

- **Logarithms.** The posterior {eq}`eq-mm-profiles` is computed from $\sum_l \log \theta_{i c_l} + \sum_j \log G_{ij\boldsymbol{c}}$ with a stable log-sum-exp; a membership and a probability $G_{ij\boldsymbol{c}}$ are floored at $10^{-300}$ before their logarithms.
- **Safeguards of the maximisation.** Each term of {eq}`eq-mm-mstep` is floored at $0$, and a row of zeros, such as an item with no shown cell when $\beta = 1$, becomes uniform.
- **Cost.** The expectation goes through the voters in batches, so that a batch holds about two million (voter, item, profile) triples. The settings refuse a shape with more than {py:data}`~civiqhub_selene.blocks.mixed.MAX_PROFILES` profiles ($4096$): the cost of an iteration grows as $M \cdot P \cdot Q^L$.
- **Counts of the value.** With survey weights, the table of co-clusters of the value is computed again from counts without weights, so that it reports the number of votes observed.

## Limits

- The number of profiles $Q^L$ grows fast: $Q = 4$ groups and $L = 6$ blocks are already $4096$ profiles, the largest number the code accepts.
- The result is a local maximum that depends on the start, and carries no uncertainty.
- The integrated completed likelihood is not defined for this model in the engine; a weighted version would be a Hypothesis.

## References

- Airoldi, E. M., Blei, D. M., Fienberg, S. E. and Xing, E. P. (2008). Mixed membership stochastic blockmodels. *Journal of Machine Learning Research*, 9. The code takes from it the mixed membership. The role per block and the exact expectation over the profiles follow the reference engine of the CiviqHub explainer "LBM and MM-OLBM step by step", as the docstring of the model says.
- Dempster, A. P., Laird, N. M. and Rubin, D. B. (1977), for expectation-maximisation: see [The response models](response-models.md).
- Corneli, M., Bouveyron, C. and Latouche, P. (2020), for the ordinal response model.
- Skinner, C. J. (1989) and Pfeffermann, D. (1993), for the weighted pseudo-likelihood, as the code cites them.
