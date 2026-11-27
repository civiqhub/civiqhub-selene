# Choosing the shape

Checked against the code of `develop` at commit `c29d115`.

## In plain words

A block model is estimated for a shape given in advance: $Q$ groups and $L$ blocks. More groups and more blocks always describe the votes at least as well, so the objective a model maximises cannot choose the shape: it would always ask for more. The engine gives two scores that can: the integrated completed likelihood of a single-membership result, which pays for each parameter, and the log-likelihood of votes held out of the estimation, which only rewards what predicts votes the model has not seen. No function of the engine chooses a shape yet: a user compares shapes by these scores, as the example below does by hand.

## Model

### The integrated completed likelihood

A {py:class}`~civiqhub_selene.blocks.results.SingleCoclustering` carries the integrated completed likelihood of Biernacki, Celeux and Govaert (2000) as {py:attr}`~civiqhub_selene.blocks.results.SingleCoclustering.integrated_likelihood`:

$$
\mathrm{ICL} = \mathcal{L}_{\mathrm{c}} - \operatorname{pen}_F - \tfrac{1}{2} (Q - 1) \log M - \tfrac{1}{2} (L - 1) \log P,
$$ (eq-shape-icl)

where $\mathcal{L}_{\mathrm{c}}$ is the classified log-likelihood of the arrangement found ([The latent block model](latent-block-model.md)), $\operatorname{pen}_F$ the penalty of the parameters of the response model, {py:meth}`~civiqhub_selene.blocks.responses.ResponseModel.complexity_penalty`, and the last two terms the penalties of the $Q - 1$ free shares of the groups and the $L - 1$ free shares of the blocks. Each term is half a logarithm of a sample size per free parameter. The response models count their parameters with the numbers of votes and of shown cells of the matrix, scale by scale ([The response models: the penalty of the parameters](response-models.md#the-penalty-of-the-parameters), {eq}`eq-response-penalty`). With $\mathcal{G}$ the scales of the items and $K_g$ the levels of the scale $g$:

$$
\begin{aligned}
\operatorname{pen}_F &= \tfrac{1}{2} \, Q L \sum_{g \in \mathcal{G}} (K_g - 1) \log n_{\text{votes}} + \tfrac{1}{2} \, Q L \log n_{\text{shown}} && \text{(categorical)}, \\
\operatorname{pen}_F &= Q L \log n_{\text{votes}} + \tfrac{1}{2} \, Q L \log n_{\text{shown}} && \text{(ordinal probit, fixed thresholds)}, \\
\operatorname{pen}_F &= Q L \log n_{\text{votes}} + \tfrac{1}{2} \, Q L \log n_{\text{shown}} + \tfrac{1}{2} \log n_{\text{votes}} \sum_{g \in \mathcal{G}} \max(K_g - 3, 0) && \text{(ordinal probit, estimated thresholds)}.
\end{aligned}
$$

With one scale, the sum of the categorical model is $K - 1$.

A larger ICL is better. It compares shapes of the same model, with the same response model, on the same votes. With survey weights, $\mathcal{L}_{\mathrm{c}}$ is the weighted pseudo-likelihood; the code notes that using the same penalty on it is a hypothesis, since a weighted ICL is not a settled criterion.

The mixed-membership model has no ICL in the engine: its result carries its observed log-likelihood $\mathcal{L}$, and its shape is compared by held-out votes.

### The log-likelihood of held-out votes

Split the shown cells $\Omega$ into cells kept for the estimation, $\Omega_{\mathrm{in}}$, and cells held out, $\Omega_{\mathrm{out}}$. {py:meth}`~civiqhub_selene.blocks.VoteMatrix.hold_out` returns the matrix where the cells of $\Omega_{\mathrm{out}}$ are not shown, their levels kept ([The data](data.md)). The model is estimated on that matrix, then scored on the cells it did not see, by the logarithm of the probability its result gives to the state of each of them ({py:meth}`~civiqhub_selene.blocks.results.PredictsVotes.log_probability`, which both values follow). For a single-membership result, it is the probability of the co-cluster of the voter and the item, on the scale $g_j$ of the item ({py:meth}`~civiqhub_selene.blocks.results.SingleCoclustering.log_probability`):

$$
\mathcal{L}_{\mathrm{out}} = \sum_{(i, j) \in \Omega_{\mathrm{out}}} \log F_{R_i C_j g_j s_{ij}}.
$$ (eq-shape-held-out-single)

For a mixed-membership result, it is the probability averaged over the profiles of the voter, as the estimation saw them, and over the blocks of the item ({py:meth}`~civiqhub_selene.blocks.results.MixedCoclustering.log_probability`):

$$
\mathcal{L}_{\mathrm{out}} = \sum_{(i, j) \in \Omega_{\mathrm{out}}} \log \sum_{\boldsymbol{c}} \eta_{i\boldsymbol{c}} \sum_{l=1}^{L} \phi_{jl} \, F_{c_l l g_j s_{ij}}.
$$ (eq-shape-held-out-mixed)

With one scale, $F_{qlgs}$ is $F_{qls}$ ([Notation](notation.md)). In both, the quantities are those of the result estimated on $\Omega_{\mathrm{in}}$. The state of a cell is read from its level alone, without the mask of shown cells: a held-out cell is scored on the vote it had, or on its abstention. A larger $\mathcal{L}_{\mathrm{out}}$ is better. It only rewards what predicts votes the model did not see, so it does not grow with the shape the way the objective does. It applies to both models, and it is the criterion the mixed-membership model declares.

### What each model declares

Each model names the criterion that should choose its shape by default, as its `default_criterion`: `"integrated_likelihood"` for {py:class}`~civiqhub_selene.blocks.LatentBlockModel` and `"held_out_likelihood"` for {py:class}`~civiqhub_selene.blocks.MixedMembershipBlockModel`. {py:class}`~civiqhub_selene.blocks.results.Selection` holds the trace of a choice of shape (the shapes compared, their scores, the one kept) and a result can carry one in {py:attr}`~civiqhub_selene.blocks.results.Coclustering.selection`; no function of the engine fills it yet. The comparison is made by hand.

## An example

The votes have three planted groups and two planted blocks. The objective of the latent block model keeps growing with the shape; its ICL is largest at the planted shape, and it is the objective minus the penalties of {eq}`eq-shape-icl`:

```python
import math

import numpy as np
from civiqhub_selene.blocks import LatentBlockModel, MixedMembershipBlockModel
from civiqhub_selene.blocks.synthetic import make_votes, single_memberships

votes, _ = make_votes(single_memberships((20, 20, 20)), single_memberships((6, 6)), seed=3)
print(LatentBlockModel.default_criterion, MixedMembershipBlockModel.default_criterion)

for n_groups, n_blocks in [(2, 2), (3, 2), (4, 2), (3, 3)]:
    model = LatentBlockModel(n_groups=n_groups, n_blocks=n_blocks)
    value = model.estimate(votes, seed=0).value
    print((n_groups, n_blocks), round(value.objective, 1), round(value.integrated_likelihood, 1))

n_voters, n_items = votes.shape
by_hand = (
    value.objective
    - model.response_model.complexity_penalty((3, 3), votes)
    - 0.5 * (3 - 1) * math.log(n_voters)
    - 0.5 * (3 - 1) * math.log(n_items)
)
print(math.isclose(by_hand, value.integrated_likelihood))
```

```text
integrated_likelihood held_out_likelihood
(2, 2) -967.2 -1007.8
(3, 2) -890.1 -951.4
(4, 2) -886.6 -968.6
(3, 3) -883.0 -973.5
True
```

The mixed-membership model is compared on a tenth of the shown cells, held out at random:

```python
rng = np.random.default_rng(0)
held_out = votes.shown & (rng.random(votes.shape) < 0.1)
training = votes.hold_out(held_out)
print(int(held_out.sum()))

for n_groups, n_blocks in [(2, 2), (3, 2)]:
    model = MixedMembershipBlockModel(n_groups=n_groups, n_blocks=n_blocks)
    value = model.estimate(training, seed=0).value
    print((n_groups, n_blocks), round(float(value.log_probability(votes)[held_out].sum()), 1))
```

```text
65
(2, 2) -86.1
(3, 2) -78.8
```

The planted shape predicts the held-out votes better. One split of the cells gives one score; the engine computes no standard error for it.

## What the code does exactly

| Formula or rule | Code |
| --- | --- |
| {eq}`eq-shape-icl` | `LatentBlockModel._finish`, `src/civiqhub_selene/blocks/single.py:282-287`; docstring `:171-174` |
| $\operatorname{pen}_F$ | `Categorical.complexity_penalty`, `blocks/responses.py:336-348`; `OrdinalProbit.complexity_penalty`, `:751-767` |
| `hold_out` | `VoteMatrix.hold_out`, `blocks/data.py:1050-1072` |
| {eq}`eq-shape-held-out-single` | `SingleCoclustering.log_probability`, `blocks/results.py:459-478` |
| {eq}`eq-shape-held-out-mixed` | `MixedCoclustering.log_probability`, `blocks/results.py:562-595` |
| Protocols of the two criteria | `blocks/results.py:45-61` |
| `Selection` | `blocks/results.py:95-171` |
| `default_criterion` | `blocks/single.py:200`, `blocks/mixed.py:326` |

The code also does what the formulas do not say:

- **The logarithms** of $M$, $P$, $n_{\text{votes}}$ and $n_{\text{shown}}$ are taken on numbers of at least $1$.
- **The probabilities of a result** come from its table of co-clusters: its vote rates and shares, with the floors of the response models; in the mixed model, the probability of a cell is floored at $10^{-300}$ before its logarithm.
- **The objective in the ICL** is that of the start that was kept.

## Limits

- No function chooses the shape: the comparison is made by hand, and its result is not recorded in the value.
- The ICL exists for the single-membership model only, and its penalty on a weighted objective is a hypothesis of the code.
- One split of the cells gives one score of held-out votes, without a standard error.

## References

- Biernacki, C., Celeux, G. and Govaert, G. (2000). Assessing a mixture model for clustering with the integrated completed likelihood. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 22(7), 719–725. doi:10.1109/34.865189
