# The response models

Checked against the code of `develop` at commit `97a0f15`.

## In plain words

A block model sorts voters into groups and items into blocks. The response model says how the cells of one co-cluster, a group on a block, spread between the states "shown without a vote" and each level of the scale. The engine ships two of them. {py:class}`~civiqhub_selene.blocks.responses.Categorical` lets each co-cluster have any distribution of its votes over the levels. {py:class}`~civiqhub_selene.blocks.responses.OrdinalProbit` reads each vote as a hidden opinion cut into levels, so that two numbers sum up how a group answers a block: where its opinion stands and how spread it is. Both give each co-cluster its own rate of votes.

## Model

### What a block model asks of a response model

The protocol {py:class}`~civiqhub_selene.blocks.responses.ResponseModel` is what a block model calls. For $Q \times L$ co-clusters, it gives starting parameters ({py:meth}`~civiqhub_selene.blocks.responses.ResponseModel.initial_parameters`); the logarithm of the probability $F_{qls}$ of each state $s$ in each co-cluster ({py:meth}`~civiqhub_selene.blocks.responses.ResponseModel.log_probabilities`); new parameters from counts $N_{qls}$, integer or fractional ({py:meth}`~civiqhub_selene.blocks.responses.ResponseModel.update`); the penalty of its parameters in the integrated likelihood ({py:meth}`~civiqhub_selene.blocks.responses.ResponseModel.complexity_penalty`); and the readable table of the co-clusters ({py:meth}`~civiqhub_selene.blocks.responses.ResponseModel.to_table`).

Both models write the probability of a state of a shown cell of the co-cluster $(q, l)$ in the same way, with a vote rate $\nu_{ql}$ and shares of the votes $p_{qlk}$ that sum to $1$ over the levels:

$$
F_{ql0} = 1 - \nu_{ql}, \qquad F_{qlk} = \nu_{ql} \, p_{qlk}, \quad k = 1, \dots, K.
$$ (eq-response-states)

They differ in how they write $p_{qlk}$.

### The categorical model

{py:class}`~civiqhub_selene.blocks.responses.Categorical` leaves $p_{ql1}, \dots, p_{qlK}$ free: $K - 1$ free parameters per co-cluster for the distribution, and no constraint on its shape. With the vote rate, a co-cluster has $K$ free parameters. It is the reference against which the cost and the benefit of the ordinal probit are judged.

### The ordinal probit

{py:class}`~civiqhub_selene.blocks.responses.OrdinalProbit` reads each vote of the co-cluster $(q, l)$ as a hidden opinion $y_{ij}$, normal with the mean $\mu_{ql}$ ({py:attr}`~civiqhub_selene.blocks.responses.OrdinalProbitParameters.latent_mean`) and the standard deviation $\sigma_{ql}$ ({py:attr}`~civiqhub_selene.blocks.responses.OrdinalProbitParameters.latent_spread`). Fixed thresholds $t_1 < \dots < t_{K-1}$ ({py:attr}`~civiqhub_selene.blocks.responses.OrdinalProbitParameters.thresholds`), with $t_0 = -\infty$ and $t_K = +\infty$, cut the opinion into levels: the vote is at the level $k$ when $t_{k-1} < y_{ij} \leq t_k$. With the standardised thresholds

$$
z_{qlk} = \frac{t_k - \mu_{ql}}{\sigma_{ql}}, \qquad k = 0, \dots, K,
$$

the share of the votes at the level $k$ is the normal mass between two thresholds:

$$
p_{qlk} = \Phi(z_{qlk}) - \Phi(z_{ql,k-1}).
$$ (eq-response-probit)

By default the thresholds are $t_k = k + \tfrac{1}{2}$ for $k = 1, \dots, K - 1$ ({py:attr}`~civiqhub_selene.blocks.responses.OrdinalProbit.thresholds` left to `None`); a strictly increasing tuple of $K - 1$ numbers replaces them. They are never estimated. With the default thresholds, the level $k$ is the interval of opinions around $k$, so that $\mu_{ql}$ reads directly on the axis of levels, from $1$ to $K$: a group whose mean is $4.2$ on a block stands between the levels $4$ and $5$. A co-cluster has three free parameters: $\nu_{ql}$, $\mu_{ql}$ and $\sigma_{ql}$.

### Items on several scales

When the items use several scales ([The data: items on different scales](data.md#items-on-different-scales)), each scale $g$ has its own grid of thresholds $t_{g1} < \dots < t_{g,K_g - 1}$, with $t_{g0} = -\infty$ and $t_{gK_g} = +\infty$. A vote on an item of the scale $g$ is cut by the grid of its scale, and the co-cluster keeps one opinion for all its items, whatever their scale: one rate $\nu_{ql}$, one mean $\mu_{ql}$ and one standard deviation $\sigma_{ql}$. The probability of the state $s$ of a shown cell of an item of the scale $g$ in the co-cluster $(q, l)$ is

$$
F_{qlg0} = 1 - \nu_{ql}, \qquad F_{qlgk} = \nu_{ql} \, p_{qlgk}, \qquad p_{qlgk} = \Phi\Big(\frac{t_{gk} - \mu_{ql}}{\sigma_{ql}}\Big) - \Phi\Big(\frac{t_{g,k-1} - \mu_{ql}}{\sigma_{ql}}\Big), \quad k = 1, \dots, K_g.
$$ (eq-response-probit-by-scale)

With one scale this is {eq}`eq-response-probit`. The grids must sit on one axis for one opinion to explain the votes of every scale: this is an assumption of the model, that the answers of two scales that cut the axis at the same place mean the same position. {py:class}`~civiqhub_selene.blocks.responses.Thresholds` sets the axis, and the regular grid of the scale $g$ on it:

$$
\text{axis of levels:} \quad t_{gk} = k + \tfrac{1}{2}, \qquad \text{unit axis:} \quad t_{gk} = \frac{k}{K_g}, \qquad k = 1, \dots, K_g - 1.
$$ (eq-response-axes)

On the unit axis, every scale covers the axis from $0$ to $1$, each of its levels taking an equal share of it: "yes" on two answers and the two top answers of four lie above $\tfrac{1}{2}$. On the axis of levels, the first levels of every scale line up. By default ({py:attr}`~civiqhub_selene.blocks.responses.Thresholds.axis` set to `"auto"`), the engine uses the axis of levels when the items share one scale, so that a matrix of one scale is computed exactly as before, and the unit axis when they use several. {py:attr}`~civiqhub_selene.blocks.responses.Thresholds.grids` replaces the regular grid of some scales by grids given by hand, on the chosen axis.

The unit axis is the axis of levels carried by $x \mapsto (x - \tfrac{1}{2}) / K$, with $K$ the levels of the longest scale: on a matrix of one scale, both give the same likelihood and the same arrangements, and the means and standard deviations of the unit axis are those of the axis of levels carried the same way. The floor $\sigma_{\min}$ and the start of $\sigma_{ql}$ are read in units of one level of the longest scale, $\sigma_{\mathrm{u}}$ ({py:attr}`~civiqhub_selene.blocks.responses.OrdinalProbitParameters.spread_unit`): $1$ on the axis of levels, $1 / K$ on the unit axis.

The categorical model has one free distribution per scale: the shares $p_{qlgk}$ over the $K_g$ levels of the scale $g$, with nothing to tie the distributions of two scales together, and one vote rate $\nu_{ql}$ for all.

## Estimation

A block model hands the response model counts $N_{qls}$ and asks for the parameters that maximise the expected log-likelihood of the cells of each co-cluster,

$$
\sum_{s=0}^{K} N_{qls} \log F_{qls} = N_{ql0} \log (1 - \nu_{ql}) + \Big(\sum_{k=1}^{K} N_{qlk}\Big) \log \nu_{ql} + \sum_{k=1}^{K} N_{qlk} \log p_{qlk}.
$$ (eq-response-expected)

The vote rate appears in the first two terms only, and the shares in the last one only: the two are estimated apart.

### The vote rate

The first two terms are those of a binomial: their maximum is the share of the cells of the co-cluster that carry a vote,

$$
\hat{\nu}_{ql} = \frac{\sum_{k=1}^{K} N_{qlk}}{\sum_{s=0}^{K} N_{qls}},
$$ (eq-response-rate)

bounded to $[\varepsilon, 1 - \varepsilon]$ with $\varepsilon$ = `min_probability` ({py:attr}`~civiqhub_selene.blocks.responses.OrdinalProbit.min_probability`, {py:attr}`~civiqhub_selene.blocks.responses.Categorical.min_probability`), so that no state ever gets the probability $0$. A co-cluster without any cell keeps its previous rate. Both models share this update.

### The shares of the categorical model

Maximising $\sum_k N_{qlk} \log p_{qlk}$ under $\sum_k p_{qlk} = 1$ gives the share of the votes at each level. The code adds the pseudo-count $\lambda$ = {py:attr}`~civiqhub_selene.blocks.responses.Categorical.smoothing` to each level first:

$$
\hat{p}_{qlk} = \frac{N_{qlk} + \lambda}{\sum_{h=1}^{K} \left(N_{qlh} + \lambda\right)}.
$$ (eq-response-categorical)

With $\lambda > 0$, a level without any vote keeps a small share. It is the maximum of $\sum_k (N_{qlk} + \lambda) \log p_{qlk}$, the posterior mode of the shares under a symmetric Dirichlet prior of concentration $\lambda + 1$. When $\lambda = 0$, a co-cluster without any vote keeps its previous shares.

### The opinion of the ordinal probit

The opinions $y_{ij}$ are not observed: only the interval they fall in is. The code estimates $\mu_{ql}$ and $\sigma_{ql}$ by expectation-maximisation on the hidden opinion, for every co-cluster at once, from the counts of votes $N_{ql1}, \dots, N_{qlK}$ and their sum $n_{ql} = \sum_k N_{qlk}$. An abstention says nothing about the opinion and does not enter.

**Expectation.** At the current $\mu_{ql}$ and $\sigma_{ql}$, the first two moments of the opinion of a vote at the level $k$ are those of a normal law truncated to $(t_{k-1}, t_k]$. With $\Phi'$ the standard normal density,

$$
r_{qlk} = \frac{\Phi'(z_{qlk}) - \Phi'(z_{ql,k-1})}{p_{qlk}}, \qquad
u_{qlk} = \frac{z_{qlk} \, \Phi'(z_{qlk}) - z_{ql,k-1} \, \Phi'(z_{ql,k-1})}{p_{qlk}},
$$

where a product $z \, \Phi'(z)$ is $0$ at an infinite threshold, they are

$$
m^{(1)}_{qlk} = \mu_{ql} - \sigma_{ql} \, r_{qlk}, \qquad
m^{(2)}_{qlk} = \mu_{ql}^2 - 2 \, \sigma_{ql} \, \mu_{ql} \, r_{qlk} + \sigma_{ql}^2 \, (1 - u_{qlk}).
$$ (eq-response-moments)

They follow from the standard normal law truncated to $(z_{ql,k-1}, z_{qlk}]$, whose mean is $-r_{qlk}$ and whose second moment is $1 - u_{qlk}$: the opinion of a vote at the level $k$ is $\mu_{ql}$ plus $\sigma_{ql}$ times that variable.

**Maximisation.** The mean and the variance of the opinions, given these moments:

$$
\mu_{ql} \leftarrow \frac{1}{n_{ql}} \sum_{k=1}^{K} N_{qlk} \, m^{(1)}_{qlk}, \qquad
\sigma_{ql} \leftarrow \sqrt{\max\Big(\frac{1}{n_{ql}} \sum_{k=1}^{K} N_{qlk} \, m^{(2)}_{qlk} - \mu_{ql}^2, \; \sigma_{\min}^2\Big)},
$$ (eq-response-opinion)

with the new $\mu_{ql}$ in the variance. The floor $\sigma_{\min}$ = {py:attr}`~civiqhub_selene.blocks.responses.OrdinalProbit.min_spread` keeps a unanimous co-cluster from sending its spread to $0$. A co-cluster without any vote, $n_{ql} \leq 10^{-12}$, keeps its parameters.

The steps restart from the previous $\mu_{ql}$ and $\sigma_{ql}$, and stop when the largest change of a mean or a standard deviation, over all co-clusters, is under $10^{-8}$, or after {py:attr}`~civiqhub_selene.blocks.responses.OrdinalProbit.max_inner_iterations` steps. Each update of the block model thus runs a few steps of this inner loop, not the loop to its end.

**On several scales.** The moments of a vote at the level $k$ of the scale $g$ are those of the opinion truncated to $(t_{g,k-1}, t_{gk}]$: the formulas {eq}`eq-response-moments`, on the grid of its scale, written $m^{(1)}_{qlgk}$ and $m^{(2)}_{qlgk}$. The opinion of the co-cluster pools the votes of every scale, with the counts $N_{qlgk}$ and $n_{ql} = \sum_g \sum_{k \geq 1} N_{qlgk}$:

$$
\mu_{ql} \leftarrow \frac{1}{n_{ql}} \sum_{g \in \mathcal{G}} \sum_{k=1}^{K_g} N_{qlgk} \, m^{(1)}_{qlgk}, \qquad
\sigma_{ql} \leftarrow \sqrt{\max\Big(\frac{1}{n_{ql}} \sum_{g \in \mathcal{G}} \sum_{k=1}^{K_g} N_{qlgk} \, m^{(2)}_{qlgk} - \mu_{ql}^2, \; (\sigma_{\min} \sigma_{\mathrm{u}})^2\Big)}.
$$ (eq-response-opinion-by-scale)

The code sums over the levels first, then over the scales, so that one scale gives exactly {eq}`eq-response-opinion`.

**Scales of two levels.** A vote on a scale of two levels only says on which side of one threshold the opinion lies: $p_{qlg2} = \Phi((\mu_{ql} - t_{g1}) / \sigma_{ql})$ fixes the ratio $(\mu_{ql} - t_{g1}) / \sigma_{ql}$, and not the mean and the standard deviation apart. When every vote of a co-cluster is on a scale of two levels, the code therefore holds $\sigma_{ql} = \sigma_{\mathrm{u}}$ and estimates $\mu_{ql}$ alone, by the first half of {eq}`eq-response-opinion-by-scale`; the table of the result marks these co-clusters ({py:attr}`~civiqhub_selene.blocks.CoclusterTable.has_fixed_spread`). One vote on a scale of three levels or more is enough to estimate the spread. The same holds on a matrix of one scale of two levels.

### Estimated thresholds

The grids are fixed by default. {py:class}`~civiqhub_selene.blocks.responses.Thresholds` with `mode="estimated"` estimates the inner thresholds of each scale, $t_{g2}, \dots, t_{g,K_g-2}$, and holds its first and its last, $t_{g1}$ and $t_{g,K_g-1}$, where the grid puts them: they anchor the scale on the axis, and fix the location and the unit that the opinions would otherwise share with the thresholds. A scale of $K_g$ levels has $\max(K_g - 3, 0)$ inner thresholds: a scale of fewer than four levels stays fixed. At the opinions $\mu_{ql}$ and $\sigma_{ql}$ of an update, the inner thresholds of the scale $g$ raise

$$
\sum_{q=1}^{Q} \sum_{l=1}^{L} \sum_{k=1}^{K_g} N_{qlgk} \log p_{qlgk},
$$ (eq-response-thresholds-estimated)

with $p_{qlgk}$ of {eq}`eq-response-probit-by-scale`, under the order $t_{g1} < \dots < t_{g,K_g-1}$. This sum is concave in the thresholds (Pratt 1981): each update takes, after the opinions, Newton steps on the inner thresholds, each halved until it keeps the order and does not lower {eq}`eq-response-thresholds-estimated`, and kept only then. It is a generalised expectation-maximisation: the objective of the block model never decreases. Each inner threshold adds $\tfrac{1}{2} \log n_{\text{votes}}$ to the penalty {eq}`eq-response-penalty`, a Hypothesis, since every co-cluster shares the thresholds of a scale. The estimate tests the convention of the axis: on answers that are not regular, the inner thresholds move away from the regular grid, and the integrated likelihood says whether the move is worth its parameters.

### The penalty of the parameters

The integrated likelihood of a single-membership arrangement subtracts half a logarithm of the sample size per free parameter of the response model. The code counts the parameters of the votes with the number of votes $n_{\text{votes}}$ ({py:attr}`~civiqhub_selene.blocks.VoteMatrix.n_votes`), and the vote rates with the number of shown cells $n_{\text{shown}}$ ({py:attr}`~civiqhub_selene.blocks.VoteMatrix.n_shown`), each at least $1$:

$$
\text{categorical:} \quad \tfrac{1}{2} \, Q L (K - 1) \log n_{\text{votes}} + \tfrac{1}{2} \, Q L \log n_{\text{shown}},
$$

$$
\text{ordinal probit:} \quad Q L \log n_{\text{votes}} + \tfrac{1}{2} \, Q L \log n_{\text{shown}}.
$$ (eq-response-penalty)

The ordinal probit has two parameters of opinion per co-cluster, hence $QL$ halves of a logarithm twice.

## An example

The probabilities of the states of one co-cluster, from its parameters, with the default thresholds of a scale of five levels:

```python
import numpy as np
from civiqhub_selene.blocks.responses import Categorical, OrdinalProbit, OrdinalProbitParameters

print(OrdinalProbit())
print(Categorical())

model = OrdinalProbit()
params = OrdinalProbitParameters(
    vote_rate=np.array([[0.8]]),
    latent_mean=np.array([[4.2]]),
    latent_spread=np.array([[0.9]]),
    thresholds=np.array([1.5, 2.5, 3.5, 4.5]),
)
states = np.exp(model.log_probabilities(params))[0, 0]
print(np.round(states, 4).tolist())
print(round(float(states.sum()), 12))
```

```text
OrdinalProbit(thresholds=None, min_spread=0.3, min_probability=1e-06, max_inner_iterations=10)
Categorical(smoothing=0.5, min_probability=1e-06)
[0.2, 0.0011, 0.0225, 0.1511, 0.3298, 0.2956]
1.0
```

The first number is $F_{ql0} = 1 - \nu_{ql} = 0.2$; the five others are $0.8 \, p_{qlk}$, and they sum to $1$ with it. Given counts of a co-cluster, two abstentions and $20$ votes leaning to the top of the scale, one update moves the rate to $20 / 22$ and the opinion towards the votes:

```python
counts = np.array([[[2.0, 0.0, 1.0, 3.0, 10.0, 6.0]]])
updated = model.update(counts, params)
print(round(float(updated.vote_rate[0, 0]), 4))
print(round(float(updated.latent_mean[0, 0]), 4), round(float(updated.latent_spread[0, 0]), 4))
```

```text
0.9091
4.0965 0.8388
```

On two scales, the unit axis gives the grid $\tfrac{1}{2}$ to the scale of two answers and $\tfrac{1}{4}, \tfrac{1}{2}, \tfrac{3}{4}$ to the scale of four. One opinion of mean $0.6$ and standard deviation $0.2$ gives the states of a cell of each scale, the states past the levels of a scale having no probability; then one update pools $30$ "yes" out of $40$ votes and $40$ votes leaning to the top of the scale of four into one opinion:

```python
from civiqhub_selene.blocks import OrdinalScale
from civiqhub_selene.blocks.responses import Thresholds

yes_no = OrdinalScale((0, 1))
four = OrdinalScale((1, 2, 3, 4))
rule = Thresholds(axis="unit")
print(rule.grid_of(yes_no, "unit"), rule.grid_of(four, "unit"))

params = OrdinalProbitParameters(
    vote_rate=np.array([[0.8]]),
    latent_mean=np.array([[0.6]]),
    latent_spread=np.array([[0.2]]),
    thresholds=(np.array([0.5]), np.array([0.25, 0.5, 0.75])),
    spread_unit=0.25,
)
states = np.exp(model.log_probabilities_by_scale(params))[0, 0]
print(np.round(states[0, :3], 4).tolist())
print(np.round(states[1], 4).tolist())

counts = np.zeros((1, 1, 2, 5))
counts[0, 0, 0, :3] = [5, 10, 30]
counts[0, 0, 1, :] = [5, 2, 6, 14, 18]
pooled = OrdinalProbit(max_inner_iterations=200).update(counts, params)
print(round(float(pooled.vote_rate[0, 0]), 4))
print(round(float(pooled.latent_mean[0, 0]), 4), round(float(pooled.latent_spread[0, 0]), 4))
```

```text
(0.5,) (0.25, 0.5, 0.75)
[0.2, 0.2468, 0.5532]
[0.2, 0.032, 0.2148, 0.3719, 0.1813]
0.8889
0.708 0.2801
```

## What the code does exactly

| Formula | Code |
| --- | --- |
| Protocol | `src/civiqhub_selene/blocks/responses.py:46-81` |
| {eq}`eq-response-states`, floored at $10^{-300}$ before the logarithm | `state_log_probabilities`, `blocks/coclusters.py:78-89` |
| {eq}`eq-response-rate`, bounds, empty co-cluster | `_vote_rate`, `blocks/responses.py:84-95` |
| {eq}`eq-response-categorical` | `Categorical.update`, `blocks/responses.py:180-192` |
| {eq}`eq-response-probit`, default thresholds | `blocks/_numerics.py:107-148`, `blocks/responses.py:313-317` |
| {eq}`eq-response-moments` | `truncated_moments`, `blocks/_numerics.py:151-178` |
| {eq}`eq-response-opinion`, stop at $10^{-8}$ | `fit_latent_normal`, `blocks/_numerics.py:230-297` |
| {eq}`eq-response-penalty` | `blocks/responses.py:194-204`, `:348-358` |
| Starting parameters | `blocks/responses.py:163-174`, `:300-322` |
| {eq}`eq-response-probit-by-scale`, grid of each scale | `OrdinalProbit._vote_shares`, `level_probabilities_by_scale` and `scale_bounds` in `blocks/_numerics.py` |
| {eq}`eq-response-axes`, `"auto"` | `Thresholds.grid_of` and `Thresholds.axis_for`, `blocks/responses.py` |
| {eq}`eq-response-opinion-by-scale`, spread held on two levels | `fit_latent_normal_by_scale`, `blocks/_numerics.py`; `OrdinalProbit.update` |
| {eq}`eq-response-thresholds-estimated`, ends held, Newton steps kept when they keep the order and raise the sum | `fit_thresholds` and `threshold_objective`, `blocks/_numerics.py`; `OrdinalProbit.update`, `OrdinalProbit.complexity_penalty` |

The code also does what the formulas do not say:

- **Masses in the tails.** {eq}`eq-response-probit` is computed without cancellation: when the lower threshold of an interval is above the mean ($z_{ql,k-1} > 0$), the difference is taken on the survival functions, $\Phi(-z_{ql,k-1}) - \Phi(-z_{qlk})$, which has the same value. $\Phi$ itself is computed as $\tfrac{1}{2} \operatorname{erfc}(-z / \sqrt{2})$, accurate in the lower tail.
- **Floors.** Each share $p_{qlk}$ of the ordinal probit is floored at $10^{-12}$ before it enters {eq}`eq-response-states`, and the shares are not normalised again; each probability $F_{qls}$ is floored at $10^{-300}$ before its logarithm. In the moments, a mass is floored at $10^{-300}$, $m^{(1)}_{qlk}$ is kept between $t_{k-1}$ and $t_k$, and $m^{(2)}_{qlk}$ is kept above $\big(m^{(1)}_{qlk}\big)^2$.
- **Starting parameters.** Both models start every co-cluster at the same vote rate: the votes over the shown cells of the whole matrix, weighted by $w_i$ when the votes carry weights, and bounded. The categorical model draws each distribution from a Dirichlet law of concentration $5$ on each level; the ordinal probit starts each mean at $(K + 1) / 2$ plus a normal draw of standard deviation $0.5$, and each $\sigma_{ql}$ at $1$.

## Limits

- The thresholds are the same for every item of a scale and every co-cluster, and fixed: the model reads the levels of every scale on one axis, whose alignment between scales is a convention, not an estimate.
- The ordinal probit gives each co-cluster one normal opinion: a co-cluster split in two camps reads as a wide spread around the middle, not as two modes.
- The vote rate is a property of the co-cluster: whether a voter votes does not depend on its opinion in the model.

## References

- Corneli, M., Bouveyron, C. and Latouche, P. (2020). Co-clustering of ordinal data via latent continuous random variables and not missing at random entries. *Journal of Computational and Graphical Statistics*, 29(4), 771–785. doi:10.1080/10618600.2020.1739533. The code takes from it the hidden normal opinion cut by thresholds.
- Pratt, J. W. (1981). Concavity of the log likelihood. *Journal of the American Statistical Association*, 76(373), 103–106. doi:10.1080/01621459.1981.10477614. The concavity of the likelihood of an ordered probit in its thresholds, which makes the Newton steps on the inner thresholds safe.
- Dempster, A. P., Laird, N. M. and Rubin, D. B. (1977). Maximum likelihood from incomplete data via the EM algorithm. *Journal of the Royal Statistical Society, Series B*, 39(1), 1–22. doi:10.1111/j.2517-6161.1977.tb01600.x
