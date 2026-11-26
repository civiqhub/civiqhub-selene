# Cost and numerical safeguards

Checked against the code of `develop` at commit `97a0f15`.

## In plain words

This page says how the cost of an estimation grows with the size of the votes and the shape, as the code computes it, and which numerical safeguards make the formulas of the other pages computable with floating-point numbers. The orders of growth are read from the code. The times at the end are measurements on one machine, not a guarantee.

## Cost

### The latent block model

An iteration of {py:class}`~civiqhub_selene.blocks.LatentBlockModel` reads the array $S_{ijs}$ of $M \cdot P \cdot (K + 1)$ numbers a few times ([The latent block model](latent-block-model.md)):

- the counts $N_{qls}$, one contraction of $S$ with the indicators of the groups and of the blocks: of the order of $M P (K + 1) \, Q L$ operations as written;
- the scores of the voters $A_{iq}$: of the order of $M P (K + 1) \, Q$;
- the scores of the items $B_{jl}$: of the order of $M P (K + 1) \, L$;
- the objective: of the order of $M P (K + 1)$;
- the update of the response model, on $Q L (K + 1)$ counts: negligible next to the rest; the ordinal probit runs at most {py:attr}`~civiqhub_selene.blocks.responses.OrdinalProbit.max_inner_iterations` steps over the $Q L$ co-clusters.

An iteration therefore costs of the order of $M P (K + 1)(Q L + Q + L)$, and an estimation that times the number of iterations of each start and the number of starts {py:attr}`~civiqhub_selene.blocks.base.BlockModel.n_starts`. The memory is that of $S$: $M P (K + 1)$ floating-point numbers, computed once per matrix.

### The mixed-membership model

The exact expectation step of {py:class}`~civiqhub_selene.blocks.MixedMembershipBlockModel` goes through the $Q^L$ profiles for every shown cell ([The mixed-membership model](mixed-membership.md)): the probabilities $G_{ij\boldsymbol{c}}$ and the expected counts cost of the order of $M P (K + 1) \, Q^L$ operations per iteration, plus terms of the order of $M L Q^{L+1}$ for the roles and $P (K + 1) L Q^{L+1}$ for the counts. The number of profiles is bounded by {py:data}`~civiqhub_selene.blocks.mixed.MAX_PROFILES`, $4096$; a shape with more is refused. The voters are read in batches, so that a batch holds about two million (voter, item, profile) triples: the memory of the expectation step stays of the order of $2 \times 10^6$ numbers per array, whatever $M$.

### The spectral start

{py:class}`~civiqhub_selene.blocks.SpectralStart` decomposes the matrix $\mathbf{X}$ of the votes ([The starts](initialization.md)): with the encoding `"states"`, a dense matrix of $M$ rows and $P (K + 1)$ columns for the voters, and one of $P$ rows and $M (K + 1)$ columns for the items. A dense singular value decomposition of an $m \times n$ matrix costs of the order of $m \, n \, \min(m, n)$ operations. The decomposition does not depend on the start: it is computed once per matrix, encoding and regulariser, and kept in a cache. Each start then only runs k-means, {py:attr}`~civiqhub_selene.blocks.SpectralStart.n_init` times, on $M$ points of $Q$ coordinates and $P$ points of $L$ coordinates.

## Numerical safeguards

The formulas of the other pages hold in exact arithmetic. The code keeps them computable:

| Safeguard | Where it acts |
| --- | --- |
| Log-sum-exp: the largest term is taken out before the exponential | Posterior of the profiles; soft memberships of the latent block model |
| Floor of $10^{-300}$ before a logarithm | Shares $\pi_q$ and $\rho_l$; probabilities $F_{qls}$; memberships and probabilities $G_{ij\boldsymbol{c}}$ of the mixed model; probability of a held-out cell; masses of the truncated moments |
| Floor of $10^{-12}$ on each share $p_{qlk}$ of the ordinal probit | No level of a co-cluster has the probability $0$ |
| Vote rates kept in $[\varepsilon, 1 - \varepsilon]$, $\varepsilon$ = `min_probability` | No state of a co-cluster has the probability $0$ |
| $\sigma_{ql} \geq \sigma_{\min}$ = `min_spread` | A unanimous co-cluster does not send its spread to $0$ |
| Masses of the upper tail taken on the survival functions, and $\Phi$ written with `erfc` | No cancellation in $\Phi(z_{qlk}) - \Phi(z_{ql,k-1})$ far in a tail |
| Truncated moments kept between their thresholds, and the second above the square of the first | The inner loop of the ordinal probit stays in its domain |
| A co-cluster without any cell keeps its vote rate, one without any vote its opinion or its shares | No division by $0$ in an update |
| A row of memberships that sums to $0$ becomes uniform; each term of the maximisation is floored at $0$ | The mixed model never divides by $0$ |
| A degree of $0$ gives a zero row or column of the Laplacian | The spectral start reads a voter without any shown cell |
| An objective that is not a number never wins | The best start is always a computed one |

## Measurements

One estimation of each model, with its default settings and the seed $0$, on votes drawn by {py:func}`~civiqhub_selene.blocks.synthetic.make_votes` with three planted groups and two planted blocks, $K = 5$ and every cell shown. The median of three runs, on one machine: an Intel Xeon processor at 2.10 GHz with 4 cores, Python 3.13.18, NumPy 2.5.5. These are measurements, not a guarantee: other votes, another machine or another version of NumPy give other times, and the number of iterations depends on the votes.

| Voters $M$ | Items $P$ | `LatentBlockModel(3, 2)` | with `SpectralStart()` | `MixedMembershipBlockModel(3, 2)` |
| --- | --- | --- | --- | --- |
| 150 | 24 | 0.13 s | 0.10 s | 1.3 s |
| 600 | 48 | 1.8 s | 0.29 s | 4.5 s |
| 1500 | 96 | 2.1 s | 1.3 s | 47 s |

The mixed-membership model costs more, by its $Q^L = 9$ profiles and by its number of iterations, which ran from 54 to 150 in the start kept, against 6 to 13 for the latent block model. The spectral start shortens the starts of the latent block model.

## What the code does exactly

| Rule | Code |
| --- | --- |
| Contractions of an iteration of the latent block model | `src/civiqhub_selene/blocks/single.py:48-71`, `:181-215`; `count_states`, `blocks/_numerics.py:47-77` |
| Expectation of the mixed model, batches | `_expect`, `blocks/mixed.py:103-161`; `_CHUNK_CELLS`, `:34-35`; `MAX_PROFILES`, `:28-29`, `:296-303` |
| Decomposition, cache | `blocks/spectral.py:48`, `:315-375` |
| Log-sum-exp | `logsumexp`, `blocks/_numerics.py:80-88` |
| Floors $10^{-300}$ | `blocks/single.py:24`; `blocks/coclusters.py:88`; `blocks/mixed.py:31`; `blocks/results.py:531`; `blocks/_numerics.py:34` |
| Floor $10^{-12}$, empty co-clusters | `blocks/responses.py:42-43`, `:84-95`, `:373-379`; `blocks/_numerics.py:35` |
| Survival functions, `erfc` | `blocks/_numerics.py:91-97`, `:107-121` |
| Truncated moments, `min_spread` | `blocks/_numerics.py:151-178`, `:230-297` |
| Memberships of the mixed model | `blocks/mixed.py:95-100`, `:328-336` |
| Zero degrees | `blocks/spectral.py:371-375` |
| An objective that is not a number | `blocks/iterative.py:40-43` |

## Limits

- The orders of growth count the operations as the code writes them; NumPy may run a contraction faster or slower than its count suggests.
- The mixed-membership model is limited to $4096$ profiles, and its cost grows as $Q^L$.
- The spectral start holds dense matrices: its memory grows as $M \cdot P \cdot (K + 1)$.
