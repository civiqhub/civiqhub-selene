# Cost and numerical safeguards

Checked against the code of `develop` at commit `c29d115`.

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

### Items on several scales

With several scales ([The data: items on different scales](data.md#items-on-different-scales)), $K$ is the number of levels of the longest scale. The array $S_{ijs}$ keeps that width in every column: in a column on a shorter scale, the states past its levels are zeros, held and read like the others. The cost and the memory above therefore count $K + 1$ states per cell, whatever the scale of the item. The counts $N_{qlgs}$ are taken on the columns of each scale in turn, and the scores of the items read each column on the table of its scale: the work stays of the same order, plus one pass per scale of $\mathcal{G}$. The update of the response model reads $Q L \, \lvert \mathcal{G} \rvert \, (K + 1)$ counts.

### The mixed-membership model

The exact expectation step of {py:class}`~civiqhub_selene.blocks.MixedMembershipBlockModel` goes through the $Q^L$ profiles for every shown cell ([The mixed-membership model](mixed-membership.md)): the probabilities $G_{ij\boldsymbol{c}}$ and the expected counts cost of the order of $M P (K + 1) \, Q^L$ operations per iteration, plus terms of the order of $M L Q^{L+1}$ for the roles and $P (K + 1) L Q^{L+1}$ for the counts. The number of profiles is bounded by {py:data}`~civiqhub_selene.blocks.mixed.MAX_PROFILES`, $4096$; a shape with more is refused. With several scales, the probability of a cell under each profile is computed for the items of each scale on the table of their scale, one scale after the other, and the expected counts are gathered back scale by scale: the order of the cost is the one above, $K$ the levels of the longest scale. The voters are read in batches, so that a batch holds about two million (voter, item, profile) triples: the memory of the expectation step stays of the order of $2 \times 10^6$ numbers per array, whatever $M$.

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
| Contractions of an iteration of the latent block model | `_voter_scores` and `_item_scores`, `src/civiqhub_selene/blocks/single.py:53-97`; `LatentBlockModel._step`, `:215-258`; `count_states` and `count_states_by_scale`, `blocks/_numerics.py:55-137` |
| $S_{ijs}$ of the width of the longest scale | `VoteMatrix._states`, `blocks/data.py:1120-1131` |
| Expectation of the mixed model, scale by scale, batches | `_expect`, `blocks/mixed.py:116-209`, the loops over the scales `:157-162` and `:188-201`; `_CHUNK_CELLS`, `:37-38`; `MAX_PROFILES`, `:33-34`, `:352-358` |
| Decomposition, cache | `blocks/spectral.py:48`, `:327-420` |
| Log-sum-exp | `logsumexp`, `blocks/_numerics.py:140-150` |
| Floors $10^{-300}$ | `blocks/single.py:29`; `blocks/coclusters.py:32`, `:204`; `blocks/mixed.py:36`; `blocks/results.py:42`, `:594`; `blocks/_numerics.py:43` |
| Floor $10^{-12}$, empty co-clusters | `blocks/responses.py:57-58`, `:138-151`, `:831-841`; `blocks/_numerics.py:44` |
| Survival functions, `erfc` | `norm_cdf` and `interval_mass`, `blocks/_numerics.py:153-183` |
| Truncated moments, `min_spread` | `truncated_moments`, `blocks/_numerics.py:213-240`; `fit_latent_normal_by_scale`, `:416-498` |
| Memberships of the mixed model | `_normalised`, `blocks/mixed.py:108-113`; `MixedMembershipBlockModel._step`, `:384-392` |
| Zero degrees | `_inverse_root`, `blocks/spectral.py:416-420` |
| An objective that is not a number | `_comparable`, `blocks/iterative.py:40-43` |

## Limits

- The orders of growth count the operations as the code writes them; NumPy may run a contraction faster or slower than its count suggests.
- The mixed-membership model is limited to $4096$ profiles, and its cost grows as $Q^L$.
- The spectral start holds dense matrices: its memory grows as $M \cdot P \cdot (K + 1)$.
- On several scales, $S_{ijs}$ and the dense matrices of the spectral start take the width of the longest scale in every column: a short scale costs as much as the longest. A test of the repository runs the mixed-membership model on $4{,}000$ voters and $270$ items on seven scales of $2$ to $11$ answers, with synthetic votes, and checks that its memory stays under $512$ MiB (`test_the_memory_stays_bounded_at_the_size_of_a_national_survey`, `tests/unit/test_mixed_membership_block_model.py:411-435`).
