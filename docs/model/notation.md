# Notation

This page fixes the notation of the whole section. It was checked against the code of `develop` at commit `97a0f15`. Every other page of the section, and the formulas the docstrings and the notebooks write, use these letters with these meanings. A letter names one object only, everywhere.

Each symbol is followed by the name the code gives it, as a link to the reference. A name that starts with an underscore is internal to the code: it is cited, without a link, so that the reader can find it in the source.

## Conventions

- **Indices start at 1 in the formulas, at 0 in the code.** The voter $i$ of a formula, from $1$ to $M$, is the row $i - 1$ of an array; the group $q$, from $1$ to $Q$, is the value $q - 1$ of {py:attr}`~civiqhub_selene.blocks.results.Coclustering.group_index`. Levels are the exception: they run from $1$ to $K$ in both, and $0$ means "no vote".
- **Logarithms are natural.**
- **Vectors are bold lower-case letters**, such as the profile $\boldsymbol{c}$ or the memberships $\boldsymbol{\theta}_i = (\theta_{i1}, \dots, \theta_{iQ})$ of a voter.
- **When one co-cluster is meant, its indices may be dropped**: $p_k$ stands for $p_{qlk}$ in a formula about the co-cluster $(q, l)$.
- $\mathbf{1}\{\cdot\}$ is 1 when the condition holds and 0 otherwise.
- $e$ and $\pi$ written alone are the numbers $2.718\ldots$ and $3.141\ldots$; the share $\pi_q$ of a group always carries its index.

## The votes

| Symbol | Meaning | Name in the code | Shape |
| --- | --- | --- | --- |
| $M$ | Number of voters | First entry of {py:attr}`~civiqhub_selene.blocks.VoteMatrix.shape` | |
| $P$ | Number of items | Second entry of {py:attr}`~civiqhub_selene.blocks.VoteMatrix.shape` | |
| $i$, $j$ | A voter, from $1$ to $M$; an item, from $1$ to $P$ | Rows of {py:attr}`~civiqhub_selene.blocks.VoteMatrix.voters`, columns of {py:attr}`~civiqhub_selene.blocks.VoteMatrix.items` | |
| $K$ | Number of levels of the scale: the scores, in order; with several scales, the levels of the longest one | {py:attr}`~civiqhub_selene.blocks.OrdinalScale.n_levels`, {py:attr}`~civiqhub_selene.blocks.VoteMatrix.n_levels` | |
| $\mathcal{G}$ | The scales of the items, one for each distinct set of scores; a single scale when every item has the same | {py:attr}`~civiqhub_selene.blocks.ItemScales.scales`, {py:attr}`~civiqhub_selene.blocks.VoteMatrix.scales` | |
| $g$ | A scale of $\mathcal{G}$ | | |
| $g_j$ | The scale of the item $j$ | {py:attr}`~civiqhub_selene.blocks.ItemScales.scale_index`, {py:meth}`~civiqhub_selene.blocks.VoteMatrix.scale_of` | $(P,)$ |
| $K_g$ | Number of levels of the scale $g$ | {py:attr}`~civiqhub_selene.blocks.ItemScales.levels_by_item`, item by item | |
| $k$ | A level, from $1$ to $K$ | | |
| $\Omega$ | The set of shown cells $(i, j)$: the item $j$ was shown to the voter $i$ | {py:attr}`~civiqhub_selene.blocks.VoteMatrix.shown` | $(M, P)$ |
| $s$ | A state of a shown cell: $0$ for shown without a vote, $k$ for a vote at level $k$ | | |
| $s_{ij}$ | The state of the shown cell $(i, j)$ | {py:attr}`~civiqhub_selene.blocks.VoteMatrix.levels`, read on $\Omega$ only | $(M, P)$ |
| $S_{ijs}$ | $\mathbf{1}\{(i, j) \in \Omega \text{ and } s_{ij} = s\}$; zero for every $s$ on a cell that is not shown | `VoteMatrix._states` (internal) | $(M, P, K + 1)$ |
| $w_i$ | Survey weight of the voter $i$, scaled to average $1$; $w_i = 1$ when the votes carry no weight | {py:attr}`~civiqhub_selene.blocks.VoteMatrix.voter_weights`, scaled in `VoteMatrix._likelihood_weights` (internal) | $(M,)$ |

A cell has three states that the notation keeps apart: not shown ($(i, j) \notin \Omega$), shown without a vote ($s_{ij} = 0$), voted ($s_{ij} = k \geq 1$). A cell that is not shown is never a zero and never a neutral vote.

## The arrangement

| Symbol | Meaning | Name in the code | Shape |
| --- | --- | --- | --- |
| $Q$ | Number of groups of voters | {py:attr}`~civiqhub_selene.blocks.base.BlockModel.n_groups` | |
| $L$ | Number of blocks of items | {py:attr}`~civiqhub_selene.blocks.base.BlockModel.n_blocks` | |
| $q$, $l$ | A group, from $1$ to $Q$; a block, from $1$ to $L$; $(q, l)$ is a co-cluster | | |
| $R_i$ | The group of the voter $i$, when each voter belongs to one group | {py:attr}`~civiqhub_selene.blocks.results.Coclustering.group_index` | $(M,)$ |
| $C_j$ | The block of the item $j$, when each item belongs to one block | {py:attr}`~civiqhub_selene.blocks.results.Coclustering.block_index` | $(P,)$ |
| $N_{qls}$ | Count of the co-cluster $(q, l)$ in the state $s$: the shown cells of its voters and items in that state, each weighted by $w_i$; an expected, fractional count in the mixed-membership model | {py:meth}`~civiqhub_selene.blocks.GroupedVotes.counts` (without weights); `_numerics.count_states` (internal) | $(Q, L, K + 1)$ |
| $N_{qlgs}$ | The same count, on the items of the scale $g$ only; with one scale, $N_{qls}$ | {py:meth}`~civiqhub_selene.blocks.GroupedVotes.counts_by_scale` (without weights); `_numerics.count_states_by_scale` (internal) | $(Q, L, \lvert\mathcal{G}\rvert, K + 1)$ |
| $A_{iq}$ | Score of the voter $i$ for the group $q$, the blocks being fixed, in the latent block model | `single._voter_scores` (internal) | $(M, Q)$ |
| $B_{jl}$ | Score of the item $j$ for the block $l$, the groups being fixed, in the latent block model | `single._item_scores` (internal) | $(P, L)$ |
| $\pi_q$ | Share of the voters in the group $q$ | `group_shares` of `single._HardState` (internal) | $(Q,)$ |
| $\rho_l$ | Share of the items in the block $l$ | `block_shares` of `single._HardState` (internal) | $(L,)$ |
| $\theta_{iq}$ | Degree of membership of the voter $i$ in the group $q$; $\sum_q \theta_{iq} = 1$ | {py:attr}`~civiqhub_selene.blocks.results.Coclustering.group_memberships` of a {py:class}`~civiqhub_selene.blocks.results.MixedCoclustering` | $(M, Q)$ |
| $\phi_{jl}$ | Degree of membership of the item $j$ in the block $l$; $\sum_l \phi_{jl} = 1$ | {py:attr}`~civiqhub_selene.blocks.results.Coclustering.block_memberships` of a {py:class}`~civiqhub_selene.blocks.results.MixedCoclustering` | $(P, L)$ |
| $\hat{\theta}_{iq}$, $\hat{\phi}_{jl}$ | The soft reading of a single-membership arrangement: the score of each voter for each group, and of each item for each block, turned into shares that sum to 1 | {py:attr}`~civiqhub_selene.blocks.results.Coclustering.group_memberships` and {py:attr}`~civiqhub_selene.blocks.results.Coclustering.block_memberships` of a {py:class}`~civiqhub_selene.blocks.results.SingleCoclustering` | $(M, Q)$, $(P, L)$ |
| $c_l$ | The role of a voter on the block $l$: the group it follows on that block | Its probabilities are $\kappa_{ilq}$ | |
| $\boldsymbol{c}$ | A profile: $\boldsymbol{c} = (c_1, \dots, c_L)$, one of the $Q^L$ choices of a group per block | {py:attr}`~civiqhub_selene.blocks.results.MixedCoclustering.profiles`; their probabilities are $\eta_{i\boldsymbol{c}}$ | $(Q^L, L)$ |
| $G_{ij\boldsymbol{c}}$ | Probability of the state of the shown cell $(i, j)$ under the profile $\boldsymbol{c}$, the block of the cell summed out | `den` in `mixed._expect` (internal) | |
| $\eta_{i\boldsymbol{c}}$ | Posterior probability of the profile $\boldsymbol{c}$ of the voter $i$ | {py:attr}`~civiqhub_selene.blocks.results.MixedCoclustering.profile_probabilities` | $(M, Q^L)$ |
| $\kappa_{ilq}$ | Posterior probability that the voter $i$ follows the group $q$ on the block $l$ | {py:attr}`~civiqhub_selene.blocks.results.MixedCoclustering.role_by_block` | $(M, L, Q)$ |
| $H_{jl}$ | Expected number of shown cells of the item $j$ that take the block $l$, weighted by $w_i$ | `item_mass` in `mixed._expect` (internal) | $(P, L)$ |
| $\alpha$ | Concentration of the symmetric Dirichlet prior on $\boldsymbol{\theta}_i$ | {py:attr}`~civiqhub_selene.blocks.MixedMembershipBlockModel.voter_concentration` | |
| $\beta$ | Concentration of the symmetric Dirichlet prior on $\boldsymbol{\phi}_j$ | {py:attr}`~civiqhub_selene.blocks.MixedMembershipBlockModel.item_concentration` | |

## The response models

| Symbol | Meaning | Name in the code | Shape |
| --- | --- | --- | --- |
| $F_{qls}$ | Probability that a shown cell of the co-cluster $(q, l)$ is in the state $s$; $\sum_s F_{qls} = 1$ | The exponential of {py:meth}`~civiqhub_selene.blocks.responses.ResponseModel.log_probabilities` | $(Q, L, K + 1)$ |
| $\nu_{ql}$ | Vote rate: probability that a shown cell of the co-cluster $(q, l)$ carries a vote; $F_{ql0} = 1 - \nu_{ql}$ | {py:attr}`~civiqhub_selene.blocks.responses.CategoricalParameters.vote_rate`, {py:attr}`~civiqhub_selene.blocks.responses.OrdinalProbitParameters.vote_rate` | $(Q, L)$ |
| $p_{qlk}$ | Share of the votes at the level $k$ in the co-cluster $(q, l)$; $F_{qlk} = \nu_{ql} \, p_{qlk}$ | {py:attr}`~civiqhub_selene.blocks.responses.CategoricalParameters.vote_shares`; computed from $\mu_{ql}$, $\sigma_{ql}$ and $t_k$ by the ordinal probit | $(Q, L, K)$ |
| $\mu_{ql}$ | Mean of the hidden opinion in the co-cluster $(q, l)$, on the axis of levels | {py:attr}`~civiqhub_selene.blocks.responses.OrdinalProbitParameters.latent_mean` | $(Q, L)$ |
| $\sigma_{ql}$ | Standard deviation of the hidden opinion in the co-cluster $(q, l)$ | {py:attr}`~civiqhub_selene.blocks.responses.OrdinalProbitParameters.latent_spread` | $(Q, L)$ |
| $t_k$ | The thresholds that cut the hidden opinion into levels, $t_1 < \dots < t_{K-1}$, with $t_0 = -\infty$ and $t_K = +\infty$ | {py:attr}`~civiqhub_selene.blocks.responses.OrdinalProbit.thresholds`, {py:attr}`~civiqhub_selene.blocks.responses.OrdinalProbitParameters.thresholds` | $(K - 1,)$ |
| $t_{gk}$ | The thresholds of the scale $g$, $t_{g1} < \dots < t_{g,K_g - 1}$, with $t_{g0} = -\infty$ and $t_{gK_g} = +\infty$; with one scale, $t_k$ | {py:class}`~civiqhub_selene.blocks.responses.Thresholds`, {py:attr}`~civiqhub_selene.blocks.responses.OrdinalProbitParameters.thresholds_by_scale` | $(K_g - 1,)$ per scale |
| $F_{qlgs}$, $p_{qlgk}$ | $F_{qls}$ and $p_{qlk}$ for a cell of an item of the scale $g$; with one scale, $F_{qls}$ and $p_{qlk}$ | {py:meth}`~civiqhub_selene.blocks.responses.ResponseModel.log_probabilities_by_scale`; {py:attr}`~civiqhub_selene.blocks.CoclusterTable.vote_shares_by_scale` | $(Q, L, \lvert\mathcal{G}\rvert, K + 1)$, $(Q, L, K_g)$ per scale |
| $m^{(1)}_{qlgk}$, $m^{(2)}_{qlgk}$ | The moments of the hidden opinion of a vote at the level $k$ of the scale $g$ | `truncated_moments_by_scale` in `_numerics` (internal) | $(Q, L, \lvert\mathcal{G}\rvert, K)$ |
| $\sigma_{\mathrm{u}}$ | One level of the longest scale on the axis of the thresholds: $1$ on the axis of levels, $1 / K$ on the unit axis | {py:attr}`~civiqhub_selene.blocks.responses.OrdinalProbitParameters.spread_unit` | |
| $\Phi$ | The standard normal distribution function | `_numerics.norm_cdf` (internal) | |
| $\Phi'$ | The standard normal density | `_numerics.norm_pdf` (internal) | |
| $y_{ij}$ | The hidden opinion behind the vote of the cell $(i, j)$, in the ordinal probit | | |
| $z_{qlk}$ | A standardised threshold, $(t_k - \mu_{ql}) / \sigma_{ql}$ | `_numerics._standardised_bounds` (internal) | $(Q, L, K + 1)$ |
| $r_{qlk}$, $u_{qlk}$ | The ratios of densities over $p_{qlk}$ that give the truncated moments | `truncated_moments` in `_numerics` (internal) | $(Q, L, K)$ |
| $m^{(1)}_{qlk}$, $m^{(2)}_{qlk}$ | First and second moments of the hidden opinion of a vote at the level $k$ in the co-cluster $(q, l)$ | `truncated_moments` in `_numerics` (internal) | $(Q, L, K)$ |
| $n_{ql}$ | Number of votes of the co-cluster $(q, l)$, $\sum_{k \geq 1} N_{qlk}$ | | $(Q, L)$ |
| $\lambda$ | The pseudo-count of the categorical model | {py:attr}`~civiqhub_selene.blocks.responses.Categorical.smoothing` | |
| $\varepsilon$ | The bound that keeps a vote rate away from $0$ and $1$ | {py:attr}`~civiqhub_selene.blocks.responses.Categorical.min_probability`, {py:attr}`~civiqhub_selene.blocks.responses.OrdinalProbit.min_probability` | |
| $\sigma_{\min}$ | The floor of $\sigma_{ql}$ | {py:attr}`~civiqhub_selene.blocks.responses.OrdinalProbit.min_spread` | |
| $n_{\text{votes}}$, $n_{\text{shown}}$ | The numbers of votes and of shown cells of the matrix | {py:attr}`~civiqhub_selene.blocks.VoteMatrix.n_votes`, {py:attr}`~civiqhub_selene.blocks.VoteMatrix.n_shown` | |

## The start

| Symbol | Meaning | Name in the code | Shape |
| --- | --- | --- | --- |
| $\tau$ | The regulariser of the spectral start, as a multiple of the average degree of each side | {py:attr}`~civiqhub_selene.blocks.SpectralStart.regularization` | |
| $\gamma$ | The leverage threshold of thresholded spectral clustering | {py:attr}`~civiqhub_selene.blocks.SpectralStart.leverage_threshold` | |
| $\psi$ | The random permutation of a random start | | |
| $\mathbf{X}$ | The matrix the spectral start decomposes: the states, one column per (item, state), or the centred levels | `spectral._decomposition` (internal) | $(M, P(K + 1))$ or $(M, P)$ |
| $\mathbf{D}_{\mathrm{r}}$, $\mathbf{D}_{\mathrm{c}}$ | The diagonal matrices of the degrees of the rows and of the columns of $\mathbf{X}$ | `spectral._laplacian` (internal) | |
| $\bar{d}_{\mathrm{r}}$, $\bar{d}_{\mathrm{c}}$ | The average degree of the rows, of the columns | | |
| $\tau_{\mathrm{r}}$, $\tau_{\mathrm{c}}$ | The regularisers of the rows and of the columns, $\tau \bar{d}_{\mathrm{r}}$ and $\tau \bar{d}_{\mathrm{c}}$ | | |
| $\mathbf{I}$ | The identity matrix | | |
| $\mathbf{X}_\tau$ | The regularised Laplacian of $\mathbf{X}$ | `spectral._laplacian` (internal) | |
| $\mathbf{U}$, $\mathbf{V}$ | The leading singular vectors on the side of the voters and of the items; $\mathbf{U}_{i\cdot}$ is the row of the voter $i$ | {py:attr}`~civiqhub_selene.blocks.spectral.SpectralEmbedding.voter_vectors`, {py:attr}`~civiqhub_selene.blocks.spectral.SpectralEmbedding.item_vectors` | $(M, d)$, $(P, d')$ |
| $\delta_k$ | The singular values of $\mathbf{X}_\tau$, in decreasing order | {py:attr}`~civiqhub_selene.blocks.spectral.SpectralEmbedding.voter_singular_values`, {py:attr}`~civiqhub_selene.blocks.spectral.SpectralEmbedding.item_singular_values` | |

## The quantities a result reports

| Symbol | Meaning | Name in the code |
| --- | --- | --- |
| $\mathcal{L}_{\mathrm{c}}$ | The classified log-likelihood of a single-membership arrangement, which its estimation maximises | {py:attr}`~civiqhub_selene.blocks.results.Coclustering.objective` of a {py:class}`~civiqhub_selene.blocks.results.SingleCoclustering` |
| $\mathcal{L}$ | The observed log-likelihood of the mixed-membership model | {py:attr}`~civiqhub_selene.blocks.results.MixedCoclustering.log_likelihood` |
| $\mathcal{J}$ | The objective of the mixed-membership model: $\mathcal{L}$ plus the log-densities of the Dirichlet priors on the memberships, up to a constant, the prior of each voter weighted by $w_i$ | {py:attr}`~civiqhub_selene.blocks.results.Coclustering.objective` of a {py:class}`~civiqhub_selene.blocks.results.MixedCoclustering` |
| $\mathrm{ICL}$ | The integrated completed likelihood, which compares shapes | {py:attr}`~civiqhub_selene.blocks.results.SingleCoclustering.integrated_likelihood` |
| $\operatorname{pen}_F$ | The penalty of the parameters of the response model in the ICL | {py:meth}`~civiqhub_selene.blocks.responses.ResponseModel.complexity_penalty` |
| $\Omega_{\mathrm{in}}$, $\Omega_{\mathrm{out}}$ | The shown cells kept for the estimation, and those held out | {py:meth}`~civiqhub_selene.blocks.VoteMatrix.hold_out` |
| $\mathcal{L}_{\mathrm{out}}$ | The log-likelihood of the held-out cells under a result estimated on the others | {py:meth}`~civiqhub_selene.blocks.results.PredictsVotes.log_probability`, summed over $\Omega_{\mathrm{out}}$ |
| $\bar{\rho}_l$ | The share of the block $l$ in a result: of the items whose block is $l$, or the mean of $\phi_{jl}$, normalised | {py:attr}`~civiqhub_selene.blocks.partition.Cluster.share` of {py:attr}`~civiqhub_selene.blocks.results.Coclustering.blocks` |
| $\operatorname{agree}(q, r)$ | How much two groups agree: one minus the total variation distance of their votes, averaged over the blocks | {py:func}`~civiqhub_selene.viz.graphs.group_graph` |
| $\mathcal{I}$ | The identifiers two partitions cover | {py:attr}`~civiqhub_selene.blocks.Partition.ids` |
| $T_{qr}$ | The number of identifiers in the cluster $q$ of one partition and $r$ of the other | Inside {py:meth}`~civiqhub_selene.blocks.Partition.agreement_with` |
| $\chi$ | A renumbering of the clusters of a partition | |
| $\operatorname{match}$ | The share of identifiers two partitions arrange in the same way, after the best renumbering | {py:meth}`~civiqhub_selene.blocks.Partition.agreement_with` |

The pages that define these quantities give their formulas and the lines of the code that compute them.

## The ordinal family

| Symbol | Meaning | Name in the code |
| --- | --- | --- |
| $N_k$ | The number of votes at the level $k$, from $1$ to $5$, of one item or of one co-cluster: $N_{qlk}$ when the co-cluster $(q, l)$ is meant | {py:attr}`~civiqhub_selene.ordinal.VoteCounts.counts` |
| $n$ | The number of votes of the counts, $\sum_{k=1}^{5} N_k$ | {py:attr}`~civiqhub_selene.ordinal.VoteCounts.total` |
| $f_k$ | The share of the votes at the level $k$, an exact fraction | {py:attr}`~civiqhub_selene.ordinal.VoteCounts.shares` |
| $f_{\leq k}$ | The share of the votes at the level $k$ or below, for $k$ from $1$ to $4$ | {py:attr}`~civiqhub_selene.ordinal.VoteCounts.cumulative_shares` |
