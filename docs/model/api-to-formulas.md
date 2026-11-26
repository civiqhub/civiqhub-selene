# From the API to the formulas

Checked against the code of `develop` at commit `c29d115`.

## In plain words

Each public name of the engine that holds or computes a quantity of the model is listed here, with the symbol the pages of this section write for it, the equations that define it, the section that derives them, and the file and the lines of the code. The docstrings of the engine link to this page. A name that carries no formula is listed too, and its row says so.

## How to read the tables

- **Name**: the public name, linked to its entry in the reference of the API.
- **Symbol**: the symbols of [Notation](notation.md) that the name holds or computes.
- **Equations**: the numbered equations that define them, each after the page that numbers it: the numbers restart on each page.
- **Section**: the page and the section that derive them.
- **Code**: the file under `src/civiqhub_selene/` and its lines, at the commit above. The lines move with the code; the names do not.

A dash means that the name has no symbol or no numbered equation of its own.

## `civiqhub_selene.blocks`

The names imported from `civiqhub_selene.blocks`.

| Name | Symbol | Equations | Section | Code |
| --- | --- | --- | --- | --- |
| {py:class}`~civiqhub_selene.blocks.Cocluster` | $\nu_{ql}$, $p_{qlk}$, $\sum_{k \geq 1} N_{qlk}$, $\mu_{ql}$, $\sigma_{ql}$ of one co-cluster $(q, l)$ | The response models {eq}`eq-response-states` | [Reading the results: the table of co-clusters](reading-results.md#the-table-of-co-clusters) | `blocks/coclusters.py:47-187` |
| {py:class}`~civiqhub_selene.blocks.CoclusterTable` | The same quantities, for every $q$ and $l$ | The response models {eq}`eq-response-states` | [Reading the results: the table of co-clusters](reading-results.md#the-table-of-co-clusters) | `blocks/coclusters.py:235-529` |
| {py:class}`~civiqhub_selene.blocks.FixedStart` | $R_i$ and $C_j$, given | — | [The starts: the fixed start](initialization.md#the-fixed-start) | `blocks/starts.py:178-254` |
| {py:class}`~civiqhub_selene.blocks.GroupedVotes` | $N_{qls}$ and $N_{qlgs}$, without survey weights | The data {eq}`eq-data-counts`, {eq}`eq-data-counts-by-scale` | [The data: the votes of a group on a block](data.md#the-votes-of-a-group-on-a-block) | `blocks/data.py:1169-1352` |
| {py:class}`~civiqhub_selene.blocks.ItemBlock` | The items of one block; in a result, its share $\bar{\rho}_l$ | Reading the results {eq}`eq-results-agreement` | [Reading the results: how much two groups agree](reading-results.md#how-much-two-groups-agree) | `blocks/partition.py:92-102`; shares of a result, `blocks/results.py:281-287`, `:301-311`, `:523-528` |
| {py:class}`~civiqhub_selene.blocks.ItemScales` | $\mathcal{G}$, $g_j$, $K_g$ | The data {eq}`eq-data-counts-by-scale` | [The data: items on different scales](data.md#items-on-different-scales) | `blocks/data.py:133-367` |
| {py:class}`~civiqhub_selene.blocks.LatentBlockModel` | $Q$, $L$, $\mathcal{L}_{\mathrm{c}}$, $A_{iq}$, $B_{jl}$ | The latent block model {eq}`eq-lbm-objective`, {eq}`eq-lbm-voter-score`, {eq}`eq-lbm-item-score` | [The latent block model](latent-block-model.md) | `blocks/single.py:131-301` |
| {py:class}`~civiqhub_selene.blocks.MixedMembershipBlockModel` | $Q$, $L$, $\alpha$, $\beta$, $\mathcal{J}$ | The mixed-membership model {eq}`eq-mm-objective`, {eq}`eq-mm-profiles`, {eq}`eq-mm-mstep` | [The mixed-membership model](mixed-membership.md) | `blocks/mixed.py:212-460` |
| {py:class}`~civiqhub_selene.blocks.OrdinalScale` | $K$, and the levels $k = 1, \dots, K$ of the scores | — | [The data: the scale](data.md#the-scale) | `blocks/data.py:45-110` |
| {py:class}`~civiqhub_selene.blocks.Partition` | $T_{qr}$, $\chi$, $\operatorname{match}$ | Reading the results {eq}`eq-results-match` | [Reading the results: how much two partitions agree](reading-results.md#how-much-two-partitions-agree) | `blocks/partition.py:117-398`; `agreement_with`, `:330-361` |
| {py:class}`~civiqhub_selene.blocks.RandomStart` | $R_i = \psi(i) \bmod Q$, and likewise $C_j$ | — | [The starts: the random start](initialization.md#the-random-start) | `blocks/starts.py:65-83` |
| {py:class}`~civiqhub_selene.blocks.SpectralStart` | $\mathbf{X}_\tau$, $\tau$, $\gamma$ | The starts {eq}`eq-starts-unit-axis`, {eq}`eq-starts-laplacian` | [The starts: the spectral start](initialization.md#the-spectral-start) | `blocks/starts.py:86-175` |
| {py:class}`~civiqhub_selene.blocks.Vote` | One vote, whose score gives the level $s_{ij}$ | The data {eq}`eq-data-states` | [The data: three states per cell](data.md#three-states-per-cell) | `blocks/data.py:370-375` |
| {py:class}`~civiqhub_selene.blocks.VoteMatrix` | $M$, $P$, $K$, $s_{ij}$, $\Omega$, $S_{ijs}$, $w_i$ | The data {eq}`eq-data-states` | [The data: three states per cell](data.md#three-states-per-cell) | `blocks/data.py:487-1166`; $S_{ijs}$, `:1120-1144`; $w_i$, `:946-955` |
| {py:class}`~civiqhub_selene.blocks.VoterCovariates` | — | — | No formula: attributes of the voters, aligned on the matrix, that no model of the engine reads | `blocks/covariates.py:22-86` |
| {py:class}`~civiqhub_selene.blocks.VoterGroup` | The voters of one group; in a result, its share | — | [Reading the results: memberships and main groups](reading-results.md#memberships-and-main-groups) | `blocks/partition.py:79-89`; shares of a result, `blocks/results.py:281-299`, `:523-528` |
| {py:func}`~civiqhub_selene.blocks.read_survey_csv` | $s_{ij}$, $\Omega$ and $w_i$, read from a file | The data {eq}`eq-data-states` | [The data: three states per cell](data.md#three-states-per-cell) | `blocks/survey.py:18-141` |

The share of a group or of a block in a result is the share of the voters or of the items whose main cluster it is, in a single-membership result, and the mean of their memberships, in a mixed one.

## `civiqhub_selene.blocks.responses`

| Name | Symbol | Equations | Section | Code |
| --- | --- | --- | --- | --- |
| {py:class}`~civiqhub_selene.blocks.responses.ResponseModel` | $F_{qls}$, $\operatorname{pen}_F$ | The response models {eq}`eq-response-states`, {eq}`eq-response-expected`, {eq}`eq-response-penalty` | [The response models: what a block model asks of a response model](response-models.md#what-a-block-model-asks-of-a-response-model) | `blocks/responses.py:64-117` |
| {py:class}`~civiqhub_selene.blocks.responses.Categorical` | $\nu_{ql}$, $p_{qlk}$, $\lambda$, $\varepsilon$ | The response models {eq}`eq-response-rate`, {eq}`eq-response-categorical`, {eq}`eq-response-penalty` | [The response models: the categorical model](response-models.md#the-categorical-model) | `blocks/responses.py:245-359` |
| {py:class}`~civiqhub_selene.blocks.responses.CategoricalParameters` | $\nu_{ql}$, $p_{qlk}$ | The response models {eq}`eq-response-states` | [The response models: the categorical model](response-models.md#the-categorical-model) | `blocks/responses.py:197-242` |
| {py:class}`~civiqhub_selene.blocks.responses.OrdinalProbit` | $\nu_{ql}$, $\mu_{ql}$, $\sigma_{ql}$, $t_k$, $\sigma_{\min}$, $\varepsilon$ | The response models {eq}`eq-response-rate`, {eq}`eq-response-probit`, {eq}`eq-response-moments`, {eq}`eq-response-opinion`, {eq}`eq-response-penalty` | [The response models: the ordinal probit](response-models.md#the-ordinal-probit) | `blocks/responses.py:566-841` |
| {py:class}`~civiqhub_selene.blocks.responses.OrdinalProbitParameters` | $\nu_{ql}$, $\mu_{ql}$, $\sigma_{ql}$, $t_k$, $t_{gk}$, $\sigma_{\mathrm{u}}$ | The response models {eq}`eq-response-probit`, {eq}`eq-response-probit-by-scale` | [The response models: the ordinal probit](response-models.md#the-ordinal-probit) | `blocks/responses.py:491-563` |
| {py:class}`~civiqhub_selene.blocks.responses.Thresholds` | $t_{gk}$, the axis, the estimated inner thresholds | The response models {eq}`eq-response-axes`, {eq}`eq-response-opinion-by-scale`, {eq}`eq-response-thresholds-estimated` | [The response models: items on several scales](response-models.md#items-on-several-scales) | `blocks/responses.py:378-488` |

## `civiqhub_selene.blocks.results`

| Name | Symbol | Equations | Section | Code |
| --- | --- | --- | --- | --- |
| {py:class}`~civiqhub_selene.blocks.results.Coclustering` | The arrangement, the memberships, the table of co-clusters, the objective | The latent block model {eq}`eq-lbm-objective`; The mixed-membership model {eq}`eq-mm-objective` | [Reading the results](reading-results.md) | `blocks/results.py:174-441` |
| {py:class}`~civiqhub_selene.blocks.results.SingleCoclustering` | $\hat{\theta}_{iq}$, $\hat{\phi}_{jl}$, $\mathrm{ICL}$, $\mathcal{L}_{\mathrm{out}}$ | The latent block model {eq}`eq-lbm-memberships`; Choosing the shape {eq}`eq-shape-icl`, {eq}`eq-shape-held-out-single` | [The latent block model: the memberships it reports](latent-block-model.md#the-memberships-it-reports) | `blocks/results.py:448-484` |
| {py:class}`~civiqhub_selene.blocks.results.MixedCoclustering` | $\theta_{iq}$, $\phi_{jl}$, $\kappa_{ilq}$, $\eta_{i\boldsymbol{c}}$, $\mathcal{L}$, $\mathcal{L}_{\mathrm{out}}$ | The mixed-membership model {eq}`eq-mm-likelihood`; Reading the results {eq}`eq-results-mixedness`; Choosing the shape {eq}`eq-shape-held-out-mixed` | [The mixed-membership model: what the value holds](mixed-membership.md#what-the-value-holds) | `blocks/results.py:487-604` |
| {py:class}`~civiqhub_selene.blocks.results.HasIntegratedLikelihood` | $\mathrm{ICL}$ | Choosing the shape {eq}`eq-shape-icl` | [Choosing the shape: the integrated completed likelihood](choosing-the-shape.md#the-integrated-completed-likelihood) | `blocks/results.py:45-52` |
| {py:class}`~civiqhub_selene.blocks.results.PredictsVotes` | $\log F_{R_i C_j g_j s_{ij}}$ for each cell, and its mixed form | Choosing the shape {eq}`eq-shape-held-out-single`, {eq}`eq-shape-held-out-mixed` | [Choosing the shape: the log-likelihood of held-out votes](choosing-the-shape.md#the-log-likelihood-of-held-out-votes) | `blocks/results.py:55-61` |
| {py:class}`~civiqhub_selene.blocks.results.Selection` | — | — | No formula: the trace of a comparison of shapes, which no function of the engine fills ([Choosing the shape: what each model declares](choosing-the-shape.md#what-each-model-declares)) | `blocks/results.py:95-171` |

## The numbers a result reports

| Name | Symbol | Equations | Section | Code |
| --- | --- | --- | --- | --- |
| {py:attr}`~civiqhub_selene.blocks.results.Coclustering.objective` | $\mathcal{L}_{\mathrm{c}}$ for a single-membership result, $\mathcal{J}$ for a mixed one | The latent block model {eq}`eq-lbm-objective`; The mixed-membership model {eq}`eq-mm-objective` | [The estimation: two algorithms, one loop, two objectives](estimation.md#two-algorithms-one-loop-two-objectives) | `blocks/results.py:213`; `blocks/single.py:243-255`; `blocks/mixed.py:431-460` |
| {py:attr}`~civiqhub_selene.blocks.results.SingleCoclustering.integrated_likelihood` | $\mathrm{ICL}$ | Choosing the shape {eq}`eq-shape-icl` | [Choosing the shape: the integrated completed likelihood](choosing-the-shape.md#the-integrated-completed-likelihood) | `blocks/results.py:457`; `blocks/single.py:282-287` |
| {py:attr}`~civiqhub_selene.blocks.results.MixedCoclustering.log_likelihood` | $\mathcal{L}$ | The mixed-membership model {eq}`eq-mm-cell`, {eq}`eq-mm-likelihood` | [The mixed-membership model: the observed log-likelihood, exactly](mixed-membership.md#the-observed-log-likelihood-exactly) | `blocks/results.py:507`; `blocks/mixed.py:116-209`, `:394-427` |
| {py:attr}`~civiqhub_selene.blocks.results.Coclustering.group_memberships` | $\hat{\theta}_{iq}$ for a single-membership result, $\theta_{iq}$ for a mixed one | The latent block model {eq}`eq-lbm-memberships`; The mixed-membership model {eq}`eq-mm-mstep` | [Reading the results: memberships and main groups](reading-results.md#memberships-and-main-groups) | `blocks/results.py:210`; `blocks/single.py:260-301`; `blocks/mixed.py:384-392`, `:394-427` |
| {py:attr}`~civiqhub_selene.blocks.results.Coclustering.block_memberships` | $\hat{\phi}_{jl}$ for a single-membership result, $\phi_{jl}$ for a mixed one | The latent block model {eq}`eq-lbm-memberships`; The mixed-membership model {eq}`eq-mm-mstep` | [Reading the results: memberships and main groups](reading-results.md#memberships-and-main-groups) | `blocks/results.py:211`; `blocks/single.py:260-301`; `blocks/mixed.py:384-392`, `:394-427` |
| {py:attr}`~civiqhub_selene.blocks.results.MixedCoclustering.role_by_block` | $\kappa_{ilq}$ | The mixed-membership model {eq}`eq-mm-profiles` | [The mixed-membership model: expectation](mixed-membership.md#expectation) | `blocks/results.py:504`; `blocks/mixed.py:116-209`, `:394-427` |
| {py:attr}`~civiqhub_selene.blocks.results.MixedCoclustering.profile_probabilities` | $\eta_{i\boldsymbol{c}}$ | The mixed-membership model {eq}`eq-mm-profiles` | [The mixed-membership model: expectation](mixed-membership.md#expectation) | `blocks/results.py:506`; `blocks/mixed.py:116-209`, `:394-427` |
| {py:attr}`~civiqhub_selene.blocks.results.MixedCoclustering.mixedness` | $1 - \max_q \theta_{iq}$ | Reading the results {eq}`eq-results-mixedness` | [Reading the results: mixedness and changes of role](reading-results.md#mixedness-and-changes-of-role) | `blocks/results.py:530-540` |

The memberships, the roles and the posterior of the profiles of a mixed result are those of the last state of the start that was kept: the memberships after its last maximisation, and the expectation computed at them.

## `civiqhub_selene.blocks.spectral`

| Name | Symbol | Equations | Section | Code |
| --- | --- | --- | --- | --- |
| {py:data}`~civiqhub_selene.blocks.spectral.Encoding` | How the votes are written as $\mathbf{X}$ | The starts {eq}`eq-starts-unit-axis`, with several scales | [The starts: the spectral start](initialization.md#the-spectral-start) | `blocks/spectral.py:41-42` |
| {py:data}`~civiqhub_selene.blocks.spectral.ENCODINGS` | The encodings of $\mathbf{X}$, `"states"` and `"scores"` | — | [The starts: the spectral start](initialization.md#the-spectral-start) | `blocks/spectral.py:44-45` |
| {py:class}`~civiqhub_selene.blocks.spectral.SpectralEmbedding` | $\mathbf{U}$, $\mathbf{V}$, $\delta_k$ | The starts {eq}`eq-starts-laplacian` | [The starts: the spectral start](initialization.md#the-spectral-start) | `blocks/spectral.py:51-211` |
| {py:func}`~civiqhub_selene.blocks.spectral.spectral_embedding` | $\mathbf{X}$, $\mathbf{X}_\tau$, $\mathbf{U}$, $\mathbf{V}$ | The starts {eq}`eq-starts-unit-axis`, {eq}`eq-starts-laplacian` | [The starts: the spectral start](initialization.md#the-spectral-start) | `blocks/spectral.py:214-306` |
| {py:func}`~civiqhub_selene.blocks.spectral.n_singular_values` | $\min(M, P (K + 1))$ and $\min(P, M (K + 1))$ under `"states"`, $\min(M, P)$ on both sides under `"scores"` | — | [The starts: the spectral start](initialization.md#the-spectral-start) | `blocks/spectral.py:309-321` |

## `civiqhub_selene.blocks.synthetic`

| Name | Symbol | Equations | Section | Code |
| --- | --- | --- | --- | --- |
| {py:func}`~civiqhub_selene.blocks.synthetic.make_votes` | $s_{ij}$ drawn from $\theta_{iq}$, $\phi_{jl}$, $\nu_{ql}$, $\mu_{ql}$, $\sigma_{ql}$, on the scale $g_j$ of each item | The response models {eq}`eq-response-states`, {eq}`eq-response-probit`, {eq}`eq-response-probit-by-scale`, {eq}`eq-response-axes` | [The data: a generative process](data.md#a-generative-process-make_votes) | `blocks/synthetic.py:132-259` |
| {py:func}`~civiqhub_selene.blocks.synthetic.single_memberships` | $\theta_{iq}$ or $\phi_{jl}$ of $0$ and $1$ only | — | [The data: a generative process](data.md#a-generative-process-make_votes) | `blocks/synthetic.py:69-86` |
| {py:class}`~civiqhub_selene.blocks.synthetic.PlantedTruth` | $\theta_{iq}$, $\phi_{jl}$, and the role $c_l$ of each voter on each block | — | [The mixed-membership model: how the votes are drawn](mixed-membership.md#how-the-votes-are-drawn) | `blocks/synthetic.py:49-66` |
| {py:data}`~civiqhub_selene.blocks.synthetic.DEFAULT_VOTE_RATE` | $\nu_{ql}$ for $Q = 3$ and $L = 2$ | The response models {eq}`eq-response-states` | [The data: a generative process](data.md#a-generative-process-make_votes) | `blocks/synthetic.py:33-34` |
| {py:data}`~civiqhub_selene.blocks.synthetic.DEFAULT_LATENT_MEAN` | $\mu_{ql}$ for $Q = 3$ and $L = 2$ | The response models {eq}`eq-response-probit` | [The data: a generative process](data.md#a-generative-process-make_votes) | `blocks/synthetic.py:36-41` |
| {py:data}`~civiqhub_selene.blocks.synthetic.DEFAULT_LATENT_SPREAD` | $\sigma_{ql}$ for $Q = 3$ and $L = 2$ | The response models {eq}`eq-response-probit` | [The data: a generative process](data.md#a-generative-process-make_votes) | `blocks/synthetic.py:43-44` |

## `civiqhub_selene.ordinal`

| Name | Symbol | Equations | Section | Code |
| --- | --- | --- | --- | --- |
| {py:data}`~civiqhub_selene.ordinal.SCORES` | The five scores, from $-2$ to $+2$: a scale with $K = 5$ | — | [The ordinal family: the five counts](ordinal.md#the-five-counts) | `ordinal/counts.py:20` |
| {py:class}`~civiqhub_selene.ordinal.VoteCounts` | $N_k$, $n$, $f_k$, $f_{\leq k}$, for one item or one co-cluster | The data {eq}`eq-data-counts`; The ordinal family {eq}`eq-ordinal-shares`, {eq}`eq-ordinal-cumulative` | [The ordinal family](ordinal.md) | `ordinal/counts.py:34-161` |

## `civiqhub_selene.viz`

| Name | Symbol | Equations | Section | Code |
| --- | --- | --- | --- | --- |
| {py:func}`~civiqhub_selene.viz.group_graph` | $\operatorname{agree}(q, r)$, $\bar{\rho}_l$ | Reading the results {eq}`eq-results-agreement` | [Reading the results: how much two groups agree](reading-results.md#how-much-two-groups-agree) | `viz/graphs.py:280-316`; `_agreement_by_block`, `:77-86` |
| {py:func}`~civiqhub_selene.viz.plot_mixedness` | The histogram of $1 - \max_q \theta_{iq}$ over the voters | Reading the results {eq}`eq-results-mixedness` | [Reading the results: mixedness and changes of role](reading-results.md#mixedness-and-changes-of-role) | `viz/draw.py:865-899` |

The other figures of `civiqhub_selene.viz` are described in the reference of the API.

## The foundation

| Name | Symbol | Equations | Section | Code |
| --- | --- | --- | --- | --- |
| {py:class}`~civiqhub_selene.types.Result`, {py:class}`~civiqhub_selene.types.Provenance`, {py:class}`~civiqhub_selene.types.Uncertainty`, {py:class}`~civiqhub_selene.estimator.Estimator` and the other names of `civiqhub_selene` | — | — | No formula: they carry the value of every method, its provenance and its uncertainty ([Architecture: the contract of a method](../architecture.md#the-contract-of-a-method)) | `types.py`, `estimator.py` |

## What has no row

No public name computes the dissensus index or the collective regime of the ordinal family, and none chooses the shape of a block model: these have no row because the engine holds no such name.
