# Service de calcul

Checked against the code of `develop` at commit `c29d115`.

The SELENE service, under `services/api` in the repository, runs computations of the engine for one trusted backend, inside a private network. This page says, with the symbols and the equations of the [mathematical model](model/index.md), how the service turns a submission of the kind `coclustering.single` into a computation of the engine, and the result of the engine into the result of the contract. The contract itself is `services/api/openapi.yaml`, and its operation is described in the [README of the service](https://github.com/civiqhub/civiqhub-selene/blob/develop/services/api/README.md), which this page does not repeat. Everything here uses synthetic data only.

## What the service computes

A job of the kind `coclustering.single` is one estimation of {py:class}`~civiqhub_selene.blocks.LatentBlockModel` on one snapshot of votes, with the settings and the seed of the submission ([The latent block model](model/latent-block-model.md)). Each voter gets one group and each item one block. The service answers with the result `selene.coclustering.single.result/1` for the payload schema 1, and `selene.coclustering.single.result/2` for the payload schema 2, whose items each have a declared scale.

What the service does not do:

- **No durable state.** A job and its result live in memory for a bounded time and are lost when the service restarts. When to compute again, how to group the changes of the votes, which result is the last valid one and what is published belong to the backend, which owns the durable tracking, the retries and the publication (TechADR-061).
- **One model, without weights.** The kind runs the single-membership model only: the service runs neither {py:class}`~civiqhub_selene.blocks.MixedMembershipBlockModel` nor survey weights, and the contract has no field of weights.
- **No uncertainty.** The method defines none, and the result carries none ([Reading the results: uncertainty](model/reading-results.md#uncertainty)).
- **No other method.** The dissensus index and the collective regimes of the ordinal family are defined in the design notes of the engine and not implemented ([From the API to the formulas: what has no row](model/api-to-formulas.md#what-has-no-row)); no function chooses the shape, `select_shape` does not exist ([Choosing the shape](model/choosing-the-shape.md)); the estimation has no variational form.

## From a submission to the data of the model

The service builds the {py:class}`~civiqhub_selene.blocks.VoteMatrix` of the engine with {py:meth}`~civiqhub_selene.blocks.VoteMatrix.from_records`, from the normalised payload ([The data](model/data.md)):

| Field of the contract | Object of the engine | Symbol | Model |
| --- | --- | --- | --- |
| `voters`, in their order | {py:attr}`~civiqhub_selene.blocks.VoteMatrix.voters`, the rows | $i$, $M$ | [The data: three states per cell](model/data.md#three-states-per-cell) |
| `items`, in their order; `id` and `revision` | {py:attr}`~civiqhub_selene.blocks.VoteMatrix.items`, the columns, by `id` | $j$, $P$ | [The data: three states per cell](model/data.md#three-states-per-cell) |
| `scale`, schema 1: the product scores $-2$ to $+2$ | {py:class}`~civiqhub_selene.blocks.OrdinalScale` `((-2, -1, 0, 1, 2))`, shared by every item | $K = 5$ | [The data: the scale](model/data.md#the-scale) |
| `scales` and `items[].scale`, schema 2 | one {py:class}`~civiqhub_selene.blocks.OrdinalScale` per declared scale, and the scale of each item, read by the engine as {py:class}`~civiqhub_selene.blocks.ItemScales` | $\mathcal{G}$, $g_j$, $K_g$ | [The data: items on different scales](model/data.md#items-on-different-scales) |
| `votes`, `[row, column, score]` | one vote of the voter `voters[row]` on the item `items[column]`; the engine reads the score as its rank on the scale of the item | $s_{ij} = k \geq 1$ | The data {eq}`eq-data-states` |
| `exposure.mode` `"complete"` | every cell is shown: a cell without a vote is an abstention | $\Omega$ is every cell | The data {eq}`eq-data-states` |
| `exposure.mode` `"listed"` | the voted cells and the cells of `shown_without_vote` are shown, every other cell is not | $\Omega$, the voted and the listed cells | [The data: the cells that are not shown](model/data.md#the-cells-that-are-not-shown) |
| `exposure.mode` `"unknown"` | refused before any computation: `422 unsupported_exposure` | — | — |

The service never maps a score to a rank by arithmetic: the engine does it, from the scale. A score of $0$ is a neutral vote, not an absence of vote. A cell is in one of three states, voted, shown without a vote, or not shown, and the absence of a vote is never read as an exposure ([The data: three states per cell](model/data.md#three-states-per-cell)).

The service normalises the payload before it computes: the votes are sorted by cell, the listed cells are sorted, and the scales of schema 2 are listed in the order in which the items first use them, which is the order of the engine. The order of the votes and the order of declaration of the scales therefore change neither the computation nor the content fingerprint. The identifiers of the scales do not change the computation either, but they are part of the content, and the result gives them back.

## From the settings to the estimator

The settings of the payload, `parameters`, become the arguments of the estimator. Their defaults are those of the engine ([The estimation: the settings every block model shares](model/estimation.md#the-settings-every-block-model-shares)):

| Setting | Default | Argument of the engine | Symbol | Model |
| --- | --- | --- | --- | --- |
| `n_groups` | `2` | `LatentBlockModel(n_groups=...)` | $Q$ | [The latent block model](model/latent-block-model.md) |
| `n_blocks` | `2` | `LatentBlockModel(n_blocks=...)` | $L$ | [The latent block model](model/latent-block-model.md) |
| `n_starts` | `6` | `LatentBlockModel(n_starts=...)` | $n_{\text{starts}}$ | [The estimation: several starts](model/estimation.md#several-starts-and-the-one-kept) |
| `max_iterations` | `200` | `LatentBlockModel(max_iterations=...)` | $T_{\max}$ | [The estimation: one start](model/estimation.md#one-start) |
| `tolerance` | `1e-6` | `LatentBlockModel(tolerance=...)` | — | [The estimation: one start](model/estimation.md#one-start) |
| `response_model` | `"ordinal_probit"` | {py:class}`~civiqhub_selene.blocks.responses.OrdinalProbit` for `"ordinal_probit"`, {py:class}`~civiqhub_selene.blocks.responses.Categorical` for `"categorical"` | $F_{qls}$ | The response models {eq}`eq-response-probit`, {eq}`eq-response-categorical` |
| `initialization` | `"random"` | {py:class}`~civiqhub_selene.blocks.RandomStart` for `"random"`, {py:class}`~civiqhub_selene.blocks.SpectralStart` for `"spectral"` | $R_i$, $C_j$ | [The starts](model/initialization.md) |
| `thresholds`, schema 2 | `{"axis": "auto", "mode": "fixed"}` | `OrdinalProbit()` for the default; `OrdinalProbit(thresholds=Thresholds(axis=..., mode=...))` otherwise; not read with `"categorical"` | $t_{gk}$ | [The response models: items on several scales](model/response-models.md#items-on-several-scales) |
| `thresholds.axis`, schema 2 | `"auto"` | {py:attr}`~civiqhub_selene.blocks.responses.Thresholds.axis`: `"auto"`, `"levels"` or `"unit"` | $t_{gk}$ | The response models {eq}`eq-response-axes` |
| `thresholds.mode`, schema 2 | `"fixed"` | {py:attr}`~civiqhub_selene.blocks.responses.Thresholds.mode`: `"fixed"` or `"estimated"` | $t_{gk}$ | The response models {eq}`eq-response-thresholds-estimated` |

The default thresholds of schema 2 give `OrdinalProbit()` itself, so that a computation of schema 2 on the product scale has the provenance of the same computation of schema 1. The seed, `seed`, is required, from $0$ to $2^{53} - 1$; it is the `seed` of {py:meth}`~civiqhub_selene.estimator.Estimator.estimate` ([The estimation: the seed](model/estimation.md#the-seed)).

## Admission budget

Before it starts a computation, the service refuses a submission whose voters, items or cells exceed the limits of the instance, or whose budget exceeds its `max_work`, with `422 limit_exceeded`. The limits of an instance are given by `GET /v1/service`. With

| Symbol | Meaning |
| --- | --- |
| $n_{\text{starts}}$ | The setting `n_starts` |
| $T_{\max}$ | The setting `max_iterations` |
| $K_{\max}$ | The number of scores of the longest scale of the submission: $5$ for schema 1 |
| $W$ | The budget of the submission |

the budget is

$$
W = n_{\text{starts}} \cdot T_{\max} \cdot M P \cdot (K_{\max} + 1) \cdot (Q + L).
$$

$W$ is a bound used to admit a submission, read from its sizes and settings before anything is computed. It is not the cost of an estimation, which depends on the iterations actually run: that cost is described in [Cost and numerical safeguards](model/complexity.md).

## From the engine result to the contract result

The engine returns a {py:class}`~civiqhub_selene.types.Result`; the service reads its {py:meth}`~civiqhub_selene.types.Result.to_dict`, of the schema `civiqhub_selene.result/0`, and keeps the fields of the contract. The value of a {py:class}`~civiqhub_selene.blocks.results.SingleCoclustering` is written by the engine in the form of one scale when the items share one, and scale by scale otherwise ([Reading the results](model/reading-results.md)); `result/1` keeps the form of one scale, and `result/2` always gives the form by scale, even for one scale.

| Field of the contract | Result | Field of `Result.to_dict()` | Symbol | Model |
| --- | --- | --- | --- | --- |
| `value.voters` | 1, 2 | `value.voters` | $i$ | [The data](model/data.md) |
| `value.items` | 1, 2 | `value.items` | $j$ | [The data](model/data.md) |
| `value.scores` | 1 | `value.scores` | the scores of the scale, $K$ | [The data: the scale](model/data.md#the-scale) |
| `value.scales` | 2 | `value.scales`, or `[value.scores]` for one scale, each with the identifier declared in the payload | $\mathcal{G}$, $K_g$ | [The data: items on different scales](model/data.md#items-on-different-scales) |
| `value.scale_index` | 2 | `value.scale_index`, or $0$ for every item for one scale | $g_j$ | [The data: items on different scales](model/data.md#items-on-different-scales) |
| `value.group_index` | 1, 2 | `value.group_index` | $R_i$ | The latent block model {eq}`eq-lbm-objective` |
| `value.block_index` | 1, 2 | `value.block_index` | $C_j$ | The latent block model {eq}`eq-lbm-objective` |
| `value.group_memberships` | 1, 2 | `value.group_memberships` | $\hat{\theta}_{iq}$ | The latent block model {eq}`eq-lbm-memberships` |
| `value.block_memberships` | 1, 2 | `value.block_memberships` | $\hat{\phi}_{jl}$ | The latent block model {eq}`eq-lbm-memberships` |
| `value.coclusters.scores` | 1 | `value.coclusters.scores` | the scores of the scale | [Reading the results: the table of co-clusters](model/reading-results.md#the-table-of-co-clusters) |
| `value.coclusters.vote_rate` | 1, 2 | `value.coclusters.vote_rate` | $\nu_{ql}$ | The response models {eq}`eq-response-rate` |
| `value.coclusters.vote_shares` | 1 | `value.coclusters.vote_shares` | $p_{qlk}$ | The response models {eq}`eq-response-states` |
| `value.coclusters.vote_shares_by_scale` | 2 | `value.coclusters.vote_shares_by_scale`, or `[value.coclusters.vote_shares]` for one scale | $p_{qlgk}$ | The response models {eq}`eq-response-probit-by-scale` |
| `value.coclusters.n_votes` | 1, 2 | `value.coclusters.n_votes` | $\sum_{k \geq 1} N_{qlk}$ | The data {eq}`eq-data-counts` |
| `value.coclusters.latent_mean` | 1, 2 | `value.coclusters.latent_mean`, ordinal probit only | $\mu_{ql}$ | The response models {eq}`eq-response-opinion` |
| `value.coclusters.latent_spread` | 1, 2 | `value.coclusters.latent_spread`, ordinal probit only | $\sigma_{ql}$ | The response models {eq}`eq-response-opinion` |
| `value.coclusters.thresholds` | 2 | `value.coclusters.thresholds`, ordinal probit only | $t_{gk}$ | The response models {eq}`eq-response-axes` |
| `value.coclusters.has_fixed_spread` | 2 | `value.coclusters.has_fixed_spread`, ordinal probit only | $\sigma_{ql} = \sigma_{\mathrm{u}}$ held | [The response models: the opinion of the ordinal probit](model/response-models.md#the-opinion-of-the-ordinal-probit) |
| `value.objective` | 1, 2 | `value.objective` | $\mathcal{L}_{\mathrm{c}}$ | The latent block model {eq}`eq-lbm-objective` |
| `value.integrated_likelihood` | 1, 2 | `value.integrated_likelihood` | $\mathrm{ICL}$ | Choosing the shape {eq}`eq-shape-icl` |
| `value.history` | 1, 2 | `value.history` | the objective after each iteration of the start kept | [The estimation: one start](model/estimation.md#one-start) |
| `value.computation` | 1, 2 | `value.computation` | — | [The estimation: what a result says about itself](model/estimation.md#what-a-result-says-about-itself) |

Both results also carry `schema`, `available` and the `provenance` of the engine: the method and its version, the version of the engine, the settings, the fingerprint of the input, the generator and the seed, the versions of the dependencies and the fingerprint of the computation ([The estimation: determinism and provenance](model/estimation.md#determinism-and-provenance)). The value holds individual memberships: it is a result for the backend, not a view to publish.

**An explained absence.** When the engine gives no value, the job still succeeds: `state` is `succeeded`, and the result carries `available: false` and the reason of the engine, with its detail, in place of `value`. The engine gives each reason under a condition of the votes and the shape ({py:func}`~civiqhub_selene.blocks.base.availability`, [The data: when a result exists](model/data.md#when-a-result-exists)):

| `unavailable.reason` | Condition of the engine |
| --- | --- |
| `no_votes` | The matrix holds no vote |
| `not_enough_data` | Fewer than $\max(2, Q)$ voters with a vote, or fewer than $\max(2, L)$ items with a vote |
| `unsupported_input` | The estimator does not read items on several scales and the matrix has several; {py:class}`~civiqhub_selene.blocks.LatentBlockModel` reads them, so a submission of schema 1 or 2 does not get this reason from it |

**No uncertainty.** The result carries no interval and no standard error: the method defines none ([Reading the results: uncertainty](model/reading-results.md#uncertainty)).

## Reproducibility and identity

- **The content fingerprint.** The service computes `content_fingerprint`: `sha256:` and the SHA-256 of the canonical JSON (sorted keys, no spaces, UTF-8) of the kind, the payload schema version and the normalised payload, every setting at its effective value. The submission key and `caller_reference` are not content. Two submissions of the same content have the same fingerprint, whatever the order of their votes or of the declaration of their scales.
- **The fingerprint of the computation.** The `provenance` of the result carries the fingerprint of the input and the fingerprint of the computation of the engine: the method and its version, the settings, the generator and the seed, and the fingerprint of the votes ([The estimation: determinism and provenance](model/estimation.md#determinism-and-provenance)). The same snapshot, settings, seed and wheel of the engine give the same result.
- **Tolerances.** On another machine, the same computation gives the same partition, up to the names of the groups and of the blocks, and numbers within $10^{-6}$ relative or $10^{-9}$ absolute. The history of the iterations is not promised: another processor may reach the same optimum by another path.
- **Schema 2 on one scale.** A submission of schema 2 whose items share the product scale is the computation of schema 1: the same fingerprint of the input, the same fingerprint of the computation and the same numbers; only the form of the result changes.

## Job life

A job is `running`, then `succeeded` or `failed`; no other state exists in the contract. The caller chooses a `submission_key`: the same key with the same content returns the same job, and the same key with another content is refused. A finished job is kept for a bounded time and a bounded number of jobs, and in memory only: after that, or after a restart, the job is unknown. The [README of the service](https://github.com/civiqhub/civiqhub-selene/blob/develop/services/api/README.md#the-job-envelope) gives the operations, the errors, the limits and the retention of an instance.
