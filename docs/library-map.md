# Finding your way

This page says where each thing lives: which module to import, which page of the reference to read, which file to open. To understand why the engine is built this way, read [Architecture](architecture.md).

## Four parts

| Part | What it is for | Needs |
| --- | --- | --- |
| [`civiqhub_selene`](reference/civiqhub_selene.md) | The contract every method shares. You meet it through every result, whatever the method | The standard library |
| [`civiqhub_selene.ordinal`](reference/civiqhub_selene.ordinal.md) | The ordinal reading: the votes of one item on the agreement scale | The standard library |
| [`civiqhub_selene.blocks`](reference/civiqhub_selene.blocks.md) | Co-clustering: the vote matrix, the block models, their starts and their values | NumPy, behind the `blocks` extra |
| [`civiqhub_selene.viz`](reference/civiqhub_selene.viz.md) | Graphs and figures of co-clustering results | NetworkX and Matplotlib, behind the `viz` extra |

Importing `civiqhub_selene` or `civiqhub_selene.ordinal` never loads NumPy. Importing `civiqhub_selene.blocks` without NumPy raises {py:class}`~civiqhub_selene.errors.MissingExtraError`, which says how to install the extra; so does importing `civiqhub_selene.viz` without NetworkX or Matplotlib. No part of the engine imports `viz`.

## I want to...

### Hold votes

| I want to | Use | Module |
| --- | --- | --- |
| Count the votes of one item | {py:class}`~civiqhub_selene.ordinal.counts.VoteCounts` | [`ordinal.counts`](reference/civiqhub_selene.ordinal.counts.md) |
| Hold the votes of many voters on many items | {py:class}`~civiqhub_selene.blocks.data.VoteMatrix` | [`blocks.data`](reference/civiqhub_selene.blocks.data.md) |
| Use another scale than -2 to +2 | {py:class}`~civiqhub_selene.blocks.data.OrdinalScale` | [`blocks.data`](reference/civiqhub_selene.blocks.data.md) |
| Draw synthetic votes with a known truth | {py:func}`~civiqhub_selene.blocks.synthetic.make_votes` | [`blocks.synthetic`](reference/civiqhub_selene.blocks.synthetic.md) |
| Read a matrix through groups and blocks | {py:meth}`VoteMatrix.group_by <civiqhub_selene.blocks.data.VoteMatrix.group_by>`, {py:class}`~civiqhub_selene.blocks.data.GroupedVotes` | [`blocks.data`](reference/civiqhub_selene.blocks.data.md) |

In a vote matrix, a cell is voted, shown without a vote, or not shown. An empty cell is never a zero nor a neutral answer.

### Run a method

| I want to | Use | Module |
| --- | --- | --- |
| Co-cluster with one group per voter | {py:class}`~civiqhub_selene.blocks.single.LatentBlockModel` | [`blocks.single`](reference/civiqhub_selene.blocks.single.md) |
| Co-cluster with mixed memberships: a voter follows one group per block | {py:class}`~civiqhub_selene.blocks.mixed.MixedMembershipBlockModel` | [`blocks.mixed`](reference/civiqhub_selene.blocks.mixed.md) |
| Start a model from regularised spectral co-clustering | {py:class}`~civiqhub_selene.blocks.starts.SpectralStart` | [`blocks.starts`](reference/civiqhub_selene.blocks.starts.md) |
| Start a model from a previous result, or from random arrangements | {py:class}`~civiqhub_selene.blocks.starts.FixedStart`, {py:class}`~civiqhub_selene.blocks.starts.RandomStart` | [`blocks.starts`](reference/civiqhub_selene.blocks.starts.md) |
| Choose the number of groups and of blocks | {py:func}`~civiqhub_selene.blocks.selection.select_shape`, {py:class}`~civiqhub_selene.blocks.selection.ShapeSelection` | [`blocks.selection`](reference/civiqhub_selene.blocks.selection.md) |
| Read the spectrum of the votes before choosing a shape | {py:func}`~civiqhub_selene.blocks.spectral.spectral_embedding`, {py:class}`~civiqhub_selene.blocks.spectral.SpectralEmbedding` | [`blocks.spectral`](reference/civiqhub_selene.blocks.spectral.md) |
| Choose how the votes of a co-cluster are modelled | {py:class}`~civiqhub_selene.blocks.responses.OrdinalProbit`, {py:class}`~civiqhub_selene.blocks.responses.Categorical` | [`blocks.responses`](reference/civiqhub_selene.blocks.responses.md) |
| Get a model from a plain word such as `"single"` or `"mixed"` | {py:func}`~civiqhub_selene.blocks.base.estimator_for`, {py:func}`~civiqhub_selene.blocks.base.memberships` | [`blocks.base`](reference/civiqhub_selene.blocks.base.md) |
| List the methods that are installed | {py:func}`~civiqhub_selene.registry.default_registry` | [`registry`](reference/civiqhub_selene.registry.md) |

Every method is run the same way: `estimator.estimate(data, seed=0)`, documented once on {py:meth}`Estimator.estimate <civiqhub_selene.estimator.Estimator.estimate>`.

### Read a result

| I want to | Use | Module |
| --- | --- | --- |
| Know whether there is a value, and read it | {py:class}`~civiqhub_selene.types.Result`, {py:class}`~civiqhub_selene.types.Unavailable` | [`types`](reference/civiqhub_selene.types.md) |
| Know which method, settings, seed and input gave it | {py:class}`~civiqhub_selene.types.Provenance`, {py:meth}`Result.fingerprint <civiqhub_selene.types.Result.fingerprint>` | [`types`](reference/civiqhub_selene.types.md) |
| Read a co-clustering | {py:class}`~civiqhub_selene.blocks.results.Coclustering`, {py:class}`~civiqhub_selene.blocks.results.SingleCoclustering`, {py:class}`~civiqhub_selene.blocks.results.MixedCoclustering` | [`blocks.results`](reference/civiqhub_selene.blocks.results.md) |
| Read the groups of voters and the blocks of items | {py:class}`~civiqhub_selene.blocks.partition.Partition` | [`blocks.partition`](reference/civiqhub_selene.blocks.partition.md) |
| Read the vote rate and the scores of each co-cluster | {py:class}`~civiqhub_selene.blocks.coclusters.CoclusterTable` | [`blocks.coclusters`](reference/civiqhub_selene.blocks.coclusters.md) |
| Handle an error of the engine | {py:class}`~civiqhub_selene.errors.SeleneError` and its subclasses; each carries a `hint` | [`errors`](reference/civiqhub_selene.errors.md) |

### Draw a result

| I want to | Use | Module |
| --- | --- | --- |
| Get a result as a NetworkX graph, to draw it elsewhere or compute on it | {py:func}`~civiqhub_selene.viz.graphs.coclustering_graph`, {py:func}`~civiqhub_selene.viz.graphs.membership_graph`, {py:func}`~civiqhub_selene.viz.graphs.role_graph`, {py:func}`~civiqhub_selene.viz.graphs.group_graph`, {py:func}`~civiqhub_selene.viz.graphs.vote_graph` | [`viz.graphs`](reference/civiqhub_selene.viz.graphs.md) |
| Draw the co-clusters, the memberships, the roles, the agreement between groups | {py:func}`~civiqhub_selene.viz.draw.draw_coclustering`, {py:func}`~civiqhub_selene.viz.draw.draw_memberships`, {py:func}`~civiqhub_selene.viz.draw.draw_roles`, {py:func}`~civiqhub_selene.viz.draw.draw_groups` | [`viz.draw`](reference/civiqhub_selene.viz.draw.md) |
| Draw the vote matrix sorted by groups and blocks | {py:func}`~civiqhub_selene.viz.draw.plot_vote_matrix` | [`viz.draw`](reference/civiqhub_selene.viz.draw.md) |
| Draw the spectrum and the embedding of the spectral start | {py:func}`~civiqhub_selene.viz.draw.plot_spectrum`, {py:func}`~civiqhub_selene.viz.draw.plot_embedding` | [`viz.draw`](reference/civiqhub_selene.viz.draw.md) |
| Draw the convergence, or how mixed the voters are | {py:func}`~civiqhub_selene.viz.draw.plot_history`, {py:func}`~civiqhub_selene.viz.draw.plot_mixedness` | [`viz.draw`](reference/civiqhub_selene.viz.draw.md) |

### Write a method

| I want to | Use | Module |
| --- | --- | --- |
| Write an estimator | {py:class}`~civiqhub_selene.estimator.Estimator` | [`estimator`](reference/civiqhub_selene.estimator.md) |
| Prove that it honours the contract | {py:class}`~civiqhub_selene.testing.EstimatorContract` | [`testing`](reference/civiqhub_selene.testing.md) |
| Write a block model | {py:class}`~civiqhub_selene.blocks.base.BlockModel`, {py:class}`~civiqhub_selene.blocks.iterative.IterativeEstimator` | [`blocks.base`](reference/civiqhub_selene.blocks.base.md), [`blocks.iterative`](reference/civiqhub_selene.blocks.iterative.md) |
| Write a response model | {py:class}`~civiqhub_selene.blocks.responses.ResponseModel` | [`blocks.responses`](reference/civiqhub_selene.blocks.responses.md) |
| Write another way to start a block model | {py:class}`~civiqhub_selene.blocks.starts.Initialization` | [`blocks.starts`](reference/civiqhub_selene.blocks.starts.md) |
| Declare it so that the registry finds it | The entry-point group {py:data}`~civiqhub_selene.registry.GROUP` | [`registry`](reference/civiqhub_selene.registry.md) |

[Architecture](architecture.md) walks through these steps.

## Two names for one object

Many objects have two import paths: the module that defines them, and the package that exports them.

```python
from civiqhub_selene import Result
from civiqhub_selene.types import Result as SameResult

print(Result is SameResult)
```

```text
True
```

Import from the package when the name is exported there: it is the shorter path, and the one that stays stable. The reference documents each object once, on the page of the module that defines it; the page of a package lists what the package exports.

## Where the files are

| Path | Content |
| --- | --- |
| [`src/civiqhub_selene/`](https://github.com/civiqhub/civiqhub-selene/tree/develop/src/civiqhub_selene) | The library. One file per module of the reference: `civiqhub_selene.blocks.data` is `src/civiqhub_selene/blocks/data.py` |
| [`tests/`](https://github.com/civiqhub/civiqhub-selene/tree/develop/tests) | Unit tests (one file per module), property-based tests, and reference vectors with their tolerance |
| [`docs/`](https://github.com/civiqhub/civiqhub-selene/tree/develop/docs) | The pages of this site |
| [`examples/`](https://github.com/civiqhub/civiqhub-selene/tree/develop/examples) | Notebooks that run the block models from end to end, on synthetic data: see [Examples](examples.md) |
| [`services/api/`](https://github.com/civiqhub/civiqhub-selene/tree/develop/services/api) | The HTTP wrapper of the engine. It installs the library and never reimplements it |

In the reference, the **[source]** link next to a name opens the code that defines it.

A module whose name starts with an underscore, such as `civiqhub_selene.blocks._numerics`, is internal: it has no page in the reference and may change without notice.

## Not written yet

The dissensus index and the collective regime of the ordinal family, the consensus between groups and the `cocluster` function are planned. The [changelog](changelog.md) lists what each version adds.
