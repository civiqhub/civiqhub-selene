# Premiers pas

This page installs the library and runs two computations on synthetic data. The tests of the repository run every Python block of this page and compare what it prints with the output shown.

## Install

```bash
pip install civiqhub-selene
```

The core and the ordinal family use the standard library only. Co-clustering needs NumPy, behind an optional extra:

```bash
pip install "civiqhub-selene[blocks]"
```

The published version 0.0.1 holds the structure of the project and no computation. Until the next release, install the development branch to run what this site describes:

```bash
pip install "civiqhub-selene[blocks] @ git+https://github.com/civiqhub/civiqhub-selene@develop"
```

The distribution is named `civiqhub-selene`; the package is imported as `civiqhub_selene`. Python 3.11 or later is required.

## Count the votes of one item

A vote is a score on the agreement scale: an integer from -2 (strong disagreement) to +2 (strong agreement). {py:class}`~civiqhub_selene.ordinal.VoteCounts` holds how many votes each score received, with exact shares.

```python
from civiqhub_selene.ordinal import VoteCounts

counts = VoteCounts.from_scores([2, 2, 1, -1, -2, 2])
print(counts)
print(counts.total, counts[2], counts.shares[4])
```

```text
VoteCounts(1, 1, 0, 1, 3; 6 votes)
6 3 1/2
```

## Run a first co-clustering

Co-clustering arranges voters in groups and items in blocks, so that the votes of one group on one block look alike. The pair of a group and a block is a co-cluster.

The votes below are synthetic: {py:func}`~civiqhub_selene.blocks.synthetic.make_votes` draws them from a truth that it plants, here three groups of 16 voters and two blocks of 7 items.

```python
from civiqhub_selene.blocks import LatentBlockModel
from civiqhub_selene.blocks.synthetic import make_votes, single_memberships

votes, truth = make_votes(
    single_memberships((16, 16, 16)),
    single_memberships((7, 7)),
    seed=0,
)
print(votes)

model = LatentBlockModel(n_groups=3, n_blocks=2)
result = model.estimate(votes, seed=0)
print(result)
```

```text
VoteMatrix(48 voters x 14 items, 408 votes, 5 levels)
100.0% of the 672 cells shown, 60.7% of the shown cells voted
Result(latent_block_model 0.1.0)
SingleCoclustering(48 voters in 3 groups x 14 items in 2 blocks)
groups of 16, 15, 17 voters; blocks of 7, 7 items
group 0 x block 0: vote rate 0.88, typical score +2, 98 votes
group 0 x block 1: vote rate 0.44, typical score -2, 49 votes
group 1 x block 0: vote rate 0.57, typical score -2, 60 votes
group 1 x block 1: vote rate 0.94, typical score +2, 99 votes
group 2 x block 0: vote rate 0.39, typical score -1, 47 votes
group 2 x block 1: vote rate 0.46, typical score 0, 55 votes
```

Three things to notice:

- **The model is a value.** {py:class}`~civiqhub_selene.blocks.single.LatentBlockModel` is a frozen dataclass: its settings are its fields, and it reads nothing from the environment.
- **The seed is explicit.** The same votes, the same settings and the same seed give the same result.
- **The call is the same for every method.** {py:meth}`~civiqhub_selene.estimator.Estimator.estimate` is written once, for all estimators.

To build a matrix from your own votes, see the constructors of {py:class}`~civiqhub_selene.blocks.data.VoteMatrix`: `from_records`, `from_table`, `from_arrays` and `from_responses`.

### Survey responses

Use `VoteMatrix.from_responses` for a respondent-by-item table of ordinal scores.
Supply a mapping from each item identifier to its `OrdinalScale`. The keys give
the column order unless `items` supplies it explicitly. Pass unique voter
identifiers in table order. Each cell holds an integer score on the source scale
or a missing value (`None` or NaN). A boolean `shown` mask distinguishes an
unshown question from an abstention. By default every question is shown. A score
in a cell marked as not shown is refused.

The model that SEL-8.2 (`ItemScales`) and SEL-8.3 aim at is a continuous Gaussian
latent, of parameters μ and σ, for each co-cluster of
{py:class}`~civiqhub_selene.blocks.responses.OrdinalProbit`. Each item `j` would
then have its own observation scale, with `K_j` ranks and its own thresholds, so
a shown cell has `K_j + 1` modalities. The CEM or EM loop would run on those
latents. Today the thresholds of `OrdinalProbit` are fixed and common to every
item. `from_responses` keeps the original ranks of each column and the direction
of agreement; it does not recode them onto a common grid.

The response models still use one shared `K`. Distinct observation scales are
refused, naming the first item that differs, until SEL-8.2 (`ItemScales`). A
homogeneous table matches `from_records` on the same scores, voters and items.

Give the optional `voter_weights` in row order, using one finite positive weight
per respondent. A survey row with zero weight can be removed before creating
the matrix. Both block models rescale the weights to average 1. In the
single-membership model they multiply each voter's response and group-membership
log terms. In the mixed-membership model they multiply each voter's marginal
log-likelihood and voter-specific regularisation term. Shared parameter updates
use the corresponding weighted counts. Item membership regularisation and
reported response counts keep their usual, unweighted definitions. The spectral
start and the reported `group_memberships` of the single-membership model are
not weighted. When the block-share term is neutral (a single block), the fitted
response parameters match those of a row duplication.

Build a `VoterCovariates` table with `VoterCovariates.from_records(votes, records)`.
Give one record per voter identifier and the same fields in each record, using
`None` for missing covariates. The constructor aligns the records to
`votes.voters`. The covariate table is available for analysis beside the matrix.

For a wide CSV with one row per voter, use `read_survey_csv` with the same
per-item scales. The reader is part of the input contract of the library. It
accepts an open text stream. A prepared semicolon CSV can specify ID, weight and
covariate columns separately:

```python
from io import StringIO

from civiqhub_selene.blocks import OrdinalScale, read_survey_csv

records = StringIO("responseid;Q1;Q2;weight;mode\nA;2;5;1.25;web\nB;;2;0.75;phone\n")
five = OrdinalScale((1, 2, 3, 4, 5))
survey_votes, survey_covariates = read_survey_csv(
    records,
    item_scales={"Q1": five, "Q2": five},
    weight_column="weight",
    covariate_columns=("mode",),
    delimiter=";",
    unshown_label="",
)
```

For a file, open it with `newline=""` and `encoding="utf-8-sig"` in place of
`StringIO`. Declare the full scale of each question, even if some ranks are
unobserved. The example uses `unshown_label=""` to exclude blank cells from
the likelihood, as when the prepared data have no indication that the question
was presented. By default, blank scores count as abstentions. Covariate values
are read as text and an empty covariate is represented by `None`.

## Read the result

A {py:class}`~civiqhub_selene.types.Result` carries a value and its provenance. Here the value is a {py:class}`~civiqhub_selene.blocks.results.SingleCoclustering`.

```python
coclustering = result.value
print(coclustering.groups)
print(coclustering.group_of("v1"))
print(coclustering[1, 1])

provenance = result.provenance
print(provenance.method, provenance.seed)
print(result.fingerprint())
```

```text
Partition(48 voters in 3 mixed groups: 16, 15, 17)
VoterGroup(index=0, 16 voters, share=0.333)
Cocluster(group=1, block=1, vote_rate=0.943, typical_score=+2, n_votes=99)
latent_block_model 0.1.0 0
d00f9fe3f69a743e
```

The fingerprint identifies the computation: the method and its version, the settings, the seed and the input. Two results with the same fingerprint come from the same computation.

## A result can be an explained absence

When there is nothing to compute, a method does not raise and does not return a default: it returns a result without a value, which says why.

```python
from civiqhub_selene.blocks import VoteMatrix

no_votes = VoteMatrix.from_records([], voters=["ana", "ben", "cleo"], items=["p1", "p2"])
absent = model.estimate(no_votes)
print(absent.is_available)
print(absent.unavailable.reason, "-", absent.unavailable.detail)
```

```text
False
no_votes - the matrix holds no vote
```

Reading `absent.value` raises {py:class}`~civiqhub_selene.errors.UnavailableError`. Test `is_available` first.

## Start from a spectral arrangement

A block model only reaches a local optimum: where it starts decides much of where it ends. By default each start is random. {py:class}`~civiqhub_selene.blocks.starts.SpectralStart` starts from the arrangement of regularised spectral co-clustering (Qin and Rohe 2013; Rohe, Qin and Yu 2016), which usually reaches the best optimum in one start:

```python
from civiqhub_selene.blocks import SpectralStart

spectral = LatentBlockModel(n_groups=3, n_blocks=2, n_starts=1, initialization=SpectralStart())
found = spectral.estimate(votes, seed=0).value
print(found.groups.agreement_with(coclustering.groups))
```

```text
1.0
```

The start is a setting of the model: it enters the provenance of the result. {py:func}`~civiqhub_selene.blocks.spectral.spectral_embedding` gives the embedding itself, to read its spectrum before choosing a shape.

## Let a voter follow several groups

{py:class}`~civiqhub_selene.blocks.mixed.MixedMembershipBlockModel` gives each voter a degree of membership to each group, and one role per block: the group it follows on that block. Below, eight voters are drawn half in group 0 and half in group 1.

```python
import numpy as np

from civiqhub_selene.blocks import MixedMembershipBlockModel

shared = np.vstack([single_memberships((16, 16, 16)), [[0.5, 0.5, 0.0]] * 8])
mixed_votes, _ = make_votes(shared, single_memberships((7, 7)), seed=0)
mixed = MixedMembershipBlockModel(n_groups=3, n_blocks=2, initialization=SpectralStart())
value = mixed.estimate(mixed_votes, seed=0).value
print(repr(value))
print(value.role_changes())
```

```text
MixedCoclustering(56 voters in 3 groups x 14 items in 2 blocks)
[[17  0  4]
 [ 0 15  2]
 [ 2  0 16]]
```

`role_changes()` counts the voters by the group they follow on block 0 (rows) and on block 1 (columns): off the diagonal, a voter changes role from one block to the other.

## Draw the result

{py:mod}`civiqhub_selene.viz` turns a result into NetworkX graphs and draws them with Matplotlib. It needs the `viz` extra:

```bash
pip install "civiqhub-selene[viz]"
```

A graph carries the numbers of the result: one node per group and per block, one edge per co-cluster.

```python
from civiqhub_selene import viz

graph = viz.coclustering_graph(coclustering)
print(graph.number_of_nodes(), graph.number_of_edges())
```

```text
5 6
```

Each `draw_*` and `plot_*` function draws on Matplotlib axes and returns them:

<!-- not run: it writes files -->

```python
viz.draw_coclustering(coclustering).figure.savefig("coclustering.png")
viz.plot_vote_matrix(votes, coclustering).figure.savefig("matrix.png")
viz.draw_roles(value).figure.savefig("roles.png")
```

The [examples](examples.md) run the whole inference, from a CSV file to the figures.

## Next

- [Examples](examples.md) are notebooks that run the single-membership and the mixed-membership models from end to end.
- [Finding your way](library-map.md) says which module holds what.
- [Architecture](architecture.md) explains the contract behind `estimate` and how to add a method.
- The [reference](reference/index.md) documents every public name.
