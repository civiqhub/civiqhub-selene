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

To build a matrix from your own votes, see the constructors of {py:class}`~civiqhub_selene.blocks.data.VoteMatrix`: `from_records`, `from_table` and `from_arrays`.

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

## Choose the number of groups and blocks

Above, the shape was given: 3 groups and 2 blocks. {py:func}`~civiqhub_selene.blocks.selection.select_shape` estimates the model at every shape of a grid, keeps the best one by a criterion, and returns the model at that shape. For the single-membership model the criterion is the integrated likelihood; for the mixed-membership one, the log-likelihood of held-out votes.

```python
from civiqhub_selene.blocks import select_shape

chosen = select_shape(votes, spectral, n_groups=(1, 2, 3, 4), n_blocks=(1, 2, 3), seed=0)
print(chosen.value.selection.criterion, chosen.value.selection.best)
print(repr(chosen.value))
```

```text
integrated_likelihood (3, 2)
SingleCoclustering(48 voters in 3 groups x 14 items in 2 blocks)
```

`chosen.value.selection` keeps the score of every shape of the grid; `print` shows them as a table.

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
