# Échelles par proposition

The questions of a survey are not asked on one scale: one question takes yes or no, another four answers from "strongly disagree" to "strongly agree", a third the five scores of the agreement scale. The engine keeps the scale of each item, reads each vote on it, and never takes the level of one scale for the level of another. This page shows how to declare the scales, what a model does with them, and how to read a result. The tests of the repository run every Python block of this page and compare what it prints with the output shown. The data are synthetic.

## Declare the scale of each item

A scale is an {py:class}`~civiqhub_selene.blocks.OrdinalScale`: its scores, in the order of agreement. Give the scale of each item by its identifier, to {py:meth}`~civiqhub_selene.blocks.VoteMatrix.from_responses`, {py:meth}`~civiqhub_selene.blocks.VoteMatrix.from_records`, {py:meth}`~civiqhub_selene.blocks.VoteMatrix.from_arrays` or {py:func}`~civiqhub_selene.blocks.read_survey_csv`. Each score is checked on the scale of its item:

```python
from civiqhub_selene.blocks import OrdinalScale, VoteMatrix

yes_no = OrdinalScale((0, 1))
agree4 = OrdinalScale((1, 2, 3, 4))
agree5 = OrdinalScale()

votes = VoteMatrix.from_responses(
    [
        [1, 4, 2, None],
        [0, 1, -2, 1],
        [1, 3, 1, 1],
    ],
    voters=["r1", "r2", "r3"],
    item_scales={"trust": yes_no, "safety": agree4, "reform": agree5, "vote": yes_no},
)
print(votes)
print(votes.item_scales)
print(votes.scale_of("safety"))
```

```text
VoteMatrix(3 voters x 4 items, 11 votes, 3 scales of 2 to 5 levels)
100.0% of the 12 cells shown, 91.7% of the shown cells voted
ItemScales(3 scales of 2 to 5 levels over 4 items)
scale 0: 0, +1 (2 items)
scale 1: +1, +2, +3, +4 (1 item)
scale 2: -2, -1, 0, +1, +2 (1 item)
OrdinalScale(scores=(1, 2, 3, 4))
```

{py:attr}`~civiqhub_selene.blocks.VoteMatrix.item_scales` is an {py:class}`~civiqhub_selene.blocks.ItemScales`: the distinct scales, in the order of their first item, and the scale of each item. Two items whose scales have the same scores share one scale. Build one by hand with {py:meth}`~civiqhub_selene.blocks.ItemScales.from_scales`, from the scale of each item in the order of the columns.

Before the votes enter the engine, each answer of a survey takes one of the three states of a cell ([The data](model/data.md#three-states-per-cell)). A recommendation, which is not a decision of the engine: an answer such as "don't know" or a refusal is an abstention, a cell shown without a vote, whose rate each co-cluster models; "not concerned", a question filtered out or asked to another sample is a cell not shown, which says nothing. The choice changes the results, and it is made with the data.

## Run a model

Both block models read items on several scales, with their defaults:

```python
import numpy as np

from civiqhub_selene.blocks import LatentBlockModel
from civiqhub_selene.blocks.synthetic import make_votes, single_memberships

scales = [(yes_no, agree4, agree5)[item % 3] for item in range(18)]
survey, truth = make_votes(
    single_memberships((60, 60)),
    single_memberships((9, 9)),
    vote_rate=[[0.9, 0.7], [0.7, 0.9]],
    latent_mean=[[0.8, 0.3], [0.25, 0.7]],
    latent_spread=[[0.15, 0.15], [0.15, 0.15]],
    item_scales=scales,
    seed=7,
)
value = LatentBlockModel(n_groups=2, n_blocks=2).estimate(survey, seed=0).value
print(value.coclusters)
```

```text
CoclusterTable(2 groups x 2 blocks, 3 scales of 2 to 5 levels)
group 0 x block 0: vote rate 0.90, typical scores +1, +4, +2, 484 votes
group 0 x block 1: vote rate 0.69, typical scores 0, +2, -1, 373 votes
group 1 x block 0: vote rate 0.76, typical scores 0, +1, -1, 411 votes
group 1 x block 1: vote rate 0.91, typical scores +1, +3, +1, 494 votes
```

{py:func}`~civiqhub_selene.blocks.synthetic.make_votes` drew the 18 items on three scales, from opinions planted on the unit axis; the model finds a group that agrees with the first block and a group that agrees with the second, and gives one typical score per scale.

## Read a result

A co-cluster has one opinion for all its items, whatever their scale: one vote rate, one latent mean and one latent spread. With several scales, the ordinal probit reads them on the unit axis by default, where every scale covers the axis from 0 to 1 in equal shares ([The response models: items on several scales](model/response-models.md#items-on-several-scales)). The planted means were 0.8, 0.3, 0.25 and 0.7, and the spreads 0.15:

```python
table = value.coclusters
print(np.round(table.latent_mean, 2).tolist())
print(np.round(table.latent_spread, 2).tolist())
print([grid.tolist() for grid in table.thresholds])
```

```text
[[0.79, 0.31], [0.25, 0.69]]
[[0.15, 0.16], [0.14, 0.15]]
[[0.5], [0.25, 0.5, 0.75], [0.2, 0.4, 0.6, 0.8]]
```

The thresholds are the grids that cut the opinion into the answers of each scale: $\tfrac{1}{2}$ for yes or no, quarters for four answers, fifths for five. Everything that is read per answer is read per scale:

```python
first = table[0, 0]
print(first)
print(first.typical_score_by_scale)
print([round(share, 3) for share in first.vote_shares_by_scale[0]])
```

```text
Cocluster(group=0, block=0, vote_rate=0.896, typical_score_by_scale=(+1, +4, +2), n_votes=484)
(1, 4, 2)
[0.032, 0.968]
```

Group 0 answers the first block "yes" in 97% of its votes on the questions of two answers. The readings of one scale refuse a result of several, in the same words everywhere, instead of returning the distribution of one scale as if it were the others':

```python
try:
    value.scale
except ValueError as error:
    print(type(error).__name__, error)
```

```text
InvalidInputError this co-clustering has 3 scales, not one
```

| Reading of one scale | Reading by scale |
| --- | --- |
| `VoteMatrix.scale` | `VoteMatrix.item_scales`, `scales`, `scale_of(item)` |
| `GroupedVotes.counts()` | `GroupedVotes.counts_by_scale()` |
| `CoclusterTable.scale`, `vote_shares`, `log_probabilities()` | `CoclusterTable.scales`, `vote_shares_by_scale`, `log_probabilities_by_scale()` |
| `Cocluster.scores`, `vote_shares`, `typical_score` | `Cocluster.scores_by_scale`, `vote_shares_by_scale`, `typical_score_by_scale` |
| `Coclustering.scale` | `Coclustering.item_scales`, `scales` |

The dictionary of a result follows the same rule: with several scales, `scales` and `scale_index` take the place of `scores`, and the table writes `vote_shares_by_scale` in place of `vote_shares`, so that a program written for one scale fails on a missing key rather than misreads:

```python
data = value.to_dict()
print(data["scales"], data["scale_index"][:6])
print(sorted(data["coclusters"]))
```

```text
[[0, 1], [1, 2, 3, 4], [-2, -1, 0, 1, 2]] [0, 1, 2, 0, 1, 2]
['has_fixed_spread', 'latent_mean', 'latent_spread', 'n_votes', 'scales', 'thresholds', 'vote_rate', 'vote_shares_by_scale']
```

`has_fixed_spread` marks the co-clusters whose votes are all on scales of two answers: such votes say on which side of $\tfrac{1}{2}$ the opinion lies, not how spread it is, and the spread of these co-clusters is held at one level of the longest scale rather than estimated.

## Choose where the grids lie

The grids are a setting of the ordinal probit, {py:class}`~civiqhub_selene.blocks.responses.Thresholds`, and they enter the provenance of the result. Its axis is `"auto"` by default: the axis of levels on a matrix of one scale, so that such a matrix is computed exactly as before items could have several, and the unit axis on several. Give the axis yourself, or the grids of some scales by hand, on that axis:

```python
from civiqhub_selene.blocks.responses import OrdinalProbit, Thresholds

print(OrdinalProbit(thresholds=Thresholds(axis="levels")).thresholds)
by_hand = OrdinalProbit(thresholds={agree4: (0.2, 0.5, 0.8)})
print(by_hand.thresholds)
again = LatentBlockModel(n_groups=2, n_blocks=2, response_model=by_hand).estimate(survey, seed=0)
print([grid.tolist() for grid in again.value.coclusters.thresholds])
```

```text
Thresholds(axis='levels', grids=(), mode='fixed')
Thresholds(axis='auto', grids=((OrdinalScale(scores=(1, 2, 3, 4)), (0.2, 0.5, 0.8)),), mode='fixed')
[[0.5], [0.2, 0.5, 0.8], [0.2, 0.4, 0.6, 0.8]]
```

The unit axis assumes that two answers that cut the axis at the same place mean the same position: that "rather agree" on four answers sits where "agree" sits on five. It is a convention, not a measurement. Grids given by hand change that convention for the scales they name.

The data can test it: `mode="estimated"` estimates the inner thresholds of each scale of four answers or more, its first and its last held ([The response models: estimated thresholds](model/response-models.md#estimated-thresholds)). On answers that follow the regular grid, the estimate stays near it; on answers that do not, it moves. The integrated likelihood counts each inner threshold in its penalty ([The response models: the penalty of the parameters](model/response-models.md#the-penalty-of-the-parameters)):

```python
estimated = OrdinalProbit(thresholds=Thresholds(mode="estimated"))
fitted = LatentBlockModel(n_groups=2, n_blocks=2, response_model=estimated).estimate(survey, seed=0)
grids = fitted.value.coclusters.thresholds
print([[round(threshold, 2) for threshold in grid.tolist()] for grid in grids])
```

```text
[[0.5], [0.25, 0.51, 0.75], [0.2, 0.41, 0.58, 0.8]]
```

The votes of the survey above were drawn on the regular grids: the inner threshold of the scale of four answers stays near $\tfrac{1}{2}$, and those of the scale of five near $0.4$ and $0.6$; yes or no has none.

## What does not change

A matrix whose items share one scale is computed as before: same fingerprint, same counts, same arrangement and same numbers, bit for bit. The tables of the ordinal probit add two fields to their dictionary, `thresholds` and `has_fixed_spread`.

## Limits

- Each item has one scale, declared with the votes; the engine does not infer it.
- The grids are fixed by default, and estimated only inside each scale: its ends are held, and a scale of fewer than four answers has no inner threshold to estimate.
- An item whose answers are not ordered, or a number to be cut into classes, is prepared before it enters the matrix.
