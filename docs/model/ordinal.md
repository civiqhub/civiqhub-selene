# The ordinal family

Checked against the code of `develop` at commit `c29d115`.

## In plain words

The ordinal family reads the votes on one item, or on one co-cluster, as five numbers: how many votes each score of the agreement scale received. It needs the standard library only. What it computes from these numbers are exact fractions, so that an equality written on paper is an equality in the code. The counts hold neither the identity of the voters nor the abstentions.

## Model

### The five counts

The family has one scale of five scores, {py:data}`~civiqhub_selene.ordinal.SCORES`: from $-2$ (strongly disagree) to $+2$ (strongly agree), $0$ being the neutral answer. As on the other pages, the level $k$ of a score is its rank, from $1$ to $5$, and the score of the level $k$ is $k - 3$.

{py:class}`~civiqhub_selene.ordinal.VoteCounts` holds the numbers of votes at each level, $N_1, \dots, N_5$ ({py:attr}`~civiqhub_selene.ordinal.VoteCounts.counts`), and their sum, the number of votes $n$ ({py:attr}`~civiqhub_selene.ordinal.VoteCounts.total`):

$$
n = \sum_{k=1}^{5} N_k.
$$

Any five integers that are not negative are counts, five zeros included; a boolean or a float is refused. The {py:attr}`~civiqhub_selene.ordinal.VoteCounts.name` says what was counted; the engine never interprets it.

### The shares

When $n \geq 1$, the share of the votes at the level $k$ is

$$
f_k = \frac{N_k}{n}, \qquad k = 1, \dots, 5,
$$ (eq-ordinal-shares)

and the share of the votes at the level $k$ or below is

$$
f_{\leq k} = \sum_{h=1}^{k} f_h = \frac{1}{n} \sum_{h=1}^{k} N_h, \qquad k = 1, \dots, 4.
$$ (eq-ordinal-cumulative)

{py:attr}`~civiqhub_selene.ordinal.VoteCounts.shares` returns the five $f_k$ and {py:attr}`~civiqhub_selene.ordinal.VoteCounts.cumulative_shares` the four $f_{\leq k}$; the fifth would always be $1$. Both are exact fractions (`fractions.Fraction`): the shares sum to exactly $1$, and two shares that are equal on paper compare equal in the code.

Without any vote, $n = 0$, the shares are not defined. Both properties then raise {py:class}`~civiqhub_selene.errors.InvalidInputError` rather than return a value: the absence of votes is not a distribution.

### Changing the counts

The counts never change in place. {py:meth}`~civiqhub_selene.ordinal.VoteCounts.with_vote` returns new counts with one more vote at a score, and {py:meth}`~civiqhub_selene.ordinal.VoteCounts.without_vote` with one vote fewer, refused when there is no vote at that score: one $N_k$ changes by one, and the shares follow from the new counts. {py:meth}`~civiqhub_selene.ordinal.VoteCounts.from_scores` counts a series of scores. Indexing by a score, such as `votes[2]`, reads one count. {py:meth}`~civiqhub_selene.ordinal.VoteCounts.fingerprint` identifies the counts and their name.

### The votes of a co-cluster

{py:meth}`~civiqhub_selene.blocks.GroupedVotes.vote_counts` turns the votes of the co-cluster $(q, l)$ into five counts: $N_k = N_{qlk}$, counted without survey weights, as {eq}`eq-data-counts` writes them ([The data](data.md#the-votes-of-a-group-on-a-block)). The matrix must have one scale, of five scores: a matrix whose items are on several scales is refused with {py:class}`~civiqhub_selene.errors.InvalidInputError`, as every reading of one scale ([The data: items on different scales](data.md#items-on-different-scales)). The counts follow the ranks: the count of the $k$-th score of the scale of the matrix becomes the count of the score $k - 3$ of {py:data}`~civiqhub_selene.ordinal.SCORES`, whatever the scores of the matrix are.

The counts leave out the abstentions of the co-cluster, $N_{ql0}$: they know nothing of its vote rate $\nu_{ql}$. They are named after the co-cluster, with the indices of the code, which start at $0$.

## An example

Counts written by hand, for one item:

```python
from fractions import Fraction

from civiqhub_selene.errors import InvalidInputError
from civiqhub_selene.ordinal import SCORES, VoteCounts

votes = VoteCounts((3, 1, 0, 4, 2), name="item 7")
print(SCORES, votes.total, votes[1])
print([str(share) for share in votes.shares])
print([str(share) for share in votes.cumulative_shares])
print(sum(votes.shares) == 1, votes.cumulative_shares[2] == Fraction(2, 5))
print(votes.with_vote(0).counts, VoteCounts.from_scores([2, -2, 2, 0]).counts)
try:
    VoteCounts((0, 0, 0, 0, 0)).shares
except InvalidInputError as error:
    print(error)
```

```text
(-2, -1, 0, 1, 2) 10 4
['3/10', '1/10', '0', '2/5', '1/5']
['3/10', '2/5', '2/5', '4/5']
True True
(3, 1, 1, 4, 2) (1, 0, 1, 0, 2)
shares are undefined without a vote: VoteCounts(0, 0, 0, 0, 0; 0 votes)
```

A matrix written by hand, on a scale of the scores $1$ to $5$, read through two groups of voters and a single block of items. The voter `"cy"` did not vote on `"q2"`: the cell is an abstention.

```python
from civiqhub_selene.blocks import OrdinalScale, Partition, VoteMatrix

matrix = VoteMatrix.from_records(
    [("ana", "q1", 5), ("ana", "q2", 4), ("ben", "q1", 1), ("ben", "q2", 5), ("cy", "q1", 5)],
    scale=OrdinalScale((1, 2, 3, 4, 5)),
)
groups = Partition.from_voter_labels(matrix.voters, [0, 1, 0])
grouped = matrix.group_by(groups)
counts = grouped.vote_counts(0, 0)
print(counts.name, counts.counts, counts[2], counts.total)
print(grouped.counts()[0, 0].tolist())
```

```text
group 0 x block 0 (0, 0, 0, 1, 2) 2 3
[1.0, 0.0, 0.0, 0.0, 1.0, 2.0]
```

The two votes at the score $5$ of the matrix are counted at the score $+2$. The co-cluster has four shown cells and one abstention; its five counts hold its three votes only.

## What the code does exactly

| Formula or rule | Code |
| --- | --- |
| The five scores | {py:data}`~civiqhub_selene.ordinal.SCORES`, `src/civiqhub_selene/ordinal/counts.py:20` |
| Counts, five integers that are not negative | `VoteCounts.__post_init__`, `ordinal/counts.py:57-78` |
| $n$ | `total`, `ordinal/counts.py:96-99` |
| {eq}`eq-ordinal-shares` | `shares`, `ordinal/counts.py:101-110` |
| {eq}`eq-ordinal-cumulative` | `cumulative_shares`, `ordinal/counts.py:112-126` |
| No share without a vote | `_positive_total`, `ordinal/counts.py:151-155` |
| `from_scores`, `with_vote`, `without_vote`, indexing | `ordinal/counts.py:80-94`, `:128-140`, `:146-149` |
| Fingerprint of the counts and the name | `ordinal/counts.py:142-144` |
| Counts of a co-cluster, by rank, one scale of five scores only | `GroupedVotes.vote_counts`, `src/civiqhub_selene/blocks/data.py:1282-1303` |

## Limits

- The family reads a scale of five scores only.
- The counts hold no abstention and no vote rate; a co-cluster gives them without survey weights.
- Without any vote, the shares are not defined.

## References

The quantities of this page are defined by the code. See [References](references.md) for the works the section cites.
