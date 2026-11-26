# The starts

Checked against the code of `develop` at commit `c29d115`.

## In plain words

A block model climbs from a starting arrangement to the nearest maximum of its objective: where it starts decides where it ends. The engine ships three ways to start. {py:class}`~civiqhub_selene.blocks.RandomStart`, the default, deals the voters and the items at random. {py:class}`~civiqhub_selene.blocks.SpectralStart` reads the shape of the votes first: it places voters and items in a few dimensions where those who answer alike point the same way, and groups them there. {py:class}`~civiqhub_selene.blocks.FixedStart` starts from an arrangement given by the caller, such as the one a single-membership model found.

## Model

### What a start gives

A start object is a setting of the model, {py:attr}`~civiqhub_selene.blocks.base.BlockModel.initialization`. It follows the protocol {py:class}`~civiqhub_selene.blocks.starts.Initialization`: given the votes, the shape $(Q, L)$ and the random generator of the start, {py:meth}`~civiqhub_selene.blocks.starts.Initialization.arrangement` returns a group $R_i$ for each voter and a block $C_j$ for each item, with no empty group and no empty block. The latent block model starts from that arrangement; the mixed-membership model turns it into memberships ([The mixed-membership model](mixed-membership.md)). Being a frozen dataclass, a start object enters the provenance of the result with its settings.

### The random start

The voters are shuffled and dealt in turn to the groups: with $\psi$ a random permutation of the voters, drawn with the generator of the start,

$$
R_i = \psi(i) \bmod Q,
$$

counting from $0$ as the code does. The items are dealt to the blocks in the same way, with a second permutation drawn after the first. The groups differ in size by one at most, and none is empty. Each start of a model draws its own permutations ([The estimation](estimation.md)).

### The spectral start

{py:class}`~civiqhub_selene.blocks.SpectralStart` is regularised spectral co-clustering: DI-SIM (Rohe, Qin and Yu 2016) for a matrix of votes, with the regulariser and the projection onto the sphere of regularised spectral clustering (Qin and Rohe 2013).

**The matrix.** The votes are written as a matrix $\mathbf{X}$ with nonnegative entries, or centred entries, by its {py:attr}`~civiqhub_selene.blocks.SpectralStart.encoding`:

- `"states"`, the default: one row per voter and one column per pair (item, state), with $\mathbf{X}_{i,(j,s)} = S_{ijs}$. Two rows share an entry when the two voters are in the same state on the same item, so that $\mathbf{X}\mathbf{X}^{\top}$ counts the items on which two voters gave the same answer, abstentions included. The degree of a voter is its number of shown cells. The same is done with the transposed array for the items.
- `"scores"`: one matrix of centred levels, $\mathbf{X}_{ij} = s_{ij} - (K + 1)/2$ on a voted cell and $0$ elsewhere, so that a neutral vote, an abstention and a cell that is not shown all read $0$. Its degrees are the numbers of votes of each row and column. The left singular vectors place the voters and the right ones the items.

**Items on several scales.** A level means another answer from one scale to the next ([The data: items on different scales](data.md#items-on-different-scales)), so both encodings first place the votes of every scale on the unit axis of {py:class}`~civiqhub_selene.blocks.responses.Thresholds`, where the level $k$ of the scale $g$ covers the interval $\big(\tfrac{k-1}{K_g}, \tfrac{k}{K_g}\big]$ ([The response models: items on several scales](response-models.md#items-on-several-scales), {eq}`eq-response-axes`), and $K$ is the number of levels of the longest scale. Under `"states"`, the axis is cut into $K$ equal bins, and a vote is spread over the bins its interval covers, in proportion to the length it covers in each; an abstention keeps its own column. For the item $j$ of the scale $g_j$ and the bin $b = 1, \dots, K$:

$$
\mathbf{X}_{i,(j,0)} = S_{ij0}, \qquad
\mathbf{X}_{i,(j,b)} = \sum_{k=1}^{K_{g_j}} S_{ijk} \; K_{g_j} \, \Big\lvert \Big(\tfrac{k-1}{K_{g_j}}, \tfrac{k}{K_{g_j}}\Big] \cap \Big(\tfrac{b-1}{K}, \tfrac{b}{K}\Big] \Big\rvert,
$$

where $\lvert \cdot \rvert$ is the length of an interval: the weights of a vote sum to $1$, so that the degree of a voter is still its number of shown cells. Under `"scores"`, the centred level of a vote is stretched to the levels of the longest scale, $\mathbf{X}_{ij} = \big(s_{ij} - (K_{g_j} + 1)/2\big) \, K / K_{g_j}$: $K$ times the distance from $\tfrac{1}{2}$ to the centre of its interval. With one scale, both encodings are those above.

**The regularised Laplacian.** With $\mathbf{D}_{\mathrm{r}}$ and $\mathbf{D}_{\mathrm{c}}$ the diagonal matrices of the degrees of the rows and of the columns, and $\mathbf{I}$ the identity,

$$
\mathbf{X}_\tau = (\mathbf{D}_{\mathrm{r}} + \tau_{\mathrm{r}} \mathbf{I})^{-1/2} \, \mathbf{X} \, (\mathbf{D}_{\mathrm{c}} + \tau_{\mathrm{c}} \mathbf{I})^{-1/2},
$$ (eq-starts-laplacian)

where each regulariser is $\tau$ ({py:attr}`~civiqhub_selene.blocks.SpectralStart.regularization`, $1$ by default) times the average degree of its side: $\tau_{\mathrm{r}} = \tau \, \bar{d}_{\mathrm{r}}$ and $\tau_{\mathrm{c}} = \tau \, \bar{d}_{\mathrm{c}}$. $\tau = 0$ is the Laplacian of Dhillon (2001). A degree of $0$ with $\tau = 0$ gives a row or a column of zeros.

**The embedding.** The singular value decomposition of $\mathbf{X}_\tau$ gives singular values $\delta_1 \geq \delta_2 \geq \dots$ and singular vectors. Each voter keeps the first $Q$ coordinates of its row of left singular vectors, $\mathbf{U}_{i\cdot}$, and each item the first $L$ of its row $\mathbf{V}_{j\cdot}$, or every coordinate when there are fewer ({py:func}`~civiqhub_selene.blocks.spectral.spectral_embedding`).

**The groups.** On each side, k-means groups the rows ({py:meth}`~civiqhub_selene.blocks.spectral.SpectralEmbedding.labels`):

1. each row is divided by its length, which projects it onto the unit sphere ({py:attr}`~civiqhub_selene.blocks.SpectralStart.project`, true by default); a row of zeros stays $0$;
2. k-means reads only the rows whose length before projection, their leverage, is at least $\gamma / \sqrt{M}$ for the voters and $\gamma / \sqrt{P}$ for the items, with $\gamma$ = {py:attr}`~civiqhub_selene.blocks.SpectralStart.leverage_threshold`: the thresholded spectral clustering t-RSC of Qin and Rohe (2013). With $\gamma = 0$, the default, every row is read; when fewer than $Q$ rows pass, every row is read;
3. k-means is seeded by k-means++ and refined by Lloyd's algorithm, {py:attr}`~civiqhub_selene.blocks.SpectralStart.n_init` times ($10$ by default); the run with the smallest sum of squared distances to the centroids is kept;
4. every row, read or not, takes the cluster of its nearest centroid; if a cluster is then empty, the row farthest from its centroid, in a cluster of more than one row, moves to it. No cluster is empty.

### Why the projection, and why the states

Under a degree-corrected block model, the rows of one group of $\mathbf{U}$ share a direction and differ in length only, while the rows of two groups are orthogonal (Qin and Rohe 2013, Lemma 3.3, as {py:class}`~civiqhub_selene.blocks.spectral.SpectralEmbedding` cites it). The length of a row grows with how much the voter took part; projecting the rows onto the sphere removes it, and gathers each group on one point.

The encoding `"states"` sees a group that splits where another agrees: two groups with the same mean on a block, one unanimous and one divided, fill different columns of $\mathbf{X}$. Under the latent block model, the expectation of $\mathbf{X}$ is the indicator matrix of the groups times a matrix of the co-clusters, whose rank can reach $Q$ even when $L$ is smaller. The encoding `"scores"` reads one number per cell, centred: it is cheaper, and blind to a difference of spread at equal mean.

### A reading of the number of groups

{py:attr}`~civiqhub_selene.blocks.spectral.SpectralEmbedding.voter_singular_values` lists every singular value on the side of the voters, in decreasing order. A clear gap after the $k$-th value suggests $k$ groups. It is a reading of the matrix, not a rule: the engine chooses no shape from it.

### The fixed start

{py:class}`~civiqhub_selene.blocks.FixedStart` holds a group for each voter and a block for each item, the same for every start; it checks that they follow the matrix and leave no cluster empty. {py:meth}`~civiqhub_selene.blocks.FixedStart.from_coclustering` takes the main group and the main block of a value, and its {py:attr}`~civiqhub_selene.blocks.FixedStart.source` keeps the fingerprint of the computation they come from. It is the way to start a mixed-membership model from the result of a latent block model. The labels are settings: the provenance of the result says where it started. Several starts from a fixed arrangement differ only by the starting parameters of the response model.

## An example

The spectral start finds the planted groups and blocks of a synthetic matrix before any iteration of a model; a random start does not:

```python
import numpy as np
from civiqhub_selene.blocks import Partition, RandomStart, SpectralStart
from civiqhub_selene.blocks.spectral import spectral_embedding
from civiqhub_selene.blocks.synthetic import make_votes, single_memberships

votes, truth = make_votes(single_memberships((20, 20, 20)), single_memberships((6, 6)), seed=3)
planted_groups = Partition.from_voter_labels(votes.voters, truth.voter_memberships.argmax(axis=1))
planted_blocks = Partition.from_item_labels(votes.items, truth.item_memberships.argmax(axis=1))
print(SpectralStart())

groups, blocks = SpectralStart().arrangement(votes, (3, 2), np.random.default_rng(0))
print(round(Partition.from_voter_labels(votes.voters, groups).agreement_with(planted_groups), 3))
print(Partition.from_item_labels(votes.items, blocks).agreement_with(planted_blocks))

groups, blocks = RandomStart().arrangement(votes, (3, 2), np.random.default_rng(0))
print(round(Partition.from_voter_labels(votes.voters, groups).agreement_with(planted_groups), 3))

embedding = spectral_embedding(votes, n_voter_components=3, n_item_components=2)
print(np.round(embedding.voter_singular_values[:5], 3).tolist())
```

```text
SpectralStart(encoding='states', regularization=1.0, project=True, leverage_threshold=0.0, n_init=10)
0.967
1.0
0.417
[0.532, 0.402, 0.275, 0.227, 0.226]
```

The spectral arrangement agrees with the planted groups for 58 voters out of 60 and finds the blocks; the random one is near chance. The singular values drop after the third, the number of planted groups.

## What the code does exactly

| Formula or rule | Code |
| --- | --- |
| Protocol | `src/civiqhub_selene/blocks/starts.py:45-63` |
| Random start | `blocks/starts.py:66-84` |
| Spectral start, settings and their defaults | `blocks/starts.py:87-176` |
| Fixed start, `from_coclustering`, checks | `blocks/starts.py:179-254` |
| The matrix of each encoding | `_decomposition`, `blocks/spectral.py:327-369` |
| Votes of several scales on the unit axis, the stretch of `"scores"` | `_aligned_states`, `blocks/spectral.py:372-396`; `_decomposition`, `:349-356` |
| {eq}`eq-starts-laplacian` | `_laplacian`, `blocks/spectral.py:399-413`; `_inverse_root`, `:416-420` |
| Number of components | `n_singular_values`, `blocks/spectral.py:309-321`; `SpectralStart.arrangement`, `blocks/starts.py:151-176` |
| Projection, leverage, k-means++, Lloyd, empty clusters | `SpectralEmbedding.labels`, `blocks/spectral.py:141-197`; `:432-554` |

The code also does what the formulas do not say:

- **Cache.** The decomposition does not depend on the start: it is computed once for a matrix, an encoding and a regulariser, and kept in a small cache of four entries, keyed by the fingerprint of the matrix. Only the seeding of k-means changes from one start to the next, so the spectral starts of a model are close to one another.
- **Lloyd's algorithm** stops when no label changes, or after 100 passes; an empty cluster keeps its centroid during the passes.
- **Survey weights** are not read by the spectral start.

## Limits

- A spectral start is a start, not a result: the model still climbs from it, and can end elsewhere.
- The decomposition is dense: its cost grows with $M \cdot P \cdot (K + 1)$ under `"states"`, $K$ the levels of the longest scale.
- The gap in the singular values is a reading, not a criterion for the shape.

## References

- Qin, T. and Rohe, K. (2013). Regularized spectral clustering under the degree-corrected stochastic blockmodel. *Advances in Neural Information Processing Systems*, 26.
- Rohe, K., Qin, T. and Yu, B. (2016). Co-clustering directed graphs to discover asymmetries and directional communities. *Proceedings of the National Academy of Sciences*, 113(45).
- Dhillon, I. S. (2001), for the Laplacian without regulariser, as the code cites it.
