# The estimation

Checked against the code of `develop` at commit `97a0f15`.

## In plain words

The two block models estimate in different ways, by classification expectation-maximisation for one and by exact expectation-maximisation for the other, but they share one loop. Each model runs it from several starts, keeps the start that reaches the largest objective, and returns that one. The whole computation depends on the votes, the settings and one seed only: the same three give the same result, and the result says which three it came from.

## Model

### The settings every block model shares

{py:class}`~civiqhub_selene.blocks.base.BlockModel` holds the settings of both models, as fields of a frozen dataclass:

| Setting | Meaning | Default |
| --- | --- | --- |
| {py:attr}`~civiqhub_selene.blocks.base.BlockModel.n_groups` | $Q$, the number of groups | $2$ |
| {py:attr}`~civiqhub_selene.blocks.base.BlockModel.n_blocks` | $L$, the number of blocks | $2$ |
| {py:attr}`~civiqhub_selene.blocks.base.BlockModel.response_model` | The response model of the co-clusters | `OrdinalProbit()` |
| {py:attr}`~civiqhub_selene.blocks.base.BlockModel.n_starts` | The number of starts; the best one is kept | $6$ |
| {py:attr}`~civiqhub_selene.blocks.base.BlockModel.max_iterations` | The largest number of iterations of a start | $200$ |
| {py:attr}`~civiqhub_selene.blocks.base.BlockModel.tolerance` | The gain of objective under which a start stops | $10^{-6}$ |
| {py:attr}`~civiqhub_selene.blocks.base.BlockModel.initialization` | What gives the arrangement each start begins from | `RandomStart()` |

The example below prints them from the class itself. {py:class}`~civiqhub_selene.blocks.MixedMembershipBlockModel` adds $\alpha$, $\beta$ and the weight of its start ([The mixed-membership model](mixed-membership.md)).

### Two algorithms, one loop, two objectives

A block model writes three hooks: a start, an iteration and a finish. The loop that runs them is written once, in {py:class}`~civiqhub_selene.blocks.iterative.IterativeEstimator`, and a model cannot redefine it. What an iteration does, and what its objective is, depend on the model:

| Model | Iteration | Objective |
| --- | --- | --- |
| {py:class}`~civiqhub_selene.blocks.LatentBlockModel` | Classification expectation-maximisation: parameters, then voters, then items | $\mathcal{L}_{\mathrm{c}}$, the classified log-likelihood ([The latent block model](latent-block-model.md)) |
| {py:class}`~civiqhub_selene.blocks.MixedMembershipBlockModel` | Exact expectation-maximisation, regularised by Dirichlet priors | $\mathcal{J}$, the log-likelihood plus the log-densities of the priors ([The mixed-membership model](mixed-membership.md)) |

Both report their objective as {py:attr}`~civiqhub_selene.blocks.results.Coclustering.objective`. The two are different quantities: $\mathcal{L}_{\mathrm{c}}$ scores one arrangement with the shares of its groups and blocks, $\mathcal{J}$ sums over every profile and adds the priors. An objective only compares computations of the same method on the same votes; it never compares a latent block model with a mixed-membership model, nor two matrices of votes. Comparing shapes is the subject of [Choosing the shape](choosing-the-shape.md).

## Estimation

### One start

A start begins from the state its hook returns: the arrangement the {py:attr}`~civiqhub_selene.blocks.base.BlockModel.initialization` gives, and the starting parameters of the response model. The loop then repeats the iteration of the model. After each iteration it notes the objective of the new state, and computes the gain, the objective of the new state minus that of the state before:

- when the gain is under {py:attr}`~civiqhub_selene.blocks.base.BlockModel.tolerance`, the start stops and is marked converged; a negative gain stops it as well;
- otherwise, after {py:attr}`~civiqhub_selene.blocks.base.BlockModel.max_iterations` iterations, it stops and is not marked converged.

The value carries the record of the start that was kept as {py:attr}`~civiqhub_selene.blocks.results.Coclustering.history` (`_numerics.History`, internal): the objective noted after each iteration (the starting state is not in it), the number of iterations, and whether the start converged. This loop is the only convergence loop of the package; the inner loop of the ordinal probit runs through it as well ([The response models](response-models.md)).

### Several starts, and the one kept

The models of the engine reach a local maximum of their objective, which depends on where they start. {py:attr}`~civiqhub_selene.blocks.base.BlockModel.n_starts` starts are run, each with its own random generator, and the start whose final objective is the largest is kept; on a tie, the first one. An objective that is not a number never wins. Only the start that is kept reaches the finish hook, which builds the value. More starts make a better maximum more likely, at a cost that grows linearly with their number; a spectral start puts every start near the same arrangement.

## Determinism and provenance

### The seed

{py:meth}`~civiqhub_selene.estimator.Estimator.estimate` takes the votes and a seed, a non-negative integer, $0$ by default. Every random draw of the computation comes from it: the seed is split into one independent stream per start by `numpy.random.SeedSequence(seed).spawn(n_starts)`, and each start draws from a generator `numpy.random.default_rng` of its stream (PCG64). Nothing is read from the environment. With the same votes, the same settings and the same seed, the computation is the same, and so is its result.

### What a result says about itself

The {py:class}`~civiqhub_selene.types.Result` holds the value and a {py:class}`~civiqhub_selene.types.Provenance`: the method (name, version, reference), the version of the engine, the settings in effect, the fingerprint of the votes, the versions of Python and NumPy, the generator and the seed, and the lineage of the votes. Two fingerprints are at hand:

- the fingerprint of the computation, {py:meth}`~civiqhub_selene.types.Provenance.fingerprint`, which the value also carries as {py:attr}`~civiqhub_selene.blocks.results.Coclustering.computation`. It is a SHA-256 of the method, its version, the settings, the generator and the seed, the fingerprint of the votes and the lineage. It leaves out the versions of the engine and of the dependencies: the same request on a newer NumPy is the same request;
- the fingerprint of what was found, {py:meth}`~civiqhub_selene.blocks.results.Coclustering.fingerprint`: a SHA-256 of the voters, the items, the scores, the arrangement, the memberships, and the vote rates and shares of the co-clusters. It leaves out the objective and the history.

Before computing, the estimator asks whether a result exists on the votes ([The data](data.md)); when none does, the result carries an explained absence and the same provenance.

## An example

```python
from civiqhub_selene.blocks import LatentBlockModel
from civiqhub_selene.blocks.synthetic import make_votes, single_memberships

print(LatentBlockModel())

votes, _ = make_votes(single_memberships((20, 20, 20)), single_memberships((6, 6)), seed=3)
model = LatentBlockModel(n_groups=3, n_blocks=2)
first = model.estimate(votes, seed=0)
again = model.estimate(votes, seed=0)
other = model.estimate(votes, seed=1)

print(first.provenance.fingerprint() == again.provenance.fingerprint())
print(first.value.fingerprint() == again.value.fingerprint())
print(first.value.computation == first.provenance.fingerprint())
print(first.provenance.fingerprint() == other.provenance.fingerprint())
print(first.provenance.randomness, other.provenance.seed)
```

```text
LatentBlockModel(n_groups=2, n_blocks=2, response_model=OrdinalProbit(thresholds=None, min_spread=0.3, min_probability=1e-06, max_inner_iterations=10), n_starts=6, max_iterations=200, tolerance=1e-06, initialization=RandomStart())
True
True
True
False
Randomness(generator='PCG64', seed='0') 1
```

The same seed gives the same computation and the same value; another seed is another computation.

## What the code does exactly

| Rule | Code |
| --- | --- |
| Settings and their defaults | `src/civiqhub_selene/blocks/base.py:165-202` |
| One stream per start, best start kept, first on a tie | `IterativeEstimator._compute`, `blocks/iterative.py:82-98`; `_comparable`, `:40-43` |
| The loop and its stop rule | `iterate`, `blocks/_numerics.py:199-227`; `History`, `:181-196` (internal) |
| A model cannot redefine the loop | `blocks/iterative.py:74-80` |
| `estimate`: checks, provenance, absence, value | `src/civiqhub_selene/estimator.py:146-179` |
| Provenance and its fingerprint | `estimator.py:181-197`; `src/civiqhub_selene/types.py:203-252` |
| Fingerprint of a value | `blocks/results.py:320-341` |
| What an objective compares | Docstring of `Coclustering`, `blocks/results.py:192-193` |

## Limits

- The maximum reached is local: more starts make a better one more likely, and nothing guarantees the global maximum.
- A start can stop at the limit of iterations without converging; the history of the value says so.
- The result carries no uncertainty on the arrangement or the parameters.
