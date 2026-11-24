# Examples

The folder [`examples/`](https://github.com/civiqhub/civiqhub-selene/tree/develop/examples) of the repository holds notebooks that run SELENE from end to end: they load votes from a CSV file, start a block model, run a full inference and draw the result. They are saved with their outputs, so GitHub shows them without running them. The data are synthetic.

| Notebook | What it shows |
| --- | --- |
| [`01_spectral_start.ipynb`](https://github.com/civiqhub/civiqhub-selene/blob/develop/examples/01_spectral_start.ipynb) | Load a vote matrix. Compute the regularised spectral embedding (DI-SIM, RSC); read its spectrum, its "star" and its projection onto the sphere; find groups and blocks before any model; see why the votes are encoded as states rather than as scores |
| [`02_olbm.ipynb`](https://github.com/civiqhub/civiqhub-selene/blob/develop/examples/02_olbm.ipynb) | One group per voter, with {py:class}`~civiqhub_selene.blocks.single.LatentBlockModel`: choose the shape with the integrated likelihood, compare spectral and random starts, read the result and its provenance, draw the co-clusters, the sorted matrix and the agreement between groups, use the NetworkX graphs as data |
| [`03_mm_olbm.ipynb`](https://github.com/civiqhub/civiqhub-selene/blob/develop/examples/03_mm_olbm.ipynb) | Mixed memberships, with {py:class}`~civiqhub_selene.blocks.mixed.MixedMembershipBlockModel`: start from the spectral arrangement or from the single-membership result, read and draw the memberships, the roles from block to block and the changes of role, compare both models on held-out votes |

## What the notebooks find

On the synthetic collectives of the examples, twenty single starts of each kind give:

| Model, collective | Random starts that reach the best objective | Spectral starts that reach it |
| --- | --- | --- |
| One group per voter, `consultation` (240 voters, 24 items) | 85%, in 12 iterations on average | 100%, in 6 iterations |
| One group per voter, `assembly` (150 voters, 20 items) | 35% | 100% |
| Mixed memberships, `assembly` | 50% | 100% |

These are measures on two synthetic collectives, not a property of the methods: the notebooks show how to measure them on other votes.

## Run them

```bash
python -m pip install -e ".[viz]" jupyterlab
cd examples
jupyter lab
```

Every code cell of every notebook is run by `tests/unit/test_examples.py`. The [`README` of the folder](https://github.com/civiqhub/civiqhub-selene/blob/develop/examples/README.md) says how the data are drawn and how to save a notebook after a change.
