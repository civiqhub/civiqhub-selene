# Examples

Notebooks that run SELENE from end to end, on synthetic data only. Each one loads votes
from a CSV file, starts a block model, runs a full inference and draws the result.

| Notebook | What it shows |
| --- | --- |
| [`01_spectral_start.ipynb`](01_spectral_start.ipynb) | Load a vote matrix; compute the regularised spectral embedding (DI-SIM, RSC); read its spectrum, its "star" and its projection; find groups and blocks before any model; compare the encodings of the votes |
| [`02_olbm.ipynb`](02_olbm.ipynb) | Co-clustering with one group per voter (`LatentBlockModel`): choose the shape with the integrated likelihood, compare spectral and random starts, read the result and its provenance, draw the co-clusters, the sorted matrix, the agreement between groups, use the NetworkX graphs as data |
| [`03_mm_olbm.ipynb`](03_mm_olbm.ipynb) | Mixed memberships (`MixedMembershipBlockModel`): start from the spectral arrangement or from the single-membership result, read and draw the memberships, the roles from block to block and the changes of role, compare both models on held-out votes |
| [`04_olbm_vs_mm_olbm.ipynb`](04_olbm_vs_mm_olbm.ipynb) | One pipeline for both models, compared on the same footing: main arrangement, members planted in two groups, roles, held-out votes; on a collective with shared members, where the MM-OLBM predicts better, and on one without, where the OLBM does |

The notebooks are saved with their outputs, so they can be read on GitHub without
running them.

## Scripts

Two command-line scripts frame a co-clustering of survey data: one prepares the vote
matrices, the other reads the groups with the description of the respondents. They need
pandas; the second also needs SciPy and Matplotlib. With `--demo`, both run on synthetic
data drawn by the script itself.

| Script | What it does |
| --- | --- |
| [`preprocess_ordinal_survey.py`](preprocess_ordinal_survey.py) | Split a survey table into one metadata base and one vote base per scale size (2 to 7 levels by default), joinable on a respondent identifier; encode each cell in the three states of a `VoteMatrix` (vote, abstention, not shown); build the `VoteMatrix` of each base |
| [`profile_clusters.py`](profile_clusters.py) | Describe the groups of a co-clustering with the metadata of the respondents, for single or mixed memberships, with optional survey weights: test values of Lebart, Morineau and Piron, identity cards, heatmap, stacked bars, quantile boxes, association ranking, correspondence analysis |

```bash
python examples/preprocess_ordinal_survey.py --demo --out demo_bases
python examples/profile_clusters.py --demo --soft --out demo_profile
```

Run `--help` on each script for its options. Keep real survey files outside the
repository and pass their path on the command line.

## Run them

From the root of the repository:

```bash
python -m pip install -e ".[viz]" jupyterlab
```

```bash
cd examples
jupyter lab
```

Open a notebook and run it from the top: the paths of the data are relative to this
folder. To run them all from the command line and save them with their outputs:

```bash
python -m pip install -e ".[viz]" --group examples
```

```bash
python examples/run_notebooks.py
```

The test `tests/unit/test_examples.py` runs every code cell of every notebook, in order,
in the CI.

## Data

The files of [`data/`](data) are synthetic: [`data/make_data.py`](data/make_data.py) draws
them with `make_votes` from a planted truth, and writes them again, identical, when run
from the root of the repository. No identifier resembles real data.

| File | Content |
| --- | --- |
| `<name>.votes.csv` | One row per shown cell: `voter,item,score`. An empty score is a cell shown without a vote; a pair without a row was not shown |
| `<name>.voters.csv` | The planted membership of each voter to each group |
| `<name>.items.csv` | The planted membership of each item to each block |

| Collective | Voters | Items | Planted |
| --- | --- | --- | --- |
| `consultation` | 240, in 3 groups | 24, in 3 blocks | Each voter sees 35% to 100% of the items. On the third block, groups 0 and 1 share the same mean opinion: one agrees, the other is split |
| `assembly` | 150, of which 30 share their membership between two groups | 20, of which 4 share theirs between the two blocks | Each cell is shown with probability 0.9 |

[`loading.py`](loading.py) reads these files into a `VoteMatrix`. The engine itself reads
no file: build the matrix of your own votes the same way, with
`VoteMatrix.from_records`, and keep the three states of a cell.

Never commit data from real participants here, even anonymised: see
[`CONTRIBUTING.md`](../CONTRIBUTING.md#data).

## Change a notebook

1. Edit the notebook, then run it from the top with `python examples/run_notebooks.py`,
   so that it is saved with its outputs and without the timestamps of execution.
2. Run `ruff check examples` and `ruff format examples`: they lint and format the code
   cells too.
3. Run `pytest tests/unit/test_examples.py`.
