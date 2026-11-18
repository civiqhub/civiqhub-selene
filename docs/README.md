# Documentation

This folder holds the pages of the documentation site of `civiqhub-selene`. The site is built with [Sphinx](https://www.sphinx-doc.org) from two sources: the Markdown pages of this folder, and the docstrings of the code.

This file explains how the site is organised and how to change it. It is read on GitHub and is not a page of the site.

## Build the site

The build needs Python 3.13 or later.

```bash
python -m pip install -e ".[blocks]" --group docs
python -m sphinx -W -b html docs docs/_build/html
```

Then open `docs/_build/html/index.html`. The option `-W` turns every warning into an error: this is how the CI builds the site, in the `documentation` job.

## What is where

| Path | Content |
| --- | --- |
| `conf.py` | The configuration of Sphinx |
| `_ext/selene_docs.py` | A local Sphinx extension; its docstring says what it does |
| `index.md` | The home page and the table of contents of the site |
| `getting-started.md`, `library-map.md`, `architecture.md` | The guide: pages written by hand |
| `reference/` | The reference: one page per public module, generated from the docstrings |
| `changelog.md` | Includes `CHANGELOG.md` from the root of the repository |
| `_build/` | The output of the build. Not committed |

Sections that are planned and not written yet: the methods (definition, parameters and limits of each method), what the results do not say, how to write an estimator outside this repository, and the runnable examples.

## Five rules

1. **Pages are Markdown.** They are read by [MyST](https://myst-parser.readthedocs.io), and they must stay readable on GitHub.
2. **The reference is never written by hand.** To change what the reference says, change the docstring.
3. **Every public module has a page in the reference.** `tests/unit/test_docs_reference.py` checks it.
4. **Every Python block runs.** `tests/unit/test_docs_examples.py` runs the Python blocks of `README.md` and of every page of this folder, and compares what they print with the output the page shows.
5. **Every reference to a name of the engine resolves.** The build fails on a name that does not exist.

## Add a page to the guide

1. Create `docs/my-page.md`. It starts with one title of level 1.
2. Add `my-page` to a `toctree` of `index.md`. A page that is in no table of contents fails the build.
3. Build the site.

Link to another page with a relative Markdown link: `[Architecture](architecture.md)`. Link to a heading with the anchor GitHub gives it: `[the contract](architecture.md#the-contract-of-a-method)`.

## Refer to a name of the engine

Use a role, so that the name becomes a link to the reference and the build checks that it exists:

````markdown
{py:class}`~civiqhub_selene.blocks.VoteMatrix` holds the votes.
Call {py:meth}`~civiqhub_selene.estimator.Estimator.estimate`.
````

| Role | For |
| --- | --- |
| `{py:class}` | A class or an error |
| `{py:func}` | A function |
| `{py:meth}` | A method |
| `{py:attr}` | An attribute or a property |
| `{py:data}` | A constant |
| `{py:mod}` | A module |
| `{py:obj}` | Any of them |

The leading `~` shows the last part of the name only. Both import paths work: the package that exports the name (`civiqhub_selene.blocks.VoteMatrix`) and the module that defines it (`civiqhub_selene.blocks.data.VoteMatrix`).

On GitHub a role shows as plain text. This is accepted.

## Write a Python block

````markdown
```python
from civiqhub_selene.ordinal import VoteCounts

print(VoteCounts.from_scores([2, 2, -1]).total)
```

```text
3
```
````

- A block fenced with `python` is run by the tests. The blocks of a page share one namespace and run in the order of the page.
- A `text` block that directly follows a Python block is its output: the test compares it with what the block prints, character for character. Show an output only when it does not depend on the machine or on the version of a dependency.
- To show code that cannot run, such as a skeleton, write the comment `<!-- not run: the reason -->` on the line before the block, followed by an empty line.
- A page whose blocks use NumPy or `civiqhub_selene.blocks` is run in the test jobs that install NumPy only. Nothing has to be declared.
- Examples use synthetic data only: `make_votes`, or values written by hand.
- `ruff format` also formats the Python blocks of the Markdown files.

## Write formulas and diagrams

A formula is written between dollar signs, inline (`$a + b$`) or on its own lines between `$$`. A diagram is a block fenced with `mermaid`; GitHub draws it as well.

The site loads MathJax and Mermaid from the jsDelivr network, on the pages that hold a formula or a diagram only.

## The reference

Each public module has one page, `reference/<dotted name of the module>.md`. The page of a module is three lines:

````markdown
# `civiqhub_selene.blocks.data`

```{eval-rst}
.. automodule:: civiqhub_selene.blocks.data
```
````

It shows the docstring of the module, then every name of its `__all__` with its docstring. The page of a package shows the docstring of the package, the names the package exports (the directive `public-names` reads its `__all__`) and the list of its modules.

When you add a public module:

1. Declare its `__all__`.
2. Create its page: copy the page of a sibling module and change the name on both lines.
3. Add the dotted name of the module to the `toctree` of the page of its package, for example `reference/civiqhub_selene.blocks.md`.

When you add a name to an existing module, add it to `__all__`: the page follows. When you remove or rename a module, remove or rename its page.

A module whose name starts with an underscore is internal and has no page. An underscore method appears in the reference when it carries its own docstring: this is how a hook such as `Estimator._compute` is documented. An override without a docstring does not appear.

## When the build fails

| Message | Cause | What to do |
| --- | --- | --- |
| `py:class reference target not found: civiqhub_selene...` | A docstring or a page names something that does not exist, or that is not public | Fix the name. A name is public when it is in the `__all__` of a public module |
| `py:class reference target not found: decimal.Decimal` (a name of another package) | A signature uses a type of a package that `conf.py` does not know yet | Add the top-level package to the first pattern of `nitpick_ignore_regex` in `conf.py` |
| `py:class reference target not found: Some words` | A `Returns:` section starts with `some words: more words`; the words before the colon are read as a type | Reword the first line of the section without a colon |
| `document isn't included in any toctree` | A page is in no table of contents | Add it to a `toctree` |
| `cross-reference target not found` (`myst.xref_missing`) | A Markdown link points to a page or a heading that does not exist | Fix the link |
| `Unexpected indentation`, `Block quote ends without a blank line` | A docstring is not valid reStructuredText | Put an empty line before and after a list or an indented block; write `::` before a literal block |

The site is built without network access, and it does not link to the documentation of Python or NumPy.
