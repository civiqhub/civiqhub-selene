# SELENE

SELENE is the open-source scientific engine of [CiviqHub](https://civiqhub.com) for analysing collective deliberation. `civiqhub-selene` is its Python library: deterministic computations on explicit inputs, with no network access and no dependency on CiviqHub services.

A result describes a group and decides nothing. Every result carries its value or an explained absence, and the provenance that lets anyone compute it again.

## What the library holds

| Part | Holds | Needs |
| --- | --- | --- |
| [`civiqhub_selene`](reference/civiqhub_selene.md) | The contract every method shares: errors, results with their provenance, the base class of estimators, the registry, the contract kit | The standard library |
| [`civiqhub_selene.ordinal`](reference/civiqhub_selene.ordinal.md) | The ordinal reading of the votes on one item | The standard library |
| [`civiqhub_selene.blocks`](reference/civiqhub_selene.blocks.md) | Co-clustering: groups of voters and blocks of items that vote alike | NumPy, behind the `blocks` extra |

## Where to go

| You want to | Read |
| --- | --- |
| Install the library and run a first computation | [Getting started](getting-started.md) |
| Know which module holds what | [Finding your way](library-map.md) |
| Understand how the engine is built, or add a method | [Architecture](architecture.md) |
| Look up a class, a function or an error | [Reference](reference/index.md) |
| See what changed between versions | [Changelog](changelog.md) |

```{toctree}
:hidden:
:caption: Guide

getting-started
library-map
architecture
```

```{toctree}
:hidden:
:caption: Reference

reference/index
changelog
```

```{toctree}
:hidden:
:caption: Project

Source code <https://github.com/civiqhub/civiqhub-selene>
Contributing <https://github.com/civiqhub/civiqhub-selene/blob/develop/CONTRIBUTING.md>
License <https://github.com/civiqhub/civiqhub-selene/blob/develop/LICENSE>
```
