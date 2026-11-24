# Référence de l'API

Every public name of the engine, with its definition. These pages are generated from the docstrings of the code: they describe the version shown at the top of the site and cannot drift from it.

- Each object is documented once, on the page of the module that defines it.
- The page of a package lists the names the package exports, each linked to its definition.
- The **[source]** link next to a name opens the code.
- An underscore method shown here, such as `Estimator._compute`, is a hook: a method that the author of an estimator writes, and that a user never calls.

| Package | Holds | Needs |
| --- | --- | --- |
| [`civiqhub_selene`](civiqhub_selene.md) | The contract every method shares | The standard library |
| [`civiqhub_selene.ordinal`](civiqhub_selene.ordinal.md) | The ordinal reading of the votes on one item | The standard library |
| [`civiqhub_selene.blocks`](civiqhub_selene.blocks.md) | Co-clustering of voters and items | NumPy, behind the `blocks` extra |
| [`civiqhub_selene.viz`](civiqhub_selene.viz.md) | Graphs and figures of co-clustering results | NetworkX and Matplotlib, behind the `viz` extra |

To find a name from what you want to do, start from [Finding your way](../library-map.md).

```{toctree}
:hidden:
:maxdepth: 2

civiqhub_selene
civiqhub_selene.ordinal
civiqhub_selene.blocks
civiqhub_selene.viz
```
