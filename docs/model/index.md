# Mathematical model

This section writes down the mathematics of the engine as the code computes it: the data a block model reads, the models of co-clustering, how they are estimated and started, how a shape is compared to another, and what a result says. It is written for whoever wants to check a result, review a method or extend one: each formula comes with the name the code gives each symbol, and with the file that computes it.

The guide says how to use the library; the reference says what each name does. This section says why the numbers are what they are.

## The code is the authority

Each page of the section states the commit of the code against which its formulas were checked, and cites the file and the lines that compute each of them. Where a page and the code disagree, the code says what the library does: the page is wrong, and the difference is reported as an issue of the repository.

Every symbol is defined once, on the page [Notation](notation.md), and keeps that meaning on every page.

## What the engine does not do

A result describes a group of voters; it decides nothing. It carries its value, or an absence that says why there is none, and the provenance that lets anyone compute it again.

The section describes what the code holds, and nothing else. As of the commit each page states:

- the co-clustering models are estimated by classification expectation-maximisation (one group per voter) and by exact expectation-maximisation (mixed memberships); no variational method is part of the engine;
- no function chooses the numbers of groups and of blocks: a shape is compared to another by the quantities each result reports;
- a co-clustering result carries no uncertainty: no standard error, no interval;
- the ordinal family holds the counts of the votes of one item ({py:class}`~civiqhub_selene.ordinal.VoteCounts`) only; the dissensus index and the collective regime are defined in the design notes of the engine and are not implemented.

## Reading order

The pages build on one another, in this order. Those that are not written yet are listed without a link.

1. [Notation](notation.md): the letters of the section, the names of the code they stand for, and the conventions.
2. [The data](data.md): the matrix of votes, its three states per cell, the scale, and the counts the models read.
3. [The response models](response-models.md): what a co-cluster says about the cells it holds.
4. [The latent block model](latent-block-model.md): one group per voter, one block per item.
5. [The mixed-membership model](mixed-membership.md): a voter follows different groups on different blocks.
6. [The estimation](estimation.md): the loop every block model shares, its convergence, its starts and its determinism.
7. [The starts](initialization.md): random, spectral and fixed.
8. [Choosing the shape](choosing-the-shape.md): the integrated completed likelihood and the likelihood of held-out votes.
9. [Reading the results](reading-results.md): memberships, roles, agreement between two results, what is identifiable.
10. [Cost and numerical safeguards](complexity.md).
11. From the API to the formulas: each public object and the formulas it computes (not written yet).
12. Glossary and references (not written yet).
13. The ordinal family (not written yet).
